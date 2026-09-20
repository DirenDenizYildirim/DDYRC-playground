"""Run a soup and log measurements.

    python -m soup.run --mode ring --seed 1 --epochs 50000 --out runs/ring_s1
    python -m soup.run --config configs/bff.json --seed 1 --out runs/bff_s1
"""
import argparse
import csv
import json
import os
import platform
import sys
import time

import numpy as np

from . import compat as compat_mod
from . import core, metrics, replicator
from .analyze import HOE_RISE, SUSTAIN
from .config import add_cli_args, build_config
from .interp import TERM_NAMES
from .rng import fill_random, make_state

CSV_FIELDS = [
    "epoch", "wall_s", "sim_s",
    # order-0 entropy is always reported beside high-order entropy: on its own
    # HOE cannot tell a random soup from a one-byte crystal
    "entropy_bits", "comp_bits", "high_order_entropy",
    "brotli2_bits", "zlib_bits", "hoe_zlib",
    # computation
    "mean_steps", "frac_copy_runs", "frac_steps_in_loop", "ops_per_run",
    "copies_per_run",
    "frac_term_budget", "frac_term_off_region", "frac_term_unmatched",
    "runs", "steps", "mutations",
    # novelty
    "distinct_windows", "n_persistent", "A_t",
    "n_persistent_naive", "A_t_naive",
    # population structure, only meaningful for the tape-based modes
    "distinct_tapes", "distinct_labels",
]


class Soup:
    """Owns the memory, the RNG stream and the epoch loop for one run."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = make_state(cfg.seed)
        if cfg.mode == "ring":
            self.mem = np.zeros(cfg.M, dtype=np.uint8)
        else:
            self.mem = np.zeros((cfg.N, 64), dtype=np.uint8)
            self.perm = np.arange(cfg.N, dtype=np.int64)
        self.buf = np.zeros(cfg.region_len, dtype=np.uint8)
        self.visited = np.zeros(cfg.region_len, dtype=np.int32)
        self.gen = 0
        # an empty label array must keep the rank of the memory it shadows
        off_shape = (0,) + self.mem.shape[1:]
        self.labels = np.zeros(self.mem.shape if cfg.labels else off_shape,
                               dtype=np.uint8)
        self.lab_buf = np.zeros(cfg.region_len if cfg.labels else 0,
                                dtype=np.uint8)
        if cfg.compat != "none":
            self.seed_base = compat_mod.seed_base_for(cfg.seed)
            compat_mod.cubff_init(
                self.mem, compat_mod.seed_of_host(self.seed_base, 0))
            self.mut_prob = int(round(cfg.mu * compat_mod.CUBFF_MUTATION_DENOM))
        else:
            fill_random(self.flat, self.state)
        if cfg.load:
            saved = np.load(cfg.load)
            if saved.shape != self.mem.shape:
                raise SystemExit("checkpoint shape %s does not match this "
                                 "configuration's %s"
                                 % (saved.shape, self.mem.shape))
            self.mem[...] = saved
        if cfg.labels:
            # one lineage id per tape, so label turnover measures coalescence
            per = self.mem.shape[0] if self.mem.ndim > 1 else self.mem.size
            self.labels[...] = np.broadcast_to(
                (np.arange(per) % 251).astype(np.uint8).reshape(
                    (per,) + (1,) * (self.mem.ndim - 1)), self.mem.shape)
        if cfg.plant:
            self._plant(cfg.plant)
        self.stats = np.zeros(core.N_STATS, dtype=np.int64)
        self.mutations = 0
        self.sim_s = 0.0
        self.epoch = cfg.start_epoch

    @property
    def flat(self):
        return self.mem.reshape(-1)

    def _plant(self, n):
        """Overwrite n random sites with the hand-written replicator.

        Validation only: it changes the initial condition, so a planted run is
        not a sample of the process the rest of the results are about.  Sites
        are drawn from a separate numpy generator seeded from cfg.seed, leaving
        the simulation's own stream untouched.
        """
        prog = replicator.as_array()
        picker = np.random.default_rng(self.cfg.seed)
        for _ in range(n):
            if self.cfg.mode == "ring":
                p = int(picker.integers(self.cfg.M))
                self.flat[(p + np.arange(prog.size)) % self.cfg.M] = prog
            else:
                self.mem[int(picker.integers(self.cfg.N)), : prog.size] = prog

    def advance(self, n_epochs):
        """Run n_epochs, accumulating stats.  Returns seconds spent simulating."""
        cfg = self.cfg
        wrap = cfg.head_bound == "wrap"
        allow_copy = not cfg.no_copy
        t0 = time.perf_counter()
        if cfg.compat != "none":
            m, self.gen = compat_mod.cubff_chunk(
                self.mem, self.buf, self.perm, self.epoch, n_epochs,
                self.seed_base, self.mut_prob, cfg.compat == "cubff",
                cfg.k, self.stats, self.visited, self.gen, self.labels,
                self.lab_buf)
        elif cfg.mode == "ring":
            m, self.gen = core.ring_chunk(
                self.mem, self.buf, n_epochs, cfg.ticks_per_epoch, cfg.R,
                cfg.k, cfg.mu, self.state, self.stats, self.visited, self.gen,
                self.labels, self.lab_buf, cfg.head_source == "tape", wrap,
                allow_copy, bool(cfg.indel))
        elif cfg.mode == "blocks":
            m, self.gen = core.blocks_chunk(
                self.mem, self.buf, n_epochs, cfg.ticks_per_epoch, cfg.d,
                cfg.k, cfg.mu, self.state, self.stats, self.visited, self.gen,
                self.labels, self.lab_buf, cfg.head_source == "tape", wrap,
                allow_copy)
        else:
            m, self.gen = core.bff_chunk(
                self.mem, self.buf, self.perm, n_epochs, cfg.k, cfg.mu,
                self.state, self.stats, self.visited, self.gen, self.labels,
                self.lab_buf, wrap, allow_copy)
        dt = time.perf_counter() - t0
        self.sim_s += dt
        self.mutations += int(m)
        self.epoch += n_epochs
        return dt


def truncate_logs(out, start_epoch):
    """Drop every logged row past ``start_epoch``; return the kymo and the
    last surviving row.

    Background work in this environment does not always survive an idle
    period, so a run has to be resumable from its last tape dump without
    leaving a torn or duplicated log behind.  The row *at* the resume point
    is kept, because it carries the counters for the epochs that led up to
    it; the resumed run therefore does not re-measure that epoch, and picks
    up its A(t) from the tracker state saved beside the dump.
    """
    csv_path = os.path.join(out, "metrics.csv")
    if not os.path.exists(csv_path):
        return [], None
    with open(csv_path) as fh:
        kept = [r for r in csv.DictReader(fh)
                if int(r["epoch"]) <= start_epoch]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in kept:
            w.writerow(r)

    pat_path = os.path.join(out, "patterns.log")
    if os.path.exists(pat_path):
        with open(pat_path) as fh:
            lines, keep, blocks = fh.readlines(), True, []
        for line in lines:
            if line.startswith("# epoch "):
                keep = int(line.split()[2]) <= start_epoch
            if keep:
                blocks.append(line)
        with open(pat_path, "w") as fh:
            fh.writelines(blocks)

    kymo_path = os.path.join(out, "kymograph.npy")
    kymo = []
    if os.path.exists(kymo_path):
        old = np.load(kymo_path)
        kymo = [row.copy() for row in old[: len(kept)]]
    return kymo, (kept[-1] if kept else None)


def measure(soup, trackers):
    """One metrics row.  Consumes and resets the interpreter counters."""
    cfg = soup.cfg
    flat = soup.flat
    h = metrics.shannon_entropy_bits(flat)
    c = metrics.compressed_bits_per_byte(flat, cfg.compressor)
    c2 = metrics.compressed_bits_per_byte(flat, "brotli2")
    cz = metrics.compressed_bits_per_byte(flat, "zlib", cfg.zlib_level)
    wrap = cfg.mode == "ring"
    tape_len = None if wrap else 64
    keys, counts = metrics.count_windows_u64(flat, cfg.window_size, wrap, tape_len)
    freqs = metrics.byte_frequencies(flat)
    n_windows = int(counts.sum())
    tracker, tracker_naive = trackers
    a_t, n_pers = tracker.update(keys, counts, soup.epoch, freqs, n_windows)
    a_t_n, n_pers_n = tracker_naive.update(keys, counts, soup.epoch)
    top = metrics.top_windows(flat, cfg.top_window, wrap, tape_len, cfg.top_k)

    s = soup.stats
    runs = int(s[core.STAT_RUNS])
    steps = int(s[core.STAT_STEPS])
    row = {
        "epoch": soup.epoch,
        "entropy_bits": round(h, 6),
        "comp_bits": round(c, 6),
        "high_order_entropy": round(h - c, 6),
        "brotli2_bits": round(c2, 6),
        "zlib_bits": round(cz, 6),
        "hoe_zlib": round(h - cz, 6),
        "frac_steps_in_loop": round(s[core.STAT_REVISITS] / steps, 6) if steps else 0.0,
        "ops_per_run": round(s[core.STAT_COMMANDS] / runs, 4) if runs else 0.0,
        "copies_per_run": round(s[core.STAT_COPIES] / runs, 4) if runs else 0.0,
        "n_persistent_naive": n_pers_n,
        "A_t_naive": a_t_n,
        "distinct_tapes": (len(np.unique(soup.mem.view([("", np.uint8)] * 64)))
                           if cfg.mode != "ring" else ""),
        "distinct_labels": (int(np.unique(soup.labels).size)
                            if soup.labels.size else ""),
        "mean_steps": round(steps / runs, 4) if runs else 0.0,
        "frac_copy_runs": round(s[core.STAT_COPYRUNS] / runs, 6) if runs else 0.0,
        "frac_term_budget": round(s[core.STAT_TERM0] / runs, 6) if runs else 0.0,
        "frac_term_off_region": round(s[core.STAT_TERM1] / runs, 6) if runs else 0.0,
        "frac_term_unmatched": round(s[core.STAT_TERM2] / runs, 6) if runs else 0.0,
        "runs": runs,
        "steps": steps,
        "mutations": soup.mutations,
        "distinct_windows": int(keys.size),
        "n_persistent": n_pers,
        "A_t": a_t,
        "sim_s": round(soup.sim_s, 3),
    }
    soup.stats[:] = 0
    soup.mutations = 0
    return row, top


def main(argv=None):
    parser = argparse.ArgumentParser(description="soup v0")
    parser.add_argument("--out", required=True, help="run directory")
    parser.add_argument("--quiet", action="store_true")
    add_cli_args(parser)
    args = parser.parse_args(argv)
    cfg = build_config(args)

    out = args.out
    os.makedirs(os.path.join(out, "snapshots"), exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as fh:
        fh.write(cfg.to_json() + "\n")

    soup = Soup(cfg)
    trackers = (
        metrics.PersistenceTracker(cfg.c_min, cfg.tau_epochs, cfg.null_ratio,
                                   cfg.window_size),
        metrics.PersistenceTracker(cfg.c_min, cfg.tau_epochs, 0.0,
                                   cfg.window_size),
    )
    kymo_path = os.path.join(out, "kymograph.npy")
    csv_path = os.path.join(out, "metrics.csv")
    pat_path = os.path.join(out, "patterns.log")

    resuming = bool(cfg.load) and os.path.exists(csv_path)
    kymo, last_row = truncate_logs(out, cfg.start_epoch) if resuming else ([], None)
    tracker_path = os.path.join(out, "snapshots",
                                "trackers_%08d.npz" % cfg.start_epoch)
    # only a resume that lands exactly on a logged dump can carry on the log;
    # anything else re-measures the starting condition and starts A(t) again
    continuing = (resuming and last_row is not None
                  and int(last_row["epoch"]) == cfg.start_epoch
                  and os.path.exists(tracker_path))
    if continuing:
        metrics.load_trackers(tracker_path, trackers)
    t_start = time.perf_counter()

    def dump_tape():
        np.save(os.path.join(out, "snapshots", "epoch_%08d.npy" % soup.epoch),
                soup.mem)
        if soup.labels.size:
            np.save(os.path.join(out, "snapshots",
                                 "labels_%08d.npy" % soup.epoch), soup.labels)
        metrics.save_trackers(
            os.path.join(out, "snapshots", "trackers_%08d.npz" % soup.epoch),
            trackers)

    with open(csv_path, "a" if resuming else "w", newline="") as cfh, \
            open(pat_path, "a" if resuming else "w") as pfh:
        writer = csv.DictWriter(cfh, fieldnames=CSV_FIELDS)
        if not resuming:
            writer.writeheader()

        def snapshot():
            row, top = measure(soup, trackers)
            row["wall_s"] = round(time.perf_counter() - t_start, 3)
            writer.writerow(row)
            cfh.flush()
            pfh.write("# epoch %d  H=%.4f C=%.4f HO=%.4f  A(t)=%d\n"
                      % (row["epoch"], row["entropy_bits"], row["zlib_bits"],
                         row["high_order_entropy"], row["A_t"]))
            for raw, n in top:
                pfh.write("  %8d  %s  %s\n"
                          % (n, metrics.printable(raw), raw.hex()))
            pfh.flush()
            kymo.append(soup.flat[: cfg.kymo_width].copy())
            if not args.quiet:
                print("epoch %7d  HO=%.4f  meanSteps=%8.1f  copy=%.3f  A(t)=%d  "
                      "sim=%.1fs" % (row["epoch"], row["high_order_entropy"],
                                     row["mean_steps"], row["frac_copy_runs"],
                                     row["A_t"], row["sim_s"]), flush=True)
            return row

        if continuing:
            base = float(last_row["high_order_entropy"])
        else:
            base = snapshot()["high_order_entropy"]   # the starting condition
            dump_tape()
        streak, stop_at, detected = 0, cfg.start_epoch + cfg.epochs, False
        while soup.epoch < stop_at:
            n = min(cfg.snapshot_interval, stop_at - soup.epoch)
            soup.advance(n)
            row = snapshot()
            if cfg.stop_after_takeover and not detected:
                streak = streak + 1 if row["high_order_entropy"] >= base + HOE_RISE else 0
                if streak >= SUSTAIN:
                    detected = True
                    stop_at = min(cfg.start_epoch + cfg.epochs,
                                  soup.epoch + cfg.stop_after_takeover)
                    print("takeover detected at epoch %d; running to %d"
                          % (soup.epoch, stop_at), flush=True)
            if soup.epoch % cfg.tape_dump_interval == 0 or soup.epoch >= stop_at:
                dump_tape()
            if len(kymo) % 100 == 0:
                np.save(kymo_path, np.array(kymo, dtype=np.uint8))

    np.save(kymo_path, np.array(kymo, dtype=np.uint8))
    wall = time.perf_counter() - t_start
    # a resumed run only simulated the epochs after the checkpoint, so the
    # throughput figures have to be against those, not against soup.epoch
    ran = soup.epoch - cfg.start_epoch
    total_runs = cfg.ticks_per_epoch * ran
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump({
            "config": json.loads(cfg.to_json()),
            "wall_seconds": round(wall, 2),
            "sim_seconds": round(soup.sim_s, 2),
            "epochs": soup.epoch,
            "epochs_simulated": ran,
            "runs": total_runs,
            "epochs_per_second": round(ran / wall, 3),
            "sim_epochs_per_second": round(ran / soup.sim_s, 3) if soup.sim_s else None,
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "walk_len": cfg.walk_len,
            "region_len": cfg.region_len,
        }, fh, indent=2)
        fh.write("\n")
    if not args.quiet:
        print("done: %d epochs in %.1fs wall (%.1fs simulating), %s"
              % (soup.epoch, wall, soup.sim_s, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
