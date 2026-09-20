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

from . import core, metrics, replicator
from .analyze import HOE_RISE, SUSTAIN
from .config import add_cli_args, build_config
from .interp import TERM_NAMES
from .rng import fill_random, make_state

CSV_FIELDS = [
    "epoch", "wall_s", "sim_s",
    "entropy_bits", "zlib_bits", "high_order_entropy",
    "mean_steps", "frac_copy_runs",
    "frac_term_budget", "frac_term_off_region", "frac_term_unmatched",
    "runs", "steps", "mutations",
    "distinct_windows", "n_persistent", "A_t",
]


class Soup:
    """Owns the memory, the RNG stream and the epoch loop for one run."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = make_state(cfg.seed)
        if cfg.mode == "ring":
            self.mem = np.zeros(cfg.M, dtype=np.uint8)
            self.buf = np.zeros(2 * cfg.R + 1, dtype=np.uint8)
        else:
            self.mem = np.zeros((cfg.N, 64), dtype=np.uint8)
            self.buf = np.zeros(128, dtype=np.uint8)
            self.perm = np.arange(cfg.N, dtype=np.int64)
        fill_random(self.flat, self.state)
        if cfg.plant:
            self._plant(cfg.plant)
        self.stats = np.zeros(core.N_STATS, dtype=np.int64)
        self.mutations = 0
        self.sim_s = 0.0
        self.epoch = 0

    @property
    def flat(self):
        return self.mem.reshape(-1)

    def _plant(self, n):
        prog = replicator.as_array()
        flat = self.flat
        for _ in range(n):
            if self.cfg.mode == "ring":
                p = int(np.random.default_rng(self.cfg.seed + _).integers(self.cfg.M))
                idx = (p + np.arange(prog.size)) % self.cfg.M
                flat[idx] = prog
            else:
                t = int(np.random.default_rng(self.cfg.seed + _).integers(self.cfg.N))
                self.mem[t, : prog.size] = prog

    def advance(self, n_epochs):
        """Run n_epochs, accumulating stats.  Returns seconds spent simulating."""
        cfg = self.cfg
        t0 = time.perf_counter()
        if cfg.mode == "ring":
            m = core.ring_chunk(self.mem, self.buf, n_epochs, cfg.ticks_per_epoch,
                                cfg.R, cfg.k, cfg.mu, self.state, self.stats,
                                cfg.head_bound == "wrap")
        else:
            m = core.bff_chunk(self.mem, self.buf, self.perm, n_epochs,
                               cfg.k, cfg.mu, self.state, self.stats,
                               cfg.head_bound == "wrap")
        dt = time.perf_counter() - t0
        self.sim_s += dt
        self.mutations += int(m)
        self.epoch += n_epochs
        return dt


def measure(soup, tracker):
    """One metrics row.  Consumes and resets the interpreter counters."""
    cfg = soup.cfg
    flat = soup.flat
    h, c, ho = metrics.high_order_entropy(flat, cfg.zlib_level)
    wrap = cfg.mode == "ring"
    tape_len = None if wrap else 64
    keys, counts = metrics.count_windows_u64(flat, cfg.window_size, wrap, tape_len)
    a_t, n_pers = tracker.update(keys, counts)
    top = metrics.top_windows(flat, cfg.top_window, wrap, tape_len, cfg.top_k)

    s = soup.stats
    runs = int(s[core.STAT_RUNS])
    row = {
        "epoch": soup.epoch,
        "entropy_bits": round(h, 6),
        "zlib_bits": round(c, 6),
        "high_order_entropy": round(ho, 6),
        "mean_steps": round(s[core.STAT_STEPS] / runs, 4) if runs else 0.0,
        "frac_copy_runs": round(s[core.STAT_COPYRUNS] / runs, 6) if runs else 0.0,
        "frac_term_budget": round(s[core.STAT_TERM0] / runs, 6) if runs else 0.0,
        "frac_term_off_region": round(s[core.STAT_TERM1] / runs, 6) if runs else 0.0,
        "frac_term_unmatched": round(s[core.STAT_TERM2] / runs, 6) if runs else 0.0,
        "runs": runs,
        "steps": int(s[core.STAT_STEPS]),
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
    tracker = metrics.PersistenceTracker(cfg.c_min, cfg.tau)
    kymo = []
    kymo_path = os.path.join(out, "kymograph.npy")

    csv_path = os.path.join(out, "metrics.csv")
    pat_path = os.path.join(out, "patterns.log")
    t_start = time.perf_counter()

    def dump_tape():
        np.save(os.path.join(out, "snapshots", "epoch_%08d.npy" % soup.epoch),
                soup.mem)

    with open(csv_path, "w", newline="") as cfh, open(pat_path, "w") as pfh:
        writer = csv.DictWriter(cfh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        def snapshot():
            row, top = measure(soup, tracker)
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

        base = snapshot()["high_order_entropy"]   # epoch 0: random condition
        dump_tape()
        streak, stop_at = 0, cfg.epochs
        while soup.epoch < stop_at:
            n = min(cfg.snapshot_interval, stop_at - soup.epoch)
            soup.advance(n)
            row = snapshot()
            if cfg.stop_after_takeover and stop_at == cfg.epochs:
                streak = streak + 1 if row["high_order_entropy"] >= base + HOE_RISE else 0
                if streak >= SUSTAIN:
                    stop_at = min(cfg.epochs,
                                  soup.epoch + cfg.stop_after_takeover)
                    print("takeover detected at epoch %d; running to %d"
                          % (soup.epoch, stop_at), flush=True)
            if soup.epoch % cfg.tape_dump_interval == 0 or soup.epoch >= stop_at:
                dump_tape()
            if len(kymo) % 100 == 0:
                np.save(kymo_path, np.array(kymo, dtype=np.uint8))

    np.save(kymo_path, np.array(kymo, dtype=np.uint8))
    wall = time.perf_counter() - t_start
    total_runs = cfg.ticks_per_epoch * soup.epoch
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump({
            "config": json.loads(cfg.to_json()),
            "wall_seconds": round(wall, 2),
            "sim_seconds": round(soup.sim_s, 2),
            "epochs": soup.epoch,
            "runs": total_runs,
            "epochs_per_second": round(soup.epoch / wall, 3),
            "sim_epochs_per_second": round(soup.epoch / soup.sim_s, 3) if soup.sim_s else None,
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        }, fh, indent=2)
        fh.write("\n")
    if not args.quiet:
        print("done: %d epochs in %.1fs wall (%.1fs simulating), %s"
              % (soup.epoch, wall, soup.sim_s, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
