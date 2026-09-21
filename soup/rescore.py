"""Recompute the snapshot-derivable metrics for runs made before they existed.

    python -m soup.rescore runs/ring_s1 runs/bff_s1 ...

Entropy, the compressed size and A(t) are functions of the stored memory, so
they can be recomputed from a run's ``snapshots/*.npy`` without re-simulating.
The *execution* metrics -- mean steps, the loop fraction, termination mix --
cannot: they are counters the interpreter kept while running.  Those columns
are left out here rather than guessed, and the runs that needed them were
re-run instead.

Output is ``rescored.csv`` in the run directory; :mod:`soup.analyze` prefers it
over ``metrics.csv`` where the two overlap.

One caveat on A(t): tape dumps are much sparser than metric rows (every 1000
epochs rather than every 50), so a rescored A(t) samples the same history more
coarsely than a live one.  Rescored A(t) values are comparable with each other
and not with the live column.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

from . import metrics

FIELDS = ["epoch", "entropy_bits", "comp_bits", "high_order_entropy",
          "brotli2_bits", "zlib_bits", "hoe_zlib",
          "distinct_windows", "n_persistent", "A_t",
          "n_persistent_naive", "A_t_naive"]


def rescore(run_dir, tau_epochs, null_ratio, compressor="brotli6"):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    wrap = cfg["mode"] == "ring"
    tape_len = None if wrap else 64
    w = cfg["window_size"]
    trackers = (metrics.PersistenceTracker(cfg["c_min"], tau_epochs, null_ratio, w),
                metrics.PersistenceTracker(cfg["c_min"], tau_epochs, 0.0, w))
    rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, "snapshots", "epoch_*.npy"))):
        epoch = int(os.path.basename(path)[6:-4])
        mem = np.load(path).reshape(-1)
        h = metrics.shannon_entropy_bits(mem)
        c = metrics.compressed_bits_per_byte(mem, compressor)
        c2 = metrics.compressed_bits_per_byte(mem, "brotli2")
        cz = metrics.compressed_bits_per_byte(mem, "zlib")
        keys, counts = metrics.count_windows_u64(mem, w, wrap, tape_len)
        freqs = metrics.byte_frequencies(mem)
        n_windows = int(counts.sum())
        a_t, n_p = trackers[0].update(keys, counts, epoch, freqs, n_windows)
        a_n, n_n = trackers[1].update(keys, counts, epoch)
        rows.append({"epoch": epoch, "entropy_bits": round(h, 6),
                     "comp_bits": round(c, 6),
                     "high_order_entropy": round(h - c, 6),
                     "brotli2_bits": round(c2, 6), "zlib_bits": round(cz, 6),
                     "hoe_zlib": round(h - cz, 6),
                     "distinct_windows": int(keys.size),
                     "n_persistent": n_p, "A_t": a_t,
                     "n_persistent_naive": n_n, "A_t_naive": a_n})
    out = os.path.join(run_dir, "rescored.csv")
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tau-epochs", type=int, default=250)
    ap.add_argument("--null-ratio", type=float, default=5.0)
    ap.add_argument("--compressor", default="brotli6")
    args = ap.parse_args(argv)
    for d in args.runs:
        rows = rescore(d, args.tau_epochs, args.null_ratio, args.compressor)
        last = rows[-1]
        print("%-24s %3d snapshots  H=%.3f comp=%.3f HOE=%.3f  A(t)=%d (naive %d)"
              % (os.path.basename(d.rstrip("/")), len(rows),
                 last["entropy_bits"], last["comp_bits"],
                 last["high_order_entropy"], last["A_t"], last["A_t_naive"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
