"""Competition assay: put two soups in one soup and see whose bytes win.

Build a population of ``n`` tapes, half sampled from soup A and half from
soup B, label the two halves, run the normal bff/cubff dynamics at the
normal mutation rate, and record what fraction of memory still carries A's
label over time.  Labels ride along with '.' and ',' and are invisible to
the machine, so the assay does not perturb what it measures.

The honest control is A against A: the same soup split in half and given two
labels.  Its spread across replicates is the noise band, and any A-vs-B
curve has to leave that band before it means anything.

    python -m soup.assay --a runs/plateau_s12/snapshots/epoch_00005500.npy \
                         --b runs/plateau_s12/snapshots/epoch_00055500.npy \
                         --out analysis/assay/s12_first_vs_last --reps 5

Writes <out>.csv with one row per (rep, epoch).
"""
import argparse
import csv
import json
import os

import numpy as np

from . import core
from .config import Config

LABEL_A, LABEL_B = 1, 2


def build(mem_a, mem_b, n, rng):
    """n tapes, half from each soup, shuffled into random positions."""
    half = n // 2
    mem = np.zeros((n, 64), dtype=np.uint8)
    labels = np.zeros((n, 64), dtype=np.uint8)
    pos = rng.permutation(n)
    for side, src, label in ((0, mem_a, LABEL_A), (1, mem_b, LABEL_B)):
        take = rng.choice(src.shape[0], size=half, replace=src.shape[0] < half)
        where = pos[side * half:(side + 1) * half]
        mem[where] = src[take]
        labels[where] = label
    if n % 2:                       # odd n: the leftover slot goes to A
        mem[pos[-1]] = mem_a[rng.integers(mem_a.shape[0])]
        labels[pos[-1]] = LABEL_A
    return mem, labels


def fractions(soup):
    """(fraction of bytes labelled A, fraction of tapes that are majority A)."""
    lab = soup.labels
    frac_bytes = float((lab == LABEL_A).mean())
    per_tape = (lab == LABEL_A).sum(axis=1)
    frac_tapes = float((per_tape > 32).mean() + 0.5 * (per_tape == 32).mean())
    return frac_bytes, frac_tapes


def one_replicate(mem_a, mem_b, n, epochs, every, seed, mu, k):
    """Run one assay replicate; yields a row per measurement."""
    from .run import Soup                     # local: keeps import cost off CLI

    cfg = Config(mode="bff", compat="cubff", N=n, k=k, mu=mu,
                 seed=seed, epochs=epochs, labels=1)
    soup = Soup(cfg)
    rng = np.random.default_rng(seed)
    mem, labels = build(mem_a, mem_b, n, rng)
    soup.mem[...] = mem
    soup.labels[...] = labels
    while True:
        fb, ft = fractions(soup)
        s = soup.stats
        runs = int(s[core.STAT_RUNS])
        yield {
            "epoch": soup.epoch,
            "frac_bytes_a": round(fb, 6),
            "frac_tapes_a": round(ft, 6),
            "distinct_tapes": int(len(np.unique(
                soup.mem.view([("", np.uint8)] * 64)))),
            "mean_steps": round(s[core.STAT_STEPS] / runs, 2) if runs else 0.0,
            "copies_per_run": round(s[core.STAT_COPIES] / runs, 3) if runs else 0.0,
        }
        soup.stats[:] = 0
        if soup.epoch >= epochs:
            return
        soup.advance(min(every, epochs - soup.epoch))


FIELDS = ["rep", "epoch", "frac_bytes_a", "frac_tapes_a", "distinct_tapes",
          "mean_steps", "copies_per_run"]


def run_assay(path_a, path_b, out, n=8192, epochs=300, every=10, reps=5,
              seed0=1000, mu=0.000244140625, k=8192, quiet=False):
    mem_a = np.load(path_a)
    mem_b = np.load(path_b)
    if mem_a.ndim != 2 or mem_a.shape[1] != 64:
        raise SystemExit("%s is not a tape population" % path_a)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        for rep in range(reps):
            for row in one_replicate(mem_a, mem_b, n, epochs, every,
                                     seed0 + rep, mu, k):
                row["rep"] = rep
                w.writerow(row)
                fh.flush()
                if not quiet:
                    print("rep %d epoch %4d  A=%.4f bytes  %.4f tapes"
                          % (rep, row["epoch"], row["frac_bytes_a"],
                             row["frac_tapes_a"]), flush=True)
    with open(out + ".json", "w") as fh:
        json.dump({"a": path_a, "b": path_b, "n": n, "epochs": epochs,
                   "every": every, "reps": reps, "seed0": seed0, "mu": mu,
                   "k": k}, fh, indent=2)
    return out + ".csv"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--a", required=True, help=".npy tape population, label A")
    p.add_argument("--b", required=True, help=".npy tape population, label B")
    p.add_argument("--out", required=True, help="output path without .csv")
    p.add_argument("--n", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--mu", type=float, default=0.000244140625)
    p.add_argument("--k", type=int, default=8192)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    print(run_assay(a.a, a.b, a.out, a.n, a.epochs, a.every, a.reps,
                    a.seed0, a.mu, a.k, a.quiet))


if __name__ == "__main__":
    main()
