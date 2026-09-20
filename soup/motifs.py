"""Motif families: cluster the dominant 16-byte windows and follow them.

The top windows of a soup that has been taken over are nearly all shifts of
one string, so counting windows overstates how many distinct things are
there.  Clustering them by edit distance collapses the shifts, and a
family's *coverage* -- the fraction of tapes holding at least one of its
windows -- is a frequency with a denominator worth quoting.

A sweep is the user-facing event: a family passes ``rise`` coverage while
the family that was leading falls below ``fall``.

    python -m soup.motifs runs/plateau_s12 --out analysis/motifs/s12
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np
from numba import njit

from . import metrics

WINDOW = 16
TOP = 200            # windows clustered per snapshot
MAX_DIST = 8         # edit distance at which two 16-byte windows are kin;
                     # see null_distances() for where the number comes from
RISE, FALL = 0.10, 0.05


@njit(cache=True)
def edit_distance(a, b):
    """Levenshtein distance between two byte strings."""
    n, m = a.size, b.size
    prev = np.arange(m + 1, dtype=np.int64)
    cur = np.empty(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        cur[0] = i
        for j in range(1, m + 1):
            sub = prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 1)
            cur[j] = min(sub, prev[j] + 1, cur[j - 1] + 1)
        prev, cur = cur, prev
    return prev[m]


@njit(cache=True)
def _nearest_family(cand, members, family_of, n_fam):
    """Family index whose closest member is nearest to ``cand``, and that gap."""
    best_fam, best_d = -1, 1 << 30
    for i in range(members.shape[0]):
        d = edit_distance(cand, members[i])
        if d < best_d:
            best_d, best_fam = d, family_of[i]
    return best_fam, best_d


def null_distances(mem, pairs=2000, w=WINDOW, seed=0):
    """Edit distances between unrelated windows drawn from the soup's own
    byte-frequency null (the soup shuffled).  MAX_DIST is set below the low
    tail of this distribution, so a pair that clusters is a pair no i.i.d.
    model of the same soup would have produced.
    """
    rng = np.random.default_rng(seed)
    flat = np.asarray(mem).reshape(-1).copy()
    rng.shuffle(flat)
    windows, _, _ = window_ids(flat.reshape(-1, 64), w)
    idx = rng.choice(windows.shape[0], 2 * pairs)
    return np.array([
        edit_distance(np.ascontiguousarray(windows[idx[i]]),
                      np.ascontiguousarray(windows[idx[i + pairs]]))
        for i in range(pairs)])


def window_ids(mem, w=WINDOW):
    """Distinct windows, their total counts, and a (n_tapes, 49) id matrix."""
    if mem.ndim != 2 or mem.shape[1] != 64:
        raise SystemExit("motif families are defined on tape populations")
    mat = np.ascontiguousarray(metrics.window_matrix(mem, w, False, 64))
    recs = mat.view(np.dtype((np.void, w))).reshape(-1)
    uniq, inv, counts = np.unique(recs, return_inverse=True,
                                  return_counts=True)
    ids = inv.reshape(mem.shape[0], -1)
    windows = np.frombuffer(uniq.tobytes(), dtype=np.uint8).reshape(-1, w)
    return windows, counts, ids


def cluster(windows, counts, top=TOP, max_dist=MAX_DIST):
    """Greedy single-linkage clustering of the ``top`` commonest windows.

    Returns a list of families, each a list of indices into ``windows``,
    ordered by the count of their most frequent member.
    """
    order = np.argsort(counts, kind="stable")[::-1][:top]
    fams = []
    members = np.zeros((0, windows.shape[1]), dtype=np.uint8)
    family_of = np.zeros(0, dtype=np.int64)
    for idx in order:
        cand = np.ascontiguousarray(windows[idx])
        if members.shape[0]:
            fam, d = _nearest_family(cand, members, family_of, len(fams))
        else:
            fam, d = -1, 1 << 30
        if d > max_dist:
            fam = len(fams)
            fams.append([])
        fams[fam].append(int(idx))
        members = np.vstack([members, cand[None, :]])
        family_of = np.append(family_of, fam)
    return fams


def coverage(ids, member_ids):
    """Fraction of tapes holding at least one window from the family."""
    return float(np.isin(ids, np.asarray(member_ids)).any(axis=1).mean())


def snapshot_families(mem, top=TOP, max_dist=MAX_DIST):
    """[(coverage, count, representative_bytes, member_ids)], best first."""
    windows, counts, ids = window_ids(mem)
    out = []
    for fam in cluster(windows, counts, top, max_dist):
        rep = max(fam, key=lambda i: counts[i])
        out.append({
            "coverage": coverage(ids, fam),
            "count": int(counts[fam].sum()),
            "top_count": int(counts[rep]),
            "size": len(fam),
            "rep": windows[rep].tobytes(),
            "rep_id": int(rep),
            "members": fam,
        })
    out.sort(key=lambda f: -f["coverage"])
    return out


class Registry:
    """Stable family ids across snapshots, matched by representative."""

    def __init__(self, max_dist=MAX_DIST):
        self.reps = []
        self.max_dist = max_dist

    def id_of(self, rep):
        cand = np.frombuffer(rep, dtype=np.uint8)
        best, best_d = -1, 1 << 30
        for i, known in enumerate(self.reps):
            d = int(edit_distance(cand, known))
            if d < best_d:
                best, best_d = i, d
        if best_d <= self.max_dist:
            return best
        self.reps.append(cand)
        return len(self.reps) - 1


EPOCH_RE = re.compile(r"epoch_(\d+)\.npy$")


def snapshot_paths(run_dir):
    paths = sorted(glob.glob(os.path.join(run_dir, "snapshots", "epoch_*.npy")))
    return [(int(EPOCH_RE.search(p).group(1)), p) for p in paths]


FIELDS = ["epoch", "family", "coverage", "count", "top_count", "size",
          "rep_hex", "rep_printable", "is_leader"]


def track(run_dir, out, top=TOP, max_dist=MAX_DIST, keep=6, rise=RISE,
          fall=FALL, quiet=False):
    """Family coverage per snapshot, plus the sweeps it implies."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    reg = Registry(max_dist)
    leader, sweeps, series = None, [], []
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        for epoch, path in snapshot_paths(run_dir):
            fams = snapshot_families(np.load(path), top, max_dist)
            cov = {}
            for fam in fams[:keep]:
                fid = reg.id_of(fam["rep"])
                cov[fid] = max(cov.get(fid, 0.0), fam["coverage"])
                w.writerow({
                    "epoch": epoch, "family": fid,
                    "coverage": round(fam["coverage"], 6),
                    "count": fam["count"], "top_count": fam["top_count"],
                    "size": fam["size"], "rep_hex": fam["rep"].hex(),
                    "rep_printable": metrics.printable(fam["rep"]),
                    "is_leader": int(leader == fid),
                })
            fh.flush()
            best = max(cov, key=lambda f: cov[f]) if cov else None
            if leader is None:
                if best is not None and cov[best] >= rise:
                    leader = best
            elif cov.get(leader, 0.0) < fall:
                risers = [f for f in cov if f != leader and cov[f] >= rise]
                if risers:
                    new = max(risers, key=lambda f: cov[f])
                    sweeps.append({"epoch": epoch, "from": leader, "to": new,
                                   "from_coverage": round(cov.get(leader, 0.0), 6),
                                   "to_coverage": round(cov[new], 6)})
                    leader = new
            series.append({"epoch": epoch, "leader": leader,
                           "leader_coverage": round(cov.get(leader, 0.0), 6)
                           if leader is not None else None,
                           "n_families": len(cov)})
            if not quiet:
                print("epoch %8d  families=%d leader=%s cov=%.4f"
                      % (epoch, len(cov), leader,
                         cov.get(leader, 0.0) if leader is not None else 0.0),
                      flush=True)
    with open(out + ".json", "w") as fh:
        json.dump({"run": run_dir, "window": WINDOW, "top": top,
                   "max_dist": max_dist, "rise": rise, "fall": fall,
                   "n_sweeps": len(sweeps), "sweeps": sweeps,
                   "leader": series}, fh, indent=2)
    return out + ".csv", sweeps


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir")
    p.add_argument("--out", required=True, help="output path without .csv")
    p.add_argument("--top", type=int, default=TOP)
    p.add_argument("--max-dist", type=int, default=MAX_DIST)
    p.add_argument("--keep", type=int, default=6)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    path, sweeps = track(a.run_dir, a.out, a.top, a.max_dist, a.keep,
                         quiet=a.quiet)
    print("%s  %d sweeps" % (path, len(sweeps)))


if __name__ == "__main__":
    main()
