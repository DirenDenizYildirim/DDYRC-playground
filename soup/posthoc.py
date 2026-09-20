"""Per-snapshot measurements on a soup that has already been taken over.

Three questions, one row per saved soup:

self-sufficiency
    Pair a soup tape with a *fresh uniform-random* tape and run once.  Does
    it still copy?  How much of the 128-byte pair ends up carrying the soup
    tape's provenance label?  The controls are soup-vs-soup (the pairing the
    population actually evolved under) and random-vs-random (the floor).

where the variation sits
    Take the leading motif family, align every tape that carries it on its
    first occurrence, and report the per-offset byte entropy.  Offsets 0..15
    are the seed window itself, so low entropy there is circular and is
    reported only for completeness; offsets outside that range are free, and
    how far the low-entropy region extends past the seed is the real answer.

bulk structure
    distinct tapes, and the pooled entropy of bytes inside vs outside the
    motif region.

    python -m soup.posthoc runs/plateau_s12 --out analysis/posthoc/s12
"""
import argparse
import csv
import json
import os

import numpy as np

from . import metrics, motifs
from .interp import run_region_full

PAIR = 128
TAPE = 64
SAMPLES = 512
CONSERVED_BITS = 2.0     # an offset counts as conserved below this entropy


def _run_pair(buf, labels, k, visited, gen, heads_from_tape=True):
    """One pair run, under the head rule of the mode being analysed.

    ``heads_from_tape`` is cubff's rule (heads read from the first two bytes,
    execution from pc = 2) -- the baseline.  False is this repo's own spec,
    ip = head0 = head1 = 0.
    """
    if heads_from_tape:
        h0, h1, pc = int(buf[0]) % PAIR, int(buf[1]) % PAIR, 2
    else:
        h0, h1, pc = 0, 0, 0
    return run_region_full(buf, pc, h0, h1, k, True, visited, gen, labels, True)


def self_sufficiency(mem, samples=SAMPLES, k=8192, seed=0,
                     heads_from_tape=True):
    """Copy rate and colonisation for soup-vs-random, with both controls.

    ``copy_rate`` is the fraction of pairs in which *any* copy happened --
    the usual bff statistic, and not directional: a random partner can do
    the copying.  ``colonisation`` is the directional one: the fraction of
    the 128-byte pair carrying the focal tape's provenance label at the end,
    which is 0.5 when neither side gains.
    """
    rng = np.random.default_rng(seed)
    n = mem.shape[0]
    visited = np.zeros(PAIR, dtype=np.int32)
    out = {}
    for name in ("soup_vs_random", "soup_vs_soup", "random_vs_random"):
        copied = np.zeros(samples)
        colonised = np.zeros(samples)
        steps = np.zeros(samples)
        for s in range(samples):
            if name == "random_vs_random":
                focal = rng.integers(0, 256, size=TAPE, dtype=np.uint8)
            else:
                focal = mem[rng.integers(n)]
            if name == "soup_vs_soup":
                other = mem[rng.integers(n)]
            else:
                other = rng.integers(0, 256, size=TAPE, dtype=np.uint8)
            buf = np.empty(PAIR, dtype=np.uint8)
            lab = np.empty(PAIR, dtype=np.uint8)
            first = rng.integers(2) == 0     # focal tape's side of the pair
            buf[0:TAPE], buf[TAPE:] = (focal, other) if first else (other, focal)
            lab[0:TAPE] = 1 if first else 2
            lab[TAPE:] = 2 if first else 1
            st, ncopy, _term, _rev, _ncmd = _run_pair(
                buf, lab, k, visited, s + 1, heads_from_tape)
            copied[s] = ncopy > 0
            colonised[s] = float((lab == 1).mean())
            steps[s] = st
        out[name] = {
            "copy_rate": round(float(copied.mean()), 6),
            "colonisation": round(float(colonised.mean()), 6),
            "mean_steps": round(float(steps.mean()), 2),
        }
    return out


def motif_mask(mem, member_ids, ids, w=motifs.WINDOW):
    """(n_tapes, 64) bool: bytes covered by a window of the family."""
    hit = np.isin(ids, np.asarray(member_ids))
    mask = np.zeros(mem.shape, dtype=bool)
    for j in range(w):
        mask[:, j:j + hit.shape[1]] |= hit
    return mask


def first_occurrence(ids, member_ids):
    """Per-tape index of the first family window, or -1."""
    hit = np.isin(ids, np.asarray(member_ids))
    any_hit = hit.any(axis=1)
    first = np.where(any_hit, hit.argmax(axis=1), -1)
    return first


def alignment_profile(mem, ids, member_ids, span=32):
    """Per-offset byte entropy across tapes aligned on the motif.

    Offsets run from -span to +span relative to the start of the first
    family window in each tape; offsets that fall off the end of a tape are
    dropped, so each offset's entropy is over the tapes where it exists.
    """
    first = first_occurrence(ids, member_ids)
    rows = np.flatnonzero(first >= 0)
    offsets = np.arange(-span, span + 1)
    ent = np.full(offsets.size, np.nan)
    counts = np.zeros(offsets.size, dtype=np.int64)
    for i, off in enumerate(offsets):
        pos = first[rows] + off
        ok = (pos >= 0) & (pos < mem.shape[1])
        if ok.sum() < 32:
            continue
        vals = mem[rows[ok], pos[ok]]
        ent[i] = max(0.0, metrics.shannon_entropy_bits(vals))
        counts[i] = int(ok.sum())
    return offsets, ent, counts, rows.size


def conserved_span(offsets, ent, bits=CONSERVED_BITS):
    """Width of the contiguous low-entropy run containing offset 0."""
    zero = int(np.flatnonzero(offsets == 0)[0])
    if not np.isfinite(ent[zero]) or ent[zero] >= bits:
        return 0, 0, 0
    lo = zero
    while lo > 0 and np.isfinite(ent[lo - 1]) and ent[lo - 1] < bits:
        lo -= 1
    hi = zero
    while hi < ent.size - 1 and np.isfinite(ent[hi + 1]) and ent[hi + 1] < bits:
        hi += 1
    return int(offsets[lo]), int(offsets[hi]), int(offsets[hi] - offsets[lo] + 1)


FIELDS = ["epoch", "distinct_tapes",
          "leader_coverage", "leader_rep_hex",
          "ss_copy_rate", "ss_colonisation", "ss_mean_steps",
          "ctrl_soup_copy_rate", "ctrl_soup_colonisation",
          "ctrl_rand_copy_rate", "ctrl_rand_colonisation",
          "frac_bytes_in_motif", "H_inside", "H_outside",
          "conserved_lo", "conserved_hi", "conserved_width",
          "n_aligned_tapes"]


def snapshot_row(mem, epoch, samples=SAMPLES, k=8192, seed=0, span=32,
                 profiles=None, heads_from_tape=True):
    fams = motifs.snapshot_families(mem)
    windows, counts, ids = motifs.window_ids(mem)
    ss = self_sufficiency(mem, samples, k, seed, heads_from_tape)
    row = {
        "epoch": epoch,
        "distinct_tapes": int(len(np.unique(
            np.ascontiguousarray(mem).view([("", np.uint8)] * 64)))),
        "ss_copy_rate": ss["soup_vs_random"]["copy_rate"],
        "ss_colonisation": ss["soup_vs_random"]["colonisation"],
        "ss_mean_steps": ss["soup_vs_random"]["mean_steps"],
        "ctrl_soup_copy_rate": ss["soup_vs_soup"]["copy_rate"],
        "ctrl_soup_colonisation": ss["soup_vs_soup"]["colonisation"],
        "ctrl_rand_copy_rate": ss["random_vs_random"]["copy_rate"],
        "ctrl_rand_colonisation": ss["random_vs_random"]["colonisation"],
    }
    if not fams:
        row.update({f: "" for f in FIELDS if f not in row})
        return row
    lead = fams[0]
    mask = motif_mask(mem, lead["members"], ids)
    inside, outside = mem[mask], mem[~mask]
    # aligned on the single representative window, not the whole family:
    # different tapes match different members, so a family-wide alignment
    # would put different bytes at offset 0 and hide the conserved core
    offsets, ent, n_at, n_aligned = alignment_profile(
        mem, ids, [lead["rep_id"]], span)
    lo, hi, width = conserved_span(offsets, ent)
    row.update({
        "leader_coverage": round(lead["coverage"], 6),
        "leader_rep_hex": lead["rep"].hex(),
        "frac_bytes_in_motif": round(float(mask.mean()), 6),
        "H_inside": round(metrics.shannon_entropy_bits(inside), 4)
                    if inside.size else "",
        "H_outside": round(metrics.shannon_entropy_bits(outside), 4)
                     if outside.size else "",
        "conserved_lo": lo, "conserved_hi": hi, "conserved_width": width,
        "n_aligned_tapes": n_aligned,
    })
    if profiles is not None:
        profiles[str(epoch)] = {
            "offsets": offsets.tolist(),
            "entropy": [None if not np.isfinite(e) else round(float(e), 4)
                        for e in ent],
            "n": n_at.tolist(),
            "rep_hex": lead["rep"].hex(),
        }
    return row


def analyse(run_dir, out, samples=SAMPLES, k=8192, seed=0, span=32,
            every=1, quiet=False, heads_from_tape=True):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    profiles = {}
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        for i, (epoch, path) in enumerate(motifs.snapshot_paths(run_dir)):
            if i % every:
                continue
            row = snapshot_row(np.load(path), epoch, samples, k, seed, span,
                               profiles, heads_from_tape)
            w.writerow(row)
            fh.flush()
            if not quiet:
                print("epoch %8d  copy(vs random)=%.3f colonisation=%.3f "
                      "conserved=%s" % (epoch, row["ss_copy_rate"],
                                        row["ss_colonisation"],
                                        row["conserved_width"]), flush=True)
    with open(out + "_profiles.json", "w") as fh:
        json.dump({"run": run_dir, "span": span, "samples": samples,
                   "conserved_bits": CONSERVED_BITS,
                   "profiles": profiles}, fh)
    return out + ".csv"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=SAMPLES)
    p.add_argument("--k", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--span", type=int, default=32)
    p.add_argument("--every", type=int, default=1, help="use every Nth dump")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--head-source", default="tape", choices=["tape", "fixed"],
                   help="tape = cubff's rule (the baseline); fixed = the spec")
    a = p.parse_args(argv)
    print(analyse(a.run_dir, a.out, a.samples, a.k, a.seed, a.span, a.every,
                  a.quiet, a.head_source == "tape"))


if __name__ == "__main__":
    main()
