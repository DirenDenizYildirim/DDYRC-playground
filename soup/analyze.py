"""Post-hoc analysis of finished runs.

The detectors below are applied identically to every run.  They are stated
here rather than tuned per run, and the thresholds are printed alongside the
answers so a reader can disagree with them.

    python -m soup.analyze runs/ring_s1 runs/ring_s2 --json analysis/ring.json
"""
import argparse
import json
import os

import numpy as np

from .metrics import printable
from .plots import read_metrics

# detector constants
#
# The baseline for high-order entropy is the epoch-0 row: the initial condition
# is i.i.d. uniform bytes, for which HOE is 0 up to the zlib header overhead
# (measured: |HOE(0)| < 0.01 in every run here).  A rise of a full bit per byte
# is ~100x that noise floor and ~1/6 of the signal from a verified replicator
# takeover (the planted control reaches 5.7), so it is a permissive threshold
# for "something took over", not a tuned one.
HOE_RISE = 1.0            # bits/byte above the epoch-0 baseline counts as a rise
HOE_SOFT = 0.5            # a weaker "structure is present" marker
SUSTAIN = 5               # snapshots the condition must hold for
H_COLLAPSE = 4.0          # bits/byte: order-0 entropy this low means a monoculture


def _first_sustained(x, ok, sustain=SUSTAIN):
    """Index of the first element where ``ok`` holds for ``sustain`` in a row."""
    flag = ok(x)
    for i in range(len(flag) - sustain + 1):
        if flag[i:i + sustain].all():
            return i
    return None


def detect(m):
    """Transition detectors on one run's metric table."""
    epoch, hoe, h = m["epoch"], m["high_order_entropy"], m["entropy_bits"]
    baseline = float(hoe[0])            # the initial i.i.d. random condition
    thresh = baseline + HOE_RISE

    i_rise = _first_sustained(hoe, lambda x: x >= thresh)
    i_soft = _first_sustained(hoe, lambda x: x >= baseline + HOE_SOFT)
    i_coll = _first_sustained(h, lambda x: x <= H_COLLAPSE)

    a = m["A_t"]
    half = len(a) // 2
    span_early = epoch[half] - epoch[0]
    span_late = epoch[-1] - epoch[half]
    out = {
        "epochs": int(epoch[-1]),
        "hoe_baseline": round(baseline, 4),
        "hoe_threshold": round(thresh, 4),
        "hoe_final": round(float(hoe[-1]), 4),
        "hoe_max": round(float(hoe.max()), 4),
        "hoe_argmax_epoch": int(epoch[int(hoe.argmax())]),
        "entropy_final": round(float(h[-1]), 4),
        "entropy_min": round(float(h.min()), 4),
        "zlib_final": round(float(m["zlib_bits"][-1]), 4),
        "takeover_epoch": None if i_rise is None else int(epoch[i_rise]),
        "hoe_soft_epoch": None if i_soft is None else int(epoch[i_soft]),
        "entropy_collapse_epoch": None if i_coll is None else int(epoch[i_coll]),
        "mean_steps_final": round(float(m["mean_steps"][-1]), 1),
        "frac_copy_final": round(float(m["frac_copy_runs"][-1]), 4),
        "A_t_final": int(a[-1]),
        "A_t_rate_first_half": round(float((a[half] - a[0]) / span_early * 1000), 3),
        "A_t_rate_second_half": round(float((a[-1] - a[half]) / span_late * 1000), 3),
        "n_persistent_final": int(m["n_persistent"][-1]),
    }
    out["A_t_plateaus"] = bool(out["A_t_rate_second_half"] < 0.5 * out["A_t_rate_first_half"])
    return out


def final_patterns(run_dir, top=5):
    """The last block of patterns.log."""
    with open(os.path.join(run_dir, "patterns.log")) as fh:
        lines = fh.read().splitlines()
    start = max(i for i, l in enumerate(lines) if l.startswith("#"))
    return lines[start:start + 1 + top]


def analyse(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    d = detect(read_metrics(run_dir))
    d["run"] = os.path.basename(run_dir.rstrip("/"))
    d["mode"] = cfg["mode"]
    d["seed"] = cfg["seed"]
    d["N_or_M"] = cfg["N"] if cfg["mode"] == "bff" else cfg["M"]
    d["wall_seconds"] = summary["wall_seconds"]
    d["sim_seconds"] = summary["sim_seconds"]
    d["epochs_per_second"] = summary["epochs_per_second"]
    d["patterns"] = final_patterns(run_dir)
    return d


COLUMNS = [("run", "run"), ("takeover_epoch", "takeover"),
           ("hoe_soft_epoch", "HOE>0.5"),
           ("entropy_collapse_epoch", "H collapse"),
           ("entropy_final", "H final"), ("zlib_final", "zlib final"),
           ("hoe_final", "HOE final"), ("hoe_max", "HOE max"),
           ("mean_steps_final", "steps/run"),
           ("frac_copy_final", "copy frac"), ("A_t_final", "A(t)"),
           ("A_t_rate_first_half", "dA/1k (1st half)"),
           ("A_t_rate_second_half", "dA/1k (2nd half)")]


def table(rows):
    head = "| " + " | ".join(c[1] for c in COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in COLUMNS) + "|"
    body = []
    for r in rows:
        body.append("| " + " | ".join(
            "-" if r[k] is None else str(r[k]) for k, _ in COLUMNS) + " |")
    return "\n".join([head, rule] + body)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    rows = [analyse(d) for d in args.runs]
    print(table(rows))
    print()
    for r in rows:
        print("### %s  (mode=%s seed=%s size=%s)" % (r["run"], r["mode"],
                                                     r["seed"], r["N_or_M"]))
        print("  wall %.1fs, %.1f epochs/s" % (r["wall_seconds"],
                                               r["epochs_per_second"]))
        for line in r["patterns"]:
            print("  " + line)
        print()
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
