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

# --- end-state classifier ----------------------------------------------------
#
# Every finished run is put in exactly one class, from the median of its last
# five snapshots.  The thresholds are stated here, applied to every run
# identically, and printed with the answers.
#
#   random   memory still looks like the initial condition
#   crystal  low-entropy fixed point: loops purged, every run a straight walk
#   program  a repeated multi-byte pattern that contains brackets, and runs
#            that spend real time executing it
#
# "well above the window length" is the discriminator that matters: a run that
# only walks forward and falls off the end takes walk_len steps, so mean steps
# at several times that means loops are actually running.
CLASS_TAIL = 5            # snapshots averaged for the verdict
RANDOM_H = 7.0            # order-0 entropy this high means nothing took over
RANDOM_HOE = 0.5
CRYSTAL_H = 3.0
CRYSTAL_LOOP = 0.10       # fraction of steps spent re-running an instruction
CRYSTAL_STEPS = 1.5       # x walk_len
PROGRAM_HOE = 1.0
PROGRAM_STEPS = 3.0       # x walk_len
BRACKETS = set(b"[]")


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
        "A_t_naive_final": int(m["A_t_naive"][-1]) if "A_t_naive" in m else None,
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


def _tail(m, key, n=CLASS_TAIL):
    if key not in m or m[key].size == 0:
        return None
    return float(np.median(m[key][-min(n, m[key].size):]))


def top_patterns(run_dir, tail=CLASS_TAIL):
    """Top windows from the last ``tail`` blocks of patterns.log, as bytes.

    The tail, not just the final block: a program takeover's single most
    frequent window is often a run of one padding byte even while the program
    itself is everywhere, so one snapshot is a noisy view of what dominates.
    """
    try:
        with open(os.path.join(run_dir, "patterns.log")) as fh:
            lines = fh.read().splitlines()
    except OSError:
        return []
    starts = [i for i, l in enumerate(lines) if l.startswith("#")]
    out = []
    for b in starts[-tail:]:
        for line in lines[b + 1:]:
            if line.startswith("#"):
                break
            try:
                out.append(bytes.fromhex(line.split()[-1]))
            except (ValueError, IndexError):
                pass
    return out


CODE_LIKE_DISTINCT = 4   # distinct byte values a window needs to look like code


def looks_like_code(windows):
    """True if some dominant window carries control flow and real variety.

    A run of one byte is not code even when that byte is '['; a window needs a
    bracket *and* several distinct values before it counts as a program body.
    """
    return any(set(w) & BRACKETS and len(set(w)) >= CODE_LIKE_DISTINCT
               for w in windows)


def classify(m, walk_len, windows):
    """Label a finished run: random, crystal, program, or mixed."""
    h = _tail(m, "entropy_bits")
    hoe = _tail(m, "high_order_entropy")
    steps = _tail(m, "mean_steps")
    loop = _tail(m, "frac_steps_in_loop")
    if h is None or hoe is None:
        return "unknown", {}
    facts = {"H": h, "HOE": hoe, "mean_steps": steps, "frac_loop": loop,
             "walk_len": walk_len,
             "code_like_pattern": looks_like_code(windows)}
    if (hoe >= PROGRAM_HOE and steps is not None
            and steps >= PROGRAM_STEPS * walk_len
            and facts["code_like_pattern"]):
        return "program", facts
    if (h < CRYSTAL_H and (loop is None or loop < CRYSTAL_LOOP)
            and (steps is None or steps <= CRYSTAL_STEPS * walk_len)):
        return "crystal", facts
    if h >= RANDOM_H and hoe < RANDOM_HOE:
        return "random", facts
    return "mixed", facts


def analyse(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    spath = os.path.join(run_dir, "summary.json")
    # a run stopped by hand has metrics but no summary; report what it has
    summary = json.load(open(spath)) if os.path.exists(spath) else {}
    m = read_metrics(run_dir)
    rescored_path = os.path.join(run_dir, "rescored.csv")
    if os.path.exists(rescored_path):
        m_res = read_metrics(run_dir, "rescored.csv")
        for key in ("entropy_bits", "comp_bits", "high_order_entropy",
                    "A_t", "A_t_naive", "n_persistent"):
            if key in m_res:
                m[key + "_rescored"] = m_res[key]
    d = detect(m)
    d["run"] = os.path.basename(run_dir.rstrip("/"))
    d["mode"] = cfg["mode"]
    d["seed"] = cfg["seed"]
    d["N_or_M"] = cfg["M"] if cfg["mode"] == "ring" else cfg["N"]
    d["variant"] = ("d=%d" % cfg["d"] if cfg["mode"] == "blocks" else
                    ",".join(v for v, on in (("no-copy", cfg.get("no_copy")),
                                             ("indel", cfg.get("indel")),
                                             ("halt", cfg.get("head_bound") == "halt"))
                             if on) or "-")
    walk = summary.get("walk_len")
    if walk is None:
        walk = cfg["R"] + 1 if cfg["mode"] == "ring" else 128
    d["klass"], facts = classify(m, walk, top_patterns(run_dir))
    d["class_facts"] = facts
    d["frac_steps_in_loop"] = _tail(m, "frac_steps_in_loop")
    d["compat"] = cfg.get("compat", "none")
    if "A_t_rescored" in m:
        d["A_t_rescored"] = int(m["A_t_rescored"][-1])
        d["A_t_naive_rescored"] = int(m["A_t_naive_rescored"][-1])
        d["hoe_rescored"] = round(float(m["high_order_entropy_rescored"][-1]), 4)
    d["wall_seconds"] = summary.get("wall_seconds")
    d["sim_seconds"] = summary.get("sim_seconds")
    d["epochs_per_second"] = summary.get("epochs_per_second")
    d["complete"] = bool(summary)
    d["patterns"] = final_patterns(run_dir)
    return d


COLUMNS = [("run", "run"), ("klass", "class"),
           ("takeover_epoch", "takeover"),
           ("entropy_final", "H"), ("hoe_final", "HOE"), ("hoe_max", "HOE max"),
           ("mean_steps_final", "steps/run"),
           ("frac_steps_in_loop", "in-loop"),
           ("frac_copy_final", "copy frac"),
           ("A_t_final", "A(t)"), ("A_t_naive_final", "A(t) naive")]


def table(rows):
    head = "| " + " | ".join(c[1] for c in COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in COLUMNS) + "|"
    body = []
    for r in rows:
        body.append("| " + " | ".join(
            "-" if r[k] is None else str(r[k]) for k, _ in COLUMNS) + " |")
    return "\n".join([head, rule] + body)


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion.

    Wilson rather than the normal approximation because the counts here are
    small and the proportion can sit at 0 or 1, where the normal interval has
    zero width and is simply wrong.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def rate_report(rows):
    """Transition rate per configuration, with a confidence interval."""
    groups = {}
    for r in rows:
        key = (r["mode"], r.get("compat", "none"), r["N_or_M"],
               r.get("variant", "-"))
        groups.setdefault(key, []).append(r)
    out = []
    for key, rs in sorted(groups.items()):
        k = sum(1 for r in rs if r["klass"] == "program")
        p, lo, hi = wilson(k, len(rs))
        epochs = sorted(r["takeover_epoch"] for r in rs
                        if r["takeover_epoch"] is not None)
        out.append({"mode": key[0], "compat": key[1], "size": key[2],
                    "variant": key[3],
                    "n": len(rs), "programs": k,
                    "rate": round(p, 3), "ci_low": round(lo, 3),
                    "ci_high": round(hi, 3),
                    "takeover_epochs": epochs})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--rates", action="store_true",
                    help="also report the program-class rate per configuration")
    args = ap.parse_args(argv)
    dirs = [d for d in args.runs
            if os.path.isfile(os.path.join(d, "config.json"))]
    skipped = [d for d in args.runs if d not in dirs]
    if skipped:
        print("skipping %d path(s) that are not run directories\n" % len(skipped))
    rows = [analyse(d) for d in dirs]
    print(table(rows))
    print()
    if args.rates:
        print("| configuration | n | program | rate | 95% CI (Wilson) | takeover epochs |")
        print("|---|---|---|---|---|---|")
        for g in rate_report(rows):
            print("| %s%s N=%s%s | %d | %d | %.2f | %.2f-%.2f | %s |"
                  % (g["mode"],
                     "" if g["compat"] == "none" else " (%s)" % g["compat"],
                     g["size"],
                     "" if g["variant"] == "-" else " %s" % g["variant"],
                     g["n"], g["programs"], g["rate"],
                     g["ci_low"], g["ci_high"],
                     ", ".join(str(e) for e in g["takeover_epochs"]) or "-"))
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
            json.dump({"runs": rows, "rates": rate_report(rows)}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
