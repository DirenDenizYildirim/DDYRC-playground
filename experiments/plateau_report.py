"""Turn the assay / motif / post-takeover outputs into numbers and figures.

Reads whatever ``experiments.phase2`` has produced so far and writes

    <out>/summary.json     every number, machine-readable
    <out>/summary.md       the same as tables, ready to paste into RESULTS
    <out>/assay_*.png      the competition curves
    <out>/advantage.png    final advantage against soup age
    <out>/families.png     motif family coverage over time
    <out>/selfsuff.png     self-sufficiency and conserved width over time
    <out>/profiles_*.png   per-offset entropy around the leading motif

    python -m experiments.plateau_report --out analysis/plateau_report
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from soup import figures

SEEDS = (12, 13, 14, 19)
EPOCH_RE = re.compile(r"_(\d{8})$")


def finals(path, field="frac_bytes_a"):
    """Per-replicate value at the last assay epoch."""
    d = figures.read_csv(path)
    last = d["epoch"].max()
    return d[field][d["epoch"] == last]


def band(values):
    v = np.asarray(values, dtype=float)
    return {"mean": round(float(v.mean()), 5),
            "min": round(float(v.min()), 5),
            "max": round(float(v.max()), 5),
            "n": int(v.size)}


def gather_seed(assay_dir):
    """Everything one continued run's assays say."""
    out = {"dir": assay_dir, "vs_first": {}, "vs_prev": {}, "controls": {}}
    for name in ("control_first", "control_last"):
        p = os.path.join(assay_dir, name + ".csv")
        if os.path.exists(p):
            out["controls"][name] = band(finals(p))
    for kind in ("vs_first", "vs_prev"):
        for p in sorted(glob.glob(os.path.join(assay_dir, kind + "_*.csv"))):
            m = EPOCH_RE.search(os.path.basename(p).rsplit(".csv", 1)[0])
            if m:
                out[kind][int(m.group(1))] = band(finals(p))
    ctrl = [v for c in out["controls"].values() for v in (c["min"], c["max"])]
    out["control_band"] = ([round(min(ctrl), 5), round(max(ctrl), 5)]
                           if ctrl else None)
    return out


def verdict(entry, control_band):
    """Does this comparison leave the neutral band?"""
    if control_band is None:
        return "no control"
    if entry["min"] > control_band[1]:
        return "later wins"
    if entry["max"] < control_band[0]:
        return "earlier wins"
    return "inside the band"


def md_table(rows, headers):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--assays", default="analysis/plateau")
    ap.add_argument("--motifs", default="analysis/motifs")
    ap.add_argument("--posthoc", default="analysis/posthoc")
    ap.add_argument("--out", default="analysis/plateau_report")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    summary, md = {}, ["# Plateau assays", ""]
    groups, adv_first, adv_prev = [], [], []
    for seed in a.seeds:
        tag = "plateau_s%d" % seed
        d = os.path.join(a.assays, tag)
        if not os.path.isdir(d):
            continue
        g = gather_seed(d)
        summary[tag] = g
        cb = g["control_band"]
        rows = []
        for epoch in sorted(g["vs_first"]):
            e = g["vs_first"][epoch]
            p = g["vs_prev"].get(epoch)
            rows.append([epoch, "%.4f" % e["mean"],
                         "%.4f-%.4f" % (e["min"], e["max"]),
                         verdict(e, cb),
                         "%.4f" % p["mean"] if p else "-",
                         verdict(p, cb) if p else "-"])
        md += ["## seed %d" % seed, "",
               "neutral band from the two self-controls: %s" %
               ("%.4f - %.4f" % tuple(cb) if cb else "not measured"), "",
               md_table(rows, ["epoch", "vs first (mean)", "vs first (range)",
                               "verdict", "vs previous (mean)", "verdict"]),
               ""]
        items = [(os.path.join(d, "vs_first_%08d.csv" % e), "epoch %d" % e)
                 for e in sorted(g["vs_first"])]
        ctrl = os.path.join(d, "control_first.csv")
        groups.append(("seed %d: each soup against the first" % seed, items,
                       ctrl if os.path.exists(ctrl) else None))
        xs = np.array(sorted(g["vs_first"]))
        if xs.size:
            adv_first.append(("seed %d" % seed, xs,
                              np.array([g["vs_first"][e]["mean"] for e in xs]),
                              np.array([g["vs_first"][e]["min"] for e in xs]),
                              np.array([g["vs_first"][e]["max"] for e in xs])))
        xp = np.array(sorted(g["vs_prev"]))
        if xp.size:
            adv_prev.append(("seed %d" % seed, xp,
                             np.array([g["vs_prev"][e]["mean"] for e in xp]),
                             np.array([g["vs_prev"][e]["min"] for e in xp]),
                             np.array([g["vs_prev"][e]["max"] for e in xp])))

    made = []
    if groups:
        made.append(figures.plot_assays(
            groups, os.path.join(a.out, "assay_vs_first.png"),
            "each saved soup against the first post-takeover soup"))
    if adv_first:
        made.append(figures.plot_advantage(
            adv_first, os.path.join(a.out, "advantage_vs_first.png"),
            "advantage over the first post-takeover soup"))
    if adv_prev:
        made.append(figures.plot_advantage(
            adv_prev, os.path.join(a.out, "advantage_vs_prev.png"),
            "advantage over the soup one step earlier"))

    fam = [("seed %d" % s, os.path.join(a.motifs, "s%d.csv" % s))
           for s in a.seeds
           if os.path.exists(os.path.join(a.motifs, "s%d.csv" % s))]
    if fam:
        made.append(figures.plot_families(
            fam, os.path.join(a.out, "families.png"),
            "motif family coverage, dotted lines are sweeps"))
        summary["sweeps"] = {}
        for label, path in fam:
            meta = path.rsplit(".csv", 1)[0] + ".json"
            if os.path.exists(meta):
                summary["sweeps"][label] = json.load(open(meta))["sweeps"]

    ph = [("seed %d" % s, os.path.join(a.posthoc, "s%d.csv" % s))
          for s in a.seeds
          if os.path.exists(os.path.join(a.posthoc, "s%d.csv" % s))]
    if ph:
        made.append(figures.plot_selfsufficiency(
            ph, os.path.join(a.out, "selfsuff.png")))
        summary["posthoc"] = {}
        rows = []
        for label, path in ph:
            d = figures.read_csv(path)
            summary["posthoc"][label] = {
                k: [None if not np.isfinite(x) else round(float(x), 5)
                    for x in d[k]]
                for k in ("epoch", "ss_colonisation", "ss_colonisation_sem",
                          "ctrl_soup_colonisation", "conserved_width",
                          "leader_coverage", "distinct_tapes")
                if k in d}
            for i in (0, len(d["epoch"]) // 2, len(d["epoch"]) - 1):
                rows.append([label, int(d["epoch"][i]),
                             "%.4f" % d["ss_colonisation"][i],
                             "%.4f" % d["ctrl_soup_colonisation"][i],
                             int(d["distinct_tapes"][i]),
                             int(d["conserved_width"][i])])
            prof = path.rsplit(".csv", 1)[0] + "_profiles.json"
            if os.path.exists(prof):
                made.append(figures.plot_profiles(
                    prof, os.path.join(a.out, "profiles_%s.png"
                                       % label.replace(" ", ""))))
        md += ["## self-sufficiency and structure", "",
               md_table(rows, ["run", "epoch", "colonisation vs random",
                               "soup-vs-soup control", "distinct tapes",
                               "conserved bytes"]), ""]

    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(a.out, "summary.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    for m in made:
        print(m)
    print(os.path.join(a.out, "summary.md"))


if __name__ == "__main__":
    main()
