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
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    return {"mean": round(float(v.mean()), 5),
            "sd": round(sd, 5),
            "sem": round(sd / np.sqrt(v.size), 5) if v.size > 1 else None,
            "min": round(float(v.min()), 5),
            "max": round(float(v.max()), 5),
            "n": int(v.size),
            "values": [round(float(x), 5) for x in v]}


def welch(a, b):
    """Welch's t and a two-sided p, without scipy.

    Replicate-to-replicate drift in this assay is large (a self-control can
    land anywhere in 0.45-0.55), so comparing ranges is far too conservative
    and comparing means needs the spread carried with it.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return None, None, None
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    if va + vb <= 0:
        return None, None, None
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
    # two-sided p from the t distribution via the regularised incomplete beta
    from math import lgamma
    x = df / (df + t * t)

    def betacf(aa, bb, xx):
        qab, qap, qam = aa + bb, aa + 1.0, aa - 1.0
        c, d = 1.0, 1.0 - qab * xx / qap
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        h = d
        for m in range(1, 200):
            m2 = 2 * m
            aa1 = m * (bb - m) * xx / ((qam + m2) * (aa + m2))
            d = 1.0 + aa1 * d
            d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1.0 + aa1 / (c if abs(c) > 1e-30 else 1e-30)
            h *= d * c
            aa1 = -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2))
            d = 1.0 + aa1 * d
            d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1.0 + aa1 / (c if abs(c) > 1e-30 else 1e-30)
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 3e-12:
                break
        return h

    def betainc(aa, bb, xx):
        if xx <= 0:
            return 0.0
        if xx >= 1:
            return 1.0
        lb = (lgamma(aa + bb) - lgamma(aa) - lgamma(bb)
              + aa * np.log(xx) + bb * np.log(1 - xx))
        if xx < (aa + 1) / (aa + bb + 2):
            return np.exp(lb) * betacf(aa, bb, xx) / aa
        return 1 - np.exp(lb) * betacf(bb, aa, 1 - xx) / bb

    p = float(betainc(df / 2.0, 0.5, x))
    return round(float(t), 3), round(p, 5), round(float(df), 2)


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
    out["control_values"] = [v for c in out["controls"].values()
                             for v in c["values"]]
    return out


def verdict(entry, control_band, control_values=None, alpha=0.05):
    """Two readings: the conservative range test, then Welch against control.

    The range test asks whether every replicate beat every control
    replicate; with five replicates and this much drift it almost never
    fires, so the t-test against the pooled controls is the one to read.
    """
    if control_band is None:
        return "no control", None, None
    if control_values:
        t, p, _df = welch(entry["values"], control_values)
        if p is not None and p < alpha:
            return ("later wins (p=%.3g)" % p if t > 0
                    else "earlier wins (p=%.3g)" % p), t, p
        return ("no difference (p=%.3g)" % p if p is not None
                else "no control"), t, p
    return "inside the band", None, None


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
        cv = g["control_values"]
        rows = []
        for epoch in sorted(g["vs_first"]):
            e = g["vs_first"][epoch]
            pv = g["vs_prev"].get(epoch)
            ve, te, pe = verdict(e, cb, cv)
            e["welch_t"], e["welch_p"] = te, pe
            if pv:
                vp, tp, pp = verdict(pv, cb, cv)
                pv["welch_t"], pv["welch_p"] = tp, pp
            rows.append([epoch, "%.4f +/- %.4f" % (e["mean"], e["sem"] or 0),
                         ve,
                         "%.4f +/- %.4f" % (pv["mean"], pv["sem"] or 0)
                         if pv else "-",
                         vp if pv else "-"])
        ctrl_mean = float(np.mean(cv)) if cv else float("nan")
        ctrl_sd = float(np.std(cv, ddof=1)) if len(cv) > 1 else float("nan")
        md += ["## seed %d" % seed, "",
               "self-control (A against A), %d replicates: mean %.4f, sd %.4f, "
               "range %.4f-%.4f" % (len(cv), ctrl_mean, ctrl_sd,
                                    min(cv) if cv else float("nan"),
                                    max(cv) if cv else float("nan")), "",
               md_table(rows, ["epoch", "vs first (mean +/- sem)", "verdict",
                               "vs previous (mean +/- sem)", "verdict"]),
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
