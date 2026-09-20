"""Figures for the post-takeover analysis: assays, motif families, profiles.

Separate from soup.plots, which draws a run's metric columns.  Everything
here reads the CSV/JSON that soup.assay, soup.motifs and soup.posthoc write.

    python -m soup.figures assay analysis/assay/s12_*.csv --out fig.png
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

from .plots import GRID, INK, INK_2, SERIES, style

import matplotlib.pyplot as plt

NEUTRAL = "#8d8b85"


def read_csv(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for key in rows[0]:
        try:
            out[key] = np.array([float(r[key]) if r[key] != "" else np.nan
                                 for r in rows])
        except ValueError:
            out[key] = np.array([r[key] for r in rows])
    return out


def assay_band(path, field="frac_bytes_a"):
    """(epochs, mean, lo, hi) across replicates of one assay."""
    d = read_csv(path)
    epochs = np.unique(d["epoch"])
    vals = np.array([d[field][d["epoch"] == e] for e in epochs])
    return epochs, vals.mean(axis=1), vals.min(axis=1), vals.max(axis=1)


def _ctrl(ax, control, label="A vs A control"):
    if control is None:
        return
    e, m, lo, hi = assay_band(control)
    ax.fill_between(e, lo, hi, color=NEUTRAL, alpha=0.22, lw=0)
    ax.plot(e, m, color=NEUTRAL, lw=1.6, ls=(0, (4, 2)), label=label)


def assay_panel(ax, paths, labels, control=None, title="", field="frac_bytes_a"):
    ax.axhline(0.5, color=GRID, lw=1.2, zorder=0)
    _ctrl(ax, control)
    cmap = plt.get_cmap("viridis")
    for i, (p, lab) in enumerate(zip(paths, labels)):
        e, m, lo, hi = assay_band(p, field)
        c = cmap(0.08 + 0.82 * (i / max(len(paths) - 1, 1)))
        ax.fill_between(e, lo, hi, color=c, alpha=0.14, lw=0)
        ax.plot(e, m, lw=2.0, color=c, label=lab, solid_capstyle="round")
    ax.set_xlabel("assay epoch")
    ax.set_ylabel("fraction of bytes held by the later soup")
    ax.set_title(title, loc="left", color=INK)
    ax.margins(x=0.02)
    return ax


def plot_assays(groups, out_path, suptitle=None, field="frac_bytes_a"):
    """groups: [(panel_title, [(path, label)], control_path or None)]."""
    style()
    n = len(groups)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 4.3 * rows),
                             squeeze=False)
    for ax, (title, items, control) in zip(axes.ravel(), groups):
        assay_panel(ax, [p for p, _ in items], [l for _, l in items],
                    control, title, field)
        ax.legend(fontsize=8, loc="best", labelcolor=INK_2, ncol=2)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=13,
                     fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if suptitle else 1))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_advantage(series, out_path, title, xlabel="soup epoch",
                   ylabel="final share of bytes"):
    """series: [(label, xs, means, los, his)] -- advantage against soup age."""
    style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0.5, color=GRID, lw=1.2, zorder=0)
    for i, (label, xs, m, lo, hi) in enumerate(series):
        c = SERIES[i % len(SERIES)]
        ax.fill_between(xs, lo, hi, color=c, alpha=0.14, lw=0)
        ax.plot(xs, m, "o-", lw=2.0, ms=4, color=c, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=INK)
    ax.legend(fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_families(motif_csvs, out_path, suptitle=None, keep=5):
    """Family coverage over time, one panel per run, sweeps marked."""
    style()
    n = len(motif_csvs)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 4.0 * rows),
                             squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ax, (label, path) in zip(axes.ravel(), motif_csvs):
        d = read_csv(path)
        fams = d["family"]
        totals = {f: d["coverage"][fams == f].max() for f in np.unique(fams)}
        top = sorted(totals, key=lambda f: -totals[f])[:keep]
        for i, f in enumerate(top):
            sel = fams == f
            ax.plot(d["epoch"][sel], d["coverage"][sel], lw=1.9,
                    color=cmap(i % 10), label="family %d" % f)
        meta = path.rsplit(".csv", 1)[0] + ".json"
        if os.path.exists(meta):
            for sw in json.load(open(meta))["sweeps"]:
                ax.axvline(sw["epoch"], color=INK_2, lw=1.0, ls=":")
        ax.axhline(0.10, color=GRID, lw=1.0)
        ax.set_title(label, loc="left", color=INK)
        ax.set_xlabel("epoch")
        ax.set_ylabel("fraction of tapes carrying the family")
        ax.legend(fontsize=8, ncol=2, labelcolor=INK_2)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=13,
                     fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if suptitle else 1))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_selfsufficiency(posthoc_csvs, out_path, suptitle=None):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3))
    for i, (label, path) in enumerate(posthoc_csvs):
        d = read_csv(path)
        c = SERIES[i % len(SERIES)]
        axes[0].plot(d["epoch"], d["ss_colonisation"], lw=2.0, color=c,
                     label=label)
        axes[1].plot(d["epoch"], d["conserved_width"], lw=2.0, color=c,
                     label=label)
    axes[0].axhline(0.5, color=GRID, lw=1.2)
    axes[0].set_ylabel("share of the pair taken from a fresh random tape")
    axes[0].set_title("self-sufficiency", loc="left", color=INK)
    axes[1].set_ylabel("conserved bytes around the motif")
    axes[1].set_title("conserved region", loc="left", color=INK)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(fontsize=9, labelcolor=INK_2)
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=13,
                     fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if suptitle else 1))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_profiles(profile_json, out_path, epochs=None, title=""):
    """Per-offset entropy around the leading motif, a few epochs overlaid."""
    style()
    data = json.load(open(profile_json))["profiles"]
    keys = sorted(data, key=int)
    if epochs:
        keys = [k for k in keys if int(k) in set(epochs)]
    else:
        keys = [keys[i] for i in np.linspace(0, len(keys) - 1,
                                             min(5, len(keys))).astype(int)]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    cmap = plt.get_cmap("viridis")
    ax.axvspan(0, 15, color=GRID, alpha=0.7, lw=0)
    for i, k in enumerate(keys):
        p = data[k]
        y = np.array([np.nan if v is None else v for v in p["entropy"]])
        ax.plot(p["offsets"], y, lw=1.9,
                color=cmap(0.08 + 0.82 * (i / max(len(keys) - 1, 1))),
                label="epoch %s" % k)
    ax.set_xlabel("byte offset from the motif's first occurrence")
    ax.set_ylabel("entropy across tapes (bits)")
    ax.set_title(title or "shaded: the 16-byte seed window (conserved by "
                          "construction)", loc="left", color=INK)
    ax.legend(fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("kind", choices=["assay", "families", "selfsuff",
                                     "profiles"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--control", default=None)
    ap.add_argument("--title", default=None)
    a = ap.parse_args(argv)
    paths = [p for pat in a.paths for p in sorted(glob.glob(pat)) or [pat]]
    names = [os.path.basename(p).rsplit(".", 1)[0] for p in paths]
    if a.kind == "assay":
        print(plot_assays([(a.title or "", list(zip(paths, names)),
                            a.control)], a.out))
    elif a.kind == "families":
        print(plot_families(list(zip(names, paths)), a.out, a.title))
    elif a.kind == "selfsuff":
        print(plot_selfsufficiency(list(zip(names, paths)), a.out, a.title))
    else:
        print(plot_profiles(paths[0], a.out, title=a.title or ""))


if __name__ == "__main__":
    main()
