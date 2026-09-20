"""Figures: high-order entropy, A(t), and the kymograph.

    python -m soup.plots runs/ring_s1 runs/ring_s2 runs/ring_s3 --out analysis/ring

Colour is assigned by the job it does, not by taste:
  * seeds are a categorical identity -> the first three categorical slots,
    which are the ones that stay separable for colour-vision deficiency when
    every pair can appear together;
  * in the kymograph, instruction bytes carry identity (three functional
    families -> three categorical slots) and no-op bytes carry magnitude
    (one recessive grey ramp), so instructions are never confusable with
    data and the data still shows its own texture.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

# --- design tokens (light surface) -------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # blue, orange, aqua

# kymograph families
FAM_COPY = "#2a78d6"     # . ,
FAM_LOOP = "#eb6834"     # [ ]
FAM_MOVE = "#1baf7a"     # < > { } - +

COPY_BYTES = b".,"
LOOP_BYTES = b"[]"
MOVE_BYTES = b"<>{}-+"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.dpi": 130,
    })


def read_metrics(run_dir):
    with open(os.path.join(run_dir, "metrics.csv")) as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for key in rows[0]:
        try:
            out[key] = np.array([float(r[key]) for r in rows])
        except ValueError:
            pass
    return out


def _direct_labels(ax, series, x, min_gap_frac=0.055):
    """Label lines at their right end, nudged apart so they never collide."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    items = sorted(series, key=lambda s: s[1])          # by final y
    placed = []
    for label, y, color in items:
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.annotate(label, xy=(x, y), xytext=(6, 0), textcoords="offset points",
                    color=color, fontsize=9, va="center", fontweight="semibold",
                    annotation_clip=False)


def line_panel(ax, runs, field, title, ylabel, logy=False):
    ends = []
    for i, (label, m) in enumerate(runs):
        color = SERIES[i % len(SERIES)]
        ax.plot(m["epoch"], m[field], lw=2.0, color=color, label=label,
                solid_capstyle="round")
        ends.append((label, m[field][-1], color))
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.margins(x=0.02)
    if len(runs) > 1:
        # legend below the axes so it can never sit on top of a transition
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
                  ncol=min(len(runs), 4), fontsize=9, labelcolor=INK_2,
                  handlelength=1.6, columnspacing=1.6)
        _direct_labels(ax, ends, max(m["epoch"][-1] for _, m in runs))
    return ax


def plot_series(runs, out_path, field, title, ylabel, logy=False, note=None):
    style()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    line_panel(ax, runs, field, title, ylabel, logy)
    if note:
        fig.text(0.01, -0.06, note, color=INK_2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --- kymograph ---------------------------------------------------------------

def kymo_rgb(kymo):
    """Map bytes to RGB: three instruction families + a grey ramp for no-ops."""
    lut = np.zeros((256, 3), dtype=np.float64)
    # no-ops: a light grey ramp, recessive but still showing byte texture
    v = np.linspace(0.94, 0.74, 256)
    lut[:, 0] = lut[:, 1] = v
    lut[:, 2] = v * 0.995
    for raw, hexcolor in ((COPY_BYTES, FAM_COPY), (LOOP_BYTES, FAM_LOOP),
                          (MOVE_BYTES, FAM_MOVE)):
        rgb = np.array([int(hexcolor[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        for b in raw:
            lut[b] = rgb
    return lut[kymo]


def plot_kymograph(run_dir, out_path, epochs_per_row, raw_path=None):
    kymo = np.load(os.path.join(run_dir, "kymograph.npy"))
    rgb = kymo_rgb(kymo)
    if raw_path:
        plt.imsave(raw_path, rgb)          # exactly one pixel per byte

    style()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.imshow(rgb, aspect="auto", interpolation="nearest", origin="upper",
              extent=[0, kymo.shape[1], kymo.shape[0] * epochs_per_row, 0])
    ax.set_xlabel("memory position (bytes)")
    ax.set_ylabel("epoch")
    ax.set_title("kymograph: %d-byte slice of memory over time"
                 % kymo.shape[1], loc="left", color=INK)
    ax.grid(False)
    handles = [
        Line2D([], [], marker="s", ls="", ms=9, color=FAM_COPY, label="copy  . ,"),
        Line2D([], [], marker="s", ls="", ms=9, color=FAM_LOOP, label="loop  [ ]"),
        Line2D([], [], marker="s", ls="", ms=9, color=FAM_MOVE, label="head/arith  < > { } - +"),
        Line2D([], [], marker="s", ls="", ms=9, color="#d8d7d2", label="no-op byte (shade = value)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=4, fontsize=8.5, labelcolor=INK_2, handletextpad=0.4,
              columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --- byte composition --------------------------------------------------------

INSTR_ORDER = b"<>{}-+.,[]"


def byte_composition(run_dir):
    """Frequency of each instruction byte over time, from the raw .npy dumps."""
    import glob
    epochs, freqs, noop = [], [], []
    for f in sorted(glob.glob(os.path.join(run_dir, "snapshots", "*.npy"))):
        mem = np.load(f).reshape(-1)
        counts = np.bincount(mem, minlength=256) / mem.size
        epochs.append(int(os.path.basename(f)[6:-4]))
        freqs.append([counts[b] for b in INSTR_ORDER])
        noop.append(1.0 - sum(counts[b] for b in INSTR_ORDER))
    return np.array(epochs), np.array(freqs), np.array(noop)


def plot_composition(run_dirs, out_path, title):
    """Small multiples: one panel per instruction byte, shared y scale.

    Ten series cannot be told apart by hue, so they are faceted instead --
    identity comes from the panel, not the colour.
    """
    style()
    data = [(lab, byte_composition(d)) for lab, d in run_dirs]
    ymax = max(f.max() for _, (_, f, _) in data) * 1.12
    fig, axes = plt.subplots(2, 5, figsize=(11.5, 4.6), sharex=True, sharey=True)
    for j, b in enumerate(INSTR_ORDER):
        ax = axes[j // 5][j % 5]
        for i, (lab, (ep, fr, _)) in enumerate(data):
            ax.plot(ep, fr[:, j], lw=1.8, color=SERIES[i % len(SERIES)], label=lab)
        ax.axhline(1 / 256, color="#b9b8b3", lw=1.0, ls=(0, (3, 3)))
        ax.set_title("  %s" % chr(b), loc="left", color=INK, fontsize=13)
        ax.set_ylim(0, ymax)
        ax.tick_params(labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel("epoch", fontsize=9)
    axes[0][0].set_ylabel("fraction of memory", fontsize=9)
    axes[1][0].set_ylabel("fraction of memory", fontsize=9)
    fig.suptitle(title + "   (dashed line = 1/256, the random-soup level)",
                 x=0.005, ha="left", color=INK, fontsize=12, fontweight="bold")
    if len(data) > 1:
        axes[1][2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.32),
                          ncol=len(data), fontsize=9, labelcolor=INK_2)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --- entry point -------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="plots for one or more soup runs")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    import json
    loaded = []
    for d in args.runs:
        cfg = json.load(open(os.path.join(d, "config.json")))
        loaded.append(("seed %d" % cfg["seed"], read_metrics(d), d, cfg))

    runs = [(lab, m) for lab, m, _, _ in loaded]
    tag = args.label or loaded[0][3]["mode"]
    plot_series(runs, os.path.join(args.out, "high_order_entropy.png"),
                "high_order_entropy",
                "%s: high-order entropy" % tag,
                "bits/byte  (H - zlib)",
                note="entropy of the byte distribution minus zlib(9) bits per byte")
    plot_series(runs, os.path.join(args.out, "a_t.png"), "A_t",
                "%s: A(t), cumulative novel persistent 8-byte patterns" % tag,
                "distinct windows")
    plot_series(runs, os.path.join(args.out, "mean_steps.png"), "mean_steps",
                "%s: mean steps executed per run" % tag, "steps")
    plot_series(runs, os.path.join(args.out, "frac_copy_runs.png"),
                "frac_copy_runs",
                "%s: fraction of runs executing at least one copy" % tag,
                "fraction")
    plot_composition([(lab, d) for lab, _, d, _ in loaded],
                     os.path.join(args.out, "byte_composition.png"),
                     "%s: instruction-byte composition of memory" % tag)
    for lab, _, d, cfg in loaded:
        plot_kymograph(d, os.path.join(args.out, "kymograph_%s.png"
                                       % lab.replace(" ", "")),
                       cfg["snapshot_interval"],
                       raw_path=os.path.join(args.out, "kymograph_%s_raw.png"
                                             % lab.replace(" ", "")))
    print("wrote plots to", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
