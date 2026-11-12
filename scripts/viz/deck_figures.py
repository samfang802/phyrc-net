#!/usr/bin/env python3
"""Figures for the 2026-11-10 progress deck (results/deck_20261110/figures). No torch, CPU only.

Numbers come from the result json files where they exist, otherwise from the docs tables cited next
to each figure (docs/ARCHITECTURE_UPDATE_V11.md §9.4, docs/redesign/phase1_bidirectional_refiner.md §2.1,
docs/redesign/phase1_ilf.md §1.1).

  python scripts/viz/deck_figures.py --out results/deck_20261110/figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

plt.rcParams.update({
    "font.size": 18, "axes.titlesize": 20, "axes.labelsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16,
    "legend.fontsize": 16, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "font.family": "DejaVu Sans",
})


def _save(fig, out, name):
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[deck]", p)


def _bars(ax, labels, vals, colors, ref=None, ref_label=None, fmt="{:.2f}"):
    x = np.arange(len(labels))
    ax.bar(x, vals, width=0.62, color=colors, edgecolor="white", linewidth=2)
    for xi, v in zip(x, vals):
        ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=15, color=INK2)
    ax.set_xticks(x, labels)
    if ref is not None:
        ax.axhline(ref, color=INK2, ls="--", lw=2)
        ax.text(len(labels) - 0.45, ref, " " + ref_label, ha="left", va="center", fontsize=15, color=INK2,
                clip_on=False)


# --------------------------------------------------------------------------- slide 1: copy rates
def fig_copy_rates(out):
    d = json.load(open("results/eval_suite/final/test/diversity/diversity_test.json"))
    ref = d["reference"]["other_real_val"]
    names = ["V10", "V10.6", "V11", "V12", "cGAN"]
    crack = [d["models"][n]["copy"]["copy_rate"]["value"] for n in names]
    spall = [d["models"][n]["copy"]["spall_copy_rate"]["value"] for n in names]
    cols = [MUTED, MUTED, BLUE, BLUE, ORANGE]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    _bars(axs[0], names, crack, cols, ref["copy_rate"], "R307")
    axs[0].set_title("Crack copy rate ↓")
    _bars(axs[1], names, spall, cols, ref["spall_copy_rate"], "R307")
    axs[1].set_title("Spall-content copy rate ↓")
    for ax in axs:
        ax.set_ylim(0, 1.0)
    fig.tight_layout()
    _save(fig, out, "copy_rates.png")


# --------------------------------------------------------------------------- slide 3: GRU vs encoder
def fig_encoder_error(out):
    # docs/DAMAGE_STATE_ENCODER.md §8 (anylabel labels, 3-seed mean for A_info)
    rows = [("R_spall LOCO RMSE", 0.069, 0.064), ("R307 spall err 6%", 0.23, 0.05), ("R307 spall err 8%", 0.28, 0.17),
            ("CTR1 spall err 6%", -0.15, -0.17), ("CTR1 spall err 8%", -0.12, -0.20),
            ("DI LOCO RMSE", 0.133, 0.146), ("Peak-force RMSE", 0.077, 0.104)]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = np.arange(len(rows))[::-1]
    h = 0.36
    g = [abs(r[1]) for r in rows]
    z = [abs(r[2]) for r in rows]
    ax.barh(y + h / 2, g, h, color=MUTED, label="Original GRU g", edgecolor="white", linewidth=2)
    ax.barh(y - h / 2, z, h, color=BLUE, label="Damage-state encoder z", edgecolor="white", linewidth=2)
    for yi, a, b, r in zip(y, g, z, rows):
        ax.text(a + 0.005, yi + h / 2, f"{r[1]:+.2f}" if "err" in r[0] else f"{r[1]:.3f}", va="center", fontsize=14, color=INK2)
        ax.text(b + 0.005, yi - h / 2, f"{r[2]:+.2f}" if "err" in r[0] else f"{r[2]:.3f}", va="center", fontsize=14, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("|error|  (lower is better)")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 0.34)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    _save(fig, out, "encoder_vs_gru.png")


# --------------------------------------------------------------------------- slide 4: appearance code
def fig_app_fog(out):
    # docs/ARCHITECTURE_UPDATE_V11.md §9.4 (test)
    names = ["gn", "v11", "z only", "app\nnull", "app\nrandom", "app\nmean"]
    fog = [0.135, 0.324, 0.206, 0.184, 0.179, 0.081]
    E = [11.0, 7.7, 9.8, 9.3, 9.4, 10.5]
    cols = [MUTED, ORANGE, MUTED, MUTED, MUTED, BLUE]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    _bars(axs[0], names, fog, cols, 0.086, "GT")
    axs[0].set_title("High-DI fog share ↓")
    axs[0].set_ylim(0, 0.38)
    _bars(axs[1], names, E, cols, 10.8, "GT", fmt="{:.1f}")
    axs[1].set_title("Fine-texture energy E (→ GT)")
    axs[1].set_ylim(0, 13)
    fig.tight_layout()
    _save(fig, out, "app_code_fog.png")


# --------------------------------------------------------------------------- slide 5: hsp_P
def fig_hspP(out):
    f = pd.read_csv("results/phase1_fiber_clock/frames_with_clocks.csv")
    cols = {"R315": MUTED, "R330": MUTED, "R615": MUTED, "R1015": MUTED, "CTR1": ORANGE, "R307": BLUE}
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for c, col in cols.items():
        g = f[f.exp == c].sort_values("max_drift")
        if g.empty:
            continue
        lw, lab = (3, c) if c in ("CTR1", "R307") else (1.6, None)
        axs[0].plot(g.max_drift, g.hsp_P, color=col, lw=lw, label=lab)
        gg = g.groupby("level").agg(R=("R_spall", "mean"), h=("hsp_P", "mean")).reset_index()
        axs[1].plot(gg.h, gg.R, "o-", color=col, lw=lw, ms=8 if lab else 5, label=lab)
    axs[0].plot([], [], color=MUTED, lw=1.6, label="4 train columns")
    axs[0].set_xlabel("Max drift so far (%)")
    axs[0].set_ylabel("hsp_P (mm)")
    axs[0].set_title("Spalling height from fiber section")
    axs[0].legend(frameon=False, loc="upper left")
    axs[1].set_xlabel("hsp_P (mm)")
    axs[1].set_ylabel("Real spall area")
    axs[1].set_title("Spall area vs clock (per drift level)")
    fig.tight_layout()
    _save(fig, out, "hspP_clock.png")


# --------------------------------------------------------------------------- slide 6: gravity template
def fig_gravity(out):
    sys.argv = [sys.argv[0]]
    import importlib.util
    spec = importlib.util.spec_from_file_location("bcl", os.path.join(_ROOT, "scripts", "eval", "build_clock_layouts.py"))
    bcl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bcl)
    T = bcl.gravity_template()
    lay = np.load("results/phase1_fiber_layout/layouts/grav_hspP_test.npz")
    noi = np.load("results/phase1_fiber_layout/layouts/gravnoise_hspP_test.npz")
    fr = pd.read_csv("results/phase1_fiber_clock/frames_with_clocks.csv").set_index("filename")
    fn = list(lay["filename"])
    picks = []
    for code, lab in ((3000, "3%"), (4000, "4%"), (6000, "6%"), (8000, "8%")):
        c = [i for i, n in enumerate(fn) if n.startswith(f"CTR1_NE_{code:05d}_")]
        i = max(c, key=lambda i: fr.loc[fn[i], "hsp_P"])
        picks.append((i, lab))
    fig = plt.figure(figsize=(14, 4.4))
    gs = fig.add_gridspec(1, 6, width_ratios=[1.25, 0.3, 1, 1, 1, 1], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(T, cmap="Blues", vmin=0, vmax=max(T.max(), 1e-6))
    ax.set_title("Template T (spall freq.)", fontsize=17)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=13)
    for j, (i, lab) in enumerate(picks):
        a = fig.add_subplot(gs[2 + j])
        a.imshow(lay["layout"][i, 0] > 0.5, cmap="Blues", vmin=0, vmax=1.4)
        a.set_title(f"{lab}  A={lay['layout'][i, 0].mean():.2f}", fontsize=16)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for s in a.spines.values():
            s.set_visible(True); s.set_color(MUTED)
    _save(fig, out, "gravity_template.png")
    # jitter: same area, 4 draws at 8%
    i = picks[-1][0]
    j = list(noi["filename"]).index(fn[i])
    fig, axs = plt.subplots(1, 4, figsize=(9.6, 2.9))
    for k, a in enumerate(axs):
        a.imshow(noi["layout"][j, k] > 0.5, cmap="Blues", vmin=0, vmax=1.4)
        a.set_title(f"k = {k}", fontsize=16)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.suptitle("Template + boundary jitter (same area)", fontsize=18, y=1.04)
    _save(fig, out, "gravity_jitter.png")


# --------------------------------------------------------------------------- slide 9: refiner S use
def fig_sprobe(out):
    r = json.load(open("results/refiner_s_probe/test/report.json"))
    sel = [("Real data", None, INK2), ("Teacher forcing", "S1_gt_15k", AQUA),
           ("V11 refiner", "full_6k", MUTED), ("V12 refiner", "v12_6k", BLUE)]
    bands = ["lo", "mid", "hi"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axs[0]
    w = 0.2
    for j, (lab, key, col) in enumerate(sel):
        g = r[next(iter(r))]["real_gain"] if key is None else r[key]["gain"]["other"]
        vals = [g[f"gain_{b}"] for b in bands]
        ax.bar(np.arange(3) + (j - 1.5) * w, vals, w, color=col, label=lab, edgecolor="white", linewidth=2)
    ax.set_xticks(range(3), ["low freq", "mid freq", "high freq"])
    ax.set_ylabel("Output change / S change")
    ax.set_title("Single frame: swap S, how much output follows")
    ax.legend(frameon=False, fontsize=14)
    # rollout roll test, docs/redesign/phase1_ilf.md §1.1 (val)
    ax = axs[1]
    labs = ["original", "no shared\nnoise", "S from other\ntrajectory", "both"]
    keep = [0.878, 0.869, 0.752, 0.713]
    _bars(ax, labs, keep, [BLUE, BLUE, ORANGE, ORANGE], 0.948, "real", fmt="{:.2f}")
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("keep@1")
    ax.set_title("Rollout: keep per frame")
    ax.tick_params(axis="x", labelsize=13)
    fig.tight_layout()
    _save(fig, out, "refiner_s_use.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/deck_20261110/figures")
    ap.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    figs = {"copy": fig_copy_rates, "encoder": fig_encoder_error, "app": fig_app_fog, "hspP": fig_hspP,
            "gravity": fig_gravity, "sprobe": fig_sprobe}
    for k, fn in figs.items():
        if not a.only or k in a.only:
            fn(a.out)


if __name__ == "__main__":
    main()
