#!/usr/bin/env python3
"""Spread-direction plot: generated samples, train reals, test reals and cGAN in a 2-D DINOv3 feature plane.

Axes (fixed by reals and a fixed set of basis models, not by the model being plotted, so plots are comparable):
  x  real damage axis: mean(test reals with R_spall > 0.05) − mean(the rest), unit length.
  y  off-CTR1 direction: remove from every feature its part in span{x axis, top 10 PCs of the test reals}
     (the directions CTR1 itself varies along); PC1 of the remainder of the basis models (one sample per
     condition each) is the y axis, signed so the train reals have positive mean. y ≈ 0 is "looks like
     CTR1", |y| large is a column appearance CTR1 never shows.
All images are in the reference domain (photo → inpaint → CLAHE → JPEG q75), as in eval_diversity.py: our
models' samples go through JPEG q75 (--jpeg), the cGAN (trained on raw photos) through the generator's CLAHE
transform and then JPEG q75.

  python scripts/eval/plot_spread_direction.py --no_clahe \
      --model grav=results/eval_suite/v12gtrefiner/test/samples/v12gtrefiner_006000_grav.npz \
      --model gravnoise=results/eval_suite/v12gtrefiner/test/samples/v12gtrefiner_006000_gravnoise.npz \
      --jpeg grav=75 gravnoise=75 --out_dir results/eval_suite/diversity/spread_direction
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.metrics import dino_mmd as dm  # noqa: E402

_spec = importlib.util.spec_from_file_location("_kid", os.path.join(_ROOT, "scripts", "eval", "eval_dinov3_kid.py"))
_kid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_kid)

# (name, npz, CLAHE first, JPEG quality): the same reference-domain preprocessing as eval_diversity.py.
BASIS = [("V10.6C", "results/eval_suite/v106C/test/samples/v106C_6k.npz", False, 75),
         ("base", "results/eval_suite/v105/test/samples/gn.npz", False, 75),
         ("cGAN", "results/eval_suite/v106/test/progression_sel/cGAN_eq.npz", True, 75)]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", action="append", required=True, metavar="NAME=NPZ")
    p.add_argument("--jpeg", nargs="*", default=[], metavar="NAME=QUALITY")
    p.add_argument("--clahe", nargs="*", default=[], metavar="NAME", help="models drawn in the raw-photo domain (applied before --jpeg)")
    p.add_argument("--data_path", default="Data/DamagePattern/id_list_val_R307.json")
    p.add_argument("--image_dir", default="Data/DamagePattern_inpaint_selfcondnoDI/images")
    p.add_argument("--labels", default="Data/DamagePattern/anylabel_r_spall.csv")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--no_clahe", action="store_true")
    p.add_argument("--dino_model_id", default=dm.DEFAULT_DINO)
    p.add_argument("--cache_dir", default=_kid.SHARED_CACHE, help="feature cache shared with eval_dinov3_kid.py")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", required=True)
    cli = p.parse_args()
    os.makedirs(cli.out_dir, exist_ok=True)
    jpeg_q = {n: int(q) for n, q in (j.split("=", 1) for j in cli.jpeg)}

    feats = _kid._Cache(dm.DinoFeatures(cli.dino_model_id, device=cli.device), cli.cache_dir, 0, 0,
                        cache_dir=cli.cache_dir)

    def fake_feats(path, use_clahe, q):
        r = _kid._load_npz(path)
        fake = r["fake"]
        G, K = fake.shape[:2]
        u8 = fake.reshape(-1, *fake.shape[2:])
        if use_clahe:
            u8 = dm.clahe_uint8(u8, cli.image_size)
        if q:
            u8 = _kid._jpeg_u8(u8, q)
        return list(r["filename"]), feats(u8)[0].reshape(G, K, -1)

    basis = {n: fake_feats(pth, c, q) for n, pth, c, q in BASIS}
    names = basis["cGAN"][0]
    load = lambda fns: dm.load_real_uint8(list(fns), cli.image_dir, cli.image_size, not cli.no_clahe)
    real = feats(load(names))[0].astype(np.float64)
    train = feats(load(json.load(open(cli.data_path))["train"]))[0].astype(np.float64)
    rs = pd.read_csv(cli.labels).set_index("filename").loc[names, "R_spall"].to_numpy()

    mu = real.mean(0)
    damaged = rs > 0.05
    ax_d = real[damaged].mean(0) - real[~damaged].mean(0)
    ax_d /= np.linalg.norm(ax_d)
    vt = np.linalg.svd(real - mu, full_matrices=False)[2]
    B = np.linalg.qr(np.vstack([ax_d, vt[:10]]).T)[0]
    off = lambda x: (x - mu) - ((x - mu) @ B) @ B.T
    pooled = np.vstack([off(g[:, 0].astype(np.float64)) for _, g in basis.values()])
    ax_o = np.linalg.svd(pooled - pooled.mean(0), full_matrices=False)[2][0]
    if (off(train) @ ax_o).mean() < 0:
        ax_o = -ax_o
    proj = lambda x: ((x - mu) @ ax_d, (x - mu) @ ax_o)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#1f2328", "#5b6470", "#e3e6e4"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    rx, ry = proj(real)
    tx, ty = proj(train)
    cx, cy = proj(basis["cGAN"][1][:, 0].astype(np.float64))
    stats = {"test_real": {"var_x": rx.var(), "var_y": ry.var(), "mean_y": ry.mean()},
             "train": {"var_x": tx.var(), "var_y": ty.var(), "mean_y": ty.mean()},
             "cGAN": {"var_x": cx.var(), "var_y": cy.var(), "mean_y": cy.mean()}}
    for item in cli.model:
        name, path = item.split("=", 1)
        fns, g = fake_feats(path, name in cli.clahe, jpeg_q.get(name))
        if fns != names:
            raise ValueError(f"{name}: rows differ from the cGAN npz")
        G, K = g.shape[:2]
        g = g.astype(np.float64)
        gx, gy = proj(g[:, 0])
        ax_all, ay_all = proj(g.reshape(G * K, -1))
        ax_all, ay_all = ax_all.reshape(G, K), ay_all.reshape(G, K)
        stats[name] = {"k": K, "var_x": gx.var(), "var_y": gy.var(), "mean_y": gy.mean(),
                       "within_var_x": float(ax_all.var(1).mean()), "within_var_y": float(ay_all.var(1).mean())}

        for show_all in ([False, True] if K > 1 else [False]):
            fig, a = plt.subplots(figsize=(9.5, 6.6))
            a.scatter(tx, ty, s=30, marker="^", facecolor="none", edgecolor="#2a78d6", lw=0.9, alpha=0.45, label="train columns (real)", zorder=1)
            a.scatter(cx, cy, s=26, marker="D", facecolor="none", edgecolor="#8a4fd0", lw=0.9, alpha=0.6, label="cGAN", zorder=2)
            a.scatter(rx, ry, s=46, facecolor="none", edgecolor="#6b7178", lw=1.2, label="test real (CTR1)", zorder=3)
            px, py, pc = ((ax_all.ravel(), ay_all.ravel(), np.repeat(rs, K)) if show_all else (gx, gy, rs))
            sc = a.scatter(px, py, s=14 if show_all else 34, c=np.clip(pc, 0, 0.6), cmap="YlOrRd", vmin=0, vmax=0.6,
                           alpha=0.6 if show_all else 0.8, edgecolor="none", zorder=4)
            kk = K if show_all else 1
            a.set_title(f"{name} · feature spread, reference domain · K = {kk}", loc="left", color=INK, fontsize=14)
            txt = (f"model  var x {gx.var():.2f}  y {gy.var():.2f}  mean y {gy.mean():+.2f}\n"
                   f"test   var x {rx.var():.2f}  y {ry.var():.2f}\n"
                   f"cGAN   var x {cx.var():.2f}  y {cy.var():.2f}  mean y {cy.mean():+.2f}")
            if K > 1:
                txt += f"\nwithin-condition var x {stats[name]['within_var_x']:.2f}  y {stats[name]['within_var_y']:.2f}"
            a.text(0.98, 0.98, txt, transform=a.transAxes, fontsize=10.5, color=MUTED, family="DejaVu Sans Mono", ha="right", va="top")
            a.grid(color=GRID, lw=1)
            for s in ["top", "right"]:
                a.spines[s].set_visible(False)
            a.set_xlabel("real damage axis  (clean → damaged)", color=MUTED)
            a.set_ylabel("off-CTR1 direction (column appearance)", color=MUTED)
            a.legend(loc="lower left", frameon=False, fontsize=11)
            fig.colorbar(sc, ax=a, fraction=0.04, pad=0.02).set_label("R_spall of the condition", color=MUTED)
            out = os.path.join(cli.out_dir, f"spread_direction_{name}_k{kk}.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("saved", out)

    stats = {n: {k: (float(v) if not isinstance(v, int) else v) for k, v in s.items()} for n, s in stats.items()}
    json.dump(stats, open(os.path.join(cli.out_dir, "spread_direction_stats.json"), "w"), indent=2)
    for n, s in stats.items():
        print(f"{n:20s} " + "  ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items()))


if __name__ == "__main__":
    main()
