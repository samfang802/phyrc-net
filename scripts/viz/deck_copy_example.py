#!/usr/bin/env python3
"""Copy examples for the 2026-11-10 deck: the cGAN sample whose cracks best match a training image, and the
cGAN spall patch with the highest NCC to a training spall patch. Same domain and settings as eval_diversity.py
(inpainted reference set, no CLAHE, JPEG q75, tolerance 3 px, 16 px patches every 8 px).

  python scripts/viz/deck_copy_example.py --out results/deck_20261110/figures
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
from src.metrics import damage_texture as dtx  # noqa: E402
from src.metrics import dino_mmd as dm  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, "scripts", "eval", f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_kid = _load("eval_dinov3_kid")
_seg = _load("phase0_layout_oracle_spallseg")
ORANGE, BLUE, INK2 = "#eb6834", "#2a78d6", "#52514e"


def patches(u8, masks, size=16, stride=8, min_cover=0.6, min_std=6.0):
    x = torch.from_numpy(np.asarray(u8)).float()[:, None]
    m = torch.from_numpy(np.asarray(masks)).float()[:, None]
    P = F.unfold(x, size, stride=stride).transpose(1, 2)
    cover = F.unfold(m, size, stride=stride).transpose(1, 2).mean(-1)
    sd = P.std(-1)
    ok = (cover >= min_cover) & (sd > min_std)
    n_side = (u8.shape[-1] - size) // stride + 1
    img, loc = torch.nonzero(ok, as_tuple=True)
    Pz = (P - P.mean(-1, keepdim=True)) / (sd[..., None] + 1e-6) / size
    yx = np.stack([(loc // n_side).numpy() * stride, (loc % n_side).numpy() * stride], 1)
    return Pz[ok], img.numpy(), yx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/eval_suite/cgan_clahe/test/samples/cGAN_clahe_eq.npz")
    ap.add_argument("--image_dir", default="Data/DamagePattern_inpaint_selfcondnoDI/images")
    ap.add_argument("--data_path", default="Data/DamagePattern/id_list_val_R307.json")
    ap.add_argument("--segmenter", default="results/phase0_layout_oracle/spallseg/best.pt")
    ap.add_argument("--out", default="results/deck_20261110/figures")
    ap.add_argument("--tag", default="cGAN", help="model name in titles; file suffix unless cGAN")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    run = _kid._load_npz(a.npz)
    names = list(run["filename"])
    K = run["fake"].shape[1]
    fake = _kid._jpeg_u8(run["fake"], 75).reshape(-1, *run["fake"].shape[-2:])   # row c*K + k
    tag, suf = a.tag, ("" if a.tag == "cGAN" else f"_{a.tag}")
    load = lambda f: dm.load_real_uint8(list(f), a.image_dir, 128, False)
    test_u8 = load(names)
    train_names = json.load(open(a.data_path))["train"]
    train_u8 = load(train_names)
    net, thr = _seg.load_segmenter(a.segmenter, dev)
    truth = _seg.predict_masks(net, test_u8, dev, thr).reshape(len(names), -1).mean(1)

    # crack: no-spall conditions, best F1 against every train image
    ns = np.concatenate([np.arange(c * K, (c + 1) * K) for c in np.flatnonzero(truth <= 0.02)])
    fc, tc = dtx.crack_map(fake[ns]), dtx.crack_map(train_u8)
    Fm = dtx.crack_f1(fc, tc, 3)
    i, j = np.unravel_index(np.argmax(Fm), Fm.shape)
    gi = ns[i]
    gc, gk = gi // K, gi % K
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.6))
    axs[0].imshow(fake[gi], cmap="gray", vmin=0, vmax=255)
    axs[0].set_title(f"{tag} sample\n({names[gc][:13]}, k={gk})", fontsize=17)
    axs[1].imshow(train_u8[j], cmap="gray", vmin=0, vmax=255)
    axs[1].set_title(f"Best-matching train image\n({train_names[j][:13]})", fontsize=17)
    ov = np.ones(fc[i].shape + (3,))
    ov[tc[j]] = [0.16, 0.47, 0.84]
    ov[fc[i]] = [0.92, 0.41, 0.20]
    ov[fc[i] & tc[j]] = [0.1, 0.1, 0.1]
    axs[2].imshow(ov)
    axs[2].set_title(f"Crack overlap (F1 = {Fm[i, j]:.2f})", fontsize=17)
    axs[2].legend(handles=[mpatches.Patch(color=ORANGE, label=tag), mpatches.Patch(color=BLUE, label="train"),
                           mpatches.Patch(color="#1a1a1a", label="both")], loc="lower center",
                  bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=15)
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"copy_crack_example{suf}.png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[copy] crack", tag, names[gc], gk, train_names[j], float(Fm[i, j]))

    # spall content: highest-NCC cGAN patch vs every train spall patch
    sp = np.concatenate([np.arange(c * K, (c + 1) * K) for c in np.flatnonzero(truth > 0.02)])
    fmask = _seg.predict_masks(net, fake[sp], dev, thr)
    tmask = _seg.predict_masks(net, train_u8, dev, thr)
    has = np.flatnonzero(tmask.reshape(len(train_u8), -1).mean(1) > 0.02)
    Pf, If, Lf = patches(fake[sp], fmask)
    Pt, It, Lt = patches(train_u8[has], tmask[has])
    S = (Pf.to(dev) @ Pt.to(dev).T)
    best, arg = S.max(1)
    p = int(best.argmax())
    q = int(arg[p])
    fi, ti = sp[If[p]], has[It[q]]
    fc_, fk_ = fi // K, fi % K
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.8))
    for ax, img, (y, x), title, col in ((axs[0], fake[fi], Lf[p], f"{tag} ({names[fc_][:13]}, k={fk_})", ORANGE),
                                         (axs[1], train_u8[ti], Lt[q], f"train ({train_names[ti][:13]})", BLUE)):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.add_patch(mpatches.Rectangle((x - 0.5, y - 0.5), 16, 16, fill=False, ec=col, lw=3))
        ax.set_title(title, fontsize=17)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Most similar spall patch (NCC = {float(best[p]):.2f})", fontsize=18, y=1.06)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"copy_spall_example{suf}.png"), dpi=200, bbox_inches="tight", facecolor="white")
    print("[copy] spall", tag, names[fc_], fk_, train_names[ti], float(best[p]))


if __name__ == "__main__":
    main()
