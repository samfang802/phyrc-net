#!/usr/bin/env python3
"""Laplacian pyramid on 5 test GT images (no CFM inference).

    x^(3) = down(down(x))
    x^(2) = down(x) - up(x^(3))
    x^(1) = x - up(down(x))
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def down(x):
    return F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)


def up(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def laplacian_bands(x01):
    """x01: [C,H,W] in [0,1] → x, x1, x2, x3, recon all as [H,W] display arrays."""
    if x01.ndim == 2:
        x01 = x01.unsqueeze(0)
    x = x01.unsqueeze(0).float()
    x_d1 = down(x)
    x3 = down(x_d1)
    x2 = x_d1 - up(x3, size=x_d1.shape[-2:])
    x1 = x - up(x_d1, size=x.shape[-2:])
    recon = x1 + up(x2 + up(x3, size=x2.shape[-2:]), size=x.shape[-2:])

    def to_hw(t):
        g = t[0].mean(dim=0).detach().cpu()
        return g

    x_hw = to_hw(x).clamp(0, 1).numpy()
    x3_full = to_hw(up(up(x3, size=x2.shape[-2:]), size=x.shape[-2:])).clamp(0, 1).numpy()
    x2_full = to_hw(up(x2, size=x.shape[-2:]))
    x1_hw = to_hw(x1)
    recon_hw = to_hw(recon).clamp(0, 1).numpy()
    return {
        "x": x_hw,
        "x3": x3_full,
        "x2": (0.5 + 2.0 * x2_full).clamp(0, 1).numpy(),
        "x1": (0.5 + 2.0 * x1_hw).clamp(0, 1).numpy(),
        "recon": recon_hw,
        "err": float((to_hw(x) - to_hw(recon)).abs().max()),
        "h3": int(x3.shape[-2]),
        "h2": int(x2.shape[-2]),
        "h1": int(x1.shape[-2]),
    }


def save_one(gt01, di_value, out_path):
    b = laplacian_bands(gt01)
    fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.2))
    panels = [
        (b["x"], r"$x$  (GT)"),
        (b["x3"], rf"$x^{{(3)}}=down(down(x))$  [{b['h3']}px↑]"),
        (b["x2"], rf"$x^{{(2)}}=down(x)-up(x^{{(3)}})$  [{b['h2']}px↑]"),
        (b["x1"], rf"$x^{{(1)}}=x-up(down(x))$  [{b['h1']}px]"),
        (b["recon"], r"$x^{{(1)}}+up(x^{{(2)}}+up(x^{{(3)}}))$"),
    ]
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle(
        f"Laplacian pyramid (GT only, no inference)  ·  test CTR1  ·  DI={float(di_value):.2f}  "
        f"·  recon max|err|={b['err']:.2e}",
        fontsize=11,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[lap] saved {out_path}  recon_err={b['err']:.2e}", flush=True)


def _to_nchw01(img):
    x = img.detach().float().cpu() if torch.is_tensor(img) else torch.as_tensor(np.asarray(img), dtype=torch.float32)
    if x.ndim == 3 and x.shape[0] not in (1, 3):
        x = x.permute(2, 0, 1)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if float(x.max()) > 1.5:
        x = x / 255.0
    return x.clamp(0.0, 1.0)


def main():
    import test_cfm_checkpoint as tcc

    def gt_only(self, anchors, di_values):
        items = []
        for a, di in zip(anchors, di_values):
            items.append({"anchor": a, "sampled_vis": a["real"], "di_value": float(di)})
        return items

    def wrapped(items, di_values, step, result_dir, *args, **kwargs):
        out_dir = os.path.join(result_dir, "inference_samples", "haar_dwt_iwt")
        for item, di in zip(items, di_values):
            gt = _to_nchw01(item["anchor"]["real"])
            di_tag = f"{float(di):.2f}".replace(".", "p")
            save_one(gt, di, os.path.join(out_dir, f"laplacian_gt_di{di_tag}.png"))
        print("[lap] skip inference panel (GT Laplacian only)", flush=True)
        return os.path.join(out_dir, "laplacian_gt_di0p00.png")

    tcc.CFMADM = getattr(tcc, "CFMADM", None)
    tcc._run_test_inference_panel
    orig_run = tcc._run_test_inference_panel

    def run_gt_panel(model, dataloader, split, di_values, step, result_dir, device, anchor_exp, extra_tag=""):
        from test_cfm_checkpoint import (
            _collect_split_anchors,
            _pool_key,
            _resolve_anchor_exp,
        )

        del _resolve_anchor_exp
        pool_key = _pool_key(split)
        model.validation_outputs = {
            "gt_DI": [],
            "pred_DI": [],
            "train_anchor_candidates": [],
            "val_anchor_candidates": [],
            "test_anchor_candidates": [],
        }
        for cache_attr in (
            "_fixed_train_inference_anchors",
            "_fixed_val_inference_anchors",
            "_fixed_train_inference_anchor",
            "_fixed_val_inference_anchor",
        ):
            if hasattr(model, cache_attr):
                delattr(model, cache_attr)
        print(f"[lap] Collecting {split} GT anchors (no ODE)...", flush=True)
        _collect_split_anchors(model, dataloader, split)
        candidates = model.validation_outputs[pool_key]
        anchors = model._select_anchors_for_di_values(
            candidates, di_values, split=split, anchor_exp=anchor_exp
        )
        items = gt_only(model, anchors, di_values)
        return wrapped(items, di_values, step, result_dir)

    tcc._run_test_inference_panel = run_gt_panel
    sys.argv = [
        "test_cfm_checkpoint.py",
        "--checkpoint",
        os.path.join(ROOT, "results/DamagePatternGeneration/CFMADM/selfcondnoDI/checkpoint_step_014000.pt"),
        "--result_dir",
        os.path.join(ROOT, "results/DamagePatternGeneration/CFMADM/selfcondnoDI"),
        "--fixed_di_values",
        "0,0.4,0.6,0.8,1",
        "--split",
        "test",
        "--skip_metrics",
        "--num_workers",
        "1",
    ]
    tcc.main()


if __name__ == "__main__":
    main()
