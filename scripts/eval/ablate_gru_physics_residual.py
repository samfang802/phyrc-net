#!/usr/bin/env python3
"""Ablate 1024-d HysGRU physics features: A=full vs B=zero-out GRU → Δ=|A-B|.

Fixed: same z0, same DI (+direction), same spatial mask (all-ones).
A: [DI, Mask, g_1024]
B: [DI, Mask, 0]
Δ: |A-B|  = pure visual contribution of the 1024-d physics pathway.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_energy_guidance_rows import (  # noqa: E402
    _build_dataloader,
    _collect_gallery,
    _load_ckpt,
    _parse_di,
    _tensor_to_vis,
    sample_ode_guided,
)
from argument.argument_mixed import set_dataset_args
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match


def _parse_args():
    p = argparse.ArgumentParser(description="GRU physics feature ablation residual panel")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="results/DamagePatternGeneration/CFMADM/2026-07-26_21-24-55/checkpoint_step_014000.pt",
    )
    p.add_argument("--result_dir", type=str, default="")
    p.add_argument("--fixed_di_values", type=str, default="0.0,0.4,0.75,0.95")
    p.add_argument("--val_anchor_exp", type=str, default="R615")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--ode_steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=0, help="Fixed z0 RNG seed")
    p.add_argument(
        "--delta_gain",
        type=float,
        default=4.0,
        help="Display gain for residual row (clipped to [0,1])",
    )
    return p.parse_args()


def _save_ablation_panel(rows_a, rows_b, rows_delta, di_values, out_path, step, val_exp, gain):
    n_cols = len(di_values)
    fig, axes = plt.subplots(3, n_cols, figsize=(max(4, n_cols * 2.4), 7.2))
    if n_cols == 1:
        axes = axes.reshape(3, 1)

    labels = [
        r"A: full $[DI,Mask,g_{1024}]$",
        r"B: zero-out $[DI,Mask,0]$",
        rf"$\Delta=|A-B|$ (×{gain:g})",
    ]
    for r, (label, images) in enumerate(zip(labels, [rows_a, rows_b, rows_delta])):
        for c, (img, di) in enumerate(zip(images, di_values)):
            arr = img.detach().cpu().permute(1, 2, 0).numpy()
            axes[r, c].imshow(np.clip(arr, 0, 1))
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"DI={di:.2f}", fontsize=10)
            if c == 0:
                axes[r, c].text(
                    -0.08,
                    0.5,
                    label,
                    transform=axes[r, c].transAxes,
                    va="center",
                    ha="right",
                    fontsize=7.5,
                    rotation=90,
                )

    fig.suptitle(
        f"step {step} · val {val_exp} · fixed $z_0$/DI/mask "
        r"· physics ablation $\Delta=|A-B|$",
        fontsize=11,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[ablate] Saved: {out_path}", flush=True)


def main():
    cli = _parse_args()
    ckpt = os.path.abspath(cli.checkpoint)
    result_dir = os.path.abspath(cli.result_dir.strip() or os.path.dirname(ckpt))
    args = _load_args_from_result_dir(result_dir)
    args.result_dir = result_dir
    if cli.batch_size > 0:
        args.batch_size = cli.batch_size
    di_values = _parse_di(cli.fixed_di_values)
    args.inference_num_images = len(di_values)
    args.fixed_di_values = cli.fixed_di_values
    args.inference_val_anchor_exp = cli.val_anchor_exp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = _resolve_num_workers(cli.num_workers)
    ode_steps = cli.ode_steps if cli.ode_steps > 0 else None
    print(
        f"[ablate] device={device} seed={cli.seed} condition_mode={getattr(args,'condition_mode',None)}",
        flush=True,
    )
    if str(getattr(args, "condition_mode", "")).lower() != "split":
        print(
            "[ablate] WARNING: this ablation is designed for condition_mode=split "
            "(1024-d via CrossAttn). Current run may not isolate physics tokens.",
            flush=True,
        )

    extractor_args = _load_args_from_result_dir(args.extractor_path)
    extractor = getattr(importlib.import_module("src.model.HystereticPrediction"), extractor_args.architecture)(**vars(extractor_args)).to(device)
    extractor_ckpt = _resolve_required_match(
        os.path.join(args.extractor_path, "**/best.ckpt"), "extractor checkpoint"
    )
    extractor.load_state_dict(
        torch.load(extractor_ckpt, map_location=device, weights_only=False)["state_dict"]
    )
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False

    print("[ablate] Building val set...", flush=True)
    _, val_dl = _build_dataloader(
        args, extractor, extractor_args, "val", args.batch_size, num_workers
    )
    set_dataset_args(args, val_dl)
    val_gallery = _collect_gallery(val_dl)

    model = getattr(importlib.import_module("src.model.ImageGeneration"), args.architecture)(
        regressor=None,
        **vars(args),
    ).to(device)
    step = _load_ckpt(model, ckpt, device)
    model.eval()

    model.validation_outputs = {
        "gt_DI": [],
        "pred_DI": [],
        "train_anchor_candidates": [],
        "val_anchor_candidates": [],
        "test_anchor_candidates": [],
    }
    for item in val_gallery:
        model.validation_outputs["val_anchor_candidates"].append(
            {
                "split": "val",
                "real": _tensor_to_vis(item["image"]),
                "hidden_state": item["hidden"],
                "direction": item["direction"],
                "info": item["info"],
                "exp": item["exp"],
                "timestep": item["timestep"],
                "gt_di": item["di"],
                "sample_uid": f"val_{item['exp']}_{item['timestep']}",
            }
        )
    if hasattr(model, "_fixed_val_inference_anchors"):
        delattr(model, "_fixed_val_inference_anchors")
    val_anchors = model._select_anchors_for_di_values(
        model.validation_outputs["val_anchor_candidates"],
        di_values,
        split="val",
        anchor_exp=cli.val_anchor_exp,
    )

    # One shared z0 for all DI columns (fair ablation across DI).
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)
    z0_shared = torch.randn(
        (1, model.model_in_channels, model.model_spatial_size, model.model_spatial_size),
        device=device,
    )

    rows_a, rows_b, rows_delta = [], [], []
    stats = []

    for di_value, anchor in zip(di_values, val_anchors):
        print(
            f"[ablate] DI={di_value:.2f} exp={anchor['exp']} t={anchor['timestep']} "
            f"GT_DI={anchor['gt_di']:.3f}",
            flush=True,
        )
        hidden = anchor["hidden_state"].unsqueeze(0).to(device)
        direction = anchor["direction"].unsqueeze(0).to(device)
        info = anchor["info"].unsqueeze(0).to(device)
        di = torch.tensor([[float(di_value)]], device=device, dtype=info.dtype)
        c, gru_feat = model._unpack_conditions(info, hidden, direction, di)
        if gru_feat is None:
            raise RuntimeError(
                "gru_feat is None — need condition_mode=split for this ablation"
            )
        gru_zero = torch.zeros_like(gru_feat)

        z_a = sample_ode_guided(
            model, c, z0_shared.clone(), gru_feat=gru_feat, num_steps=ode_steps
        )
        z_b = sample_ode_guided(
            model, c, z0_shared.clone(), gru_feat=gru_zero, num_steps=ode_steps
        )
        img_a = _tensor_to_vis(model._decode_from_target_space(z_a))[0].detach().cpu()
        img_b = _tensor_to_vis(model._decode_from_target_space(z_b))[0].detach().cpu()
        delta = (img_a - img_b).abs()
        delta_vis = (delta * cli.delta_gain).clamp(0.0, 1.0)

        mean_l1 = float(delta.mean().item())
        max_l1 = float(delta.max().item())
        # Per-channel mean residual
        mean_per_ch = [float(delta[c].mean().item()) for c in range(delta.shape[0])]
        stats.append(
            {
                "di": float(di_value),
                "exp": anchor["exp"],
                "timestep": int(anchor["timestep"]),
                "gt_di": float(anchor["gt_di"]),
                "delta_mean_l1": mean_l1,
                "delta_max": max_l1,
                "delta_mean_per_channel": mean_per_ch,
            }
        )
        print(
            f"[ablate]   Δ mean L1={mean_l1:.4f} max={max_l1:.4f}",
            flush=True,
        )

        rows_a.append(img_a)
        rows_b.append(img_b)
        rows_delta.append(delta_vis)

    save_dir = os.path.join(result_dir, "inference_samples")
    out_png = os.path.join(save_dir, f"gru_ablation_residual_step_{step:05d}.png")
    _save_ablation_panel(
        rows_a, rows_b, rows_delta, di_values, out_png, step, cli.val_anchor_exp, cli.delta_gain
    )

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": ckpt,
        "global_step": step,
        "val_anchor_exp": cli.val_anchor_exp,
        "fixed_di_values": di_values,
        "seed": cli.seed,
        "delta_gain_display": cli.delta_gain,
        "mask": "all-ones",
        "protocol": "A=full gru_feat; B=zeros_like(gru_feat); same z0/DI/direction/mask",
        "per_di": stats,
    }
    out_json = os.path.join(save_dir, f"gru_ablation_residual_step_{step:05d}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[ablate] Summary: {out_json}", flush=True)
    print("[ablate] Done.", flush=True)


if __name__ == "__main__":
    main()
