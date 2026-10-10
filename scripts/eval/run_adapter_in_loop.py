#!/usr/bin/env python3
"""Adapter-in-the-loop on a frozen CFMADM checkpoint (no U-Net retrain, no x_clean).

1. Freeze velocity_net from ``checkpoint_step_*.pt``.
2. Train ``ResidualPhysicsAdapter`` so ``gru_feat' = adapter(gru_feat)`` minimizes
   OT-CFM velocity MSE (adapter-in-the-loop).
3. Run fixed-DI test panel with adapted physics features.

Example:
  CUDA_VISIBLE_DEVICES=6 python3 -u scripts/eval/run_adapter_in_loop.py \\
    --checkpoint results/DamagePatternGeneration/CFMADM/selfcondnoDI/checkpoint_step_014000.pt \\
    --adapter_steps 1500 --batch_size 32
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Repo root on sys.path when launched as scripts/*.py
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from argument.argument_mixed import set_dataset_args
from src.dataset import PatternLoopDataset
from src.model.exp.oracle_adapter import ResidualPhysicsAdapter
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match
from test_cfm_checkpoint import (
    _collect_split_anchors,
    _load_periodic_checkpoint,
    _parse_di_values,
    _pool_key,
    _resolve_anchor_exp,
    _save_gt_inference_panel,
)


def _parse_args():
    p = argparse.ArgumentParser(description="Adapter-in-the-loop on frozen CFMADM")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="./results/DamagePatternGeneration/CFMADM/selfcondnoDI/checkpoint_step_014000.pt",
    )
    p.add_argument("--result_dir", type=str, default="", help="defaults to checkpoint parent")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--adapter_steps", type=int, default=1500, help="Adam steps on adapter only")
    p.add_argument("--adapter_lr", type=float, default=1e-3)
    p.add_argument("--adapter_hidden", type=int, default=512)
    p.add_argument("--adapter_dropout", type=float, default=0.0)
    p.add_argument("--fixed_di_values", type=str, default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--split", type=str, default="test", choices=("test", "val", "train"))
    p.add_argument("--anchor_exp", type=str, default="CTR1")
    p.add_argument("--skip_panel", action="store_true")
    p.add_argument(
        "--adapter_out",
        type=str,
        default="",
        help="Where to save adapter.pt (default: <result_dir>/adapter_in_loop/adapter.pt)",
    )
    return p.parse_args()


def _freeze_velocity_net(model):
    model.eval()
    model.velocity_net.eval()
    for p in model.velocity_net.parameters():
        p.requires_grad_(False)


def _train_adapter(model, adapter, loader, device, steps, lr):
    adapter.train()
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(lr), weight_decay=1e-4)
    it = iter(loader)
    running = 0.0
    log_every = max(1, steps // 10)

    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        (
            real_data,
            hidden_state,
            direction,
            _exp,
            _timestep,
            DI,
            info,
            valid_mask,
            condition_mask,
            *_rest,
        ) = model._unpack_generation_batch(batch)

        real_data = real_data.to(device)
        hidden_state = hidden_state.to(device)
        direction = direction.to(device)
        DI = DI.to(device)
        info = info.to(device)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device)
        if condition_mask is not None:
            condition_mask = condition_mask.to(device)

        with torch.no_grad():
            c, gru_feat = model._unpack_conditions(info, hidden_state, direction, DI)
            target = model._encode_to_target_space(real_data)

        gru_adapted = adapter(gru_feat)
        loss = model.flow_matching_loss(
            target,
            c,
            valid_mask=valid_mask,
            condition_mask=condition_mask,
            gru_feat=gru_adapted,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        running += float(loss.detach())
        if step % log_every == 0 or step == 1 or step == steps:
            avg = running / float(log_every if step % log_every == 0 else step)
            running = 0.0
            delta = (gru_adapted.detach() - gru_feat).norm(dim=-1).mean()
            print(
                f"[adapter] step {step}/{steps}  fm_loss={float(loss.detach()):.5f}  "
                f"||Δgru||={float(delta):.4f}",
                flush=True,
            )

    adapter.eval()
    return adapter


def _run_panel_with_adapter(model, dataloader, split, di_values, step, result_dir, device, anchor_exp):
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

    print(f"[adapter] Collecting {split} anchors...", flush=True)
    _collect_split_anchors(model, dataloader, split)
    candidates = model.validation_outputs[pool_key]
    anchors = model._select_anchors_for_di_values(
        candidates, di_values, split=split, anchor_exp=anchor_exp
    )
    if not anchors:
        raise RuntimeError(f"No anchors for exp={anchor_exp} split={split}")

    model.to(device)
    model.eval()
    model.use_tto = False
    items = model._run_fixed_di_inference_for_anchors(anchors, di_values)

    # Custom filename tag for adapter-in-loop.
    fig_path = _save_gt_inference_panel(
        items,
        di_values,
        step,
        result_dir,
        split,
        anchor_exp,
        ode_steps=int(model.cfm_ode_steps),
        ref_guidance_weight=0.0,
        ref_guidance_mode="none",
        use_tto=False,
    )
    # Rename / copy with adapter tag.
    base, ext = os.path.splitext(fig_path)
    tagged = f"{base}_adapter{ext}"
    if tagged != fig_path:
        os.replace(fig_path, tagged)
        fig_path = tagged
    print(f"[adapter] Panel saved: {fig_path}", flush=True)
    return fig_path


def main():
    cli = _parse_args()
    checkpoint_path = os.path.abspath(cli.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    result_dir = os.path.abspath(cli.result_dir.strip() or os.path.dirname(checkpoint_path))
    args = _load_args_from_result_dir(result_dir)
    args.result_dir = result_dir
    args.batch_size = int(cli.batch_size)
    # Adapter stage does not need AR pairing / virtual explosion.
    args.use_ar_self_conditioning = False
    args.use_ar_chain_ot = False
    args.train_virtual_repeat = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    di_values = _parse_di_values(cli.fixed_di_values)
    print(f"[adapter] ckpt={checkpoint_path}", flush=True)
    print(f"[adapter] result_dir={result_dir}", flush=True)
    print(
        f"[adapter] steps={cli.adapter_steps} lr={cli.adapter_lr} "
        f"batch={cli.batch_size} di={di_values}",
        flush=True,
    )

    extractor_path = args.extractor_path
    extractor_args = _load_args_from_result_dir(extractor_path)
    extractor = getattr(importlib.import_module("src.model.HystereticPrediction"), extractor_args.architecture)(**vars(extractor_args))
    extractor = extractor.to(device)
    extractor_ckpt = _resolve_required_match(
        os.path.join(extractor_path, "**/best.ckpt"), "extractor checkpoint"
    )
    extractor.load_state_dict(
        torch.load(extractor_ckpt, map_location=device, weights_only=False)["state_dict"]
    )
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False

    train_transform = build_damage_image_transform_with_loss_mask(
        image_size=args.image_size,
        use_augmentation=bool(args.img_aug),
        output_scale=True,
        normalize_mean=(0.5, 0.5, 0.5),
        normalize_std=(0.5, 0.5, 0.5),
        use_grayscale_input=args.use_grayscale_input,
        use_clahe=args.use_clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid_size=args.clahe_tile_grid_size,
        aug_prob=getattr(args, "aug_prob", 0.5),
        aug_blur_prob=getattr(args, "aug_blur_prob", 0.15),
        aug_brightness=getattr(args, "aug_brightness", 0.2),
        aug_contrast=getattr(args, "aug_contrast", 0.2),
        aug_saturation=getattr(args, "aug_saturation", 0.0),
        aug_hue=getattr(args, "aug_hue", 0.0),
        aug_hflip_prob=getattr(args, "aug_hflip_prob", 0.5),
        aug_blur_kernel=getattr(args, "aug_blur_kernel", 3),
        aug_blur_sigma_min=getattr(args, "aug_blur_sigma_min", 0.1),
        aug_blur_sigma_max=getattr(args, "aug_blur_sigma_max", 0.5),
    )
    val_transform = build_damage_image_transform_with_loss_mask(
        image_size=args.image_size,
        use_augmentation=False,
        output_scale=True,
        normalize_mean=(0.5, 0.5, 0.5),
        normalize_std=(0.5, 0.5, 0.5),
        use_grayscale_input=args.use_grayscale_input,
        use_clahe=args.use_clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid_size=args.clahe_tile_grid_size,
    )

    dataset_kwargs = dict(vars(args))
    dataset_kwargs["use_loss_mask"] = True
    dataset_kwargs["use_damage_mask_condition"] = True
    dataset_kwargs["use_object_loss_mask"] = False
    dataset_kwargs["use_ar_self_conditioning"] = False
    dataset_kwargs["use_ar_chain_ot"] = False

    num_workers = _resolve_num_workers(cli.num_workers)
    print("[adapter] Building train dataset...", flush=True)
    train_ds = PatternLoopDataset(
        root="./Data",
        mode="train",
        transform=train_transform,
        extractor=extractor,
        extractor_args=extractor_args,
        **dataset_kwargs,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    set_dataset_args(args, train_dl)
    print(f"[adapter] train={len(train_ds)}", flush=True)

    model = getattr(importlib.import_module("src.model.ImageGeneration"), args.architecture)(
        regressor=None,
        **vars(args),
    )
    step, _epoch = _load_periodic_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    _freeze_velocity_net(model)

    physics_dim = int(getattr(model, "physics_dim", 1024))
    adapter = ResidualPhysicsAdapter(
        dim=physics_dim,
        hidden_dim=int(cli.adapter_hidden),
        dropout=float(cli.adapter_dropout),
    ).to(device)
    print(
        f"[adapter] ResidualPhysicsAdapter dim={physics_dim} "
        f"hidden={cli.adapter_hidden} (U-Net frozen)",
        flush=True,
    )

    adapter = _train_adapter(
        model, adapter, train_dl, device, int(cli.adapter_steps), float(cli.adapter_lr)
    )

    out_dir = os.path.join(result_dir, "adapter_in_loop")
    os.makedirs(out_dir, exist_ok=True)
    adapter_path = cli.adapter_out.strip() or os.path.join(
        out_dir, f"adapter_step{cli.adapter_steps}.pt"
    )
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "physics_dim": physics_dim,
            "hidden_dim": int(cli.adapter_hidden),
            "adapter_steps": int(cli.adapter_steps),
            "adapter_lr": float(cli.adapter_lr),
            "base_checkpoint": checkpoint_path,
            "global_step_base": int(step),
        },
        adapter_path,
    )
    print(f"[adapter] Saved {adapter_path}", flush=True)

    # Attach for inference hook.
    model.physics_adapter = adapter
    model.use_physics_adapter = True

    if cli.skip_panel:
        print("[adapter] skip_panel; done.", flush=True)
        return

    print(f"[adapter] Building {cli.split} panel dataset...", flush=True)
    panel_ds = PatternLoopDataset(
        root="./Data",
        mode=cli.split,
        transform=val_transform,
        extractor=extractor,
        extractor_args=extractor_args,
        **dataset_kwargs,
    )
    panel_dl = DataLoader(
        panel_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    panel_path = _run_panel_with_adapter(
        model,
        panel_dl,
        cli.split,
        di_values,
        step,
        result_dir,
        device,
        _resolve_anchor_exp(model, cli.split, cli.anchor_exp),
    )
    # Also copy into adapter_in_loop for easy find.
    import shutil

    dst = os.path.join(out_dir, os.path.basename(panel_path))
    shutil.copy2(panel_path, dst)
    print(f"[adapter] Done. Panel: {panel_path}", flush=True)
    print(f"[adapter] Copy: {dst}", flush=True)


if __name__ == "__main__":
    main()
