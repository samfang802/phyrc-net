#!/usr/bin/env python3
"""Train a g-only 16x16 spall-area head for low-frequency guidance.

The head never sees the image. Targets are the existing extent masks of the
full 128 frame, in the same [-1, 1] scale the generator uses.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from argument.argument_mixed import set_dataset_args
from src.dataset import PatternLoopDataset
from src.model.cfm.lowfreq_guide import (
    SpallAreaHead,
    full_frame_coords,
    suppress_interference,
)
from src.model.cfm.spall_mask import extent_mask
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--result_dir",
        type=str,
        default="results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit",
    )
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--out",
        type=str,
        default="results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit/spall_area_head_ifix.pt",
    )
    return p.parse_args()


def _build_loader(args, mode, batch_size, num_workers, shuffle):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor_path = args.extractor_path
    extractor_args = _load_args_from_result_dir(extractor_path)
    extractor = getattr(
        importlib.import_module("src.model.HystereticPrediction"),
        extractor_args.architecture,
    )(**vars(extractor_args)).to(device)
    extractor_ckpt = _resolve_required_match(
        os.path.join(extractor_path, "**/best.ckpt"), "extractor checkpoint"
    )
    extractor.load_state_dict(
        torch.load(extractor_ckpt, map_location=device, weights_only=False)["state_dict"]
    )
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad_(False)
    transform = build_damage_image_transform_with_loss_mask(
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
    dataset_kwargs["eval_real_condition_mask"] = True
    dataset_kwargs["use_ar_temporal_pairing"] = False
    ds = PatternLoopDataset(
        root="./Data",
        mode=mode,
        transform=transform,
        extractor=extractor,
        extractor_args=extractor_args,
        **dataset_kwargs,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle,
    )
    set_dataset_args(args, dl)
    return dl


def _batch_xy(batch, grid, device):
    image, hidden = batch[0], batch[1]
    image = image.to(device, non_blocking=True)
    hidden = hidden.to(device, non_blocking=True)
    valid = batch[-1].to(device, non_blocking=True)
    interference = (1.0 - valid).clamp(0.0, 1.0)
    with torch.no_grad():
        target = suppress_interference(extent_mask(image, grid), interference)
    coords = full_frame_coords(image.shape[0], grid, device, image.dtype)
    return hidden, coords, target, interference


def main():
    cli = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = _load_args_from_result_dir(cli.result_dir)
    args.result_dir = os.path.abspath(cli.result_dir)
    os.makedirs(os.path.join(args.result_dir, "data_check"), exist_ok=True)
    num_workers = _resolve_num_workers(cli.num_workers)
    print("[spall-area] building train set", flush=True)
    loader = _build_loader(args, "train", cli.batch_size, num_workers, shuffle=True)
    physics_dim = int(args.physics_dim)
    head = SpallAreaHead(physics_dim=physics_dim, grid=cli.grid).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cli.lr, weight_decay=1e-4)
    step = 0
    pos_weight = None
    head.train()
    while step < cli.steps:
        for batch in loader:
            hidden, coords, target, interference = _batch_xy(batch, cli.grid, device)
            if pos_weight is None:
                rate = float(target.mean().clamp(min=0.02, max=0.5))
                pos_weight = target.new_tensor((1.0 - rate) / rate)
                blocked = torch.nn.functional.adaptive_avg_pool2d(interference, (cli.grid, cli.grid))
                print(
                    f"[spall-area] target_rate={float(target.mean()):.3f} "
                    f"pos_weight={float(pos_weight):.2f} "
                    f"interf_cells={(blocked >= 0.5).float().mean():.3f}",
                    flush=True,
                )
            logits = head(hidden, coords, interference)
            bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            prob = torch.sigmoid(logits)
            tv = (prob[:, :, 1:, :] - prob[:, :, :-1, :]).abs().mean()
            tv = tv + (prob[:, :, :, 1:] - prob[:, :, :, :-1]).abs().mean()
            loss = bce + 0.05 * tv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 100 == 0 or step == 1:
                with torch.no_grad():
                    pred = torch.sigmoid(logits)
                    flat_p = pred.flatten(1)
                    flat_t = target.flatten(1)
                    p0 = flat_p - flat_p.mean(dim=1, keepdim=True)
                    t0 = flat_t - flat_t.mean(dim=1, keepdim=True)
                    corr = (p0 * t0).sum(dim=1) / (
                        p0.norm(dim=1) * t0.norm(dim=1) + 1e-8
                    )
                print(
                    f"[spall-area] step={step} bce={float(bce):.4f} tv={float(tv):.4f} "
                    f"pred={float(pred.mean()):.3f} tgt={float(target.mean()):.3f} "
                    f"corr={float(corr.mean()):.3f}",
                    flush=True,
                )
            if step >= cli.steps:
                break
    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "physics_dim": physics_dim,
            "grid": int(cli.grid),
            "steps": int(cli.steps),
            "pos_weight": float(pos_weight) if pos_weight is not None else 1.0,
        },
        cli.out,
    )
    print(f"[spall-area] saved {os.path.abspath(cli.out)}", flush=True)

    head.eval()
    shown = 0
    tiles = []
    with torch.no_grad():
        for batch in loader:
            hidden, coords, target, interference = _batch_xy(batch, cli.grid, device)
            pred = torch.sigmoid(head(hidden, coords, interference))
            n = min(8, pred.shape[0])
            blocked = torch.nn.functional.adaptive_avg_pool2d(interference, (cli.grid, cli.grid))
            for i in range(n):
                tiles.append((pred[i, 0].cpu(), target[i, 0].cpu(), blocked[i, 0].cpu()))
            shown += n
            if shown >= 8:
                break
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 8, figsize=(16, 6))
        for i, (pred, tgt, blocked) in enumerate(tiles[:8]):
            axes[0, i].imshow(pred.numpy(), vmin=0, vmax=1, cmap="magma")
            axes[1, i].imshow(tgt.numpy(), vmin=0, vmax=1, cmap="magma")
            axes[2, i].imshow(blocked.numpy(), vmin=0, vmax=1, cmap="magma")
            axes[0, i].axis("off")
            axes[1, i].axis("off")
            axes[2, i].axis("off")
        axes[0, 0].set_ylabel("pred")
        axes[1, 0].set_ylabel("target")
        axes[2, 0].set_ylabel("interf.")
        fig.tight_layout()
        preview = "/tmp/cascade_cmp/spall_area_preview.png"
        os.makedirs("/tmp/cascade_cmp", exist_ok=True)
        fig.savefig(preview, dpi=120)
        print(f"[spall-area] preview {preview}", flush=True)
    except Exception as exc:
        print(f"[spall-area] preview skipped: {exc}", flush=True)


if __name__ == "__main__":
    main()
