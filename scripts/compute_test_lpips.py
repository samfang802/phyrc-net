#!/usr/bin/env python3
"""Compute test-set LPIPS (VGG) for one or more CFMADM checkpoints."""

import argparse
import importlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

from argument.argument_mixed import set_dataset_args
from src.dataset import PatternLoopDataset
from src.losses.lpips import LPIPS
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match


def _parse_args():
    parser = argparse.ArgumentParser(description="Compute test-set LPIPS (VGG) for CFMADM checkpoints")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Paths to checkpoint_step_XXXXXX.pt files",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("test", "val", "train"),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--summary_path",
        type=str,
        default="results/test_lpips_summary.json",
        help="Path to save aggregated summary JSON",
    )
    return parser.parse_args()


def _load_periodic_checkpoint(model, checkpoint_path, device):
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    status = model.load_state_dict(payload["state_dict"], strict=True)
    step = int(payload.get("global_step", 0))
    epoch = int(payload.get("epoch", 0))
    print(f"[lpips] Loaded {checkpoint_path}", flush=True)
    print(f"[lpips] load_state_dict: {status}", flush=True)
    print(f"[lpips] global_step={step} epoch={epoch}", flush=True)
    return step, epoch


def _build_test_dataloader(args, extractor, extractor_args, split, batch_size, num_workers, device):
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
    dataset_kwargs["use_damage_mask_condition"] = (
        args.architecture == "CFMADM" and bool(args.use_damage_mask_condition)
    )
    if dataset_kwargs["use_damage_mask_condition"]:
        dataset_kwargs["use_object_loss_mask"] = False

    ds = PatternLoopDataset(
        root="./Data",
        mode=split,
        transform=transform,
        extractor=extractor,
        extractor_args=extractor_args,
        **dataset_kwargs,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    set_dataset_args(args, dl)
    print(f"[lpips] {split} samples={len(ds)}", flush=True)
    return dl


@torch.no_grad()
def _compute_split_lpips(model, dataloader, lpips_fn, device):
    model.eval()
    lpips_fn.eval()
    total = 0.0
    count = 0

    for batch in dataloader:
        real_data, hidden_state, direction, _exp, _timestep, DI, info = batch[:7]
        real_data = real_data.to(device)
        hidden_state = hidden_state.to(device)
        direction = direction.to(device)
        DI = DI.to(device)
        info = info.to(device)

        if len(batch) >= 9:
            condition_mask = batch[8].to(device)
        elif len(batch) >= 8 and model.use_damage_mask_condition:
            condition_mask = batch[7].to(device)
        else:
            condition_mask = None

        c, gru_feat = model._unpack_conditions(info, hidden_state, direction, DI)
        sampled = model.sample_with_ode(c, gru_feat=gru_feat)
        sampled_image = model._decode_from_target_space(sampled)

        real_vis = torch.clamp(real_data * 0.5 + 0.5, 0.0, 1.0)
        sampled_vis = torch.clamp(sampled_image * 0.5 + 0.5, 0.0, 1.0)
        batch_lpips = lpips_fn(sampled_vis, real_vis).view(-1)
        total += float(batch_lpips.sum().item())
        count += int(batch_lpips.numel())

    if count == 0:
        raise RuntimeError("No samples processed for LPIPS")
    return total / count


def _save_result(result_dir, split, step, checkpoint_path, lpips_value):
    save_dir = os.path.join(result_dir, "test_metrics")
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "split": split,
        "global_step": step,
        "checkpoint": os.path.abspath(checkpoint_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        f"{split}/lpips_vgg": float(lpips_value),
    }
    json_path = os.path.join(save_dir, f"lpips_{split}_step_{step:05d}.json")

    metrics_path = os.path.join(save_dir, f"metrics_{split}_step_{step:05d}.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            merged = json.load(f)
        merged[f"{split}/lpips_vgg"] = float(lpips_value)
        payload = merged

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if os.path.isfile(metrics_path):
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    print(f"[lpips] Saved: {json_path}", flush=True)
    return payload


def _evaluate_checkpoint(checkpoint_path, split, batch_size, num_workers, device, lpips_fn):
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    result_dir = os.path.dirname(checkpoint_path)
    args = _load_args_from_result_dir(result_dir)
    args.result_dir = result_dir
    args.batch_size = batch_size

    extractor_path = args.extractor_path
    extractor_args = _load_args_from_result_dir(extractor_path)
    extractor = importlib.import_module("src.model.HystereticPrediction").__dict__[
        extractor_args.architecture
    ](**vars(extractor_args)).to(device)
    extractor_ckpt = _resolve_required_match(
        os.path.join(extractor_path, "**/best.ckpt"), "extractor checkpoint"
    )
    extractor.load_state_dict(
        torch.load(extractor_ckpt, map_location=device, weights_only=False)["state_dict"]
    )
    extractor.eval()
    for param in extractor.parameters():
        param.requires_grad = False

    dataloader = _build_test_dataloader(
        args, extractor, extractor_args, split, batch_size, num_workers, device
    )
    model = importlib.import_module("src.model.ImageGeneration").__dict__[args.architecture](
        regressor=None,
        **vars(args),
    ).to(device)
    step, _epoch = _load_periodic_checkpoint(model, checkpoint_path, device)

    print(f"[lpips] Computing {split} LPIPS (VGG)...", flush=True)
    lpips_value = _compute_split_lpips(model, dataloader, lpips_fn, device)
    print(f"[lpips] {split}/lpips_vgg={lpips_value:.6f}", flush=True)
    return _save_result(result_dir, split, step, checkpoint_path, lpips_value)


def main():
    cli = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = _resolve_num_workers(cli.num_workers)
    lpips_fn = LPIPS().to(device)

    summary = []
    for checkpoint_path in cli.checkpoints:
        result = _evaluate_checkpoint(
            checkpoint_path,
            cli.split,
            cli.batch_size,
            num_workers,
            device,
            lpips_fn,
        )
        summary.append(result)

    summary_path = os.path.abspath(cli.summary_path)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[lpips] Summary saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
