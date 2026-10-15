#!/usr/bin/env python3
"""How far do single-frame samples sit from the real frames, and from the previous real frame?

For each (previous, target) pair of a split it samples the single-frame model for both frames and
measures global translation (phase correlation, in pixels at the model resolution) and residual
energy for four comparisons:

  gt_prev  -> gt_t      real motion between neighbouring frames
  base_t   -> gt_t      single-frame error on the same frame
  base_prev-> gt_t      what a state S_in = base(t-1) shows against x_t (the residual r = x_t - S_in)
  base_prev-> base_t    motion between two samples
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from argument.argument_mixed import set_dataset_args
from src.dataset import PatternLoopDataset
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_required_match


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--result_dir", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ode_steps", type=int, default=0, help="0 = the run's own cfm_ode_steps")
    p.add_argument("--out_dir", default="")
    return p.parse_args()


def _gray01(x):
    return ((x[:, 0].clamp(-1, 1) + 1.0) * 0.5).float().cpu().numpy()


def _shift(a, b, win):
    (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
    return float(dx), float(dy)


def _compare(a, b, win):
    dx, dy = _shift(a, b, win)
    h, w = a.shape
    aligned = cv2.warpAffine(
        a, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), borderMode=cv2.BORDER_REFLECT
    )
    return {
        "shift_px": float(np.hypot(dx, dy)),
        "abs_before": float(np.abs(b - a).mean()),
        "abs_after_align": float(np.abs(b - aligned).mean()),
    }


def _summary(rows):
    shift = np.array([r["shift_px"] for r in rows])
    before = np.array([r["abs_before"] for r in rows])
    after = np.array([r["abs_after_align"] for r in rows])
    return {
        "pairs": int(len(rows)),
        "shift_median": float(np.median(shift)),
        "shift_p90": float(np.percentile(shift, 90)),
        "shift_p99": float(np.percentile(shift, 99)),
        "frac_gt_1px": float((shift > 1).mean()),
        "frac_gt_2px": float((shift > 2).mean()),
        "mean_abs_diff": float(before.mean()),
        "mean_abs_diff_after_align": float(after.mean()),
    }


def main():
    cli = _parse_args()
    torch.manual_seed(cli.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = os.path.abspath(cli.result_dir)
    args = _load_args_from_result_dir(result_dir)
    args.result_dir = result_dir
    args.batch_size = int(cli.batch_size)
    if cli.ode_steps > 0:
        args.cfm_ode_steps = int(cli.ode_steps)

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    payload = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    w = state.get("velocity_net.scalar_mlp.0.weight")
    if w is not None and int(w.shape[1]) == 3 and bool(getattr(args, "zero_di_condition", False)):
        args.keep_zero_di_slot = True
    state = {k: v for k, v in state.items() if "u2_vss" not in k and not k.startswith("fid.")}

    extractor_args = _load_args_from_result_dir(args.extractor_path)
    extractor = getattr(importlib.import_module("src.model.HystereticPrediction"), extractor_args.architecture)(
        **vars(extractor_args)
    ).to(device)
    extractor_ckpt = _resolve_required_match(os.path.join(args.extractor_path, "**/best.ckpt"), "extractor")
    extractor.load_state_dict(torch.load(extractor_ckpt, map_location=device, weights_only=False)["state_dict"])
    extractor.eval()

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
    kwargs = dict(vars(args))
    kwargs["use_loss_mask"] = True
    kwargs["use_damage_mask_condition"] = args.architecture == "CFMADM" and bool(args.use_damage_mask_condition)
    if kwargs["use_damage_mask_condition"]:
        kwargs["use_object_loss_mask"] = False
    kwargs["use_paired_bridge"] = True  # only switches on previous-frame pairing in the dataset
    kwargs["cache_data"] = True
    ds = PatternLoopDataset(
        root="./Data", mode=cli.split, transform=transform, extractor=extractor,
        extractor_args=extractor_args, **kwargs,
    )
    dl = DataLoader(ds, batch_size=cli.batch_size, shuffle=False, num_workers=cli.num_workers)
    set_dataset_args(args, dl)

    model = getattr(importlib.import_module("src.model.ImageGeneration"), args.architecture)(
        regressor=None, **vars(args)
    ).to(device)
    status = model.load_state_dict(state, strict=False)
    print(f"[shift] load missing={len(status.missing_keys)} unexpected={len(status.unexpected_keys)}", flush=True)
    model.eval()

    size = int(args.image_size)
    win = cv2.createHanningWindow((size, size), cv2.CV_32F)
    names = ("gt_prev_to_gt_t", "base_t_to_gt_t", "base_prev_to_gt_t", "base_prev_to_base_t")
    rows = {n: [] for n in names}
    per_pair = []
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"shift-{cli.split}"):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            has_prev = batch["has_prev"].reshape(-1) >= 0.5
            if not bool(has_prev.any()):
                continue
            info = batch["info"]
            mask = model._align_spatial_to_target(batch["mask_combined"])
            c_t, g_t = model._unpack_conditions(
                info, batch["hidden_state_target"], batch["direction_target"], batch["DI_target"]
            )
            c_p, g_p = model._unpack_conditions(
                info, batch["hidden_state_prev"], batch["direction_prev"], batch["DI_prev"]
            )
            base_t = model.sample_with_ode(c_t, gru_feat=g_t, damage_mask=mask)
            base_p = model.sample_with_ode(c_p, gru_feat=g_p, damage_mask=mask)
            gt_t, gt_p = _gray01(batch["image_target"]), _gray01(batch["image_prev"])
            bt, bp = _gray01(base_t), _gray01(base_p)
            for i in torch.nonzero(has_prev).reshape(-1).tolist():
                pair = {
                    "gt_prev_to_gt_t": _compare(gt_p[i], gt_t[i], win),
                    "base_t_to_gt_t": _compare(bt[i], gt_t[i], win),
                    "base_prev_to_gt_t": _compare(bp[i], gt_t[i], win),
                    "base_prev_to_base_t": _compare(bp[i], bt[i], win),
                    "exp": str(batch["exp"][i]),
                    "timestep": int(batch["timestep_target"][i]),
                }
                per_pair.append(pair)
                for n in names:
                    rows[n].append(pair[n])

    summary = {n: _summary(rows[n]) for n in names}
    out_dir = cli.out_dir or os.path.join(result_dir, "eval_base_vs_gt_shift")
    os.makedirs(out_dir, exist_ok=True)
    out = {"split": cli.split, "checkpoint": os.path.abspath(cli.checkpoint),
           "ode_steps": int(args.cfm_ode_steps), "image_size": size, "summary": summary, "pairs": per_pair}
    path = os.path.join(out_dir, f"shift_{cli.split}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\n[shift] {cli.split}: pairs={len(per_pair)} image={size}px ode_steps={args.cfm_ode_steps}")
    for n in names:
        s = summary[n]
        print(
            f"  {n:22s} shift median {s['shift_median']:.2f}px p90 {s['shift_p90']:.2f} p99 {s['shift_p99']:.2f} "
            f"| >1px {s['frac_gt_1px']:.2f} >2px {s['frac_gt_2px']:.2f} "
            f"| mean|diff| {s['mean_abs_diff']:.4f} -> aligned {s['mean_abs_diff_after_align']:.4f}"
        )
    print(f"[shift] saved {path}")


if __name__ == "__main__":
    main()
