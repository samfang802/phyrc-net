#!/usr/bin/env python3
"""Export an NxN CFM inference grid on the test set with near-uniform DI coverage."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from argument.argument_mixed import set_dataset_args
from src.dataset import PatternLoopDataset
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match


def _parse_args():
    p = argparse.ArgumentParser(description="Uniform-DI CFM inference grid")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--result_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=("test", "val", "train"))
    p.add_argument("--grid_size", type=int, default=10, help="NxN grid (default 10 → 100 images)")
    p.add_argument(
        "--anchor_exp",
        type=str,
        default="",
        help="Optional specimen filter (empty = all exps on split)",
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--cfm_ode_steps", type=int, default=0, help="0 = use args.json")
    p.add_argument("--di_min", type=float, default=0.0)
    p.add_argument(
        "--di_max",
        type=float,
        default=-1.0,
        help="Upper DI for linspace targets; <0 → use max GT DI on filtered pool",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _strip_u2_vss(state_dict):
    drop = [k for k in list(state_dict.keys()) if "u2_vss" in str(k)]
    for k in drop:
        del state_dict[k]
    return drop


def _strip_fid(state_dict):
    drop = [k for k in list(state_dict.keys()) if str(k).startswith("fid.")]
    for k in drop:
        del state_dict[k]
    return drop


def _load_cfm(model, checkpoint_path, device):
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = payload["state_dict"]
    dropped_u2 = _strip_u2_vss(state)
    dropped_fid = _strip_fid(state)
    if dropped_u2:
        print(f"[grid] stripped {len(dropped_u2)} u2_vss keys", flush=True)
    if dropped_fid:
        print(f"[grid] stripped {len(dropped_fid)} fid.* keys", flush=True)
    status = model.load_state_dict(state, strict=False)
    missing = list(status.missing_keys)
    unexpected = list(status.unexpected_keys)
    print(f"[grid] load missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    step = int(payload.get("global_step", 0))
    epoch = int(payload.get("epoch", 0))
    print(f"[grid] Loaded step={step} epoch={epoch}", flush=True)
    return step, epoch


def _collect_all_candidates(model, dataloader, split):
    pool_key = f"{split}_anchor_candidates"
    model.validation_outputs = {
        "gt_DI": [],
        "pred_DI": [],
        "train_anchor_candidates": [],
        "val_anchor_candidates": [],
        "test_anchor_candidates": [],
    }
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"collect-{split}", leave=False):
            real_data, hidden_state, direction, exp, timestep, DI, info = batch[:7]
            real_vis = torch.clamp(real_data * 0.5 + 0.5, 0.0, 1.0)
            # Temporarily raise candidate cap by appending without early stop.
            remaining_attr = getattr(model, "_inference_candidate_cap", None)
            if remaining_attr is not None:
                # Bypass internal cap by ensuring large enough room each call.
                pass
            pool = model.validation_outputs[pool_key]
            # Direct append so we keep the full split (ignore training cap).
            take_count = real_vis.shape[0]
            for sample_idx in range(take_count):
                anchor_exp = model._extract_batch_field(exp, sample_idx)
                anchor_timestep = int(model._extract_batch_field(timestep, sample_idx))
                gt_di_value = float(model._extract_batch_field(DI, sample_idx))
                pool.append(
                    {
                        "split": split,
                        "real": real_vis[sample_idx].detach().cpu(),
                        "hidden_state": hidden_state[sample_idx].detach().cpu(),
                        "direction": direction[sample_idx].detach().cpu(),
                        "info": info[sample_idx].detach().cpu(),
                        "exp": anchor_exp,
                        "timestep": anchor_timestep,
                        "gt_di": gt_di_value,
                        "sample_uid": (
                            f"{split}_{anchor_exp}_{anchor_timestep}_"
                            f"{sample_idx}_{len(pool)}"
                        ),
                    }
                )
    return model.validation_outputs[pool_key]


def _select_uniform_di_anchors(candidates, di_targets, anchor_exp=""):
    required_exp = str(anchor_exp or "").strip()
    pool = list(candidates)
    if required_exp:
        pool = [c for c in pool if str(c["exp"]) == required_exp]
    if not pool:
        raise RuntimeError(
            f"No candidates for anchor_exp={required_exp or 'any'} "
            f"(candidates={len(candidates)})"
        )

    selected = []
    remaining = list(pool)
    for di_value in di_targets:
        search = remaining if remaining else pool
        anchor = min(
            search,
            key=lambda c: (
                abs(float(c["gt_di"]) - float(di_value)),
                int(c["timestep"]),
                str(c["sample_uid"]),
            ),
        )
        selected.append(anchor)
        remaining = [c for c in remaining if c["sample_uid"] != anchor["sample_uid"]]
    return selected


def _save_grid(images, titles, grid_size, out_path, suptitle):
    n = grid_size * grid_size
    fig, axes = plt.subplots(
        grid_size,
        grid_size,
        figsize=(grid_size * 1.35, grid_size * 1.45),
    )
    axes = np.atleast_2d(axes)
    for i in range(n):
        r, c = divmod(i, grid_size)
        ax = axes[r, c]
        img = images[i]
        if img.ndim == 3 and img.shape[0] in (1, 3):
            vis = img.permute(1, 2, 0).numpy()
            if vis.shape[-1] == 1:
                vis = vis[..., 0]
        else:
            vis = img.numpy()
        ax.imshow(vis, cmap="gray" if vis.ndim == 2 else None)
        ax.set_title(titles[i], fontsize=6, pad=2)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[grid] saved {out_path}", flush=True)


def main():
    cli = _parse_args()
    torch.manual_seed(int(cli.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cli.seed))

    checkpoint_path = os.path.abspath(cli.checkpoint)
    result_dir = os.path.abspath(cli.result_dir)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    if not os.path.isfile(os.path.join(result_dir, "args.json")):
        raise FileNotFoundError(f"args.json missing in {result_dir}")

    grid_size = int(cli.grid_size)
    n_images = grid_size * grid_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = _load_args_from_result_dir(result_dir)
    args.result_dir = result_dir
    args.batch_size = int(cli.batch_size)
    args.inference_num_images = n_images
    if int(cli.cfm_ode_steps) > 0:
        args.cfm_ode_steps = int(cli.cfm_ode_steps)

    print(f"[grid] ckpt={checkpoint_path}", flush=True)
    print(f"[grid] result_dir={result_dir}", flush=True)
    print(
        f"[grid] split={cli.split} grid={grid_size}x{grid_size} "
        f"ode={args.cfm_ode_steps} zero_di={getattr(args, 'zero_di_condition', None)}",
        flush=True,
    )

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
    for p in extractor.parameters():
        p.requires_grad = False

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

    num_workers = _resolve_num_workers(cli.num_workers)
    ds = PatternLoopDataset(
        root="./Data",
        mode=cli.split,
        transform=transform,
        extractor=extractor,
        extractor_args=extractor_args,
        **dataset_kwargs,
    )
    dl = DataLoader(ds, batch_size=cli.batch_size, shuffle=False, num_workers=num_workers)
    set_dataset_args(args, dl)
    print(f"[grid] {cli.split} samples={len(ds)}", flush=True)

    model = importlib.import_module("src.model.ImageGeneration").__dict__[args.architecture](
        regressor=None,
        **vars(args),
    ).to(device)
    step, epoch = _load_cfm(model, checkpoint_path, device)
    model.eval()

    candidates = _collect_all_candidates(model, dl, cli.split)
    required_exp = str(cli.anchor_exp or "").strip()
    pool = [c for c in candidates if (not required_exp or str(c["exp"]) == required_exp)]
    if len(pool) < n_images:
        raise RuntimeError(
            f"Need {n_images} distinct anchors, got {len(pool)} "
            f"(split={cli.split}, anchor_exp={required_exp or 'any'})"
        )

    di_min = float(cli.di_min)
    di_max = float(cli.di_max)
    if di_max < 0:
        di_max = max(float(c["gt_di"]) for c in pool)
    di_targets = np.linspace(di_min, di_max, n_images).astype(np.float64).tolist()
    print(
        f"[grid] DI targets linspace[{di_min:.3f}, {di_max:.3f}] n={n_images} "
        f"pool={len(pool)}",
        flush=True,
    )

    # Clear cached selection so mixin path isn't required.
    for cache_attr in (
        "_fixed_train_inference_anchors",
        "_fixed_val_inference_anchors",
        "_fixed_test_inference_anchors",
    ):
        if hasattr(model, cache_attr):
            delattr(model, cache_attr)

    anchors = _select_uniform_di_anchors(candidates, di_targets, anchor_exp=required_exp)
    selected_gt = np.asarray([float(a["gt_di"]) for a in anchors], dtype=np.float64)
    print(
        f"[grid] selected GT DI: min={selected_gt.min():.3f} max={selected_gt.max():.3f} "
        f"unique={len(np.unique(np.round(selected_gt, 4)))}",
        flush=True,
    )

    images = []
    titles = []
    meta_rows = []
    with torch.no_grad():
        for target_di, anchor in tqdm(
            list(zip(di_targets, anchors)),
            desc="ode-infer",
            leave=True,
        ):
            item = model._run_inference_for_anchor(anchor, target_di)
            images.append(item["sampled_vis"])
            titles.append(
                f"tDI={float(target_di):.2f}\n"
                f"GT={float(anchor['gt_di']):.2f} {anchor['exp']}"
            )
            meta_rows.append(
                {
                    "target_di": float(target_di),
                    "gt_di": float(anchor["gt_di"]),
                    "exp": str(anchor["exp"]),
                    "timestep": int(anchor["timestep"]),
                }
            )

    save_dir = os.path.join(result_dir, "inference_samples")
    ode = int(args.cfm_ode_steps)
    fig_path = os.path.join(
        save_dir,
        f"uniform_di_{cli.split}_{grid_size}x{grid_size}_step_{step:05d}_ode{ode}.png",
    )
    exp_tag = required_exp or "all"
    _save_grid(
        images,
        titles,
        grid_size,
        fig_path,
        suptitle=(
            f"step {step} · {cli.split} · {exp_tag} · ODE={ode} · "
            f"{grid_size}×{grid_size} Inference (uniform DI)"
        ),
    )

    meta_path = fig_path.replace(".png", ".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": checkpoint_path,
                "result_dir": result_dir,
                "split": cli.split,
                "anchor_exp": required_exp,
                "global_step": step,
                "epoch": epoch,
                "cfm_ode_steps": ode,
                "grid_size": grid_size,
                "di_min": di_min,
                "di_max": di_max,
                "zero_di_condition": bool(getattr(args, "zero_di_condition", False)),
                "items": meta_rows,
                "figure": fig_path,
            },
            f,
            indent=2,
        )
    print(f"[grid] meta → {meta_path}", flush=True)


if __name__ == "__main__":
    main()
