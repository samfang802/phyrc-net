#!/usr/bin/env python3
"""z0_new prior via DI filter + mutual SSIM Top-3 (no GRU similarity).

Pipeline
--------
1. Candidate pool: |DI_train - DI_target| <= di_max_diff (default 0.05)
2. Among candidates, pick 3 images maximizing mean pairwise SSIM
3. OpenCV TELEA inpaint interference → reverse ODE → Slerp → forward ODE (mask=all-ones)
4. Save: GT / z0_new panel + Top-3 train panel
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_energy_guidance_rows import (  # noqa: E402
    _build_dataloader,
    _load_ckpt,
    _parse_di,
    _slerp_average,
    _tensor_to_vis,
    reverse_ode_to_z0,
    sample_ode_guided,
)
from compare_z0_new_prior import (  # noqa: E402
    _collect_gallery_with_filename,
    _prepare_inpainted_tensor,
)
from argument.argument_mixed import set_dataset_args
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match


def _parse_args():
    p = argparse.ArgumentParser(description="z0_new via DI filter + mutual SSIM Top-3")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="results/DamagePatternGeneration/CFMADM/2026-07-26_21-24-55/checkpoint_step_014000.pt",
    )
    p.add_argument("--result_dir", type=str, default="")
    p.add_argument("--fixed_di_values", type=str, default="0.0,0.4,0.75,0.95")
    p.add_argument("--val_anchor_exp", type=str, default="R615")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--di_max_diff", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--ode_steps", type=int, default=0)
    p.add_argument("--inpaint_radius", type=int, default=5)
    return p.parse_args()


@torch.no_grad()
def _pairwise_ssim_matrix(images_chw, device):
    """images: list of CHW float tensors in [-1,1] or [0,1]; compute SSIM on [0,1]."""
    n = len(images_chw)
    vis = []
    for im in images_chw:
        x = im.float()
        if float(x.min()) < -1e-3:
            x = torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)
        else:
            x = torch.clamp(x, 0.0, 1.0)
        vis.append(x)
    metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    mat = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            a = vis[i].unsqueeze(0).to(device)
            b = vis[j].unsqueeze(0).to(device)
            s = float(metric(a, b).item())
            metric.reset()
            mat[i, j] = mat[j, i] = s
    return mat


def _select_mutual_ssim_topk(candidates, k=3, device="cuda"):
    """Pick k candidates maximizing mean pairwise SSIM (spatial alignment)."""
    if len(candidates) < k:
        raise RuntimeError(
            f"Need at least {k} DI-filtered candidates, got {len(candidates)}"
        )
    images = [c["image"] for c in candidates]
    mat = _pairwise_ssim_matrix(images, device)

    best_key = None
    best_idx = None
    n = len(candidates)
    # Exhaustive search over combinations of size k (pool is DI-filtered, usually small).
    from itertools import combinations

    for comb in combinations(range(n), k):
        pairs = [
            mat[comb[a], comb[b]] for a in range(k) for b in range(a + 1, k)
        ]
        score = float(np.mean(pairs))
        min_pair = float(np.min(pairs))
        key = (score, min_pair)  # prefer high mean; tie-break on min pairwise SSIM
        if best_key is None or key > best_key:
            best_key = key
            best_idx = comb

    mean_ssim, min_ssim = best_key
    selected = [
        {
            "item": candidates[i],
            "mean_pair_ssim": mean_ssim,
            "min_pair_ssim": min_ssim,
            "pool_index": i,
        }
        for i in best_idx
    ]
    sub = mat[np.ix_(best_idx, best_idx)]
    return selected, mean_ssim, min_ssim, sub


def _save_gt_prior_panel(gt_imgs, prior_imgs, di_values, out_path, step, val_exp):
    n_cols = len(di_values)
    fig, axes = plt.subplots(2, n_cols, figsize=(max(4, n_cols * 2.4), 4.8))
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    row_labels = ["GT (val)", r"$z_{0,new}$ (SSIM Top-3)"]
    for r, (label, images) in enumerate(zip(row_labels, [gt_imgs, prior_imgs])):
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
                    fontsize=8,
                    rotation=90,
                )
    fig.suptitle(
        f"step {step} · val {val_exp} · $|dDI|\\leq 0.05$ · mutual-SSIM Top-3 · Slerp $z_{{0,new}}$",
        fontsize=11,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[ssim_prior] Saved prior panel: {out_path}", flush=True)


def _save_topk_panel(topk_per_di, di_values, out_path, step, val_exp, top_k, inpainted_views, stats):
    n_cols = len(di_values)
    fig, axes = plt.subplots(top_k, n_cols, figsize=(max(4, n_cols * 2.6), top_k * 2.5))
    if top_k == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(top_k, 1)

    for c, (di, ranked) in enumerate(zip(di_values, topk_per_di)):
        mean_s, min_s, pool_n = stats[c]
        for r in range(top_k):
            ax = axes[r, c]
            if r >= len(ranked):
                ax.axis("off")
                continue
            item = ranked[r]["item"]
            if inpainted_views[c][r] is not None:
                arr = _tensor_to_vis(inpainted_views[c][r]).permute(1, 2, 0).numpy()
            else:
                arr = _tensor_to_vis(item["image"]).permute(1, 2, 0).numpy()
            ax.imshow(np.clip(arr, 0, 1))
            ax.axis("off")
            ax.set_title(
                f"#{r+1} {item['exp']} t={item['timestep']}\n"
                f"DI={item['di']:.2f}",
                fontsize=7,
            )
            if r == 0:
                ax.text(
                    0.5,
                    1.32,
                    f"Query DI={di:.2f}\npool={pool_n} meanSSIM={mean_s:.3f}",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )
            if c == 0:
                ax.text(
                    -0.08,
                    0.5,
                    f"Top-{r+1}",
                    transform=ax.transAxes,
                    va="center",
                    ha="right",
                    fontsize=9,
                    rotation=90,
                )

    fig.suptitle(
        f"step {step} · val {val_exp} · Top-{top_k} by mutual SSIM "
        r"($|DI_{train}-DI_{target}|\leq 0.05$, no GRU, inpainted)",
        fontsize=10,
        y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[ssim_prior] Saved Top-{top_k} panel: {out_path}", flush=True)


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
        f"[ssim_prior] device={device} | |dDI|<={cli.di_max_diff} | "
        f"select Top-{cli.top_k} by mutual SSIM (no GRU)",
        flush=True,
    )

    extractor_args = _load_args_from_result_dir(args.extractor_path)
    extractor = importlib.import_module("src.model.HystereticPrediction").__dict__[
        extractor_args.architecture
    ](**vars(extractor_args)).to(device)
    extractor_ckpt = _resolve_required_match(
        os.path.join(args.extractor_path, "**/best.ckpt"), "extractor checkpoint"
    )
    extractor.load_state_dict(
        torch.load(extractor_ckpt, map_location=device, weights_only=False)["state_dict"]
    )
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False

    print("[ssim_prior] Building train gallery...", flush=True)
    train_ds, train_dl = _build_dataloader(
        args, extractor, extractor_args, "train", args.batch_size, num_workers
    )
    set_dataset_args(args, train_dl)
    train_gallery = _collect_gallery_with_filename(train_ds)
    print(f"[ssim_prior] train gallery={len(train_gallery)}", flush=True)

    print("[ssim_prior] Building val set...", flush=True)
    val_ds, _ = _build_dataloader(
        args, extractor, extractor_args, "val", args.batch_size, num_workers
    )
    val_gallery = _collect_gallery_with_filename(val_ds)

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

    model = importlib.import_module("src.model.ImageGeneration").__dict__[args.architecture](
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

    image_root = os.path.join("./Data", "DamagePattern", "images")
    mask_dir = os.path.join("./Data", "DamagePattern", "mask")
    save_dir = os.path.join(result_dir, "inference_samples")

    gt_imgs, prior_imgs, topk_per_di, inpainted_views, stats = [], [], [], [], []

    for di_value, anchor in zip(di_values, val_anchors):
        print(
            f"[ssim_prior] DI={di_value:.2f} exp={anchor['exp']} t={anchor['timestep']} "
            f"GT_DI={anchor['gt_di']:.3f}",
            flush=True,
        )
        gt_imgs.append(anchor["real"])

        pool = [
            item
            for item in train_gallery
            if abs(float(item["di"]) - float(di_value)) <= cli.di_max_diff
        ]
        # Deduplicate by (exp, timestep), keep first.
        uniq, seen = [], set()
        for item in pool:
            key = (item["exp"], item["timestep"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(item)
        pool = uniq
        print(f"[ssim_prior]   DI-filter pool={len(pool)}", flush=True)
        if len(pool) < cli.top_k:
            raise RuntimeError(
                f"DI={di_value}: only {len(pool)} candidates with |dDI|<={cli.di_max_diff}"
            )

        ranked, mean_s, min_s, sub = _select_mutual_ssim_topk(
            pool, k=cli.top_k, device=device
        )
        topk_per_di.append(ranked)
        stats.append((mean_s, min_s, len(pool)))
        print(
            f"[ssim_prior]   mutual-SSIM Top-{cli.top_k}: mean={mean_s:.4f} min={min_s:.4f} | "
            + ", ".join(
                f"{r['item']['exp']}@t{r['item']['timestep']}(DI={r['item']['di']:.2f})"
                for r in ranked
            ),
            flush=True,
        )
        print(f"[ssim_prior]   pairwise SSIM=\n{np.array2string(sub, precision=3)}", flush=True)

        z0_list = []
        col_inpainted = []
        for entry in ranked:
            t_item = entry["item"]
            inpainted = _prepare_inpainted_tensor(
                t_item,
                val_transform,
                image_root,
                mask_dir,
                args.image_size,
                cli.inpaint_radius,
            )
            col_inpainted.append(inpainted.detach().cpu())
            img = inpainted.unsqueeze(0).to(device)
            hidden = t_item["hidden"].unsqueeze(0).to(device)
            direction = t_item["direction"].unsqueeze(0).to(device)
            info = t_item["info"].unsqueeze(0).to(device)
            di = torch.tensor([[t_item["di"]]], device=device, dtype=info.dtype)
            c_k, gru_k = model._unpack_conditions(info, hidden, direction, di)
            z1 = model._encode_to_target_space(img)
            z0_k = reverse_ode_to_z0(model, z1, c_k, gru_feat=gru_k, num_steps=ode_steps)
            z0_list.append(z0_k[0].detach())
        while len(col_inpainted) < cli.top_k:
            col_inpainted.append(None)
        inpainted_views.append(col_inpainted)

        z0_new = _slerp_average(z0_list).unsqueeze(0).to(device)

        hidden = anchor["hidden_state"].unsqueeze(0).to(device)
        direction = anchor["direction"].unsqueeze(0).to(device)
        info = anchor["info"].unsqueeze(0).to(device)
        di = torch.tensor([[float(di_value)]], device=device, dtype=info.dtype)
        c, gru_feat = model._unpack_conditions(info, hidden, direction, di)
        z = sample_ode_guided(
            model,
            c,
            z0_new.clone(),
            gru_feat=gru_feat,
            num_steps=ode_steps,
            energy_fn=None,
            guidance_scale=0.0,
            di_tensor=di,
        )
        prior_imgs.append(_tensor_to_vis(model._decode_from_target_space(z))[0].detach().cpu())

    prior_path = os.path.join(save_dir, f"z0_new_ssim_prior_step_{step:05d}.png")
    topk_path = os.path.join(save_dir, f"z0_new_ssim_topk_train_step_{step:05d}.png")
    _save_gt_prior_panel(gt_imgs, prior_imgs, di_values, prior_path, step, cli.val_anchor_exp)
    _save_topk_panel(
        topk_per_di,
        di_values,
        topk_path,
        step,
        cli.val_anchor_exp,
        cli.top_k,
        inpainted_views,
        stats,
    )
    print("[ssim_prior] Done.", flush=True)


if __name__ == "__main__":
    main()
