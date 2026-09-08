#!/usr/bin/env python3
"""Scatter: HysGRU cosine distance (X) vs visual distance (Y) over random image pairs.

X = 1 - CosineSim(g_A, g_B)  in 1024-d HysGRU hidden space
Y options:
  - 1 - SSIM(image_A, image_B)
  - 1 - IoU(damage_mask_A, damage_mask_B) where damage ≈ dark pixels (Otsu)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure

from argument.argument_mixed import set_dataset_args
from src.dataset import PatternLoopDataset
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match


def _parse_args():
    p = argparse.ArgumentParser(description="GRU cosine distance vs visual distance scatter")
    p.add_argument(
        "--result_dir",
        type=str,
        default="results/DamagePatternGeneration/CFMADM/2026-07-26_21-24-55",
        help="Run dir with args.json (for preprocess / extractor settings)",
    )
    p.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="Comma-separated splits to pool samples from (train / val / test)",
    )
    p.add_argument("--num_pairs", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Default: {result_dir}/analysis/",
    )
    return p.parse_args()


def _build_dataset(args, extractor, extractor_args, mode):
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
    dataset_kwargs["use_loss_mask"] = False
    dataset_kwargs["train_virtual_repeat"] = 1
    dataset_kwargs["use_damage_mask_condition"] = False
    dataset_kwargs["use_object_loss_mask"] = False
    return PatternLoopDataset(
        root="./Data",
        mode=mode,
        transform=transform,
        extractor=extractor,
        extractor_args=extractor_args,
        **dataset_kwargs,
    )


@torch.no_grad()
def _collect_pool(datasets):
    pool = []
    for ds in datasets:
        n = int(getattr(ds, "base_size", len(ds)))
        for i in range(n):
            sample = ds[i]
            real, hidden, direction, exp, timestep, DI, info = sample[:7]
            e = exp
            if torch.is_tensor(e):
                e = e.item() if e.ndim == 0 else str(e)
            filename = str(ds.label.iloc[i % ds.base_size]["filename"])
            img = torch.clamp(real * 0.5 + 0.5, 0.0, 1.0).cpu()
            pool.append(
                {
                    "filename": filename,
                    "exp": str(e),
                    "timestep": int(timestep.item() if torch.is_tensor(timestep) else timestep),
                    "di": float(DI.reshape(-1)[0].item() if torch.is_tensor(DI) else DI),
                    "hidden": hidden.detach().cpu().float().reshape(-1),
                    "image": img,
                }
            )
    return pool


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / a.norm().clamp_min(1e-8)
    b = b / b.norm().clamp_min(1e-8)
    return float(1.0 - torch.dot(a, b).clamp(-1.0, 1.0))


def _otsu_threshold(gray_u8: np.ndarray) -> int:
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 127
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    return int(np.nanargmax(sigma_b))


def _damage_binary(image_chw: torch.Tensor) -> torch.Tensor:
    """Heuristic damage mask: dark pixels via Otsu on grayscale (1=damage)."""
    gray = image_chw.mean(dim=0).numpy()
    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    thr = _otsu_threshold(gray_u8)
    # Damage / crack tends to be darker than concrete.
    damage = (gray_u8 < thr).astype(np.float32)
    # If almost everything is "damage", invert (bright-bg assumption failed).
    if damage.mean() > 0.65:
        damage = 1.0 - damage
    return torch.from_numpy(damage)


def _iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    a = mask_a.bool()
    b = mask_b.bool()
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    if union <= 0:
        return 1.0  # both empty → identical "no damage"
    return float(inter / union)


@torch.no_grad()
def _pair_metrics(pool, num_pairs, seed, device):
    rng = np.random.default_rng(seed)
    n = len(pool)
    if n < 2:
        raise RuntimeError("Need at least 2 samples")
    # Sample pairs with replacement of index pairs; avoid i==j.
    i_idx = rng.integers(0, n, size=num_pairs * 2)
    j_idx = rng.integers(0, n, size=num_pairs * 2)
    pairs = []
    for i, j in zip(i_idx, j_idx):
        if i == j:
            continue
        pairs.append((int(i), int(j)))
        if len(pairs) >= num_pairs:
            break
    while len(pairs) < num_pairs:
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i != j:
            pairs.append((i, j))

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    cos_dist = np.zeros(num_pairs, dtype=np.float64)
    ssim_dist = np.zeros(num_pairs, dtype=np.float64)
    iou_dist = np.zeros(num_pairs, dtype=np.float64)
    di_abs = np.zeros(num_pairs, dtype=np.float64)

    # Precompute damage masks.
    print(f"[scatter] Computing damage masks for {n} images...", flush=True)
    damage = [_damage_binary(p["image"]) for p in pool]

    print(f"[scatter] Scoring {num_pairs} pairs...", flush=True)
    for k, (i, j) in enumerate(pairs):
        a, b = pool[i], pool[j]
        cos_dist[k] = _cosine_distance(a["hidden"], b["hidden"])
        di_abs[k] = abs(a["di"] - b["di"])

        img_a = a["image"].unsqueeze(0).to(device)
        img_b = b["image"].unsqueeze(0).to(device)
        ssim_val = float(ssim_metric(img_a, img_b).item())
        ssim_metric.reset()
        ssim_dist[k] = 1.0 - ssim_val

        iou_val = _iou(damage[i], damage[j])
        iou_dist[k] = 1.0 - iou_val

        if (k + 1) % 500 == 0 or k + 1 == num_pairs:
            print(f"[scatter]   {k+1}/{num_pairs}", flush=True)

    return {
        "cosine_distance": cos_dist,
        "ssim_distance": ssim_dist,
        "iou_distance": iou_dist,
        "di_abs_diff": di_abs,
        "pair_i": np.array([p[0] for p in pairs], dtype=np.int32),
        "pair_j": np.array([p[1] for p in pairs], dtype=np.int32),
    }


def _corr(x, y):
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _scatter_panel(metrics, out_path, title_prefix, n_samples, n_pairs):
    x = metrics["cosine_distance"]
    y_ssim = metrics["ssim_distance"]
    y_iou = metrics["iou_distance"]
    r_ssim = _corr(x, y_ssim)
    r_iou = _corr(x, y_iou)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    axes[0].scatter(x, y_ssim, s=8, alpha=0.35, c="#1f4e79", edgecolors="none")
    axes[0].set_xlabel(r"Cosine distance in HysGRU space: $1-\cos(g_A,g_B)$", fontsize=10)
    axes[0].set_ylabel(r"Visual distance: $1-\mathrm{SSIM}(I_A,I_B)$", fontsize=10)
    axes[0].set_title(f"SSIM distance  (Pearson r={r_ssim:.3f})", fontsize=11)
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(x, y_iou, s=8, alpha=0.35, c="#8b3a3a", edgecolors="none")
    axes[1].set_xlabel(r"Cosine distance in HysGRU space: $1-\cos(g_A,g_B)$", fontsize=10)
    axes[1].set_ylabel(r"Damage-area distance: $1-\mathrm{IoU}(M_A,M_B)$", fontsize=10)
    axes[1].set_title(f"Damage IoU distance  (Pearson r={r_iou:.3f})", fontsize=11)
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(
        f"{title_prefix} · N_samples={n_samples} · N_pairs={n_pairs}\n"
        "Damage mask = Otsu dark-pixel heuristic (no external segmentation model)",
        fontsize=11,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[scatter] Saved: {out_path}", flush=True)
    return {"pearson_ssim": r_ssim, "pearson_iou": r_iou}


def main():
    cli = _parse_args()
    result_dir = os.path.abspath(cli.result_dir)
    args = _load_args_from_result_dir(result_dir)
    args.result_dir = result_dir
    args.batch_size = cli.batch_size
    splits = [s.strip() for s in cli.splits.split(",") if s.strip()]
    out_dir = cli.output_dir.strip() or os.path.join(result_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[scatter] device={device} splits={splits} num_pairs={cli.num_pairs}", flush=True)

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

    datasets = []
    for mode in splits:
        print(f"[scatter] Building {mode} dataset...", flush=True)
        ds = _build_dataset(args, extractor, extractor_args, mode)
        datasets.append(ds)
    # set_dataset_args needs a dataloader once
    dl = DataLoader(datasets[0], batch_size=cli.batch_size, shuffle=False, num_workers=_resolve_num_workers(cli.num_workers))
    set_dataset_args(args, dl)

    print("[scatter] Collecting sample pool...", flush=True)
    pool = _collect_pool(datasets)
    print(f"[scatter] pool size={len(pool)}", flush=True)

    metrics = _pair_metrics(pool, cli.num_pairs, cli.seed, device)
    stats = _scatter_panel(
        metrics,
        os.path.join(out_dir, f"gru_visual_scatter_pairs{cli.num_pairs}.png"),
        title_prefix=f"splits={'+'.join(splits)}",
        n_samples=len(pool),
        n_pairs=cli.num_pairs,
    )

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "splits": splits,
        "num_samples": len(pool),
        "num_pairs": cli.num_pairs,
        "seed": cli.seed,
        "x": "cosine_distance = 1 - cos(g_A, g_B)  [HysGRU 1024-d]",
        "y_ssim": "1 - SSIM(I_A, I_B)",
        "y_iou": "1 - IoU(damage_A, damage_B); damage = Otsu dark-pixel mask",
        "pearson_r_ssim": stats["pearson_ssim"],
        "pearson_r_iou": stats["pearson_iou"],
        "mean_cosine_distance": float(metrics["cosine_distance"].mean()),
        "mean_ssim_distance": float(metrics["ssim_distance"].mean()),
        "mean_iou_distance": float(metrics["iou_distance"].mean()),
        "mean_di_abs_diff": float(metrics["di_abs_diff"].mean()),
    }
    summary_path = os.path.join(out_dir, f"gru_visual_scatter_pairs{cli.num_pairs}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        os.path.join(out_dir, f"gru_visual_scatter_pairs{cli.num_pairs}.npz"),
        **metrics,
    )
    print(f"[scatter] Summary: {summary_path}", flush=True)
    print(
        f"[scatter] Pearson r: SSIM={stats['pearson_ssim']:.4f}, IoU={stats['pearson_iou']:.4f}",
        flush=True,
    )
    print("[scatter] Done.", flush=True)


if __name__ == "__main__":
    main()
