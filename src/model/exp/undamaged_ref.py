"""Other-specimen undamaged appearance prior (AdaGN embedding, not pixel concat).

Train/infer protocol: never use the same ``exp`` as the generation target.
The reference image + its interference/object mask are jittered together
(small translation + photometric) then encoded to a vector for AdaGN.
The *target* specimen mask stays a U-Net concat channel (handled elsewhere).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.model.exp.mafp import apply_mafp
from src.utils import build_damage_image_transform_with_loss_mask


class UndamagedRefEncoder(nn.Module):
    """4-channel (RGB + mask) → ``out_dim``. Last linear is zero-init."""

    def __init__(self, in_ch=4, out_dim=128):
        super().__init__()
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Conv2d(int(in_ch), 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, self.out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def _load_object_mask_hw(mask_dir, filename, image_size):
    stem, _ext = os.path.splitext(str(filename))
    for name in (str(filename), stem + ".png", stem + ".jpg"):
        path = os.path.join(mask_dir, name)
        if os.path.isfile(path):
            break
    else:
        return torch.ones(1, image_size, image_size, dtype=torch.float32)
    mask_np = np.array(Image.open(path).convert("L"), dtype=np.float32)
    mask = torch.from_numpy(mask_np).unsqueeze(0)
    if mask.shape[-2:] != (image_size, image_size):
        mask = F.interpolate(mask.unsqueeze(0), size=(image_size, image_size), mode="nearest").squeeze(0)
    return (1.0 - mask / 255.0).clamp(0.0, 1.0)


def build_undamaged_ref_pool(
    *,
    data_root="./Data",
    data_path="id_list_val_R307.json",
    image_size=128,
    di_max=0.05,
    use_grayscale_input=True,
    use_clahe=True,
    clahe_clip_limit=2.0,
    clahe_tile_grid_size="8,8",
):
    """One DI≈0 frame per *train* specimen (CPU tensors in model range [-1, 1])."""
    pattern_root = os.path.join(data_root, "DamagePattern")
    split_path = os.path.join(pattern_root, data_path)
    train_files = set(json.load(open(split_path))["train"])
    label = pd.read_csv(os.path.join(pattern_root, "label.csv"))
    label = label[label["filename"].isin(train_files)].copy()
    if label.empty:
        raise ValueError(f"No train rows in {split_path}")

    if isinstance(clahe_tile_grid_size, str):
        parts = [int(p.strip()) for p in clahe_tile_grid_size.split(",") if p.strip()]
        clahe_tile_grid_size = tuple(parts) if len(parts) == 2 else (8, 8)
    transform = build_damage_image_transform_with_loss_mask(
        image_size=int(image_size),
        use_augmentation=False,
        output_scale=True,
        normalize_mean=(0.5, 0.5, 0.5),
        normalize_std=(0.5, 0.5, 0.5),
        use_grayscale_input=bool(use_grayscale_input),
        use_clahe=bool(use_clahe),
        clahe_clip_limit=float(clahe_clip_limit),
        clahe_tile_grid_size=clahe_tile_grid_size,
    )
    mask_dir = os.path.join(pattern_root, "mask")
    img_dir = os.path.join(pattern_root, "images")

    pool_images = []
    pool_masks = []
    pool_exps = []
    for exp, group in label.groupby("exp", sort=False):
        low = group[group["DI"].astype(float) <= float(di_max)]
        row = (low if len(low) else group).sort_values("DI").iloc[0]
        filename = str(row["filename"])
        image = Image.open(os.path.join(img_dir, filename))
        mask = _load_object_mask_hw(mask_dir, filename, int(image_size))
        packed = transform(image, object_mask=mask)
        if isinstance(packed, dict):
            image_t = packed["target_image"]
            mask_t = packed.get("object_mask", mask)
        else:
            image_t = packed
            mask_t = mask
        if mask_t.ndim == 3:
            mask_t = mask_t
        else:
            mask_t = mask_t.view(1, *mask_t.shape[-2:])
        pool_images.append(image_t.detach().cpu().float())
        pool_masks.append(mask_t.detach().cpu().float())
        pool_exps.append(str(exp))

    images = torch.stack(pool_images, dim=0)
    masks = torch.stack(pool_masks, dim=0)
    print(
        f"[undamaged_ref] pool n={len(pool_exps)} exps={pool_exps} "
        f"di_max={di_max} size={tuple(images.shape)}",
        flush=True,
    )
    return images, masks, pool_exps


def sample_other_specimen_indices(target_exps, pool_exps, generator=None):
    """For each target exp, pick a pool index with a *different* exp when possible."""
    n = len(target_exps)
    pool_n = len(pool_exps)
    if pool_n == 0:
        raise ValueError("undamaged ref pool is empty")
    indices = []
    for exp in target_exps:
        exp = str(exp)
        candidates = [i for i, pe in enumerate(pool_exps) if pe != exp]
        if not candidates:
            candidates = list(range(pool_n))
        pick = torch.randint(0, len(candidates), (1,), generator=generator).item()
        indices.append(candidates[pick])
    return indices


def jitter_undamaged_ref(images, masks, *, training, tx_max=0.08, ty_max=0.02, p_diff=0.5, p_ada=0.3):
    """Spatial + photometric jitter on the reference branch only."""
    return apply_mafp(
        images,
        condition_mask=masks,
        valid_mask=None,
        p_diff=float(p_diff),
        p_ada=float(p_ada),
        tx_max=float(tx_max),
        ty_max=float(ty_max),
        training=bool(training),
    )
