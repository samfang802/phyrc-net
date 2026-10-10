"""Preview train images with grayscale or color CLAHE preprocessing (before/after)."""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import build_damage_image_transform_with_loss_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Preview CLAHE preprocessing on train images.")
    parser.add_argument("--data_path", type=str, default="id_list_no_leak.json")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_images", type=int, default=12)
    parser.add_argument("--grid_cols", type=int, default=4)
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/DamagePatternGeneration/data_check/dataloader_color_clahe_preview.png",
    )
    parser.add_argument("--clahe_clip_limit", type=float, default=2.0)
    parser.add_argument("--clahe_tile_grid_size", type=str, default="8,8")
    parser.add_argument("--use_grayscale_input", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=False)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _to_vis(image_tensor):
    image = (image_tensor.detach().cpu() * 0.5 + 0.5).clamp(0.0, 1.0)
    arr = image.permute(1, 2, 0).numpy()
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def main():
    args = parse_args()
    split = json.load(open(os.path.join("./Data/DamagePattern", args.data_path)))
    train_files = set(split["train"])
    label = pd.read_csv("./Data/DamagePattern/label.csv")
    label = label[label["filename"].isin(train_files)].reset_index(drop=True)
    if label.empty:
        raise RuntimeError("No train images found for CLAHE preview.")

    rng = np.random.default_rng(args.seed)
    target_count = min(int(args.num_images), len(label))
    sample_idx = rng.choice(len(label), size=target_count, replace=False)

    common_kwargs = dict(
        image_size=args.image_size,
        use_augmentation=False,
        output_scale=True,
        normalize_mean=(0.5, 0.5, 0.5),
        normalize_std=(0.5, 0.5, 0.5),
        use_grayscale_input=args.use_grayscale_input,
    )
    transform_base = build_damage_image_transform_with_loss_mask(**common_kwargs, use_clahe=False)
    transform_clahe = build_damage_image_transform_with_loss_mask(
        **common_kwargs,
        use_clahe=True,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid_size=args.clahe_tile_grid_size,
    )

    image_dir = "./Data/DamagePattern/images"
    pairs = []
    for idx in sample_idx:
        filename = str(label.iloc[int(idx)]["filename"])
        di_value = float(label.iloc[int(idx)]["DI"])
        image = Image.open(os.path.join(image_dir, filename))
        base = transform_base(image)["target_image"]
        clahe = transform_clahe(image)["target_image"]
        assert base.shape[0] == 3, f"Expected 3-channel output, got {tuple(base.shape)}"
        assert clahe.shape[0] == 3, f"Expected 3-channel CLAHE output, got {tuple(clahe.shape)}"
        pairs.append((_to_vis(base), _to_vis(clahe), filename, di_value))

    cols = int(args.grid_cols)
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2.6, rows * 2.0))
    axes = np.array(axes).reshape(rows, cols * 2)

    base_title = "Gray" if args.use_grayscale_input else "RGB"
    clahe_title = "Gray+CLAHE" if args.use_grayscale_input else "RGB+CLAHE (Lab-L)"

    for i in range(rows * cols):
        r = i // cols
        c0 = (i % cols) * 2
        c1 = c0 + 1
        axes[r][c0].axis("off")
        axes[r][c1].axis("off")
        if i >= len(pairs):
            continue
        base_img, clahe_img, filename, di_value = pairs[i]
        axes[r][c0].imshow(base_img)
        axes[r][c1].imshow(clahe_img)
        if r == 0:
            axes[r][c0].set_title(base_title, fontsize=9)
            axes[r][c1].set_title(clahe_title, fontsize=9)
        axes[r][c0].text(
            0.02,
            0.98,
            f"DI={di_value:.2f}",
            transform=axes[r][c0].transAxes,
            fontsize=6,
            color="yellow",
            va="top",
        )

    mode_label = "Grayscale" if args.use_grayscale_input else "Color (Lab-L CLAHE)"
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fig.suptitle(
        f"Train CLAHE Preview ({mode_label}) | clipLimit={args.clahe_clip_limit} | "
        f"tileGridSize=({args.clahe_tile_grid_size})",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(args.output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    diffs = [float(np.mean(np.abs(a - b))) for a, b, *_ in pairs]
    channel_std = [float(np.std(a - b)) for a, b, *_ in pairs]
    print(f"[preview] Saved: {args.output_path}")
    print(f"[preview] Samples: {len(pairs)}")
    print(f"[preview] Mean |before-after| delta: {np.mean(diffs):.4f}")
    print(f"[preview] Mean channel delta std: {np.mean(channel_std):.4f}")
    print(
        f"[preview] mode={'grayscale' if args.use_grayscale_input else 'color'} "
        f"clipLimit={args.clahe_clip_limit} tileGridSize={args.clahe_tile_grid_size}"
    )


if __name__ == "__main__":
    main()
