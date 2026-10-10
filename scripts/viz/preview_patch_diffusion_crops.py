"""Visualize Patch Diffusion stochastic crops and coordinate channels."""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.model.ImageGeneration import CFMADM
from src.utils import build_damage_image_transform_with_loss_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Preview Patch Diffusion crop examples.")
    parser.add_argument("--data_path", type=str, default="id_list_no_leak.json")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_images", type=int, default=3, help="Number of source images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/DamagePatternGeneration/data_check/patch_diffusion_preview",
    )
    parser.add_argument("--patch_size_probs", type=str, default="0.5,0.3,0.2")
    return parser.parse_args()


def _to_vis(image_tensor):
    image = (image_tensor.detach().cpu() * 0.5 + 0.5).clamp(0.0, 1.0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return image.permute(1, 2, 0).numpy()


def _coord_to_vis(coord_tensor):
    arr = coord_tensor.detach().cpu().numpy()
    return np.clip((arr + 1.0) * 0.5, 0.0, 1.0)


def _apply_fixed_patch_crop(model, image, valid_mask, patch_size, top, left):
    """Deterministic crop for visualization (bypass stochastic schedule)."""
    full_size = int(image.shape[-1])
    grid_x, grid_y = model._build_full_coordinate_maps(
        full_size, full_size, device=image.device, dtype=image.dtype
    )
    image_b = image.unsqueeze(0)
    mask_b = valid_mask.unsqueeze(0) if valid_mask is not None else None

    if patch_size == full_size:
        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        return image_b, mask_b, coords, top, left, patch_size

    z_crop = image_b[:, :, top : top + patch_size, left : left + patch_size]
    mask_crop = (
        mask_b[:, :, top : top + patch_size, left : left + patch_size]
        if mask_b is not None
        else None
    )
    xx = grid_x[top : top + patch_size, left : left + patch_size]
    yy = grid_y[top : top + patch_size, left : left + patch_size]
    coords = torch.stack([xx, yy], dim=0).unsqueeze(0)
    return z_crop, mask_crop, coords, top, left, patch_size


def _pick_crop_box(full_size, patch_size, rng):
    if patch_size >= full_size:
        return 0, 0
    max_offset = full_size - patch_size
    top = int(rng.integers(0, max_offset + 1))
    left = int(rng.integers(0, max_offset + 1))
    return top, left


def _save_single_example(
    model,
    image,
    valid_mask,
    filename,
    di_value,
    patch_size,
    top,
    left,
    output_path,
):
    z_crop, mask_crop, coords, top, left, patch_size = _apply_fixed_patch_crop(
        model, image, valid_mask, patch_size, top, left
    )

    full_vis = _to_vis(image)
    patch_vis = _to_vis(z_crop[0])
    coord_x = _coord_to_vis(coords[0, 0])
    coord_y = _coord_to_vis(coords[0, 1])

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
    titles = [
        f"Full {image.shape[-1]}x{image.shape[-1]}",
        f"Patch {patch_size}x{patch_size}",
        "Coord X [-1,1]",
        "Coord Y [-1,1]",
        "Loss mask",
    ]

    axes[0].imshow(full_vis, cmap="gray" if full_vis.shape[-1] == 1 else None)
    rect = patches.Rectangle(
        (left, top),
        patch_size,
        patch_size,
        linewidth=2,
        edgecolor="#ff4444",
        facecolor="none",
    )
    axes[0].add_patch(rect)
    axes[0].set_title(titles[0], fontsize=10)

    axes[1].imshow(patch_vis, cmap="gray" if patch_vis.shape[-1] == 1 else None)
    axes[1].set_title(titles[1], fontsize=10)

    axes[2].imshow(coord_x, cmap="coolwarm", vmin=0.0, vmax=1.0)
    axes[2].set_title(titles[2], fontsize=10)

    axes[3].imshow(coord_y, cmap="coolwarm", vmin=0.0, vmax=1.0)
    axes[3].set_title(titles[3], fontsize=10)

    if mask_crop is not None:
        axes[4].imshow(mask_crop[0, 0].cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    else:
        axes[4].imshow(np.ones((patch_size, patch_size)), cmap="gray", vmin=0.0, vmax=1.0)
    axes[4].set_title(titles[4], fontsize=10)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{filename} | DI={di_value:.3f} | crop top={top}, left={left}, size={patch_size}",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_overview_grid(examples, output_path):
    n = len(examples)
    fig, axes = plt.subplots(n, 5, figsize=(14, 2.8 * n))
    if n == 1:
        axes = np.array([axes])

    for row, ex in enumerate(examples):
        full_vis = _to_vis(ex["image"])
        patch_vis = _to_vis(ex["patch"])
        coord_x = _coord_to_vis(ex["coords"][0])
        coord_y = _coord_to_vis(ex["coords"][1])
        patch_size = ex["patch_size"]
        top, left = ex["top"], ex["left"]

        axes[row, 0].imshow(full_vis, cmap="gray" if full_vis.shape[-1] == 1 else None)
        rect = patches.Rectangle(
            (left, top), patch_size, patch_size, linewidth=2, edgecolor="#ff4444", facecolor="none"
        )
        axes[row, 0].add_patch(rect)
        axes[row, 0].set_title(f"{ex['label']}\nfull + box", fontsize=9)

        axes[row, 1].imshow(patch_vis, cmap="gray" if patch_vis.shape[-1] == 1 else None)
        axes[row, 1].set_title(f"patch {patch_size}x{patch_size}", fontsize=9)

        axes[row, 2].imshow(coord_x, cmap="coolwarm", vmin=0.0, vmax=1.0)
        axes[row, 2].set_title("coord X", fontsize=9)

        axes[row, 3].imshow(coord_y, cmap="coolwarm", vmin=0.0, vmax=1.0)
        axes[row, 3].set_title("coord Y", fontsize=9)

        if ex["mask"] is not None:
            axes[row, 4].imshow(ex["mask"][0, 0].cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        else:
            axes[row, 4].imshow(np.ones((patch_size, patch_size)), cmap="gray")
        axes[row, 4].set_title("loss mask", fontsize=9)

        for col in range(5):
            axes[row, col].axis("off")

    fig.suptitle("Patch Diffusion crop examples (128 / 64 / 32)", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    split = json.load(open(os.path.join("./Data/DamagePattern", args.data_path)))
    train_files = set(split["train"])
    label = pd.read_csv("./Data/DamagePattern/label.csv")
    label = label[label["filename"].isin(train_files)].reset_index(drop=True)
    if label.empty:
        raise RuntimeError("No train images found for patch preview.")

    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(label), size=min(args.num_images, len(label)), replace=False)

    transform = build_damage_image_transform_with_loss_mask(
        image_size=args.image_size,
        use_augmentation=False,
        output_scale=True,
        normalize_mean=(0.5, 0.5, 0.5),
        normalize_std=(0.5, 0.5, 0.5),
        use_grayscale_input=True,
        use_clahe=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size="8,8",
    )

    model = CFMADM(
        architecture="CFMADM",
        condition_dim=1,
        image_size=args.image_size,
        lr=1e-4,
        result_dir=args.output_dir,
        latent_conditioning=True,
        direction_conditioning=True,
        add_DI_loss=False,
        cfm_model_channels=32,
        cfm_num_res_blocks=1,
        cfm_cond_dim=128,
        cfm_dropout=0.0,
        cfm_ode_steps=2,
        cfm_solver="euler",
        use_patch_diffusion=True,
        patch_size_probs=args.patch_size_probs,
        use_velocity_min_snr_weighting=False,
    )
    model.eval()

    image_dir = "./Data/DamagePattern/images"
    patch_sizes = [128, 64, 32]
    schedule_labels = ["128x128 (50%)", "64x64 (30%)", "32x32 (20%)"]
    overview_examples = []

    for idx in sample_idx:
        row = label.iloc[int(idx)]
        filename = str(row["filename"])
        di_value = float(row["DI"])
        pil_image = Image.open(os.path.join(image_dir, filename))
        transformed = transform(pil_image)
        image = transformed["target_image"]
        loss_mask = transformed.get("loss_mask")

        for patch_size, schedule_label in zip(patch_sizes, schedule_labels):
            top, left = _pick_crop_box(args.image_size, patch_size, rng)
            out_name = f"{os.path.splitext(filename)[0]}_P{patch_size}_t{top}_l{left}.png"
            out_path = os.path.join(args.output_dir, out_name)
            _save_single_example(
                model,
                image,
                loss_mask,
                filename,
                di_value,
                patch_size,
                top,
                left,
                out_path,
            )

            z_crop, mask_crop, coords, top, left, patch_size = _apply_fixed_patch_crop(
                model, image, loss_mask, patch_size, top, left
            )
            overview_examples.append(
                {
                    "image": image,
                    "patch": z_crop[0],
                    "coords": coords[0],
                    "mask": mask_crop,
                    "patch_size": patch_size,
                    "top": top,
                    "left": left,
                    "label": f"{filename}\n{schedule_label}",
                }
            )

    overview_path = os.path.join(args.output_dir, "patch_diffusion_overview.png")
    _save_overview_grid(overview_examples[: len(patch_sizes)], overview_path)

    print(f"[preview_patch_diffusion_crops] Saved examples to: {args.output_dir}")
    print(f"[preview_patch_diffusion_crops] Overview grid: {overview_path}")


if __name__ == "__main__":
    main()
