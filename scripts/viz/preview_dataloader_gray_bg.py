import argparse
import glob
import importlib
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import PatternLoopDataset
from src.utils import build_damage_image_transform_with_loss_mask

DEFAULT_COLUMN_INFO = [
    "id",
    "concreteStrength",
    "transverse_yield",
    "transverse_strength",
    "longitudinal_corner_yield",
    "longitudinal_corner_strength",
    "longitudinal_intermediate_yield",
    "longitudinal_intermediate_strength",
    "lMeasured",
    "axialLoad",
    "fcc",
    "epscc",
    "fcu",
    "epscu",
    "Vn",
    "Vst",
    "reinforcementRatio",
    "volTransReinfRatio",
]


def _resolve_required_match(path_pattern, description):
    matches = glob.glob(path_pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"Could not find {description} with pattern: {path_pattern}")
    return matches[0]


def _load_args_from_result_dir(result_dir):
    args_path = _resolve_required_match(os.path.join(result_dir, "**/args.json"), "args.json")
    return argparse.Namespace(**json.load(open(args_path, "r")))


def _resolve_num_workers(requested_workers):
    cpu_count = os.cpu_count() or 1
    return max(0, min(int(requested_workers), cpu_count))


def parse_args():
    parser = argparse.ArgumentParser(description="Preview 100 transformed training images (grayscale + jitter).")
    parser.add_argument("--extractor_path", type=str, required=True, help="Trained hysteresis extractor result directory")
    parser.add_argument("--data_path", type=str, default="id_list_no_leak.json", help="Split json under Data/DamagePattern")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--output_path", type=str, default="results/DamagePatternGeneration/data_check/dataloader_gray_bg_preview_10x10.png")
    parser.add_argument("--img_aug", action="store_true", help="Enable train-time geometric/noise augmentation")
    parser.add_argument(
        "--use_grayscale_input",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=True,
        help="Convert input image to grayscale and expand to 3 channels",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    extractor_args = _load_args_from_result_dir(args.extractor_path)
    column_info = getattr(extractor_args, "column_info", DEFAULT_COLUMN_INFO)
    extractor = getattr(importlib.import_module("src.model.HystereticPrediction"), extractor_args.architecture)(**vars(extractor_args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = extractor.to(device)
    extractor_ckpt = _resolve_required_match(os.path.join(args.extractor_path, "**/best.ckpt"), "extractor checkpoint")
    state_dict = torch.load(extractor_ckpt, map_location=device, weights_only=False)["state_dict"]
    extractor.load_state_dict(state_dict)
    extractor.eval()
    for param in extractor.parameters():
        param.requires_grad = False

    transform = build_damage_image_transform_with_loss_mask(
        image_size=args.image_size,
        use_augmentation=bool(args.img_aug),
        output_scale=True,
        normalize_mean=(0.5, 0.5, 0.5),
        normalize_std=(0.5, 0.5, 0.5),
        use_grayscale_input=bool(args.use_grayscale_input),
        aug_brightness=0.12,
        aug_contrast=0.12,
    )

    dataset = PatternLoopDataset(
        root="./Data",
        mode="train",
        transform=transform,
        extractor=extractor,
        extractor_args=extractor_args,
        result_dir="./results/DamagePatternGeneration",
        data_path=args.data_path,
        column_info=column_info,
        train_virtual_repeat=1,
        use_loss_mask=True,
        condition_perturb=False,
        image_size=args.image_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=_resolve_num_workers(args.num_workers),
        drop_last=False,
    )

    target_count = max(1, int(args.num_images))
    images = []
    for batch in dataloader:
        batch_images = batch[0]
        for image in batch_images:
            image = (image.detach().cpu() * 0.5 + 0.5).clamp(0.0, 1.0)
            images.append(image.permute(1, 2, 0).numpy())
            if len(images) >= target_count:
                break
        if len(images) >= target_count:
            break

    if not images:
        raise RuntimeError("No images were sampled from DataLoader.")

    images = images[:target_count]
    rows = int(args.grid_size)
    cols = int(args.grid_size)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        ax.axis("off")
        if i < len(images):
            ax.imshow(images[i])

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    color_mode = "Grayscale(3ch)" if bool(args.use_grayscale_input) else "RGB"
    fig.suptitle(f"DataLoader Preview: {color_mode} + ColorJitter", fontsize=12)
    plt.tight_layout()
    fig.savefig(args.output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    stacked = np.stack(images, axis=0)
    print(f"[preview] Saved: {args.output_path}")
    print(f"[preview] Samples: {len(images)}")
    print(
        f"[preview] img_aug={bool(args.img_aug)} "
        f"use_grayscale_input={bool(args.use_grayscale_input)}"
    )
    if bool(args.use_grayscale_input):
        channel_delta = np.abs(stacked[..., 0] - stacked[..., 1]).mean() + np.abs(stacked[..., 1] - stacked[..., 2]).mean()
        print(f"[preview] Mean channel mismatch (should be ~0 for grayscale): {channel_delta:.6f}")


if __name__ == "__main__":
    main()
