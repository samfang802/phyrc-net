#!/usr/bin/env python3
"""Extract 128x128 GT images used for fixed-DI inference anchors."""

import argparse
import importlib
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.dataset import PatternLoopDataset
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match


def _parse_args():
    parser = argparse.ArgumentParser(description="Extract inference GT images at a target DI")
    parser.add_argument("--result_dir", type=str, required=True, help="Run dir with args.json")
    parser.add_argument("--target_di", type=float, default=0.75)
    parser.add_argument(
        "--train_anchor_exp",
        type=str,
        default="",
        help="Train anchor specimen (default: from args.json)",
    )
    parser.add_argument(
        "--val_anchor_exp",
        type=str,
        default="",
        help="Val anchor specimen (default: from args.json)",
    )
    parser.add_argument("--output_dir", type=str, default="", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)
    return parser.parse_args()


def _collect_candidates(model, dataloader, split, anchor_exp):
    pool_key = f"{split}_anchor_candidates"
    model.eval()
    cap = model._inference_candidate_cap()
    required_exp = str(anchor_exp).strip()
    with torch.no_grad():
        for batch in dataloader:
            real_data, hidden_state, direction, exp, timestep, DI, info = batch[:7]
            real_vis = torch.clamp(real_data * 0.5 + 0.5, 0.0, 1.0)
            model._collect_inference_anchor_candidates(
                pool_key,
                split,
                real_vis,
                hidden_state,
                direction,
                exp,
                timestep,
                DI,
                info,
            )
            pool = model.validation_outputs[pool_key]
            if len(pool) >= cap:
                if not required_exp or any(str(c["exp"]) == required_exp for c in pool):
                    break


def _select_gt_anchor(model, candidates, split, anchor_exp, target_di):
    anchors = model._select_anchors_for_di_values(
        candidates,
        [float(target_di)],
        split=split,
        anchor_exp=anchor_exp,
    )
    if not anchors:
        raise RuntimeError(f"No {split} anchor found for exp={anchor_exp} at DI={target_di}")
    return anchors[0]


def _save_gt_image(tensor_chw, out_path):
    arr = tensor_chw.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
        Image.fromarray(arr, mode="L").save(out_path)
    else:
        Image.fromarray(arr, mode="RGB").save(out_path)


def main():
    cli = _parse_args()
    result_dir = os.path.abspath(cli.result_dir)
    args = _load_args_from_result_dir(result_dir)
    args.result_dir = result_dir

    train_exp = cli.train_anchor_exp.strip() or str(args.inference_train_anchor_exp).strip()
    val_exp = cli.val_anchor_exp.strip() or str(args.inference_val_anchor_exp).strip()
    output_dir = cli.output_dir.strip() or os.path.join(
        result_dir, "inference_samples", f"gt_di{cli.target_di:.2f}"
    )
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor_args = _load_args_from_result_dir(args.extractor_path)
    extractor = getattr(importlib.import_module("src.model.HystereticPrediction"), extractor_args.architecture)(**vars(extractor_args)).to(device)
    extractor_ckpt = _resolve_required_match(
        os.path.join(args.extractor_path, "**/best.ckpt"), "extractor checkpoint"
    )
    extractor.load_state_dict(
        torch.load(extractor_ckpt, map_location=device, weights_only=False)["state_dict"]
    )
    extractor.eval()
    for param in extractor.parameters():
        param.requires_grad = False

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
    model = getattr(importlib.import_module("src.model.ImageGeneration"), args.architecture)(
        regressor=None,
        **vars(args),
    )
    model.validation_outputs = {
        "gt_DI": [],
        "pred_DI": [],
        "train_anchor_candidates": [],
        "val_anchor_candidates": [],
        "test_anchor_candidates": [],
    }

    saved = {}
    for split, anchor_exp in (("train", train_exp), ("val", val_exp)):
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
            batch_size=cli.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        model.validation_outputs[f"{split}_anchor_candidates"] = []
        if hasattr(model, f"_fixed_{split}_inference_anchors"):
            delattr(model, f"_fixed_{split}_inference_anchors")

        _collect_candidates(model, dl, split, anchor_exp)
        candidates = model.validation_outputs[f"{split}_anchor_candidates"]
        anchor = _select_gt_anchor(model, candidates, split, anchor_exp, cli.target_di)
        gt = anchor["real"]
        h, w = gt.shape[-2], gt.shape[-1]
        if h != 128 or w != 128:
            raise ValueError(f"{split} GT shape is {h}x{w}, expected 128x128")

        png_path = os.path.join(
            output_dir,
            f"gt_{split}_{anchor_exp}_di{cli.target_di:.2f}_t{anchor['timestep']}.png",
        )
        pt_path = png_path.replace(".png", ".pt")
        _save_gt_image(gt, png_path)
        torch.save(
            {
                "split": split,
                "exp": anchor["exp"],
                "timestep": int(anchor["timestep"]),
                "target_di": float(cli.target_di),
                "gt_di": float(anchor["gt_di"]),
                "image": gt,
            },
            pt_path,
        )
        saved[split] = {
            "png": png_path,
            "pt": pt_path,
            "exp": anchor["exp"],
            "timestep": int(anchor["timestep"]),
            "gt_di": float(anchor["gt_di"]),
            "shape": tuple(gt.shape),
        }
        print(
            f"[extract_gt] {split}: exp={anchor['exp']} t={anchor['timestep']} "
            f"GT_DI={anchor['gt_di']:.4f} shape={tuple(gt.shape)} -> {png_path}",
            flush=True,
        )

    print(f"[extract_gt] Done. Output dir: {output_dir}", flush=True)
    return saved


if __name__ == "__main__":
    main()
