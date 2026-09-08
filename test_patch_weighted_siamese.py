#!/usr/bin/env python3
"""Evaluate PatchWeightedSiamese checkpoint on train/val/test (single-image R²)."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import DamageIndexGrayMaskDataset
from src.model.PatchWeightedSiameseLightning import PatchWeightedSiameseLightning
from src.model.ResNet18_Siamese_Lightning import ResNet18SiameseLightning


def _load_args(result_dir: str) -> argparse.Namespace:
    path = os.path.join(result_dir, "args.json")
    with open(path, encoding="utf-8") as f:
        return argparse.Namespace(**json.load(f))


@torch.no_grad()
def _eval_split(model, dataloader, device):
    preds, gts, exps = [], [], []
    for batch in tqdm(dataloader, desc="eval", leave=False):
        image, exp, di = batch[:3]
        image = image.to(device)
        pred = model(image).view(-1)
        preds.append(pred.detach().cpu())
        gts.append(di.view(-1).cpu())
        if isinstance(exp, (list, tuple)):
            exps.extend(list(exp))
        else:
            exps.extend([str(e) for e in exp])
    pred = torch.cat(preds).numpy().reshape(-1)
    gt = torch.cat(gts).numpy().reshape(-1)
    return {
        "n": int(len(gt)),
        "r2": float(r2_score(gt, pred)) if len(gt) > 1 else float("nan"),
        "mae": float(mean_absolute_error(gt, pred)) if len(gt) else float("nan"),
        "pred": pred,
        "gt": gt,
        "exp": np.asarray(exps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to last.ckpt (or any Lightning ckpt)",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="",
        help="Run dir with args.json (default: parent of checkpoints/)",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    cli = parser.parse_args()

    ckpt = os.path.abspath(cli.ckpt)
    if cli.result_dir:
        result_dir = os.path.abspath(cli.result_dir)
    else:
        # .../run/checkpoints/last.ckpt → .../run
        result_dir = os.path.dirname(os.path.dirname(ckpt))

    args = _load_args(result_dir)
    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")

    model = PatchWeightedSiameseLightning(**vars(args))
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    status = model.load_state_dict(payload["state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    epoch = int(payload.get("epoch", -1))
    step = int(payload.get("global_step", -1))
    print(f"[test] ckpt={ckpt}", flush=True)
    print(f"[test] result_dir={result_dir} epoch={epoch} step={step} load={status}", flush=True)
    print(
        f"[test] pattern={args.pattern_dirname} data_path={args.data_path} "
        f"target={args.target_mode} image_size={args.image_size}",
        flush=True,
    )

    splits = ["train", "val", "test"] if cli.split == "all" else [cli.split]
    out_root = os.path.join(result_dir, "prediction")
    os.makedirs(out_root, exist_ok=True)
    summary = {}

    for mode in splits:
        ds = DamageIndexGrayMaskDataset(
            root="./Data",
            mode=mode,
            image_size=int(args.image_size),
            data_path=args.data_path,
            pattern_dirname=args.pattern_dirname,
            use_mask=False,
            use_augmentation=False,
            target_mode=args.target_mode,
            beta=getattr(args, "beta", 0.3),
            r_spall_csv=getattr(args, "r_spall_csv", None),
            r_bar_method=getattr(args, "r_bar_method", "poly"),
            poly_degree=getattr(args, "poly_degree", 2),
            ma_window=getattr(args, "ma_window", 51),
            return_rs=False,
        )
        dl = DataLoader(ds, batch_size=cli.batch_size, shuffle=False, num_workers=0)
        metrics = _eval_split(model, dl, device)
        summary[mode] = {
            "n": metrics["n"],
            "r2": metrics["r2"],
            "mae": metrics["mae"],
        }
        print(
            f"[test] {mode}: n={metrics['n']} R²={metrics['r2']:.4f} MAE={metrics['mae']:.4f}",
            flush=True,
        )

        mode_dir = os.path.join(out_root, mode)
        os.makedirs(mode_dir, exist_ok=True)
        fig = ResNet18SiameseLightning.draw(
            metrics["gt"],
            metrics["pred"],
            metrics["exp"],
            target_name=str(args.target_mode).upper(),
        )
        fig.axes[0].set_title(f"{mode} R²={metrics['r2']:.4f} MAE={metrics['mae']:.4f}")
        scatter_path = os.path.join(mode_dir, "scatter.png")
        fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        np.savez(
            os.path.join(mode_dir, "predictions.npz"),
            gt=metrics["gt"],
            pred=metrics["pred"],
            exp=metrics["exp"],
        )
        print(f"[test] wrote {scatter_path}", flush=True)

    json_path = os.path.join(out_root, f"metrics_{cli.split}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"ckpt": ckpt, "epoch": epoch, "step": step, "splits": summary}, f, indent=2)
    print(f"[test] wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
