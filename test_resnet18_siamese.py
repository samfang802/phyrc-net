#!/usr/bin/env python3
"""Evaluate ResNet18 Siamese checkpoint on train/val/test (single-image R²)."""

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
from src.model.ResNet18_Siamese_Lightning import ResNet18SiameseLightning


def _load_args(result_dir: str) -> argparse.Namespace:
    path = os.path.join(result_dir, "args.json")
    with open(path, encoding="utf-8") as f:
        return argparse.Namespace(**json.load(f))


@torch.no_grad()
def _eval_split(model, dataloader, device, multi_task):
    preds, gts, exps = [], [], []
    preds_rs, gts_rs = [], []
    for batch in tqdm(dataloader, desc="eval", leave=False):
        if multi_task:
            image, exp, di, rs = batch
            rs = rs.to(device)
        else:
            image, exp, di = batch[:3]
            rs = None
        image = image.to(device)
        out = model(image)
        pred_di, pred_rs = model._split_pred(out)
        preds.append(pred_di.detach().cpu().view(-1))
        gts.append(di.view(-1).cpu())
        if multi_task:
            preds_rs.append(pred_rs.detach().cpu().view(-1))
            gts_rs.append(rs.detach().cpu().view(-1))
        if isinstance(exp, (list, tuple)):
            exps.extend(list(exp))
        else:
            exps.extend([str(e) for e in exp])
    pred = torch.cat(preds).numpy().reshape(-1)
    gt = torch.cat(gts).numpy().reshape(-1)
    out = {
        "n": int(len(gt)),
        "r2": float(r2_score(gt, pred)) if len(gt) > 1 else float("nan"),
        "mae": float(mean_absolute_error(gt, pred)) if len(gt) else float("nan"),
        "pred": pred,
        "gt": gt,
        "exp": np.asarray(exps),
        "pred_rs": None,
        "gt_rs": None,
        "r2_rs": float("nan"),
        "mae_rs": float("nan"),
    }
    if preds_rs:
        pred_rs = torch.cat(preds_rs).numpy().reshape(-1)
        gt_rs = torch.cat(gts_rs).numpy().reshape(-1)
        out["pred_rs"] = pred_rs
        out["gt_rs"] = gt_rs
        out["r2_rs"] = float(r2_score(gt_rs, pred_rs)) if len(gt_rs) > 1 else float("nan")
        out["mae_rs"] = float(mean_absolute_error(gt_rs, pred_rs)) if len(gt_rs) else float("nan")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--result_dir", type=str, default="")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    cli = parser.parse_args()

    ckpt = os.path.abspath(cli.ckpt)
    if cli.result_dir:
        result_dir = os.path.abspath(cli.result_dir)
    else:
        result_dir = os.path.dirname(os.path.dirname(ckpt))

    args = _load_args(result_dir)
    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    multi_task = bool(getattr(args, "multi_task", False))

    model_kwargs = dict(vars(args))
    model_kwargs["result_dir"] = result_dir
    model = ResNet18SiameseLightning(**model_kwargs)
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
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
        f"target={args.target_mode} image_size={args.image_size} multi_task={multi_task}",
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
            use_mask=bool(args.use_mask),
            use_rgb=bool(getattr(args, "use_rgb", False)),
            use_clahe=bool(getattr(args, "use_clahe", False)),
            use_augmentation=False,
            target_mode=args.target_mode,
            beta=getattr(args, "beta", 0.3),
            r_spall_csv=getattr(args, "r_spall_csv", None),
            r_bar_method=getattr(args, "r_bar_method", "poly"),
            poly_degree=getattr(args, "poly_degree", 2),
            ma_window=getattr(args, "ma_window", 51),
            return_rs=multi_task,
        )
        dl = DataLoader(ds, batch_size=cli.batch_size, shuffle=False, num_workers=0)
        metrics = _eval_split(model, dl, device, multi_task)
        summary[mode] = {
            "n": metrics["n"],
            "r2": metrics["r2"],
            "mae": metrics["mae"],
            "r2_rs": metrics["r2_rs"],
            "mae_rs": metrics["mae_rs"],
        }
        rs_tag = (
            f" R²_rs={metrics['r2_rs']:.4f} MAE_rs={metrics['mae_rs']:.4f}"
            if multi_task
            else ""
        )
        print(
            f"[test] {mode}: n={metrics['n']} R²={metrics['r2']:.4f} "
            f"MAE={metrics['mae']:.4f}{rs_tag}",
            flush=True,
        )

        mode_dir = os.path.join(out_root, mode)
        os.makedirs(mode_dir, exist_ok=True)
        fig = ResNet18SiameseLightning.draw(
            metrics["gt"],
            metrics["pred"],
            metrics["exp"],
            target_name=str(args.target_mode).upper(),
            pred_rs=metrics["pred_rs"],
            gt_rs=metrics["gt_rs"],
        )
        title = f"{mode} R²={metrics['r2']:.4f} MAE={metrics['mae']:.4f}"
        if multi_task:
            title += f" | Rs R²={metrics['r2_rs']:.4f}"
        fig.suptitle(title, fontsize=10)
        scatter_path = os.path.join(mode_dir, "scatter.png")
        fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        save_kw = {"gt": metrics["gt"], "pred": metrics["pred"], "exp": metrics["exp"]}
        if metrics["pred_rs"] is not None:
            save_kw["gt_rs"] = metrics["gt_rs"]
            save_kw["pred_rs"] = metrics["pred_rs"]
        np.savez(os.path.join(mode_dir, "predictions.npz"), **save_kw)
        print(f"[test] wrote {scatter_path}", flush=True)

    json_path = os.path.join(out_root, f"metrics_{cli.split}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"ckpt": ckpt, "epoch": epoch, "step": step, "splits": summary}, f, indent=2)
    print(f"[test] wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
