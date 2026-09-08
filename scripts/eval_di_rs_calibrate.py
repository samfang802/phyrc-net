#!/usr/bin/env python3
"""Two-stage: frozen DIresnet18 + standalone Rs net → calibrate DI.

Fits on train:
  intercept:  DI' = DI_hat + a
  additive:   DI' = DI_hat + a + b · Rs
  affine:     DI' = a + b·Rs + c·DI_hat + d·DI_hat·Rs
"""

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
from src.model.ResNet18_CBAM_Regressor import ResNet18CBAMLightning
from src.model.ResNet18_Siamese_Lightning import ResNet18SiameseLightning


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--di_ckpt",
        type=str,
        default="results/DamageQuantification/resnet18_siamese_inpaint_nomask/DIresnet18/checkpoints/last.ckpt",
    )
    p.add_argument("--rs_ckpt", type=str, required=True)
    p.add_argument("--rs_result_dir", type=str, default="")
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


@torch.no_grad()
def _collect(di_model, rs_model, mode, args_rs, device):
    ds = DamageIndexGrayMaskDataset(
        root="./Data",
        mode=mode,
        image_size=int(args_rs.get("image_size", 128)),
        data_path=args_rs.get("data_path", "id_list_val_R307.json"),
        pattern_dirname=args_rs.get("pattern_dirname", "DamagePattern_inpaint_selfcondnoDI"),
        use_mask=bool(args_rs.get("use_mask", False)),
        use_augmentation=False,
        target_mode="di",
        return_rs=True,
        r_spall_csv=args_rs.get("r_spall_csv"),
        beta=args_rs.get("beta", 0.3),
        r_bar_method=args_rs.get("r_bar_method", "poly"),
        poly_degree=args_rs.get("poly_degree", 2),
        ma_window=args_rs.get("ma_window", 51),
    )
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    di_hat, rs_hat, di_gt, rs_gt, exps = [], [], [], [], []
    for batch in tqdm(dl, desc=mode, leave=True):
        image, exp, di, rs = batch
        image = image.to(device)
        d_out = di_model(image)
        if isinstance(d_out, tuple):
            d_out = d_out[0]
        r_out = rs_model(image)
        if isinstance(r_out, tuple):
            r_out = r_out[0]
        di_hat.append(d_out.view(-1).cpu())
        rs_hat.append(r_out.view(-1).cpu())
        di_gt.append(di.view(-1).cpu())
        rs_gt.append(rs.view(-1).cpu())
        if isinstance(exp, (list, tuple)):
            exps.extend(list(exp))
        else:
            exps.extend([str(e) for e in exp])
    return {
        "di_hat": torch.cat(di_hat).numpy().reshape(-1),
        "rs_hat": np.clip(torch.cat(rs_hat).numpy().reshape(-1), 0.0, None),
        "di_gt": torch.cat(di_gt).numpy().reshape(-1),
        "rs_gt": torch.cat(rs_gt).numpy().reshape(-1),
        "exp": np.asarray(exps),
    }


def _fit(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(design, y, rcond=None)[0]


def _design(kind: str, di_hat, rs):
    n = len(di_hat)
    ones = np.ones(n, dtype=np.float64)
    if kind == "raw":
        return None
    if kind == "intercept":
        return np.stack([di_hat, ones], 1)
    if kind == "additive":
        return np.stack([di_hat, ones, rs], 1)
    if kind == "affine":
        return np.stack([ones, rs, di_hat, di_hat * rs], 1)
    raise ValueError(kind)


def _apply(kind, w, di_hat, rs):
    if kind == "raw":
        return di_hat
    return _design(kind, di_hat, rs) @ w


def _metrics(gt, pred):
    return {
        "n": int(len(gt)),
        "r2": float(r2_score(gt, pred)) if len(gt) > 1 else float("nan"),
        "mae": float(mean_absolute_error(gt, pred)) if len(gt) else float("nan"),
        "bias": float(pred.mean() - gt.mean()) if len(gt) else float("nan"),
    }


def main():
    cli = _parse()
    rs_dir = cli.rs_result_dir or os.path.dirname(os.path.dirname(os.path.abspath(cli.rs_ckpt)))
    out_dir = cli.out_dir or os.path.join(rs_dir, "di_rs_calibrate")
    os.makedirs(out_dir, exist_ok=True)
    args_rs = json.load(open(os.path.join(rs_dir, "args.json"), encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    di_model = ResNet18SiameseLightning.load_from_checkpoint(
        cli.di_ckpt, map_location=device, strict=True
    ).to(device).eval()
    rs_model = ResNet18CBAMLightning.load_from_checkpoint(
        cli.rs_ckpt, map_location=device, strict=True
    ).to(device).eval()

    splits = {m: _collect(di_model, rs_model, m, args_rs, device) for m in ("train", "val", "test")}
    y_tr = splits["train"]["di_gt"]
    kinds = ("raw", "intercept", "additive", "affine")
    weights = {}
    for kind in kinds:
        if kind == "raw":
            weights[kind] = None
            continue
        X = _design(kind, splits["train"]["di_hat"], splits["train"]["rs_hat"])
        weights[kind] = _fit(X, y_tr).tolist()

    # Oracle additive with GT Rs (upper bound of this calibrator family).
    w_oracle = _fit(
        _design("additive", splits["train"]["di_hat"], splits["train"]["rs_gt"]),
        y_tr,
    ).tolist()

    report = {
        "di_ckpt": os.path.abspath(cli.di_ckpt),
        "rs_ckpt": os.path.abspath(cli.rs_ckpt),
        "weights": weights,
        "oracle_additive_gt_rs": w_oracle,
        "splits": {},
    }
    print(f"[cal] weights={json.dumps(weights)}", flush=True)

    for mode, pack in splits.items():
        row = {
            "rs_head": _metrics(pack["rs_gt"], pack["rs_hat"]),
        }
        for kind in kinds:
            pred = _apply(kind, None if kind == "raw" else np.asarray(weights[kind]), pack["di_hat"], pack["rs_hat"])
            row[kind] = _metrics(pack["di_gt"], pred)
        row["oracle_additive_gt_rs"] = _metrics(
            pack["di_gt"],
            _apply("additive", np.asarray(w_oracle), pack["di_hat"], pack["rs_gt"]),
        )
        report["splits"][mode] = row
        print(
            f"[cal] {mode}: DI raw R²={row['raw']['r2']:.4f}  "
            f"int={row['intercept']['r2']:.4f}  add={row['additive']['r2']:.4f}  "
            f"aff={row['affine']['r2']:.4f}  oracleGT={row['oracle_additive_gt_rs']['r2']:.4f}  "
            f"Rs R²={row['rs_head']['r2']:.4f}",
            flush=True,
        )

    # Test scatter: raw vs additive.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    pack = splits["test"]
    for ax, pred, title in (
        (axes[0], pack["di_hat"], f"raw DI R²={report['splits']['test']['raw']['r2']:.4f}"),
        (
            axes[1],
            _apply("additive", np.asarray(weights["additive"]), pack["di_hat"], pack["rs_hat"]),
            f"additive g(DÎ,Rŝ) R²={report['splits']['test']['additive']['r2']:.4f}",
        ),
    ):
        ax.axline((0, 0), slope=1, color="black")
        for g in np.unique(pack["exp"]):
            i = np.where(pack["exp"] == g)[0]
            ax.scatter(pack["di_gt"][i], pred[i], label=str(g), s=8)
        ax.set_xlabel("Ground Truth DI")
        ax.set_ylabel("Predicted DI")
        ax.set_xlim(0, 1.4)
        ax.set_ylim(0, 1.4)
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "test_raw_vs_additive.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[cal] wrote {json_path}", flush=True)
    print(f"[cal] wrote {fig_path}", flush=True)


if __name__ == "__main__":
    main()
