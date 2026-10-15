#!/usr/bin/env python3
"""Evaluate R² of CFM-generated images using a Siamese ResNet18 DI regressor.

Pipeline (test split by default):
  PatternLoopDataset → CFM sample_with_ode → grayscale ImageNet-norm → Siamese ResNet → R²
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from argument.argument_mixed import set_dataset_args
from src.dataset import PatternLoopDataset
from src.model.ResNet18_Siamese_Lightning import ResNet18SiameseLightning
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match


def _parse_args():
    p = argparse.ArgumentParser(description="CFM generation → Siamese ResNet R²")
    p.add_argument(
        "--cfm_checkpoint",
        type=str,
        default=(
            "results/DamagePatternGeneration/CFMADM/selfcondnoDI/"
            "Opensees Based damage Patterns Generation/vq9dirur/checkpoints/last.ckpt"
        ),
    )
    p.add_argument(
        "--cfm_result_dir",
        type=str,
        default="results/DamagePatternGeneration/CFMADM/selfcondnoDI",
        help="Directory containing CFM args.json",
    )
    p.add_argument(
        "--regressor_checkpoint",
        type=str,
        default=(
            "results/DamageQuantification/resnet18_siamese_inpaint_nomask/"
            "2026-08-08_23-33-47/checkpoints/last.ckpt"
        ),
    )
    p.add_argument(
        "--regressor_result_dir",
        type=str,
        default="",
        help="Directory with regressor args.json (default: parent of checkpoint)",
    )
    p.add_argument("--split", type=str, default="test", choices=("test", "val", "train"))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--cfm_ode_steps", type=int, default=0, help="0 = use args.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--also_score_gt",
        action="store_true",
        help="Also score real GT images with the same regressor (oracle baseline)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output dir (default: <cfm_result_dir>/eval_siamese_r2)",
    )
    p.add_argument("--apg_guidance_scale", type=float, default=0.0)
    p.add_argument("--apg_eta", type=float, default=0.0)
    p.add_argument("--apg_norm_threshold", type=float, default=15.0)
    p.add_argument("--apg_momentum", type=float, default=-0.5)
    p.add_argument("--autoguide_checkpoint", type=str, default="")
    p.add_argument("--autoguide_result_dir", type=str, default="")
    p.add_argument("--autoguide_weight", type=float, default=0.0)
    p.add_argument("--autoguide_eta", type=float, default=1.0, help="beta: weight of the part parallel to v (1=off)")
    p.add_argument("--autoguide_gamma", type=float, default=0.0, help="w(t)=max(w_min, w*(1-t)^gamma); 0=constant")
    p.add_argument("--autoguide_w_min", type=float, default=0.0)
    p.add_argument("--repel_prev_scale", type=float, default=0.0)
    p.add_argument("--repel_prev_mode", type=str, default="x1", choices=["x1", "state"])
    return p.parse_args()


def _strip_u2_vss(state_dict):
    drop = [k for k in list(state_dict.keys()) if "u2_vss" in str(k) or str(k).startswith("fid.")]
    for k in drop:
        del state_dict[k]
    return drop


def _load_cfm_checkpoint(model, checkpoint_path, device):
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = payload["state_dict"]
    dropped = _strip_u2_vss(state)
    if dropped:
        print(f"[eval_r2] stripped {len(dropped)} u2_vss keys", flush=True)

    status = model.load_state_dict(state, strict=False)
    missing = list(status.missing_keys)
    unexpected = list(status.unexpected_keys)
    print(f"[eval_r2] CFM load missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    step = int(payload.get("global_step", 0))
    epoch = int(payload.get("epoch", 0))
    print(f"[eval_r2] CFM loaded step={step} epoch={epoch}", flush=True)
    return step, epoch


def _load_siamese(checkpoint_path, result_dir, device):
    args = _load_args_from_result_dir(result_dir)
    model = ResNet18SiameseLightning(**vars(args))
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    status = model.load_state_dict(payload["state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[eval_r2] Siamese load: {status}", flush=True)
    print(
        f"[eval_r2] Siamese use_mask={getattr(args, 'use_mask', False)} "
        f"in_channels={model.net.in_channels} image_size={getattr(args, 'image_size', 128)}",
        flush=True,
    )
    return model, args


def _rgb01_to_gray_imagenet(rgb01: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) in [0,1] → (B,1,H,W) ImageNet-gray normalized (same as DamageIndexGrayMaskDataset)."""
    r, g, b = rgb01[:, 0:1], rgb01[:, 1:2], rgb01[:, 2:3]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return (gray - 0.485) / 0.229


def _siamese_pred(siamese, x):
    out = siamese(x)
    if isinstance(out, tuple):
        out = out[0]
    return out.view(-1)


def _eval_damage_mask(cfm, c, damage_mask):
    """Val/test U-Net mask channel is all-ones; None would become zeros in ADM forward."""
    if not bool(getattr(cfm, "use_damage_mask_condition", False)):
        return None
    spatial = int(cfm.model_spatial_size)
    if damage_mask is None:
        return torch.ones((c.shape[0], 1, spatial, spatial), device=c.device, dtype=torch.float32)
    return damage_mask.to(device=c.device, dtype=torch.float32)


def _generate_batch(cfm, info, hidden_state, direction, DI, damage_mask=None):
    """Stage-1 128 then Stage-2 256 when cascade is on; else single sample_with_ode."""
    c, gru_feat = cfm._unpack_conditions(info, hidden_state, direction, DI)
    damage_mask = _eval_damage_mask(cfm, c, damage_mask)
    if not bool(getattr(cfm, "use_cascade_sr", False)):
        sampled = cfm.sample_with_ode(c, gru_feat=gru_feat, damage_mask=damage_mask)
        return cfm._decode_from_target_space(sampled)

    stage1 = cfm._get_cascade_stage1()
    if stage1 is None:
        raise RuntimeError("use_cascade_sr=True but Stage-1 is not attached")
    c1, gru1 = stage1._unpack_conditions(info, hidden_state, direction, DI)
    z_lr = stage1.sample_with_ode(c1, gru_feat=gru1, damage_mask=damage_mask)
    img_lr = stage1._decode_from_target_space(z_lr)
    hr = int(cfm.model_spatial_size)
    lr_cond = F.interpolate(img_lr, size=(hr, hr), mode="bicubic", align_corners=False)
    sampled = cfm.sample_with_ode(c, gru_feat=gru_feat, lr_cond=lr_cond, damage_mask=damage_mask)
    return cfm._decode_from_target_space(sampled)


def _scatter(gt, pred, title, out_path):
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.axline((0, 0), slope=1, color="black", linewidth=1.0)
    ax.scatter(gt, pred, s=12, alpha=0.75)
    ax.set_xlabel("Ground Truth DI")
    ax.set_ylabel("Predicted DI (Siamese)")
    ax.set_xlim(0, 1.4)
    ax.set_ylim(0, 1.4)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def _evaluate(cfm, siamese, dataloader, device, also_score_gt=False, siamese_size=128):
    cfm.eval()
    siamese.eval()
    siamese_size = int(siamese_size)

    gt_all, pred_gen_all, pred_gt_all = [], [], []
    exp_all = []

    for batch in tqdm(dataloader, desc="generate+score"):
        real_data, hidden_state, direction, exp, _timestep, DI, info, _valid_mask, condition_mask, *_rest = (
            cfm._unpack_generation_batch(batch)
        )
        real_data = real_data.to(device)
        hidden_state = hidden_state.to(device)
        direction = direction.to(device)
        DI = DI.to(device)
        info = info.to(device)
        if condition_mask is not None:
            condition_mask = condition_mask.to(device)

        sampled_image = _generate_batch(
            cfm, info, hidden_state, direction, DI, damage_mask=condition_mask
        )
        sampled_vis = torch.clamp(sampled_image * 0.5 + 0.5, 0.0, 1.0)
        if tuple(sampled_vis.shape[-2:]) != (siamese_size, siamese_size):
            sampled_vis = F.interpolate(
                sampled_vis, size=(siamese_size, siamese_size), mode="bilinear", align_corners=False
            )

        gen_in = _rgb01_to_gray_imagenet(sampled_vis)
        pred_gen = _siamese_pred(siamese, gen_in)

        gt_all.append(DI.view(-1).detach().cpu())
        pred_gen_all.append(pred_gen.detach().cpu())
        if isinstance(exp, (list, tuple)):
            exp_all.extend([str(e) for e in exp])
        else:
            exp_all.extend([str(e) for e in exp])

        if also_score_gt:
            real_vis = torch.clamp(real_data * 0.5 + 0.5, 0.0, 1.0)
            if tuple(real_vis.shape[-2:]) != (siamese_size, siamese_size):
                real_vis = F.interpolate(
                    real_vis, size=(siamese_size, siamese_size), mode="bilinear", align_corners=False
                )
            gt_in = _rgb01_to_gray_imagenet(real_vis)
            pred_gt_all.append(_siamese_pred(siamese, gt_in).detach().cpu())

    gt = torch.cat(gt_all).numpy()
    pred_gen = torch.cat(pred_gen_all).numpy()
    out = {
        "n": int(gt.size),
        "generated_r2": float(r2_score(gt, pred_gen)),
        "generated_mae": float(mean_absolute_error(gt, pred_gen)),
        "gt": gt,
        "pred_generated": pred_gen,
        "exp": exp_all,
    }
    if also_score_gt and pred_gt_all:
        pred_gt = torch.cat(pred_gt_all).numpy()
        out["gt_image_r2"] = float(r2_score(gt, pred_gt))
        out["gt_image_mae"] = float(mean_absolute_error(gt, pred_gt))
        out["pred_gt_images"] = pred_gt
    return out


def main():
    cli = _parse_args()
    torch.manual_seed(int(cli.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cli.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfm_ckpt = os.path.abspath(cli.cfm_checkpoint)
    cfm_result_dir = os.path.abspath(cli.cfm_result_dir)
    reg_ckpt = os.path.abspath(cli.regressor_checkpoint)
    reg_result_dir = os.path.abspath(
        cli.regressor_result_dir.strip()
        or os.path.dirname(os.path.dirname(reg_ckpt))
    )
    out_dir = os.path.abspath(cli.out_dir.strip() or os.path.join(cfm_result_dir, "eval_siamese_r2"))
    if not str(cli.out_dir).strip():
        tags = []
        if float(cli.apg_guidance_scale) != 0.0:
            tags.append(f"apg{float(cli.apg_guidance_scale):.2f}".replace(".", "p"))
        if float(cli.autoguide_weight) != 0.0:
            tags.append(f"ag{float(cli.autoguide_weight):.2f}".replace(".", "p"))
            if float(cli.autoguide_gamma) != 0.0:
                tags.append(f"g{float(cli.autoguide_gamma):g}".replace(".", "p"))
        if float(cli.repel_prev_scale) != 0.0:
            tags.append(
                f"repel{float(cli.repel_prev_scale):.2f}".replace(".", "p")
                + f"_{cli.repel_prev_mode}"
            )
        if tags:
            out_dir = os.path.abspath(
                os.path.join(cfm_result_dir, "eval_siamese_r2_" + "_".join(tags))
            )
    os.makedirs(out_dir, exist_ok=True)

    print(f"[eval_r2] device={device}", flush=True)
    print(f"[eval_r2] cfm_ckpt={cfm_ckpt}", flush=True)
    print(f"[eval_r2] siamese_ckpt={reg_ckpt}", flush=True)
    print(f"[eval_r2] split={cli.split}", flush=True)

    args = _load_args_from_result_dir(cfm_result_dir)
    args.result_dir = cfm_result_dir
    args.batch_size = int(cli.batch_size)
    if int(cli.cfm_ode_steps) > 0:
        args.cfm_ode_steps = int(cli.cfm_ode_steps)
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    _peek = torch.load(cfm_ckpt, map_location="cpu", weights_only=False)
    _sd = _peek.get("state_dict", _peek)
    _w = _sd.get("velocity_net.scalar_mlp.0.weight")
    if _w is not None and int(_w.shape[1]) == 3 and bool(getattr(args, "zero_di_condition", False)):
        args.keep_zero_di_slot = True
        print("[eval_r2] keep_zero_di_slot=True (ckpt scalar_mlp in=3)", flush=True)

    extractor_path = args.extractor_path
    extractor_args = _load_args_from_result_dir(extractor_path)
    extractor = getattr(importlib.import_module("src.model.HystereticPrediction"), extractor_args.architecture)(**vars(extractor_args)).to(device)
    extractor_ckpt = _resolve_required_match(
        os.path.join(extractor_path, "**/best.ckpt"), "extractor checkpoint"
    )
    extractor.load_state_dict(
        torch.load(extractor_ckpt, map_location=device, weights_only=False)["state_dict"]
    )
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False
    print("[eval_r2] Extractor loaded", flush=True)

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
    print(f"[eval_r2] Building {cli.split} dataset...", flush=True)
    ds = PatternLoopDataset(
        root="./Data",
        mode=cli.split,
        transform=transform,
        extractor=extractor,
        extractor_args=extractor_args,
        **dataset_kwargs,
    )
    dl = DataLoader(ds, batch_size=cli.batch_size, shuffle=False, num_workers=num_workers)
    set_dataset_args(args, dl)
    print(
        f"[eval_r2] samples={len(ds)} condition_dim={args.condition_dim} "
        f"ode_steps={args.cfm_ode_steps} zero_di={getattr(args, 'zero_di_condition', False)}",
        flush=True,
    )

    cfm = getattr(importlib.import_module("src.model.ImageGeneration"), args.architecture)(
        regressor=None,
        **vars(args),
    ).to(device)
    step, epoch = _load_cfm_checkpoint(cfm, cfm_ckpt, device)
    cfm.eval()
    cfm.apg_guidance_scale = float(cli.apg_guidance_scale)
    cfm.apg_eta = float(cli.apg_eta)
    cfm.apg_norm_threshold = float(cli.apg_norm_threshold)
    cfm.apg_momentum = float(cli.apg_momentum)
    cfm.autoguide_checkpoint = str(cli.autoguide_checkpoint)
    cfm.autoguide_result_dir = str(cli.autoguide_result_dir)
    cfm.autoguide_weight = float(cli.autoguide_weight)
    cfm.autoguide_eta = float(cli.autoguide_eta)
    cfm.autoguide_gamma = float(cli.autoguide_gamma)
    cfm.autoguide_w_min = float(cli.autoguide_w_min)
    cfm.repel_prev_scale = float(cli.repel_prev_scale)
    cfm.repel_prev_mode = str(cli.repel_prev_mode).lower()
    if float(cli.apg_guidance_scale) != 0.0:
        print(
            f"[eval_r2] APG s={float(cli.apg_guidance_scale):g} eta={float(cli.apg_eta):g} "
            f"norm_thr={float(cli.apg_norm_threshold):g} mom={float(cli.apg_momentum):g}",
            flush=True,
        )
    if float(cli.repel_prev_scale) != 0.0:
        print(
            f"[eval_r2] repel_prev γ={float(cli.repel_prev_scale):g} "
            f"mode={cli.repel_prev_mode}",
            flush=True,
        )
    if bool(getattr(cfm, "use_cascade_sr", False)):
        cfm._attach_cascade_stage1()
        stage1 = cfm._get_cascade_stage1()
        if stage1 is None:
            raise RuntimeError("Failed to attach cascade Stage-1")
        stage1.to(device)
        stage1.eval()
        print(
            f"[eval_r2] Cascade Stage-1 attached image_size={getattr(stage1, 'image_size', getattr(stage1, 'model_spatial_size', '?'))}",
            flush=True,
        )

    siamese, reg_args = _load_siamese(reg_ckpt, reg_result_dir, device)
    siamese_size = int(getattr(reg_args, "image_size", 128))

    metrics = _evaluate(
        cfm,
        siamese,
        dl,
        device,
        also_score_gt=bool(cli.also_score_gt),
        siamese_size=siamese_size,
    )

    print("======================================================================")
    print(
        f"[eval_r2] split={cli.split} n={metrics['n']} "
        f"generated_R2={metrics['generated_r2']:.4f} "
        f"generated_MAE={metrics['generated_mae']:.4f}"
    )
    if "gt_image_r2" in metrics:
        print(
            f"[eval_r2] GT-image R2={metrics['gt_image_r2']:.4f} "
            f"MAE={metrics['gt_image_mae']:.4f} (oracle / same regressor)"
        )
    print("======================================================================")

    scatter_path = os.path.join(out_dir, f"{cli.split}_generated_scatter.png")
    _scatter(
        metrics["gt"],
        metrics["pred_generated"],
        title=rf"CFM generated → Siamese $R^2$={metrics['generated_r2']:.4f}",
        out_path=scatter_path,
    )

    payload = {
        "split": cli.split,
        "n": metrics["n"],
        "generated_r2": metrics["generated_r2"],
        "generated_mae": metrics["generated_mae"],
        "cfm_checkpoint": cfm_ckpt,
        "cfm_global_step": step,
        "cfm_epoch": epoch,
        "cfm_ode_steps": int(args.cfm_ode_steps),
        "cfm_solver": str(getattr(args, "cfm_solver", "")),
        "zero_di_condition": bool(getattr(args, "zero_di_condition", False)),
        "condition_mode": str(getattr(args, "condition_mode", "")),
        "apg_guidance_scale": float(cli.apg_guidance_scale),
        "apg_eta": float(cli.apg_eta),
        "apg_norm_threshold": float(cli.apg_norm_threshold),
        "apg_momentum": float(cli.apg_momentum),
        "repel_prev_scale": float(cli.repel_prev_scale),
        "repel_prev_mode": str(cli.repel_prev_mode),
        "regressor_checkpoint": reg_ckpt,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scatter_path": scatter_path,
    }
    if "gt_image_r2" in metrics:
        payload["gt_image_r2"] = metrics["gt_image_r2"]
        payload["gt_image_mae"] = metrics["gt_image_mae"]

    json_path = os.path.join(out_dir, f"{cli.split}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    npz_path = os.path.join(out_dir, f"{cli.split}_predictions.npz")
    np.savez_compressed(
        npz_path,
        gt=metrics["gt"],
        pred_generated=metrics["pred_generated"],
        exp=np.asarray(metrics["exp"]),
        **(
            {"pred_gt_images": metrics["pred_gt_images"]}
            if "pred_gt_images" in metrics
            else {}
        ),
    )
    print(f"[eval_r2] wrote {json_path}", flush=True)
    print(f"[eval_r2] wrote {scatter_path}", flush=True)
    print(f"[eval_r2] wrote {npz_path}", flush=True)


if __name__ == "__main__":
    main()
