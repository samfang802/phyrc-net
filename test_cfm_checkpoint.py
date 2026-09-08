#!/usr/bin/env python3
"""Run CFMADM validation + fixed-DI test inference from a checkpoint_step_XXXXXX.pt file."""

import argparse
import importlib
import json
import os
from datetime import datetime, timezone

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from argument.argument_mixed import set_dataset_args
from src.dataset import PatternLoopDataset
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match


def _parse_args():
    parser = argparse.ArgumentParser(description="Test CFMADM from periodic .pt checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint_step_XXXXXX.pt",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="",
        help="Run directory containing args.json (default: parent of checkpoint)",
    )
    parser.add_argument("--batch_size", type=int, default=0, help="Override batch size (0=use args.json)")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--fixed_di_values",
        type=str,
        default="0,0.4,0.6,0.8,1",
        help="Comma-separated DI values for test inference panel",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("test", "val", "train"),
        help="Dataset split for GT/inference panel (test=held-out test set)",
    )
    parser.add_argument(
        "--anchor_exp",
        type=str,
        default="",
        help="Anchor specimen id (default: R307/R615/CTR1 by split)",
    )
    parser.add_argument(
        "--metrics_split",
        type=str,
        default="test",
        choices=("test", "val", "train"),
        help="Dataset split for FID/PSNR/SSIM/loss metrics (default: test)",
    )
    parser.add_argument("--skip_metrics", action="store_true", help="Skip split metrics")
    parser.add_argument(
        "--eval_real_mask",
        action="store_true",
        help="Use on-disk object/rebar mask as U-Net condition at val/test (not all-ones)",
    )
    parser.add_argument("--use_wandb", action="store_true", help="Log to wandb (optional)")
    parser.add_argument(
        "--cfm_ode_steps",
        type=int,
        default=0,
        help="Override ODE integration steps for inference (0=use args.json)",
    )
    parser.add_argument(
        "--ref_guidance_weight",
        type=float,
        default=-1.0,
        help="Override OT reference guidance α (<0 = use args.json / default 0)",
    )
    parser.add_argument(
        "--ref_guidance_mode",
        type=str,
        default="",
        choices=("", "latent", "velocity", "inversion", "chain", "none"),
        help="Override guidance/inversion/chain mode (empty = args.json / default latent)",
    )
    parser.add_argument(
        "--chain_t_start",
        type=float,
        default=-1.0,
        help="Override chain t_start for img2img DI chaining (<0 = args.json / 0.5)",
    )
    parser.add_argument(
        "--use_tto",
        action="store_true",
        help="Enable Test-Time Optimization of gru_feat (TV sharpness + L2 to original)",
    )
    parser.add_argument("--tto_n_iters", type=int, default=10, help="TTO Adam iterations")
    parser.add_argument("--tto_draft_steps", type=int, default=5, help="Euler steps inside TTO draft")
    parser.add_argument("--tto_lr", type=float, default=1e-2, help="TTO Adam lr on gru_feat")
    parser.add_argument("--tto_lambda_l2", type=float, default=1.0, help="TTO L2 weight vs original gru_feat")
    parser.add_argument("--tto_tv_weight", type=float, default=1.0, help="TTO TV sharpness weight")
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        default="",
        help="Frozen Stage-1 CFM checkpoint (cascade SR or FreCaS 64)",
    )
    parser.add_argument(
        "--stage1_result_dir",
        type=str,
        default="",
        help="Run directory with args.json for Stage-1 (default: parent of stage1_checkpoint)",
    )
    parser.add_argument(
        "--use_frecas",
        action="store_true",
        help="FreCaS two-stage 64→128 sampling",
    )
    parser.add_argument(
        "--frecas_single_model",
        action="store_true",
        help="FreCaS route-1: same 64 checkpoint for Stage 1 and Stage 2 (no 128 ckpt)",
    )
    parser.add_argument(
        "--frecas_stage1_stop_index",
        type=int,
        default=-1,
        help="FreCaS Stage-1 stop index L (-1 → num_steps // 2)",
    )
    parser.add_argument("--frecas_gamma", type=float, default=2.0)
    parser.add_argument("--frecas_w_h", type=float, default=2.0)
    parser.add_argument("--frecas_w_c", type=float, default=0.6)
    parser.add_argument(
        "--use_freeu",
        action="store_true",
        help="Enable training-free FreeU on decoder skip merge at inference",
    )
    parser.add_argument("--freeu_b1", type=float, default=1.2)
    parser.add_argument("--freeu_b2", type=float, default=1.4)
    parser.add_argument("--freeu_s1", type=float, default=0.9)
    parser.add_argument("--freeu_s2", type=float, default=0.2)
    parser.add_argument("--freeu_r_thresh", type=float, default=0.25)
    return parser.parse_args()


def _parse_di_values(raw):
    values = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("fixed_di_values must contain at least one DI value")
    return values


def _strip_removed_u2_vss_keys(state_dict):
    """Drop FID weights; keep u2_vss when present (re-enabled via --use_vss_u2)."""
    drop = [k for k in list(state_dict.keys()) if str(k).startswith("fid.")]
    for k in drop:
        del state_dict[k]
    return drop


def _load_periodic_checkpoint(model, checkpoint_path, device, payload=None):
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    if payload is None:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = payload["state_dict"]
    dropped = _strip_removed_u2_vss_keys(state)
    if dropped:
        print(f"[test_cfm] stripped {len(dropped)} unused keys from checkpoint", flush=True)
    status = model.load_state_dict(state, strict=False)
    missing = list(status.missing_keys)
    unexpected = list(status.unexpected_keys)
    print(f"[test_cfm] load missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    step = int(payload.get("global_step", 0))
    epoch = int(payload.get("epoch", 0))
    print(f"[test_cfm] Loaded {checkpoint_path}", flush=True)
    print(f"[test_cfm] global_step={step} epoch={epoch}", flush=True)
    return step, epoch


def _pool_key(split):
    return f"{split}_anchor_candidates"


def _resolve_anchor_exp(model, split, override=""):
    override = str(override).strip()
    if override:
        return override
    if split == "train":
        return str(model.inference_train_anchor_exp).strip()
    if split == "val":
        return str(model.inference_val_anchor_exp).strip()
    return "CTR1"


def _collect_split_anchors(model, dataloader, split):
    pool_key = _pool_key(split)
    model.eval()
    cap = model._inference_candidate_cap()
    required_exp = _resolve_anchor_exp(model, split)

    with torch.no_grad():
        for batch in dataloader:
            real_data, hidden_state, direction, exp, timestep, DI, info = batch[:7]
            condition_mask = None
            if len(batch) >= 9:
                condition_mask = batch[8]
            elif len(batch) >= 8:
                condition_mask = batch[7]
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
                condition_mask=condition_mask,
            )
            pool = model.validation_outputs[pool_key]
            if len(pool) >= cap:
                if not required_exp or any(str(c["exp"]) == required_exp for c in pool):
                    break


def _save_gt_inference_panel(
    items,
    di_values,
    step,
    result_dir,
    split,
    anchor_exp,
    ode_steps=None,
    ref_guidance_weight=0.0,
    ref_guidance_mode="latent",
    chain_t_start=0.5,
    use_tto=False,
    tto_n_iters=10,
    tto_lambda_l2=1.0,
    extra_tag="",
):
    pair_count = len(di_values)
    show_lr = any(item.get("cascade_lr_vis") is not None for item in items)
    n_rows = 3 if show_lr else 2
    fig, axes = plt.subplots(n_rows, pair_count, figsize=(max(4, pair_count * 2.2), 2.2 * n_rows + 0.6))
    if pair_count == 1:
        axes = np.array([axes], dtype=object)
    if n_rows == 2 and axes.ndim == 1:
        axes = axes.reshape(2, 1)
    if n_rows == 3 and axes.ndim == 1:
        axes = axes.reshape(3, 1)

    for col, (item, di_value) in enumerate(zip(items, di_values)):
        anchor = item["anchor"]
        gt = anchor["real"]
        axes[0, col].imshow(gt.permute(1, 2, 0).numpy())
        axes[0, col].set_title(
            f"GT DI={di_value:.2f}\nt={anchor['timestep']}",
            fontsize=8,
        )
        axes[0, col].axis("off")

        row_inf = 1
        if show_lr:
            lr_vis = item.get("cascade_lr_vis")
            if lr_vis is None:
                axes[1, col].axis("off")
            else:
                axes[1, col].imshow(lr_vis.permute(1, 2, 0).clamp(0, 1).numpy())
                axes[1, col].set_title("Stage-1 → 256 LR", fontsize=8)
            axes[1, col].axis("off")
            row_inf = 2

        axes[row_inf, col].imshow(item["sampled_vis"].permute(1, 2, 0).numpy())
        axes[row_inf, col].set_title(
            f"Inf DI={di_value:.2f}\nt={anchor['timestep']}",
            fontsize=8,
        )
        axes[row_inf, col].axis("off")

    ode_tag = f" · ODE={ode_steps}" if ode_steps is not None else ""
    alpha = float(ref_guidance_weight)
    mode = str(ref_guidance_mode or "latent").lower()
    t0 = float(chain_t_start)
    if mode == "inversion":
        ref_tag = " · ODE-inversion"
    elif mode == "chain":
        ref_tag = f" · chain t_start={t0:.2f}"
    elif alpha > 0.0:
        ref_tag = f" · ref_g={alpha:.2f}({mode})"
    else:
        ref_tag = ""
    tto_tag = f" · TTO×{int(tto_n_iters)}" if use_tto else ""
    fig.suptitle(
        f"step {step} · {split} · {anchor_exp}{ode_tag}{ref_tag}{tto_tag} · "
        + (
            "GT (top) / Stage-1 LR (mid) / Stage-2 (bottom)"
            if show_lr
            else "GT (top) / Inference (bottom)"
        ),
        fontsize=10,
    )
    plt.tight_layout()

    save_dir = os.path.join(result_dir, "inference_samples")
    os.makedirs(save_dir, exist_ok=True)
    name = f"fixed_di_{split}_step_{step:05d}"
    if ode_steps is not None:
        name += f"_ode{int(ode_steps)}"
    if show_lr:
        name += "_cascade_s1"
    if use_tto:
        name += f"_tto{int(tto_n_iters)}"
        # Disambiguate λ_L2 sweeps (default 1.0 omitted for shorter names).
        if abs(float(tto_lambda_l2) - 1.0) > 1e-8:
            name += f"_l2{float(tto_lambda_l2):g}".replace(".", "p")
    if mode == "inversion":
        name += "_odeinv"
    elif mode == "chain":
        name += f"_chain_t{t0:.2f}".replace(".", "p")
    elif alpha > 0.0:
        name += f"_refg{alpha:.2f}".replace(".", "p")
        name += f"_{mode}"
    if extra_tag:
        name += extra_tag
    fig_path = os.path.join(save_dir, f"{name}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[test_cfm] Saved {pair_count * 2}-image panel: {fig_path}", flush=True)
    return fig_path


def _run_test_inference_panel(
    model, dataloader, split, di_values, step, result_dir, device, anchor_exp, extra_tag=""
):
    pool_key = _pool_key(split)

    model.validation_outputs = {
        "gt_DI": [],
        "pred_DI": [],
        "train_anchor_candidates": [],
        "val_anchor_candidates": [],
        "test_anchor_candidates": [],
    }
    for cache_attr in (
        "_fixed_train_inference_anchors",
        "_fixed_val_inference_anchors",
        "_fixed_train_inference_anchor",
        "_fixed_val_inference_anchor",
    ):
        if hasattr(model, cache_attr):
            delattr(model, cache_attr)

    print(f"[test_cfm] Collecting {split} inference anchors...", flush=True)
    _collect_split_anchors(model, dataloader, split)
    candidates = model.validation_outputs[pool_key]
    print(f"[test_cfm] {split} anchors={len(candidates)}", flush=True)
    if candidates:
        mask_means = []
        for c in candidates:
            m = c.get("condition_mask")
            if m is not None:
                mask_means.append(float(m.float().mean()))
        if mask_means:
            print(
                f"[test_cfm] condition_mask mean over anchors: "
                f"min={min(mask_means):.4f} max={max(mask_means):.4f} "
                f"avg={sum(mask_means)/len(mask_means):.4f} n={len(mask_means)}",
                flush=True,
            )
    if not candidates:
        raise RuntimeError(f"No {split} inference anchors collected")

    anchors = model._select_anchors_for_di_values(
        candidates,
        di_values,
        split=split,
        anchor_exp=anchor_exp,
    )
    if not anchors:
        raise RuntimeError(
            f"No anchors matched exp={anchor_exp} on split={split}. "
            f"Try another --split or adjust inference_*_anchor_exp in args.json."
        )

    model.to(device)
    model.eval()
    stage1 = getattr(model, "_cascade_stage1", None)
    if stage1 is not None:
        stage1.to(device)
        stage1.eval()
    use_tto = bool(getattr(model, "use_tto", False))
    if use_tto:
        print(
            f"[test_cfm] TTO ON: n_iters={getattr(model, 'tto_n_iters', 10)} "
            f"draft_steps={getattr(model, 'tto_draft_steps', 5)} "
            f"lr={getattr(model, 'tto_lr', 1e-2)} "
            f"lambda_l2={getattr(model, 'tto_lambda_l2', 1.0)} "
            f"tv_weight={getattr(model, 'tto_tv_weight', 1.0)}",
            flush=True,
        )
    items = model._run_fixed_di_inference_for_anchors(anchors, di_values)
    return _save_gt_inference_panel(
        items,
        di_values,
        step,
        result_dir,
        split,
        anchor_exp,
        ode_steps=int(model.cfm_ode_steps),
        ref_guidance_weight=float(getattr(model, "ref_guidance_weight", 0.0)),
        ref_guidance_mode=str(getattr(model, "ref_guidance_mode", "latent")),
        chain_t_start=float(getattr(model, "chain_t_start", 0.5)),
        use_tto=use_tto,
        tto_n_iters=int(getattr(model, "tto_n_iters", 10)),
        tto_lambda_l2=float(getattr(model, "tto_lambda_l2", 1.0)),
        extra_tag=extra_tag,
    )


def _save_split_metrics(metrics, result_dir, split, step, checkpoint_path, ode_steps=None):
    save_dir = os.path.join(result_dir, "test_metrics")
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "split": split,
        "global_step": step,
        "cfm_ode_steps": int(ode_steps) if ode_steps is not None else None,
        "checkpoint": os.path.abspath(checkpoint_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }
    if ode_steps is not None:
        json_path = os.path.join(
            save_dir, f"metrics_{split}_step_{step:05d}_ode{int(ode_steps)}.json"
        )
    else:
        json_path = os.path.join(save_dir, f"metrics_{split}_step_{step:05d}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[test_cfm] Saved {split} metrics: {json_path}", flush=True)
    return json_path


def _run_split_metrics(model, dataloader, split, step, result_dir, checkpoint_path, device, use_wandb):
    import types

    # Training no longer builds FID/PSNR/SSIM; attach them for offline test only.
    if not hasattr(model, "fid") or model.fid is None:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

        model.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        model.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        model.fid = FrechetInceptionDistance(
            feature=2048, reset_real_features=False, normalize=True
        ).to(device)

        def _val_step_with_metrics(self, batch, batch_idx):
            real_data, hidden_state, direction, exp, timestep, DI, info, valid_mask, condition_mask, *_rest = (
                self._unpack_generation_batch(batch)
            )
            c, gru_feat = self._unpack_conditions(
                info, hidden_state, direction, DI, exp=exp
            )
            target_space_data = self._encode_to_target_space(real_data)
            condition_mask = self._align_spatial_to_target(condition_mask)
            valid_mask = self._align_spatial_to_target(valid_mask)
            val_fm_loss = self.flow_matching_loss(
                target_space_data,
                c,
                valid_mask=valid_mask,
                condition_mask=condition_mask,
                gru_feat=gru_feat,
            )
            lr_cond = None
            if getattr(self, "use_cascade_sr", False):
                lr_cond = self._cascade_lr_spatial(target_space_data)
            sampled = self.sample_with_ode(
                c, gru_feat=gru_feat, lr_cond=lr_cond, damage_mask=condition_mask
            )
            sampled_image = self._decode_from_target_space(sampled)
            real_vis = torch.clamp(real_data * 0.5 + 0.5, 0.0, 1.0)
            sampled_vis = torch.clamp(sampled_image * 0.5 + 0.5, 0.0, 1.0)
            self.psnr.update(sampled_vis, real_vis)
            self.ssim.update(sampled_vis, real_vis)
            self.fid.update(sampled_vis, real=False)
            self.fid.update(real_vis, real=True)
            self.log("val/loss", val_fm_loss, prog_bar=False, on_step=False, on_epoch=True)

        def _val_epoch_end_with_metrics(self):
            self.log("val/psnr", self.psnr.compute())
            self.log("val/ssim", self.ssim.compute())
            self.log("val/fid", self.fid.compute())
            self.psnr.reset()
            self.ssim.reset()
            self.fid.reset()

        model.validation_step = types.MethodType(_val_step_with_metrics, model)
        model.on_validation_epoch_end = types.MethodType(_val_epoch_end_with_metrics, model)

    model._should_log_metrics = lambda: True
    model._should_run_inference = lambda: False
    model._fixed_train_inference_anchors = None
    model.validation_outputs = {
        "gt_DI": [],
        "pred_DI": [],
        "train_anchor_candidates": [],
        "val_anchor_candidates": [],
        "test_anchor_candidates": [],
    }

    original_log = model.log

    def _log_with_split_prefix(name, *args, **kwargs):
        if isinstance(name, str) and name.startswith("val/"):
            name = f"{split}/{name[4:]}"
        return original_log(name, *args, **kwargs)

    model.log = _log_with_split_prefix

    logger = None
    if use_wandb:
        logger = pl.loggers.WandbLogger(
            project="Opensees Based damage Patterns Generation",
            group="Damage Pattern Generation",
            save_dir=result_dir,
            name=f"test_{split}_{os.path.basename(checkpoint_path)}",
        )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
    )
    print(f"[test_cfm] Running {split} metrics (FID/PSNR/SSIM/loss)...", flush=True)
    trainer.validate(model, dataloaders=dataloader, verbose=True)

    metrics = {}
    for key, value in trainer.callback_metrics.items():
        key = str(key)
        if key.startswith(f"{split}/") or key.startswith("val/"):
            metric_name = key.split("/", 1)[1]
            metrics[f"{split}/{metric_name}"] = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)

    model.log = original_log
    return _save_split_metrics(
        metrics,
        result_dir,
        split,
        step,
        checkpoint_path,
        ode_steps=int(model.cfm_ode_steps),
    )


def main():
    cli = _parse_args()
    checkpoint_path = os.path.abspath(cli.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    result_dir = cli.result_dir.strip() or os.path.dirname(checkpoint_path)
    result_dir = os.path.abspath(result_dir)
    args = _load_args_from_result_dir(result_dir)
    args.result_dir = result_dir
    if cli.batch_size > 0:
        args.batch_size = cli.batch_size

    di_values = _parse_di_values(cli.fixed_di_values)
    args.inference_num_images = len(di_values)
    args.fixed_di_values = cli.fixed_di_values
    if int(cli.cfm_ode_steps) > 0:
        args.cfm_ode_steps = int(cli.cfm_ode_steps)
    if float(cli.ref_guidance_weight) >= 0.0:
        args.ref_guidance_weight = float(cli.ref_guidance_weight)
    if str(cli.ref_guidance_mode).strip():
        args.ref_guidance_mode = str(cli.ref_guidance_mode).strip().lower()
    if float(cli.chain_t_start) > 0.0:
        args.chain_t_start = float(cli.chain_t_start)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test_cfm] result_dir={result_dir}", flush=True)
    print(f"[test_cfm] split={cli.split} fixed_di_values={di_values}", flush=True)
    print(f"[test_cfm] condition_mode={getattr(args, 'condition_mode', 'full')}", flush=True)
    print(
        f"[test_cfm] cfm_ode_steps={args.cfm_ode_steps} solver={getattr(args, 'cfm_solver', 'heun')} "
        f"ref_guidance_weight={getattr(args, 'ref_guidance_weight', 0.0)} "
        f"ref_guidance_mode={getattr(args, 'ref_guidance_mode', 'latent')} "
        f"chain_t_start={getattr(args, 'chain_t_start', 0.5)} "
        f"use_tto={bool(cli.use_tto)}",
        flush=True,
    )

    extractor_path = args.extractor_path
    extractor_args = _load_args_from_result_dir(extractor_path)
    extractor = importlib.import_module("src.model.HystereticPrediction").__dict__[
        extractor_args.architecture
    ](**vars(extractor_args))
    extractor = extractor.to(device)
    extractor_ckpt = _resolve_required_match(
        os.path.join(extractor_path, "**/best.ckpt"), "extractor checkpoint"
    )
    extractor.load_state_dict(
        torch.load(extractor_ckpt, map_location=device, weights_only=False)["state_dict"]
    )
    extractor.eval()
    for param in extractor.parameters():
        param.requires_grad = False
    print("[test_cfm] Extractor loaded.", flush=True)

    val_transform = build_damage_image_transform_with_loss_mask(
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
    dataset_kwargs["eval_real_condition_mask"] = bool(cli.eval_real_mask)
    if cli.eval_real_mask:
        print("[test_cfm] eval_real_mask=ON (val/test condition uses on-disk object mask)", flush=True)

    num_workers = _resolve_num_workers(cli.num_workers)
    split = cli.split
    print(f"[test_cfm] Building {split} dataset (hidden-state extraction)...", flush=True)
    panel_ds = PatternLoopDataset(
        root="./Data",
        mode=split,
        transform=val_transform,
        extractor=extractor,
        extractor_args=extractor_args,
        **dataset_kwargs,
    )
    print(f"[test_cfm] {split}={len(panel_ds)}", flush=True)

    panel_dl = DataLoader(
        panel_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    val_dl = None
    metrics_json_path = None
    if not cli.skip_metrics:
        metrics_split = cli.metrics_split
        print(f"[test_cfm] Building {metrics_split} dataset for metrics...", flush=True)
        metrics_ds = PatternLoopDataset(
            root="./Data",
            mode=metrics_split,
            transform=val_transform,
            extractor=extractor,
            extractor_args=extractor_args,
            **dataset_kwargs,
        )
        metrics_dl = DataLoader(
            metrics_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        set_dataset_args(args, metrics_dl)
        val_dl = metrics_dl
        print(f"[test_cfm] {metrics_split} metrics samples={len(metrics_ds)}", flush=True)

    model = importlib.import_module("src.model.ImageGeneration").__dict__[args.architecture](
        regressor=None,
        **vars(args),
    )
    step, _epoch = _load_periodic_checkpoint(model, checkpoint_path, device)
    if bool(getattr(model, "use_undamaged_ref", False)):
        model._ensure_undamaged_ref_pool()
    model.use_tto = bool(cli.use_tto)
    model.tto_n_iters = int(cli.tto_n_iters)
    model.tto_draft_steps = int(cli.tto_draft_steps)
    model.tto_lr = float(cli.tto_lr)
    model.tto_lambda_l2 = float(cli.tto_lambda_l2)
    model.tto_tv_weight = float(cli.tto_tv_weight)
    if bool(cli.use_freeu):
        net = model.velocity_net
        net.use_freeu = True
        net.freeu_b1 = float(cli.freeu_b1)
        net.freeu_b2 = float(cli.freeu_b2)
        net.freeu_s1 = float(cli.freeu_s1)
        net.freeu_s2 = float(cli.freeu_s2)
        net.freeu_r_thresh = float(cli.freeu_r_thresh)
        print(
            f"[test_cfm] FreeU ON: b=({net.freeu_b1},{net.freeu_b2}) "
            f"s=({net.freeu_s1},{net.freeu_s2}) r_thresh={net.freeu_r_thresh}",
            flush=True,
        )
    model.use_frecas = bool(cli.use_frecas)
    model.frecas_single_model = bool(cli.frecas_single_model)
    if model.use_frecas:
        model.frecas_stage1_stop_index = int(cli.frecas_stage1_stop_index)
        model.frecas_gamma = float(cli.frecas_gamma)
        model.frecas_w_h = float(cli.frecas_w_h)
        model.frecas_w_c = float(cli.frecas_w_c)
        mode = "single-64" if model.frecas_single_model else "dual-checkpoint"
        print(
            f"[test_cfm] FreCaS ON ({mode}): L={model.frecas_stage1_stop_index} "
            f"gamma={model.frecas_gamma} w_h={model.frecas_w_h} w_c={model.frecas_w_c}",
            flush=True,
        )

    stage1_ckpt_cli = str(cli.stage1_checkpoint).strip() or str(
        getattr(args, "cascade_stage1_checkpoint", "") or ""
    ).strip()
    if model.use_frecas and model.frecas_single_model:
        if int(getattr(model, "model_spatial_size", 0)) != 64:
            print(
                f"[test_cfm] WARNING: frecas_single_model expects 64-base checkpoint, "
                f"got model_spatial_size={model.model_spatial_size}",
                flush=True,
            )
        if stage1_ckpt_cli:
            print("[test_cfm] frecas_single_model: ignoring --stage1_checkpoint", flush=True)
        stage1_ckpt_cli = ""
    elif bool(cli.use_frecas) and not stage1_ckpt_cli:
        raise ValueError(
            "Dual-checkpoint FreCaS requires --stage1_checkpoint (64). "
            "For route-1 single-model FreCaS, pass --frecas_single_model with a 64 checkpoint."
        )
    if stage1_ckpt_cli:
        stage1_ckpt = os.path.abspath(stage1_ckpt_cli)
        if not os.path.isfile(stage1_ckpt):
            raise FileNotFoundError(f"Stage-1 checkpoint not found: {stage1_ckpt}")
        stage1_dir = (
            cli.stage1_result_dir.strip()
            or str(getattr(args, "cascade_stage1_result_dir", "") or "").strip()
            or os.path.dirname(stage1_ckpt)
        )
        stage1_dir = os.path.abspath(stage1_dir)
        stage1_args = _load_args_from_result_dir(stage1_dir)
        stage1_args.result_dir = stage1_dir
        stage1_args.use_cascade_sr = False
        stage1_args.cascade_stage1_checkpoint = ""
        stage1 = importlib.import_module("src.model.ImageGeneration").__dict__[
            stage1_args.architecture
        ](regressor=None, **vars(stage1_args))
        _load_periodic_checkpoint(stage1, stage1_ckpt, device)
        stage1.to(device)
        stage1.eval()
        for param in stage1.parameters():
            param.requires_grad = False
        model._cascade_stage1 = stage1
        model._cascade_stage1_ref = [stage1]
        print(
            f"[test_cfm] Stage-1 attached: {stage1_ckpt} "
            f"(image_size={getattr(stage1_args, 'image_size', '?')})",
            flush=True,
        )
    elif bool(cli.use_frecas) and not model.frecas_single_model:
        raise ValueError(
            "Dual-checkpoint FreCaS requires --stage1_checkpoint (64). "
            "For route-1 single-model FreCaS, pass --frecas_single_model with a 64 checkpoint."
        )
    elif bool(getattr(args, "use_cascade_sr", False)):
        print(
            "[test_cfm] Cascade SR without --stage1_checkpoint: "
            "panels use teacher-forced GT bicubic LR (SR upper bound).",
            flush=True,
        )

    extra_tag = "_realmask" if cli.eval_real_mask else ""
    if bool(cli.use_freeu):
        extra_tag += "_freeu"
    if model.use_frecas:
        extra_tag += "_frecas"
        if model.frecas_single_model:
            extra_tag += "_64only"
    panel_path = _run_test_inference_panel(
        model,
        panel_dl,
        split,
        di_values,
        step,
        result_dir,
        device,
        _resolve_anchor_exp(model, split, cli.anchor_exp),
        extra_tag=extra_tag,
    )

    if not cli.skip_metrics and val_dl is not None:
        metrics_json_path = _run_split_metrics(
            model,
            val_dl,
            cli.metrics_split,
            step,
            result_dir,
            checkpoint_path,
            device,
            cli.use_wandb,
        )

    done_msg = f"[test_cfm] Done. Panel: {panel_path}"
    if metrics_json_path:
        done_msg += f" | Metrics: {metrics_json_path}"
    print(done_msg, flush=True)


if __name__ == "__main__":
    main()
