#!/usr/bin/env python3
"""Export CFM OT-path inpainted images for all (or selected) splits into a Data folder.

Writes a DamagePattern-compatible tree:
  <out_dir>/images/*.png
  <out_dir>/mask        -> symlink to original DamagePattern/mask
  <out_dir>/label.csv
  <out_dir>/id_list_*.json
  <out_dir>/meta.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.dataset import PatternLoopDataset
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match


def _parse_args():
    p = argparse.ArgumentParser(description="Export full CFM inpainted DamagePattern dataset")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "results/DamagePatternGeneration/CFMADM/selfcondnoDI/"
            "Opensees Based damage Patterns Generation/vq9dirur/checkpoints/last.ckpt"
        ),
    )
    p.add_argument(
        "--result_dir",
        type=str,
        default="results/DamagePatternGeneration/CFMADM/selfcondnoDI",
        help="Run dir with args.json (not the nested wandb checkpoint folder)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="Data/DamagePattern_inpaint_selfcondnoDI",
        help="Output DamagePattern-compatible folder",
    )
    p.add_argument("--splits", type=str, default="train,val,test")
    p.add_argument("--ode_steps", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="")
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip filenames that already exist under out_dir/images",
    )
    p.add_argument(
        "--stage1_checkpoint",
        type=str,
        default="",
        help="128 CFM ckpt for two-stage export (default: --checkpoint)",
    )
    p.add_argument(
        "--stage1_result_dir",
        type=str,
        default="",
        help="args.json dir for Stage-1 128 CFM (default: --result_dir)",
    )
    p.add_argument(
        "--cascade_checkpoint",
        type=str,
        default="",
        help="If set, run Stage-1 128 OT-inpaint then Stage-2 256 cascade SR",
    )
    p.add_argument(
        "--cascade_result_dir",
        type=str,
        default="",
        help="args.json dir for Stage-2 cascade (default: parent of cascade_checkpoint)",
    )
    return p.parse_args()


def _strip_u2_vss(state_dict):
    drop = [
        k
        for k in list(state_dict.keys())
        if "u2_vss" in str(k) or str(k).startswith("fid.")
    ]
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
        print(f"[export_inpaint] stripped {len(dropped)} u2_vss keys", flush=True)

    status = model.load_state_dict(state, strict=False)
    missing = list(status.missing_keys)
    unexpected = list(status.unexpected_keys)
    print(f"[export_inpaint] load missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    step = int(payload.get("global_step", 0))
    epoch = int(payload.get("epoch", 0))
    print(f"[export_inpaint] Loaded {checkpoint_path} step={step} epoch={epoch}", flush=True)
    return step, epoch


def _load_damage_pattern_mask(dataset, filename, image_size, device, dtype):
    mask_path = dataset._resolve_object_mask_path(filename)
    if mask_path is None:
        raise FileNotFoundError(f"DamagePattern mask not found for {filename}")
    mask_np = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
    if mask_tensor.shape[-2:] != (image_size, image_size):
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode="nearest",
        ).squeeze(0)
    valid_mask = 1.0 - (mask_tensor / 255.0)
    return valid_mask.clamp_(0.0, 1.0).to(device=device, dtype=dtype).unsqueeze(0)


def _tensor_to_uint8_rgb(x):
    """x: (C,H,W) in [-1,1] → uint8 HxWx3."""
    vis = torch.clamp(x.detach().float().cpu() * 0.5 + 0.5, 0.0, 1.0)
    if vis.shape[0] == 1:
        vis = vis.repeat(3, 1, 1)
    arr = (vis.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return arr


def _prepare_out_dir(out_dir, src_pattern_dir, data_path):
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    # symlink mask (relative) so ResNet reads same SAM masks
    mask_link = os.path.join(out_dir, "mask")
    src_mask = os.path.join(src_pattern_dir, "mask")
    if not os.path.lexists(mask_link):
        rel = os.path.relpath(src_mask, start=out_dir)
        os.symlink(rel, mask_link)
        print(f"[export_inpaint] mask symlink → {rel}", flush=True)

    for name in (data_path, "label.csv"):
        src = os.path.join(src_pattern_dir, name)
        dst = os.path.join(out_dir, name)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)
            print(f"[export_inpaint] copied {name}", flush=True)


def _export_split(model, ds, split, args, cli, device, out_images, image_size):
    # Only iterate unique samples once (no virtual repeat).
    n = int(ds.base_size)
    idxs = list(range(n))
    if cli.skip_existing:
        keep = []
        for i in idxs:
            fn = str(ds.label.iloc[i, 0])
            if not os.path.isfile(os.path.join(out_images, fn)):
                keep.append(i)
        print(
            f"[export_inpaint] {split}: {len(keep)}/{n} remaining (skip_existing)",
            flush=True,
        )
        idxs = keep
    else:
        print(f"[export_inpaint] {split}: exporting {n} samples", flush=True)

    if not idxs:
        return 0

    dl = DataLoader(
        torch.utils.data.Subset(ds, idxs),
        batch_size=int(cli.batch_size),
        shuffle=False,
        num_workers=_resolve_num_workers(cli.num_workers),
    )

    written = 0
    cursor = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"inpaint-{split}", leave=True):
            real_data, hidden_state, direction, exp, timestep, DI, info, *_rest = (
                model._unpack_generation_batch(batch)
            )
            bsz = int(real_data.shape[0])
            batch_indices = idxs[cursor : cursor + bsz]
            cursor += bsz

            real_data = real_data.to(device)
            hidden_state = hidden_state.to(device)
            direction = direction.to(device)
            DI = DI.to(device)
            info = info.to(device)

            c, gru_feat = model._unpack_conditions(info, hidden_state, direction, DI)
            z1_gt = model._encode_to_target_space(real_data)

            masks = []
            filenames = []
            for local_i, real_idx in enumerate(batch_indices):
                filename = str(ds.label.iloc[int(real_idx), 0])
                filenames.append(filename)
                m = _load_damage_pattern_mask(
                    ds, filename, image_size=image_size, device=device, dtype=z1_gt.dtype
                )
                if m.shape[-2:] != z1_gt.shape[-2:]:
                    m = F.interpolate(m, size=z1_gt.shape[-2:], mode="nearest")
                masks.append(m)
            inpaint_mask = torch.cat(masks, dim=0)

            z_out = model.sample_with_ode_inpaint(
                c=c,
                z1_gt=z1_gt,
                inpaint_mask=inpaint_mask,
                z0=None,
                num_steps=int(args.cfm_ode_steps),
                gru_feat=gru_feat,
            )
            decoded = model._decode_from_target_space(z_out)

            for i, filename in enumerate(filenames):
                arr = _tensor_to_uint8_rgb(decoded[i])
                Image.fromarray(arr).save(os.path.join(out_images, filename))
                written += 1

    return written


def _dilate_inpaint_hole(mask, px=5):
    """mask: 1=KEEP, 0=INPAINT. Expand hole by ``px`` pixels."""
    k = int(px) * 2 + 1
    pad = int(px)
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=pad)


def _build_cfm_model(result_dir, checkpoint_path, device, ode_steps):
    args = _load_args_from_result_dir(result_dir)
    args.result_dir = result_dir
    args.cfm_ode_steps = int(ode_steps)
    args.train_virtual_repeat = 1
    args.condition_perturb = False
    args.use_wandb = False
    args.cascade_stage1_checkpoint = ""
    args.cascade_stage1_result_dir = ""
    model = importlib.import_module("src.model.ImageGeneration").__dict__[args.architecture](
        regressor=None,
        **vars(args),
    )
    step, epoch = _load_cfm_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, args, step, epoch


def _export_split_cascade(stage1, stage2, ds, split, cli, device, out_images):
    n = int(ds.base_size)
    idxs = list(range(n))
    if cli.skip_existing:
        keep = []
        for i in idxs:
            fn = str(ds.label.iloc[i, 0])
            if not os.path.isfile(os.path.join(out_images, fn)):
                keep.append(i)
        print(
            f"[export_inpaint] {split}: {len(keep)}/{n} remaining (skip_existing)",
            flush=True,
        )
        idxs = keep
    else:
        print(f"[export_inpaint] {split}: cascade export {n} samples", flush=True)

    if not idxs:
        return 0

    dl = DataLoader(
        torch.utils.data.Subset(ds, idxs),
        batch_size=int(cli.batch_size),
        shuffle=False,
        num_workers=_resolve_num_workers(cli.num_workers),
    )
    hr = int(stage2.model_spatial_size)
    lr = int(stage1.model_spatial_size)
    written = 0
    cursor = 0
    stage1.eval()
    stage2.eval()
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"inpaint-cascade-{split}", leave=True):
            real_data, hidden_state, direction, exp, timestep, DI, info, *_rest = (
                stage2._unpack_generation_batch(batch)
            )
            bsz = int(real_data.shape[0])
            batch_indices = idxs[cursor : cursor + bsz]
            cursor += bsz

            real_data = real_data.to(device)
            hidden_state = hidden_state.to(device)
            direction = direction.to(device)
            DI = DI.to(device)
            info = info.to(device)

            c1, gru1 = stage1._unpack_conditions(info, hidden_state, direction, DI, exp=exp)
            c2, gru2 = stage2._unpack_conditions(info, hidden_state, direction, DI, exp=exp)

            z_gt_hr = stage2._encode_to_target_space(real_data)
            if tuple(z_gt_hr.shape[-2:]) != (hr, hr):
                z_gt_hr = F.interpolate(z_gt_hr, size=(hr, hr), mode="bicubic", align_corners=False)
            z_gt_lr = F.interpolate(z_gt_hr, size=(lr, lr), mode="bicubic", align_corners=False)

            masks_hr = []
            filenames = []
            for local_i, real_idx in enumerate(batch_indices):
                filename = str(ds.label.iloc[int(real_idx), 0])
                filenames.append(filename)
                m = _load_damage_pattern_mask(
                    ds, filename, image_size=hr, device=device, dtype=z_gt_hr.dtype
                )
                if m.shape[-2:] != (hr, hr):
                    m = F.interpolate(m, size=(hr, hr), mode="nearest")
                masks_hr.append(m)
            mask_hr = torch.cat(masks_hr, dim=0)
            mask_lr = F.interpolate(mask_hr, size=(lr, lr), mode="nearest")

            z_lr = stage1.sample_with_ode_inpaint(
                c=c1,
                z1_gt=z_gt_lr,
                inpaint_mask=mask_lr,
                z0=None,
                num_steps=int(stage1.cfm_ode_steps),
                gru_feat=gru1,
            )
            lr_cond = F.interpolate(z_lr, size=(hr, hr), mode="bicubic", align_corners=False)
            z_sr = stage2.sample_with_ode(
                c2,
                gru_feat=gru2,
                lr_cond=lr_cond,
                damage_mask=None,
                num_steps=int(stage2.cfm_ode_steps),
            )
            keep_lr = _dilate_inpaint_hole(mask_lr, px=5)
            keep_hr = F.interpolate(keep_lr, size=(hr, hr), mode="nearest")
            z_out = z_gt_hr * keep_hr + z_sr * (1.0 - keep_hr)
            decoded = stage2._decode_from_target_space(z_out)

            for i, filename in enumerate(filenames):
                arr = _tensor_to_uint8_rgb(decoded[i])
                Image.fromarray(arr).save(os.path.join(out_images, filename))
                written += 1

    return written


def _abspath_from_root(path):
    if not path:
        return ""
    abs_path = os.path.abspath(os.path.join(_ROOT, path))
    if os.path.exists(abs_path):
        return abs_path
    return os.path.abspath(path)


def main():
    cli = _parse_args()
    torch.manual_seed(int(cli.seed))
    np.random.seed(int(cli.seed))

    cascade_ckpt = _abspath_from_root(cli.cascade_checkpoint)
    use_cascade = bool(cascade_ckpt)
    if use_cascade and not os.path.isfile(cascade_ckpt):
        raise FileNotFoundError(cli.cascade_checkpoint)

    stage1_ckpt = _abspath_from_root(cli.stage1_checkpoint) or _abspath_from_root(cli.checkpoint)
    if not os.path.isfile(stage1_ckpt):
        raise FileNotFoundError(cli.stage1_checkpoint or cli.checkpoint)

    stage1_dir = _abspath_from_root(cli.stage1_result_dir) or _abspath_from_root(cli.result_dir)
    cascade_dir = _abspath_from_root(cli.cascade_result_dir)
    if use_cascade and not cascade_dir:
        # wandb last.ckpt lives under ".../Generation/<id>/checkpoints/"
        cascade_dir = os.path.abspath(os.path.join(os.path.dirname(cascade_ckpt), "..", "..", ".."))
        if not os.path.isfile(os.path.join(cascade_dir, "args.json")):
            cascade_dir = os.path.dirname(os.path.dirname(cascade_ckpt))
    if use_cascade and not os.path.isfile(os.path.join(cascade_dir, "args.json")):
        raise FileNotFoundError(f"cascade args.json not found under {cascade_dir}")

    out_dir = os.path.abspath(os.path.join(_ROOT, cli.out_dir))
    src_pattern = os.path.join(_ROOT, "Data", "DamagePattern")
    device = torch.device(
        cli.device if cli.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    splits = [s.strip() for s in cli.splits.split(",") if s.strip()]

    ds_args_dir = cascade_dir if use_cascade else stage1_dir
    args = _load_args_from_result_dir(ds_args_dir)
    args.result_dir = ds_args_dir
    args.cfm_ode_steps = int(cli.ode_steps)
    args.train_virtual_repeat = 1
    args.condition_perturb = False
    args.use_wandb = False
    data_path = str(getattr(args, "data_path", "id_list_val_R307.json"))

    print(f"[export_inpaint] stage1_ckpt={stage1_ckpt}", flush=True)
    print(f"[export_inpaint] stage1_dir={stage1_dir}", flush=True)
    if use_cascade:
        print(f"[export_inpaint] cascade_ckpt={cascade_ckpt}", flush=True)
        print(f"[export_inpaint] cascade_dir={cascade_dir}", flush=True)
    print(f"[export_inpaint] out={out_dir} splits={splits} ode={args.cfm_ode_steps}", flush=True)
    print(f"[export_inpaint] device={device} cascade={use_cascade}", flush=True)

    _prepare_out_dir(out_dir, src_pattern, data_path)
    out_images = os.path.join(out_dir, "images")

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
    print("[export_inpaint] Extractor loaded.", flush=True)

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

    if use_cascade:
        stage1, stage1_args, step1, epoch1 = _build_cfm_model(
            stage1_dir, stage1_ckpt, device, cli.ode_steps
        )
        stage2, stage2_args, step2, epoch2 = _build_cfm_model(
            cascade_dir, cascade_ckpt, device, cli.ode_steps
        )
        if not bool(getattr(stage2, "use_cascade_sr", False)):
            raise ValueError("cascade checkpoint must have use_cascade_sr=True")
        if int(stage1.model_spatial_size) >= int(stage2.model_spatial_size):
            raise ValueError(
                f"Stage-1 size {stage1.model_spatial_size} must be < Stage-2 {stage2.model_spatial_size}"
            )
        model = stage2
        step, epoch = step2, epoch2
    else:
        model, args, step, epoch = _build_cfm_model(stage1_dir, stage1_ckpt, device, cli.ode_steps)
        stage1 = stage2 = None
        step1 = epoch1 = None
        stage1_args = stage2_args = args

    totals = {}
    for split in splits:
        dataset_kwargs = dict(vars(args))
        dataset_kwargs["use_loss_mask"] = True
        dataset_kwargs["use_damage_mask_condition"] = (
            args.architecture == "CFMADM" and bool(args.use_damage_mask_condition)
        )
        if dataset_kwargs["use_damage_mask_condition"]:
            dataset_kwargs["use_object_loss_mask"] = False
        dataset_kwargs["train_virtual_repeat"] = 1
        dataset_kwargs["condition_perturb"] = False

        print(f"[export_inpaint] Building {split} dataset...", flush=True)
        ds = PatternLoopDataset(
            root=os.path.join(_ROOT, "Data"),
            mode=split,
            transform=val_transform,
            extractor=extractor,
            extractor_args=extractor_args,
            **dataset_kwargs,
        )
        if use_cascade:
            written = _export_split_cascade(
                stage1, stage2, ds, split, cli, device, out_images
            )
        else:
            written = _export_split(
                model,
                ds,
                split,
                args,
                cli,
                device,
                out_images,
                image_size=int(args.image_size),
            )
        totals[split] = {"base_size": int(ds.base_size), "written": int(written)}

    meta = {
        "checkpoint": stage1_ckpt,
        "result_dir": stage1_dir,
        "cascade_checkpoint": cascade_ckpt if use_cascade else "",
        "cascade_result_dir": cascade_dir if use_cascade else "",
        "stage1_global_step": int(step1) if step1 is not None else int(step),
        "stage1_epoch": int(epoch1) if epoch1 is not None else int(epoch),
        "ode_steps": int(cli.ode_steps),
        "solver": str(getattr(args, "cfm_solver", "heun")),
        "global_step": int(step),
        "epoch": int(epoch),
        "seed": int(cli.seed),
        "splits": totals,
        "data_path": data_path,
        "image_size": int(args.image_size),
        "source_images": os.path.join(src_pattern, "images"),
        "note": (
            "Two-stage: Stage-1 128 OT-path inpaint (UNet cond=all-ones, SAM hole dilated 5px), "
            "Stage-2 256 cascade SR on the inpainted LR, then KEEP region composited from 256 GT."
            if use_cascade
            else "Inpainted with OT-path CFM (UNet cond=all-ones, replace=SAM mask dilated 5px). "
            "Masks are symlinked from original DamagePattern/mask."
        ),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    n_img = len([n for n in os.listdir(out_images) if n.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"[export_inpaint] Done. images={n_img} meta → {out_dir}/meta.json", flush=True)
    print(f"[export_inpaint] totals={totals}", flush=True)


if __name__ == "__main__":
    main()
