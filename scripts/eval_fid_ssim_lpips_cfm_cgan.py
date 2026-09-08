#!/usr/bin/env python3
"""FID / SSIM / LPIPS on test CTR1: CFM ckpt vs saved cGAN images."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import to_tensor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from argument.argument_mixed import set_dataset_args
from src.dataset import PatternLoopDataset
from src.losses.lpips import LPIPS
from src.utils import build_damage_image_transform_with_loss_mask
from test_cfm_checkpoint import _load_periodic_checkpoint
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match


def _to01(x):
    return torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)


@torch.no_grad()
def _metrics_from_pairs(fakes, reals, device, lpips_fn):
    """fakes/reals: list of [3,H,W] float in [0,1]."""
    fid = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=True).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_sum = 0.0
    n = 0
    bs = 16
    for i in range(0, len(fakes), bs):
        f = torch.stack(fakes[i : i + bs]).to(device)
        r = torch.stack(reals[i : i + bs]).to(device)
        if f.shape[1] == 1:
            f = f.repeat(1, 3, 1, 1)
        if r.shape[1] == 1:
            r = r.repeat(1, 3, 1, 1)
        ssim.update(f, r)
        fid.update(f, real=False)
        fid.update(r, real=True)
        lp = lpips_fn(f, r).view(-1)
        lpips_sum += float(lp.sum().item())
        n += int(lp.numel())
    return {
        "fid": float(fid.compute().item()),
        "ssim": float(ssim.compute().item()),
        "lpips_vgg": lpips_sum / max(n, 1),
        "n": n,
    }


@torch.no_grad()
def eval_cfm(ckpt, result_dir, dataloader, device, lpips_fn, ode_steps):
    args = _load_args_from_result_dir(result_dir)
    args.result_dir = result_dir
    set_dataset_args(args, dataloader)
    model = importlib.import_module("src.model.ImageGeneration").__dict__[args.architecture](
        regressor=None, **vars(args)
    )
    step, _ = _load_periodic_checkpoint(model, ckpt, device)
    model = model.to(device).eval()
    if ode_steps:
        model.cfm_ode_steps = int(ode_steps)
    fakes, reals = [], []
    for batch in dataloader:
        real_data, hidden_state, direction, exp, timestep, DI, info, valid_mask, condition_mask, *_ = (
            model._unpack_generation_batch(batch)
        )
        real_data = real_data.to(device)
        hidden_state = hidden_state.to(device)
        direction = direction.to(device)
        DI = DI.to(device)
        info = info.to(device)
        if condition_mask is not None:
            condition_mask = condition_mask.to(device)
        c, gru_feat = model._unpack_conditions(info, hidden_state, direction, DI, exp=exp)
        sampled = model.sample_with_ode(c, gru_feat=gru_feat, damage_mask=condition_mask)
        sampled_image = model._decode_from_target_space(sampled)
        fake = _to01(sampled_image).cpu()
        real = _to01(real_data).cpu()
        for i in range(fake.shape[0]):
            fakes.append(fake[i])
            reals.append(real[i])
        print(f"[eval] CFM {len(fakes)}/{len(dataloader.dataset)}", flush=True)
    m = _metrics_from_pairs(fakes, reals, device, lpips_fn)
    m["global_step"] = int(step)
    m["ode_steps"] = int(model.cfm_ode_steps)
    m["solver"] = str(model.cfm_solver)
    return m


def eval_cgan_folder(folder, gt_dir, image_size, device, lpips_fn):
    fakes, reals = [], []
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png")))
    for name in files:
        gt_name = name.replace("_test.jpg", ".jpg").replace("_test.png", ".png")
        gp = os.path.join(gt_dir, gt_name)
        fp = os.path.join(folder, name)
        if not os.path.isfile(gp):
            raise FileNotFoundError(f"GT missing for {name} -> {gt_name}")
        fake = Image.open(fp).convert("RGB").resize((image_size, image_size), Image.BICUBIC)
        real = Image.open(gp).convert("RGB").resize((image_size, image_size), Image.BICUBIC)
        fakes.append(to_tensor(fake))
        reals.append(to_tensor(real))
    print(f"[eval] cGAN paired {len(fakes)} images", flush=True)
    return _metrics_from_pairs(fakes, reals, device, lpips_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--ode_steps", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    args_cli = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_cli.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfm_ckpt = os.path.join(
        ROOT, "results/DamagePatternGeneration/CFMADM/selfcondnoDI/checkpoint_step_014000.pt"
    )
    cfm_dir = os.path.dirname(cfm_ckpt)
    cgan_dir = os.path.join(ROOT, "cGANckpt/CTR1")
    gt_dir = os.path.join(ROOT, "Data/DamagePattern/images")

    args = _load_args_from_result_dir(cfm_dir)
    args.result_dir = cfm_dir
    os.makedirs(os.path.join(cfm_dir, "data_check"), exist_ok=True)

    extractor_path = args.extractor_path
    extractor_args = _load_args_from_result_dir(extractor_path)
    extractor = importlib.import_module("src.model.HystereticPrediction").__dict__[
        extractor_args.architecture
    ](**vars(extractor_args)).to(device)
    extractor.load_state_dict(
        torch.load(
            _resolve_required_match(os.path.join(extractor_path, "**/best.ckpt"), "extractor"),
            map_location=device,
            weights_only=False,
        )["state_dict"]
    )
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False

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
    dataset_kwargs["use_damage_mask_condition"] = bool(args.use_damage_mask_condition)
    dataset_kwargs["use_object_loss_mask"] = False
    print("[eval] Building test dataset...", flush=True)
    ds = PatternLoopDataset(
        root="./Data",
        mode="test",
        transform=transform,
        extractor=extractor,
        extractor_args=extractor_args,
        **dataset_kwargs,
    )
    dl = DataLoader(ds, batch_size=args_cli.batch_size, shuffle=False, num_workers=_resolve_num_workers(1))

    lpips_fn = LPIPS().to(device).eval()
    print("[eval] CFM ODE sampling + metrics...", flush=True)
    cfm_m = eval_cfm(cfm_ckpt, cfm_dir, dl, device, lpips_fn, args_cli.ode_steps)
    print("[eval] cGAN folder metrics...", flush=True)
    cgan_m = eval_cgan_folder(cgan_dir, gt_dir, int(args.image_size), device, lpips_fn)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split": "test",
        "notes": (
            "Test split is CTR1 (n=124). CFM: ODE vs dataset GT (CLAHE/gray as training). "
            "cGAN: saved jpg vs original GT jpg, both resized 128 RGB."
        ),
        "selfcondnoDI_step014000": {
            "checkpoint": os.path.abspath(cfm_ckpt),
            **cfm_m,
        },
        "cGAN_CTR1": {
            "images": os.path.abspath(cgan_dir),
            **cgan_m,
        },
    }
    out_dir = os.path.join(cfm_dir, "test_metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fid_ssim_lpips_vs_cgan_CTR1.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2), flush=True)
    print(f"[eval] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
