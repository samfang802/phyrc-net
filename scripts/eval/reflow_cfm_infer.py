#!/usr/bin/env python3
"""2-rectify (Reflow) a pretrained OT-CFM checkpoint, then dump 5 test samples.

Liu et al. Flow Straight and Fast:
  1) Teacher ODE: z0 ~ N(0,I) → z1 = φ_teacher(z0 | c)
  2) Fine-tune on the straight path z_t = (1-t) z0 + t z1, v = z1 - z0
  3) Few-step Euler inference
"""

from __future__ import annotations

import argparse
import copy
import importlib
import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from argument.argument_mixed import set_dataset_args
from src.dataset import PatternLoopDataset
from src.utils import build_damage_image_transform_with_loss_mask
from test_cfm_checkpoint import (
    _load_periodic_checkpoint,
    _parse_di_values,
    _resolve_anchor_exp,
    _run_test_inference_panel,
)
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match


def _parse_args():
    p = argparse.ArgumentParser(description="Reflow a CFM checkpoint and infer 5 test samples")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--result_dir", type=str, default="")
    p.add_argument("--num_pairs", type=int, default=512)
    p.add_argument("--pair_batch_size", type=int, default=8)
    p.add_argument("--pair_ode_steps", type=int, default=30)
    p.add_argument("--reflow_steps", type=int, default=600)
    p.add_argument("--reflow_batch_size", type=int, default=16)
    p.add_argument("--reflow_lr", type=float, default=2e-5)
    p.add_argument("--infer_ode_steps", type=int, default=8)
    p.add_argument("--infer_solver", type=str, default="euler", choices=("euler", "heun"))
    p.add_argument("--fixed_di_values", type=str, default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--anchor_exp", type=str, default="CTR1")
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _build_extractor(args, device):
    extractor_path = args.extractor_path
    extractor_args = _load_args_from_result_dir(extractor_path)
    extractor = getattr(importlib.import_module("src.model.HystereticPrediction"), extractor_args.architecture)(**vars(extractor_args)).to(device)
    ckpt = _resolve_required_match(os.path.join(extractor_path, "**/best.ckpt"), "extractor checkpoint")
    extractor.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["state_dict"])
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False
    return extractor, extractor_args


def _build_dataset(args, extractor, extractor_args, mode, virtual_repeat=None):
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
    kwargs = dict(vars(args))
    kwargs["use_loss_mask"] = True
    kwargs["use_damage_mask_condition"] = (
        args.architecture == "CFMADM" and bool(args.use_damage_mask_condition)
    )
    if kwargs["use_damage_mask_condition"]:
        kwargs["use_object_loss_mask"] = False
    kwargs["condition_perturb"] = False
    if virtual_repeat is not None:
        kwargs["train_virtual_repeat"] = int(virtual_repeat)
    return PatternLoopDataset(
        root="./Data",
        mode=mode,
        transform=transform,
        extractor=extractor,
        extractor_args=extractor_args,
        **kwargs,
    )


def _generate_reflow_pairs(model, dataloader, device, num_pairs, ode_steps):
    """Teacher ODE couplings (z0, z1_hat, c, gru_feat)."""
    model.eval()
    saved_steps = int(model.cfm_ode_steps)
    saved_solver = str(model.cfm_solver)
    model.cfm_ode_steps = int(ode_steps)
    model.cfm_solver = "heun"
    spatial = int(model.model_spatial_size)
    channels = int(model.model_in_channels)

    z0_all, z1_all, c_all, gru_all = [], [], [], []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="reflow-pairs")
        for batch in pbar:
            if sum(t.shape[0] for t in z0_all) >= num_pairs:
                break
            real_data, hidden_state, direction, _exp, _ts, DI, info, *_rest = (
                model._unpack_generation_batch(batch)
            )
            hidden_state = hidden_state.to(device)
            direction = direction.to(device)
            DI = DI.to(device)
            info = info.to(device)
            c, gru_feat = model._unpack_conditions(info, hidden_state, direction, DI)
            b = c.shape[0]
            z0 = torch.randn((b, channels, spatial, spatial), device=device)
            z1 = model.sample_with_ode(c, z0=z0, gru_feat=gru_feat)
            z0_all.append(z0.detach().cpu())
            z1_all.append(z1.detach().cpu())
            c_all.append(c.detach().cpu())
            gru_all.append(gru_feat.detach().cpu())
            pbar.set_postfix(n=sum(t.shape[0] for t in z0_all))

    model.cfm_ode_steps = saved_steps
    model.cfm_solver = saved_solver
    z0 = torch.cat(z0_all, dim=0)[:num_pairs]
    z1 = torch.cat(z1_all, dim=0)[:num_pairs]
    c = torch.cat(c_all, dim=0)[:num_pairs]
    gru = torch.cat(gru_all, dim=0)[:num_pairs]
    print(f"[reflow] pairs={z0.shape[0]} z1=[{float(z1.min()):.3f},{float(z1.max()):.3f}]", flush=True)
    return TensorDataset(z0, z1, c, gru)


def _reflow_finetune(model, pair_ds, device, steps, batch_size, lr):
    model.train()
    model.patch_size_probs = [1.0, 0.0, 0.0]
    loader = DataLoader(pair_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    it = iter(loader)
    pbar = tqdm(range(int(steps)), desc="reflow-ft")
    last = 0.0
    for _ in pbar:
        try:
            z0, z1, c, gru = next(it)
        except StopIteration:
            it = iter(loader)
            z0, z1, c, gru = next(it)
        z0, z1, c, gru = z0.to(device), z1.to(device), c.to(device), gru.to(device)
        opt.zero_grad(set_to_none=True)
        loss = model.flow_matching_loss(z1, c, gru_feat=gru, z0=z0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss.detach())
        pbar.set_postfix(loss=f"{last:.4f}")
    model.eval()
    print(f"[reflow] fine-tune done last_loss={last:.4f}", flush=True)


def main():
    cli = _parse_args()
    torch.manual_seed(int(cli.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cli.seed))
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    ckpt = os.path.abspath(cli.checkpoint)
    result_dir = os.path.abspath(cli.result_dir.strip() or os.path.dirname(ckpt))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    di_values = _parse_di_values(cli.fixed_di_values)

    args = _load_args_from_result_dir(result_dir)
    args.result_dir = result_dir
    args.inference_num_images = len(di_values)
    args.fixed_di_values = cli.fixed_di_values
    args.batch_size = int(cli.pair_batch_size)

    print(f"[reflow] ckpt={ckpt}", flush=True)
    print(
        f"[reflow] pairs={cli.num_pairs} ft_steps={cli.reflow_steps} "
        f"infer={cli.infer_solver}/{cli.infer_ode_steps}",
        flush=True,
    )

    extractor, extractor_args = _build_extractor(args, device)
    print("[reflow] building train set for teacher couplings...", flush=True)
    train_ds = _build_dataset(args, extractor, extractor_args, "train", virtual_repeat=1)
    train_dl = DataLoader(
        train_ds,
        batch_size=int(cli.pair_batch_size),
        shuffle=True,
        num_workers=_resolve_num_workers(cli.num_workers),
    )
    set_dataset_args(args, train_dl)
    print(f"[reflow] train unique={len(train_ds)}", flush=True)

    model = getattr(importlib.import_module("src.model.ImageGeneration"), args.architecture)(
        regressor=None,
        **vars(args),
    ).to(device)
    step, epoch = _load_periodic_checkpoint(model, ckpt, device)

    pair_ds = _generate_reflow_pairs(
        model, train_dl, device, int(cli.num_pairs), int(cli.pair_ode_steps)
    )
    _reflow_finetune(
        model,
        pair_ds,
        device,
        steps=int(cli.reflow_steps),
        batch_size=int(cli.reflow_batch_size),
        lr=float(cli.reflow_lr),
    )

    out_dir = os.path.join(result_dir, "reflow")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"checkpoint_reflow_step_{step:06d}.pt")
    torch.save(
        {"state_dict": copy.deepcopy(model.state_dict()), "global_step": step, "epoch": epoch, "reflow": True},
        save_path,
    )
    print(f"[reflow] saved {save_path}", flush=True)

    print("[reflow] building test set for 5-sample panel...", flush=True)
    test_ds = _build_dataset(args, extractor, extractor_args, cli.split, virtual_repeat=1)
    test_dl = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=_resolve_num_workers(cli.num_workers),
    )
    print(f"[reflow] {cli.split}={len(test_ds)}", flush=True)

    model.cfm_solver = str(cli.infer_solver)
    model.cfm_ode_steps = int(cli.infer_ode_steps)
    extra = f"_reflow_{cli.infer_solver}{cli.infer_ode_steps}"
    panel = _run_test_inference_panel(
        model,
        test_dl,
        cli.split,
        di_values,
        step,
        result_dir,
        device,
        _resolve_anchor_exp(model, cli.split, cli.anchor_exp),
        extra_tag=extra,
    )
    print(f"[reflow] panel={panel}", flush=True)


if __name__ == "__main__":
    main()
