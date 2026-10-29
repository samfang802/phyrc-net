#!/usr/bin/env python3
"""Can two velocity estimates be fused (Kalman / inverse-variance)? Error statistics of x̂1 along the ODE time.

For every frame of a split, x1 = the real frame, z_τ = (1-τ) z0 + τ x1, and three estimates of x1 from
x̂1 = z_τ + (1-τ) v (the same calls the sampler makes, sample_with_ode / sample_refiner_sequence):
  base   the separate single-frame network the refiner walk uses for frame 0 (refiner_base_checkpoint)
  off    the refiner with its control branch off (trunk alone: the single-frame task it was trained on)
  on     the refiner with S_{t-1} as the control hint
The hint is either the real previous frame (`real`, oracle) or the model's own previous frame from a
samples npz of the walk (`sample`, what sampling actually sees; sample k=0). First frames have no hint.

Errors e = x̂1 - x1 on the gray channel, split into a low band (block mean, refiner_lowfreq_block) and
the high-band remainder. Per τ and band: R_a = E e_a², ρ_ab = corr(e_a, e_b), and the MSE-optimal
fusion weight of a pair. For (off, on) it is written as the control-guidance weight w of
v = v_off + w (v_on - v_off) (the existing `control_guidance`): w* = (R_off - C) / (R_off + R_on - 2C),
C = E e_off e_on. w* = 1 means "use the refiner as is", w* > 1 extrapolates away from the trunk.
The errors include the irreducible part x1 - E[x1 | z_τ], shared by all estimates; that is the
quantity a fusion weight has to work with, so it is kept.

  python scripts/eval/velocity_fusion_stats.py --cfm_result_dir <refiner run> --cfm_checkpoint <pt> \
      --samples val=<npz> test=<npz> --out results/velocity_fusion/v11appr
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from argument.argument_mixed import set_dataset_args
from src.dataset import PatternLoopDataset
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_required_match

_spec = importlib.util.spec_from_file_location(
    "_eval_r2", os.path.join(_ROOT, "scripts", "eval", "eval_cfm_generated_r2_siamese.py")
)
_eval_r2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_r2)

PAIRS = (("off", "on"), ("base", "on"), ("base", "off"))


def _parse_args():
    p = argparse.ArgumentParser(description="x̂1 error statistics for velocity fusion")
    p.add_argument("--cfm_result_dir", required=True)
    p.add_argument("--cfm_checkpoint", required=True)
    p.add_argument("--samples", nargs="*", default=[], metavar="SPLIT=NPZ", help="Walk samples for the `sample` hint")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--taus", type=float, nargs="+", default=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    p.add_argument("--draws", type=int, default=2, help="Noise draws per frame and τ")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--app_code", default="mean", choices=("null", "mean"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def _build(cli, split, device):
    result_dir = os.path.abspath(cli.cfm_result_dir)
    args = _load_args_from_result_dir(result_dir)
    args.result_dir = result_dir
    os.makedirs(os.path.join(result_dir, "data_check"), exist_ok=True)
    args.batch_size = int(cli.batch_size)
    args.train_virtual_repeat = 1
    extractor_args = _load_args_from_result_dir(args.extractor_path)
    extractor = getattr(importlib.import_module("src.model.HystereticPrediction"), extractor_args.architecture)(
        **vars(extractor_args)
    ).to(device)
    ck = _resolve_required_match(os.path.join(args.extractor_path, "**/best.ckpt"), "extractor")
    extractor.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["state_dict"])
    extractor.eval()
    transform = build_damage_image_transform_with_loss_mask(
        image_size=args.image_size, use_augmentation=False, output_scale=True,
        normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5),
        use_grayscale_input=args.use_grayscale_input, use_clahe=args.use_clahe,
        clahe_clip_limit=args.clahe_clip_limit, clahe_tile_grid_size=args.clahe_tile_grid_size,
    )
    kw = dict(vars(args))
    kw["use_loss_mask"] = True
    kw["use_damage_mask_condition"] = args.architecture == "CFMADM" and bool(args.use_damage_mask_condition)
    if kw["use_damage_mask_condition"]:
        kw["use_object_loss_mask"] = False
    if int(getattr(args, "app_code_dim", 0)) > 0:
        kw["app_unseen_code"] = -1 if cli.app_code == "null" else -2
    ds = PatternLoopDataset(root="./Data", mode=split, transform=transform, extractor=extractor,
                            extractor_args=extractor_args, **kw)
    dl = DataLoader(ds, batch_size=cli.batch_size, shuffle=False, num_workers=0)
    set_dataset_args(args, dl)
    cfm = getattr(importlib.import_module("src.model.ImageGeneration"), args.architecture)(regressor=None, **vars(args)).to(device)
    _eval_r2._load_cfm_checkpoint(cfm, os.path.abspath(cli.cfm_checkpoint), device)
    cfm.eval()
    cfm._attach_refiner_base()
    base = cfm._refiner_base_ref[0].to(device).eval()
    return cfm, base, ds, dl


def _coords(model, b, device):
    if not model.use_patch_diffusion:
        return None
    gx, gy = model._build_full_coordinate_maps(model.model_spatial_size, model.model_spatial_size, device=device, dtype=torch.float32)
    return torch.stack([gx, gy], 0).unsqueeze(0).expand(b, -1, -1, -1).contiguous()


def _ones_mask(model, z):
    return torch.ones((z.shape[0], 1, *z.shape[-2:]), device=z.device, dtype=z.dtype) if model.use_damage_mask_condition else None


def _x1_hat(model, z, t, c, g, hint=None, dg=None):
    tt = torch.full((z.shape[0],), float(t), device=z.device)
    if hint is not None:
        hint = model._ensure_cascade_lr_cond(hint, spatial_hw=z.shape[-2:], batch=z.shape[0], device=z.device, dtype=z.dtype)
    v = model._predict_velocity(z, tt, c, coords=_coords(model, z.shape[0], z.device), damage_mask=_ones_mask(model, z),
                                gru_feat=g, lr_cond=hint, gru_delta=dg)
    return z + (1.0 - float(t)) * v


def _bands(e, block):
    """(B,C,H,W) error → gray (B,H,W) low band (block means, upsampled) and high band."""
    g = e.mean(1, keepdim=True)
    low = F.interpolate(F.avg_pool2d(g, block), scale_factor=block, mode="nearest")
    return {"low": low[:, 0], "high": (g - low)[:, 0], "all": g[:, 0]}


@torch.no_grad()
def run_split(cli, split, device):
    cfm, base, ds, dl = _build(cli, split, device)
    block = int(getattr(cfm, "refiner_lowfreq_block", 16) or 16)
    # Gather every frame: condition, GRU / damage feature, real image (target space).
    C, Gf, X = [], [], []
    for batch in dl:
        real, hs, direction, _e, _t, di, info, *_ = cfm._unpack_generation_batch(batch)
        c, g = cfm._unpack_conditions(info.to(device), hs.to(device), direction.to(device), di.to(device))
        C.append(c)
        Gf.append(g)
        X.append(cfm._encode_to_target_space(real.to(device)))
    C, X = torch.cat(C), torch.cat(X)
    Gf = None if Gf[0] is None else torch.cat(Gf)
    lab = ds.label.reset_index(drop=True)
    fn = lab["filename"].astype(str).tolist()
    # Previous frame of each row along its (column, direction) walk.
    prev = np.full(len(lab), -1)
    for _, rows in lab.groupby(["exp", "direction"], sort=True):
        order = rows.sort_values("timestep", kind="stable").index.to_numpy()
        prev[order[1:]] = order[:-1]
    hints = {"real": X}
    npz = dict(s.split("=", 1) for s in cli.samples).get(split)
    if npz:
        z = np.load(npz)
        row = {str(f): i for i, f in enumerate(z["filename"])}
        u8 = torch.from_numpy(z["fake"][[row[f] for f in fn], 0]).float().to(device)
        hints["sample"] = (u8 / 127.5 - 1.0).unsqueeze(1).expand(-1, X.shape[1], -1, -1).contiguous()
    base_in = int(base.velocity_net.scalar_mlp[0].in_features)
    has = np.flatnonzero(prev >= 0)
    print(f"[fuse] {split}: {len(lab)} frames, {len(has)} with a previous frame, hints {list(hints)}", flush=True)

    keys = ["base", "off"] + [f"on_{h}" for h in hints]
    acc = {}  # (τ, band, a, b) -> [sum e_a e_b, n]
    gen = torch.Generator(device=device).manual_seed(int(cli.seed))
    for t in cli.taus:
        for d in range(int(cli.draws)):
            for s in range(0, len(has), int(cli.batch_size)):
                idx = torch.from_numpy(has[s : s + int(cli.batch_size)]).to(device)
                x1, c = X[idx], C[idx]
                g = None if Gf is None else Gf[idx]
                z0 = torch.randn(x1.shape, device=device, generator=gen)
                zt = (1.0 - t) * z0 + t * x1
                dg = None
                if getattr(cfm, "use_delta_g", False) and Gf is not None:
                    dg = g - Gf[torch.from_numpy(prev[has[s : s + int(cli.batch_size)]]).to(device)]
                c_base = torch.cat([torch.zeros_like(c[:, :1]), c], 1) if c.shape[1] + 1 == base_in else c
                est = {"base": _x1_hat(base, zt, t, c_base, g), "off": _x1_hat(cfm, zt, t, c, g, None, dg)}
                pidx = torch.from_numpy(prev[has[s : s + int(cli.batch_size)]]).to(device)
                for h, H in hints.items():
                    est[f"on_{h}"] = _x1_hat(cfm, zt, t, c, g, H[pidx], dg)
                err = {k: _bands(v - x1, block) for k, v in est.items()}
                for band in ("low", "high", "all"):
                    for i, a in enumerate(keys):
                        for b in keys[i:]:
                            key = (t, band, a, b)
                            prod = float((err[a][band] * err[b][band]).sum())
                            n = err[a][band].numel()
                            acc.setdefault(key, [0.0, 0])
                            acc[key][0] += prod
                            acc[key][1] += n
        print(f"[fuse] {split} τ={t:.2f} done", flush=True)

    def m(t, band, a, b):
        k = (t, band, a, b) if (t, band, a, b) in acc else (t, band, b, a)
        return acc[k][0] / acc[k][1]

    rows = []
    for t in cli.taus:
        for band in ("low", "high", "all"):
            r = {"tau": t, "band": band}
            for k in keys:
                r[f"R_{k}"] = m(t, band, k, k)
            for h in hints:
                on = f"on_{h}"
                for a in ("off", "base"):
                    Ra, Ro, Cab = m(t, band, a, a), m(t, band, on, on), m(t, band, a, on)
                    r[f"rho_{a}_{on}"] = Cab / np.sqrt(Ra * Ro)
                    w = (Ra - Cab) / max(Ra + Ro - 2 * Cab, 1e-12)  # weight on `on`
                    r[f"w_{a}_{on}"] = w
                    fused = (1 - w) ** 2 * Ra + w ** 2 * Ro + 2 * w * (1 - w) * Cab
                    r[f"gain_{a}_{on}"] = 1.0 - fused / min(Ra, Ro)  # MSE reduction vs the better one
            Rb, Ro_, Cbo = m(t, band, "base", "base"), m(t, band, "off", "off"), m(t, band, "base", "off")
            r["rho_base_off"] = Cbo / np.sqrt(Rb * Ro_)
            rows.append(r)
    return rows


def main():
    cli = _parse_args()
    os.makedirs(cli.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"args": vars(cli), "splits": {}}
    for split in cli.splits:
        rows = run_split(cli, split, device)
        out["splits"][split] = rows
        for r in rows:
            if r["band"] != "all":
                print(split, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        json.dump(out, open(os.path.join(cli.out, "fusion_stats.json"), "w"), indent=1)
    print("[fuse] wrote", os.path.join(cli.out, "fusion_stats.json"))


if __name__ == "__main__":
    main()
