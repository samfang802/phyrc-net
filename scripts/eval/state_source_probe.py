#!/usr/bin/env python3
"""Refiner state track: does the S_{t-1} a source produces carry cracks that line up with frame t?

For every training pair (t-1, t) of one face, S_{t-1} is built several ways and its crack map
(damage_texture.crack_map, 2 px tolerance, as crack_keep in src/metrics/temporal.py) is scored
against GT_t and GT_{t-1}; DINOv3 KID says whether S looks like the base's own samples:
  base         the refiner's offline state cache (what training reads now)
  gt           GT_{t-1} (teacher forcing, refiner_state_source=gt)
  sdedit_<τ>   GT_{t-1} at data fraction τ, z = (1-τ) ε + τ GT_{t-1}, then the base ODE τ → 1
  dds_blk<f>   DDS from noise: each Euler step x̂1 ← x̂1 + U(A y − A x̂1), A = f×f block mean, y = GT_{t-1}
  dds_crk<f>   dds_blk<f>, then x̂1 ← y inside the 1 px dilated crack map of GT_{t-1}

  python3 scripts/eval/state_source_probe.py --out_dir results/state_source_probe
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.dataset import PatternLoopDataset  # noqa: E402
from src.metrics import damage_texture as dt  # noqa: E402
from src.metrics import dino_mmd as dm  # noqa: E402
from src.model.cfm import refiner_cache  # noqa: E402
from src.utils import build_damage_image_transform_with_loss_mask  # noqa: E402
from train_mixed import _load_args_from_result_dir, _resolve_required_match  # noqa: E402

RUN = "results/DamagePatternGeneration/CFMADM/2026-11-05_22-06-00_v12gtrefiner"
CACHE = "results/DamagePatternGeneration/state_cache/train_n4_heun30_s128_15f51edd063a.pt"


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--refiner_dir", default=RUN, help="Refiner run: its args.json gives the dataset and the base")
    p.add_argument("--state_cache", default=CACHE)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n", type=int, default=2, help="S samples per pair and source (cache slots 0..n-1)")
    p.add_argument("--max_pairs", type=int, default=0, help="0 = every pair")
    p.add_argument("--sdedit", default="0.3,0.5,0.7,0.85")
    p.add_argument("--dds", default="blk4,crk4,crk8")
    p.add_argument("--dds_steps", type=int, default=60)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _load_base(ckpt, result_dir, device):
    """The refiner's frozen single-frame base, as lifecycle_mixin._load_frozen_cfm builds it."""
    a = argparse.Namespace(**json.load(open(os.path.join(result_dir, "args.json"))))
    a.result_dir = result_dir
    a.refiner_base_checkpoint, a.use_paired_bridge = "", False
    state = torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]
    for k in [k for k in state if str(k).startswith("fid.") or "u2_vss" in str(k)]:
        del state[k]
    model = getattr(importlib.import_module("src.model.ImageGeneration"), a.architecture)(regressor=None, **vars(a))
    status = model.load_state_dict(state, strict=False)
    print(f"[probe] base {ckpt}: missing={len(status.missing_keys)} unexpected={len(status.unexpected_keys)}")
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device).eval()


def _dataset(args, device):
    extractor_args = _load_args_from_result_dir(args.extractor_path)
    extractor = getattr(importlib.import_module("src.model.HystereticPrediction"), extractor_args.architecture)(
        **vars(extractor_args)
    ).to(device)
    ckpt = _resolve_required_match(os.path.join(args.extractor_path, "**/best.ckpt"), "extractor")
    extractor.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["state_dict"])
    extractor.eval()
    transform = build_damage_image_transform_with_loss_mask(
        image_size=args.image_size, use_augmentation=False, output_scale=True,
        normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5),
        use_grayscale_input=args.use_grayscale_input, use_clahe=args.use_clahe,
        clahe_clip_limit=args.clahe_clip_limit, clahe_tile_grid_size=args.clahe_tile_grid_size,
    )
    kw = dict(vars(args))
    kw.update(use_loss_mask=True, use_object_loss_mask=False, condition_perturb=False, train_virtual_repeat=1,
              g_traj_mix=False, share_photometric_aug=False, opa_lambda_ord=0.0, opa_lambda_mag=0.0)
    return PatternLoopDataset(root="./Data", mode="train", transform=transform, extractor=extractor,
                              extractor_args=extractor_args, **kw)


def _gt_image(ds, row):
    fr = ds._load_frame(row["filename"], row["exp"], row["direction"], row["timestep"], row["DI"],
                        force_flip=False, apply_random_erasing=False)
    return fr["image"]


def _block_dc(x1, y, f):
    """x̂1 + U(A y − A x̂1), A = f×f block mean, U = nearest upsampling (pseudo-inverse of A)."""
    d = F.avg_pool2d(y, f) - F.avg_pool2d(x1, f)
    return x1 + F.interpolate(d, scale_factor=f, mode="nearest")


@torch.no_grad()
def _dds(base, c, g, mask, y, crack, f, steps, z0):
    """Euler DDS on the OT path (t = 0 noise, t = 1 data); with no data consistency it is plain Euler."""
    coords = None
    if base.use_patch_diffusion:
        gx, gy = base._build_full_coordinate_maps(z0.shape[-2], z0.shape[-1], device=z0.device, dtype=z0.dtype)
        coords = torch.stack([gx, gy], 0)[None].expand(z0.shape[0], -1, -1, -1).contiguous()
    z = z0.clone()
    ts = torch.linspace(0.0, 1.0, steps + 1).tolist()
    for i in range(steps):
        t, t_next = ts[i], ts[i + 1]
        tt = torch.full((z.shape[0],), t, device=z.device)
        v = base._predict_velocity(z, tt, c, coords=coords, damage_mask=mask, gru_feat=g)
        x1 = z + (1.0 - t) * v
        eps = z - t * v
        if f > 0:
            x1 = _block_dc(x1, y, f)
        if crack is not None:
            x1 = crack * y + (1.0 - crack) * x1
        z = (1.0 - t_next) * eps + t_next * x1
    return z


def _u8(x):
    return dm.to_uint8_gray(x.float().clamp(-1, 1))


def _dil(c, r):
    st = np.ones((1, 2 * r + 1, 2 * r + 1), bool)
    return ndi.binary_dilation(c, st)


def _crack_stats(s_u8, gt_t, gt_p, region, tol=2):
    """Pixel counts summed over the batch (micro-averaged by the caller), crack pixels inside ``region`` only."""
    cs, ct, cp = (dt.crack_map(u) & region for u in (s_u8, gt_t, gt_p))
    ds_, dt_, dp_ = _dil(cs, tol), _dil(ct, tol), _dil(cp, tol)
    E = dt.local_maps(s_u8)[1]
    bh = ndi.grey_closing(s_u8.astype(np.float32), size=(1, 5, 5)) - s_u8.astype(np.float32)
    return {
        "n_s": cs.sum(), "s_in_t": (cs & dt_).sum(), "n_t": ct.sum(), "t_in_s": (ct & ds_).sum(),
        "n_p": cp.sum(), "p_in_s": (cp & ds_).sum(), "s_new": (cs & ~dp_).sum(),
        "chance_area": (dt_ & region).sum(), "px": region.sum(), "E_sum": float(E[region].sum()),
        "bh_crack_sum": float(bh[cs].sum()),
    }


def _spall(ds, filenames, size, margin=4):
    """Spall of each frame from the dataset's layout cache (SDF > 0), dilated by ``margin`` px."""
    out = np.zeros((len(filenames), size, size), bool)
    lay = getattr(ds, "_layout", None) or {}
    for i, f in enumerate(filenames):
        sdf = lay.get(str(f))
        if sdf is None:
            continue
        sdf = np.asarray(sdf, dtype=np.float32)
        sdf = sdf[0] if sdf.ndim == 3 else sdf
        if sdf.shape[-1] != size:
            sdf = F.interpolate(torch.from_numpy(sdf)[None, None], size=(size, size), mode="bilinear",
                                align_corners=False)[0, 0].numpy()
        out[i] = sdf > 0
    return _dil(out, margin)


def _summ(acc):
    r = lambda a, b: float(acc[a]) / max(float(acc[b]), 1.0)
    return {
        "precision_vs_t": r("s_in_t", "n_s"),
        "recall_vs_t": r("t_in_s", "n_t"),
        "keep_from_prev": r("p_in_s", "n_p"),
        "new_crack_share": r("s_new", "n_s"),
        "crack_density": r("n_s", "px"),
        "chance_precision": r("chance_area", "px"),
        "texture_E": r("E_sum", "px"),
        "crack_contrast": r("bh_crack_sum", "n_s"),
    }


def _kid(x, y, device):
    """Unbiased KID² (poly kernel) between two feature sets."""
    x = torch.from_numpy(x).to(device, torch.float64)
    y = torch.from_numpy(y).to(device, torch.float64)
    k = lambda a, b: (a @ b.T / a.shape[1] + 1.0) ** 3
    m, n = len(x), len(y)
    kxx, kyy = k(x, x), k(y, y)
    xx = (kxx.sum() - kxx.diag().sum()) / (m * (m - 1))
    yy = (kyy.sum() - kyy.diag().sum()) / (n * (n - 1))
    return float(xx + yy - 2.0 * k(x, y).mean())


@torch.no_grad()
def main():
    cli = _parse_args()
    device = torch.device("cuda")
    rdir = os.path.abspath(cli.refiner_dir)
    args = _load_args_from_result_dir(rdir)
    args.result_dir = rdir
    os.makedirs(os.path.join(rdir, "data_check"), exist_ok=True)
    os.makedirs(cli.out_dir, exist_ok=True)

    ds = _dataset(args, device)
    base_ckpt = os.path.abspath(args.refiner_base_checkpoint)
    base = _load_base(base_ckpt, os.path.dirname(base_ckpt), device)
    cache = torch.load(cli.state_cache, map_location="cpu", weights_only=False)
    label = ds.label.reset_index(drop=True)
    missing = [f for f in label["filename"] if f not in cache]
    if missing:
        raise SystemExit(f"{len(missing)} training frames are not in {cli.state_cache} (e.g. {missing[0]})")

    prev = ds._build_prev_iloc_map()
    pairs = [(int(prev[i]), i) for i in range(len(label)) if prev[i] >= 0]
    rng = np.random.default_rng(cli.seed)
    if cli.max_pairs and len(pairs) > cli.max_pairs:
        pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), cli.max_pairs, replace=False))]
    print(f"[probe] {len(label)} training frames, {len(pairs)} pairs (t-1, t), n={cli.n}", flush=True)

    gt = {}
    for p, t in pairs:
        for i in (p, t):
            if i not in gt:
                gt[i] = _gt_image(ds, label.iloc[i])
    in_ch = int(base.model_in_channels)
    to_model = lambda x: x.mean(0, keepdim=True).repeat(in_ch, 1, 1) if x.shape[0] != in_ch else x

    taus = [float(v) for v in cli.sdedit.split(",") if v]
    dds = [v for v in cli.dds.split(",") if v]
    sources = ["gt_shuffled", "base", "gt"] + [f"sdedit_{t:g}" for t in taus] + [f"dds_{v}" for v in dds]
    regions = ("all", "off_spall")
    acc = {r: {s: {} for s in sources} for r in regions}
    feats = {s: [] for s in sources}
    n = int(cli.n)
    panel_idx = set(np.linspace(0, len(pairs) - 1, 6).round().astype(int).tolist())
    panel = {s: [] for s in sources + ["GT_{t-1}", "GT_t"]}
    gt_t_u8, ref_base_u8 = [], []
    real_keep = {r: {} for r in regions}
    checked = False

    for b0 in range(0, len(pairs), cli.batch):
        chunk = pairs[b0 : b0 + cli.batch]
        rows_p = label.iloc[[p for p, _ in chunk]]
        info, hs, dr, di, omask = (x.to(device) for x in refiner_cache._frame_conditions(ds, rows_p))
        rep = lambda x: x.repeat_interleave(n, dim=0)
        c, g = base._unpack_conditions(rep(info), rep(hs), rep(dr), rep(di))
        mask = rep(omask)
        if getattr(base, "layout_cond", False):
            mask = torch.cat([mask, refiner_cache._frame_layouts(ds, rows_p, n, mask.shape[-1]).to(device)], 1)
        y = rep(torch.stack([to_model(gt[p]) for p, _ in chunk]).to(device))
        yt_u8 = np.repeat(_u8(torch.stack([gt[t] for _, t in chunk])), n, axis=0)
        yp_u8 = np.repeat(_u8(torch.stack([gt[p] for p, _ in chunk])), n, axis=0)
        gt_t_u8.append(_u8(torch.stack([gt[t] for _, t in chunk])))
        ref_base_u8.append(np.concatenate([_u8(cache[label.iloc[p]["filename"]][[2, 3]]) for p, _ in chunk]))
        crack_np = _dil(dt.crack_map(yp_u8), 1)
        crack = torch.from_numpy(crack_np[:, None].astype(np.float32)).to(device)

        gen = torch.Generator(device=device).manual_seed(cli.seed * 100003 + b0)
        shuf = [pairs[(b0 + j + len(pairs) // 2) % len(pairs)][0] for j in range(len(chunk))]
        out = {
            "gt_shuffled": rep(torch.stack([to_model(gt[i]) for i in shuf]).to(device)),
            "base": torch.cat([cache[label.iloc[p]["filename"]][:n].float() for p, _ in chunk]).to(device),
            "gt": y,
        }
        size = y.shape[-1]
        off = ~np.repeat(_spall(ds, rows_p["filename"].tolist(), size)
                         | _spall(ds, label.iloc[[t for _, t in chunk]]["filename"].tolist(), size), n, axis=0)
        region = {"all": np.ones_like(off), "off_spall": off}
        for tau in taus:
            eps = torch.randn(y.shape, generator=gen, device=device)
            out[f"sdedit_{tau:g}"] = base.sample_with_ode(
                c, gru_feat=g, damage_mask=mask, t_start=tau, z_init=(1.0 - tau) * eps + tau * y)
        for v in dds:
            f = int(v[3:])
            z0 = torch.randn(y.shape, generator=gen, device=device)
            out[f"dds_{v}"] = _dds(base, c, g, mask, y, crack if v.startswith("crk") else None, f, cli.dds_steps, z0)
            if not checked:
                plain = _dds(base, c[:4], g[:4], mask[:4], y[:4], None, 0, cli.dds_steps, z0[:4])
                solver, steps = base.cfm_solver, base.cfm_ode_steps
                base.cfm_solver = "euler"
                try:
                    ref = base.sample_with_ode(c[:4], z0=z0[:4], gru_feat=g[:4], damage_mask=mask[:4],
                                               num_steps=cli.dds_steps)
                finally:
                    base.cfm_solver, base.cfm_ode_steps = solver, steps
                err = float((plain - ref).abs().max())
                print(f"[probe] DDS loop without data consistency vs sample_with_ode (Euler): max |Δ| = {err:.2e}")
                if err > 1e-3:
                    raise SystemExit("DDS loop does not reproduce the base sampler")
                checked = True

        for r in regions:
            for k, val in _crack_stats(yp_u8[::n], yt_u8[::n], yp_u8[::n], region[r][::n]).items():
                real_keep[r][k] = real_keep[r].get(k, 0) + val
        sel = [j for j in range(len(chunk)) if b0 + j in panel_idx]
        panel["GT_{t-1}"] += [yp_u8[j * n] for j in sel]
        panel["GT_t"] += [yt_u8[j * n] for j in sel]
        for s in sources:
            su8 = _u8(out[s])
            for r in regions:
                for k, val in _crack_stats(su8, yt_u8, yp_u8, region[r]).items():
                    acc[r][s][k] = acc[r][s].get(k, 0) + val
            feats[s].append(su8)
            panel[s] += [su8[j * n] for j in sel]
        print(f"[probe] {min(b0 + cli.batch, len(pairs))}/{len(pairs)} pairs", flush=True)

    gt_t_u8, ref_base_u8 = np.concatenate(gt_t_u8), np.concatenate(ref_base_u8)
    fx = dm.DinoFeatures(dm.DEFAULT_DINO, device="cuda")
    f_ref_base = fx(ref_base_u8)[0]
    f_ref_gt = fx(gt_t_u8)[0]
    report = {"pairs": len(pairs), "n": n, "base_checkpoint": base_ckpt,
              "state_cache": os.path.abspath(cli.state_cache), "gt_crack_keep": {}, "sources": {}}
    kid = {}
    for s in sources:
        fs = fx(np.concatenate(feats[s]))[0]
        kid[s] = {"kid_vs_base": _kid(fs, f_ref_base, "cuda"), "kid_vs_gt": _kid(fs, f_ref_gt, "cuda")}
    cols = ["precision_vs_t", "recall_vs_t", "keep_from_prev", "new_crack_share", "crack_density",
            "texture_E", "crack_contrast", "kid_vs_base", "kid_vs_gt"]
    for r in regions:
        report["gt_crack_keep"][r] = _summ(real_keep[r])["precision_vs_t"]
        report["sources"][r] = {s: {**_summ(acc[r][s]), **kid[s]} for s in sources}
        print(f"\n[{r}] GT crack_keep t-1 → t (micro): {report['gt_crack_keep'][r]:.3f}; "
              f"chance precision: {report['sources'][r]['base']['chance_precision']:.3f}")
        print("| source | " + " | ".join(cols) + " |")
        for s in sources:
            row = report["sources"][r][s]
            print(f"| {s} | " + " | ".join(f"{row[c]:.4f}" if c.startswith("kid") or c == "crack_density"
                                           else f"{row[c]:.3f}" for c in cols) + " |")
    with open(os.path.join(cli.out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    from PIL import Image

    names = ["GT_{t-1}", "GT_t"] + sources
    Image.fromarray(np.concatenate([np.concatenate(panel[k], 1) for k in names], 0)).save(
        os.path.join(cli.out_dir, "panel.png"))
    with open(os.path.join(cli.out_dir, "panel_rows.txt"), "w") as fh:
        fh.write("\n".join(names) + "\n")
    print(f"[probe] wrote {cli.out_dir}/report.json and panel.png")


if __name__ == "__main__":
    main()
