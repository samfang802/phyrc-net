#!/usr/bin/env python3
"""Meta-evaluation of the DINOv3 KID: does it rise monotonically with damage-relevant degradation?

Reference = real images of a split (default test, CTR1). Each degradation is applied to the same
reals at increasing levels and scored against the clean reals. Image i and its degraded copy share
group i, so level 0 scores exactly 0 for the U-statistics.

Degradations: Gaussian blur, Gaussian noise, crack erasure (grayscale closing removes thin dark
lines), contrast loss, JPEG, plus structural ones that barely move the gray histogram:
temporal shift along the column, in-place 90/180° block rotations, and a vertical roll of the
spall zone. Scores: DINOv3 KID (global / patch / soft-DI), RBF MMD, Fréchet distance on the
same DINOv3 features (biased, for reference) and Inception FID (torchmetrics, biased).
Spearman ρ(level, score) summarizes monotonicity. sensitivity_z is the first real distortion
level divided by that metric's real-halves standard deviation.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import linalg
from scipy.stats import spearmanr

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.metrics import dino_mmd as dm

LEVELS = {
    "blur_sigma": [0, 0.5, 1.0, 1.5, 2.0, 3.0],
    "noise_std": [0, 4, 8, 16, 24, 32],
    "crack_erase_ksize": [0, 3, 5, 7, 9, 11],
    "contrast_loss": [0, 0.1, 0.2, 0.4, 0.6, 0.8],
    "jpeg_drop": [0, 10, 30, 50, 70, 90],  # quality = 100 - level
    # Structural distortions: histogram almost unchanged, geometry / damage alignment is not.
    "temporal_shift": [2, 5, 10, 20, 35, 50],
    "local_patch_rot90": [1, 2, 4, 8, 12, 16],
    "spall_vertical_shift": [4, 8, 16, 24, 36, 48],
}
STRUCTURAL_KINDS = ("temporal_shift", "local_patch_rot90", "spall_vertical_shift")


def _temporal_shift(u8, dt):
    """Frame i becomes the same column at i+dt, or i-dt when i+dt falls off the end."""
    G = len(u8)
    src = np.arange(G) + int(dt)
    oob = (src < 0) | (src >= G)
    src = src.copy()
    src[oob] = np.arange(G)[oob] - int(dt)
    src = np.clip(src, 0, G - 1)
    return u8[src]


def _local_patch_rot90(u8, m, seed):
    """Rotate m non-overlapping blocks by 90° or 180°. Pixel histogram is unchanged."""
    out = np.empty_like(u8)
    h, w = u8.shape[1], u8.shape[2]
    gh, gw = 4, 4
    bh, bw = h // gh, w // gw
    m = int(min(m, gh * gw))
    for i, im in enumerate(u8):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(i), int(m)]))
        chosen = rng.choice(gh * gw, size=m, replace=False)
        y = im.copy()
        for b in chosen:
            r, c = divmod(int(b), gw)
            block = y[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw]
            k = 1 if rng.random() < 0.5 else 2  # 90° or 180°
            y[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw] = np.rot90(block, k)
        out[i] = y
    return out


def degrade(u8, kind, level, rng, seed=0):
    if level == 0:
        return u8.copy()
    if kind == "temporal_shift":
        return _temporal_shift(u8, level)
    if kind == "local_patch_rot90":
        return _local_patch_rot90(u8, level, seed)
    if kind == "spall_vertical_shift":
        # Height axis of each image. Texture and the gray histogram stay put; spall does not.
        return np.roll(u8, shift=int(level), axis=1)
    out = np.empty_like(u8)
    for i, im in enumerate(u8):
        if kind == "blur_sigma":
            y = cv2.GaussianBlur(im, (0, 0), sigmaX=level)
        elif kind == "noise_std":
            y = np.clip(im.astype(np.float32) + rng.normal(0, level, im.shape), 0, 255)
        elif kind == "crack_erase_ksize":
            y = cv2.morphologyEx(im, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (level, level)))
        elif kind == "contrast_loss":
            y = im.mean() + (1.0 - level) * (im.astype(np.float32) - im.mean())
        elif kind == "jpeg_drop":
            buf = io.BytesIO()
            Image.fromarray(im).save(buf, format="JPEG", quality=max(1, 100 - int(level)))
            y = np.array(Image.open(buf))
        else:
            raise ValueError(f"unknown degradation {kind}")
        out[i] = np.round(y).astype(np.uint8)
    return out


def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2.0 * covmean.real))


def inception_fid(real_u8, fake_u8, device):
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048).to(device)
    for u8, is_real in ((real_u8, True), (fake_u8, False)):
        for i in range(0, len(u8), 64):
            x = torch.from_numpy(u8[i : i + 64])[:, None].repeat(1, 3, 1, 1).to(device)
            fid.update(x, real=is_real)
    return float(fid.compute())


def _load_di(names):
    import pandas as pd

    di = pd.read_csv("Data/DamagePattern/label.csv").set_index("filename")["DI"]
    return di.loc[list(names)].to_numpy(np.float64)


def _noise_floors(real, rg, rp, di, sigma, device, rng, n_halves=50, n_fid=8, with_fid=True):
    """Std of real-vs-real half splits. kid_global's entry is the real_halves_kid noise floor."""
    G = len(rg)
    g = np.arange(G)
    ones = np.ones(G)
    S = dm.group_sums(rg, g, rg, g, G, dm.poly_kernel, device)
    rr = dm.GroupMMD(S, S, S, ones, ones)
    Sr = dm.group_sums(rg, g, rg, g, G, dm.rbf_kernel(sigma), device)
    rbf = dm.GroupMMD(Sr, Sr, Sr, ones, ones)
    pk = dm.patch_kid(rp, g, rp, g, G, device) if rp is not None else None
    soft = dm.SoftDI(S, S, S, ones, ones, di)
    bags = {"kid_global": [], "mmd_rbf_global": [], "kid_patch": [], "kid_soft_di": [], "inception_fid": []}
    fid_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    for i in range(n_halves):
        perm = rng.permutation(G)
        a = np.zeros(G, bool)
        a[perm[: G // 2]] = True
        b = ~a
        bags["kid_global"].append(rr.value(a, b))
        bags["mmd_rbf_global"].append(rbf.value(a, b))
        if pk is not None:
            bags["kid_patch"].append(pk.value(a, b))
        bags["kid_soft_di"].append(soft.value_between(a, b))
        if with_fid and i < n_fid:
            bags["inception_fid"].append(inception_fid(real[a], real[b], device))
    std = {k: float(np.std(v)) for k, v in bags.items() if v}
    std["real_halves_kid"] = std["kid_global"]
    return std


def _first_distortion(rows):
    for row in rows:
        if row["level"] != 0:
            return row
    return rows[0]


def _finite(x):
    x = float(x)
    return x if np.isfinite(x) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--data_path", default="Data/DamagePattern/id_list_val_R307.json")
    p.add_argument("--image_dir", default="Data/DamagePattern/images")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dino_model", default=None, help="Backbone id; overrides --dino_model_id")
    p.add_argument("--dino_model_id", default=dm.DEFAULT_DINO)
    p.add_argument("--n_patch", type=int, default=16)
    p.add_argument("--kinds", nargs="*", default=list(LEVELS))
    p.add_argument("--no_inception", action="store_true")
    p.add_argument("--save_examples", action="store_true", help="Save one strip per degradation")
    p.add_argument("--n_halves", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    cli = p.parse_args()
    os.makedirs(cli.out_dir, exist_ok=True)
    rng = np.random.default_rng(cli.seed)
    model_id = cli.dino_model or cli.dino_model_id

    with open(cli.data_path) as f:
        names = json.load(f)[cli.split]
    real = dm.load_real_uint8(names, cli.image_dir)
    di = _load_di(names)
    G = len(real)
    g = np.arange(G)
    fx = dm.DinoFeatures(model_id, device=cli.device)
    rg, rp = fx(real, n_patch=cli.n_patch, seed=cli.seed)
    sigma = dm.median_distance(rg)
    print("[sanity] noise floor (real halves)...", flush=True)
    floor = _noise_floors(
        real, rg, rp, di, sigma, cli.device, rng, n_halves=cli.n_halves, with_fid=not cli.no_inception
    )
    print("[sanity] real_halves std " + "  ".join(f"{k}={v:.5g}" for k, v in floor.items()), flush=True)

    score_keys = ("kid_global", "mmd_rbf_global", "kid_patch", "kid_soft_di", "inception_fid")
    report = {
        "split": cli.split,
        "n": G,
        "dino_model_id": model_id,
        "rbf_sigma": sigma,
        "soft_di_tau": dm.soft_di_tau(di),
        "noise_floor_std": floor,
        "kinds": {},
    }
    for kind in cli.kinds:
        rows = []
        strip = []
        for level in LEVELS[kind]:
            d = degrade(real, kind, level, rng, seed=cli.seed)
            strip.append(d[G // 2])
            dg, dp = fx(d, n_patch=cli.n_patch, seed=cli.seed)
            mm = dm.global_mmds(dg, g, rg, g, G, sigma, cli.device)
            kid, _ = mm["kid"].jackknife()
            soft, _ = dm.kid_soft_di(mm["kid"], di).jackknife()
            row = {
                "level": level,
                "kid_global": kid["value"],
                "kid_global_se": kid["se"],
                "mmd_rbf_global": mm["mmd_rbf"].value(),
                "kid_soft_di": soft["value"],
                "kid_soft_di_se": soft["se"],
                "fd_dino_biased": frechet(dg, rg),
            }
            if cli.n_patch > 0:
                row["kid_patch"] = dm.patch_kid(dp, g, rp, g, G, cli.device).value()
            if not cli.no_inception:
                row["inception_fid"] = inception_fid(real, d, cli.device)
            rows.append(row)
            print(f"[sanity] {kind}={level}: " + "  ".join(f"{k}={v:.5g}" for k, v in row.items() if k != "level"), flush=True)
        lv = [r["level"] for r in rows]
        mono = {}
        for k in rows[0]:
            if k in ("level", "kid_global_se", "kid_soft_di_se"):
                continue
            rho = spearmanr(lv, [r[k] for r in rows]).statistic
            mono[k] = _finite(rho)
        first = _first_distortion(rows)
        z = {}
        for k in score_keys:
            if k not in first or k not in floor or floor[k] <= 0:
                continue
            z[k] = _finite(first[k] / floor[k])
        report["kinds"][kind] = {
            "rows": rows,
            "spearman_rho": mono,
            "sensitivity_z_first_level": z,
            "first_level": first["level"],
        }
        if cli.save_examples:
            Image.fromarray(np.concatenate(strip, axis=1)).save(os.path.join(cli.out_dir, f"strip_{kind}.png"))

    out = os.path.join(cli.out_dir, "dinov3_kid_sanity.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("[sanity] Spearman ρ(level, score) and first-level z vs that metric's real-halves std:")
    for kind, v in report["kinds"].items():
        rho = "  ".join(f"{k}={r:+.2f}" if r is not None else f"{k}=nan" for k, r in v["spearman_rho"].items())
        zs = "  ".join(
            f"{k}={r:+.2f}" if r is not None else f"{k}=nan" for k, r in v["sensitivity_z_first_level"].items()
        )
        print(f"  {kind:>22s}  ρ {rho}")
        print(f"  {'':>22s}  z {zs}")
    print(f"[sanity] wrote {out}")


if __name__ == "__main__":
    main()
