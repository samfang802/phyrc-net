#!/usr/bin/env python3
"""K samples per condition from the HystereticCGAN baseline (cGANckpt/epoch=02999.ckpt) → npz.

Same layout as generate_eval_samples.py, so the result goes straight into eval_dinov3_kid.py,
inpaint_generated_samples.py and eval_damage_texture.py. Condition = [GRU hidden state ; direction],
built by PatternLoopDataset with the GRU in --extractor_path; the checkpoint records
results/HystereticPrediction/HystereticGRU/2026-04-17_01-26-41, which no longer exists, so the
default is ./phase1_extractor (same architecture, 1024-d bidirectional hidden state).

Because the GRU is a substitute, --check_saved verifies the conditioning against the images the
original run saved (cGANckpt/CTR1/<frame>_test.jpg):
  rank      DINOv3 rank of the saved image of the same frame among all saved images, for each new
            sample (1 = nearest; chance ≈ (N+1)/2). Near 1 means the condition reaches G as it did then.
  pixel     mean |new − saved| for the same frame vs for a random other frame.
  D_margin  D(G(z,c_i), c_i) − D(G(z,c_i), c_j≠i): the discriminator prefers the matching condition.
Status (2026-10): with ./phase1_extractor the check FAILS (rank at chance, drift-progression corr vs
GT 0.18 against 0.69 for the saved images). Inverting the condition so that G reproduces the saved
images works (fresh z: |Δ| 2.8 gray levels, rank median 2; cosine to the phase1 hidden state 0.66),
so G is fine and phase1_extractor is a different GRU than the one the cGAN was trained with. Until
that GRU is found, use the saved images (K=1; z changes the output by only ~3 gray levels, so more
samples would add almost nothing). The npz meta records `conditioning_valid`.

Output RGB is converted to gray with the ITU-R 601 luma (PIL "L"), like the saved jpgs were.

BatchNorm: the generator's running statistics are unusable (eval mode gives one flat gray image for
every input), so by default G runs with batch statistics (--bn batch), as during GAN training. The
output then depends on the batch, so each sample index k is generated for all conditions in one
batch (--batch_size 0 = whole split), which makes it independent of the iteration order.

Example:
  python scripts/eval/generate_cgan_samples.py --out cgan_k8.npz --k 8 --check_saved cGANckpt/CTR1
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.metrics.samples_io import save_samples_npz
from src.data.pattern_loop import PatternLoopDataset
from src.metrics import dino_mmd as dm
from src.model.legacy_cgan import load_cgan
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir

LUMA = (0.299, 0.587, 0.114)


def _parse_args():
    p = argparse.ArgumentParser(description="HystereticCGAN samples → npz")
    p.add_argument("--checkpoint", default="cGANckpt/epoch=02999.ckpt")
    p.add_argument("--extractor_path", default="./phase1_extractor", help="GRU giving the 1024-d hidden state")
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test", choices=("test", "val", "train"))
    p.add_argument("--data_path", default="id_list_val_R307.json", help="Split file under Data/DamagePattern")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=0, help="0 = all conditions in one batch (see BatchNorm note)")
    p.add_argument("--bn", default="batch", choices=("batch", "running"), help="BatchNorm statistics for G")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check_saved", default="", help="Folder of the original run's <frame>_test.jpg")
    p.add_argument("--work_dir", default="", help="Where the dataset writes its hysteresis plots (default: next to --out)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _batch_stats(module):
    """BatchNorm layers normalise with the current batch and never update their running buffers."""
    for m in module.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.train()
            m.track_running_stats = False
            m.running_mean = m.running_var = None


def _to_gray_u8(x):
    """(B,3,H,W) tanh output → (B,H,W) uint8 luma."""
    x01 = (x.float() * 0.5 + 0.5).clamp(0, 1)
    w = torch.tensor(LUMA, device=x.device).view(1, 3, 1, 1)
    return torch.round((x01 * w).sum(1) * 255.0).to(torch.uint8).cpu().numpy()


def _conditions(cli, column_info, device):
    ex_args = _load_args_from_result_dir(os.path.abspath(cli.extractor_path))
    extractor = getattr(importlib.import_module("src.model.HystereticPrediction"), ex_args.architecture)(**vars(ex_args)).to(device)
    ck = sorted(glob.glob(os.path.join(cli.extractor_path, "**", "best.ckpt"), recursive=True))
    if not ck:
        raise FileNotFoundError(f"no best.ckpt under {cli.extractor_path}")
    extractor.load_state_dict(torch.load(ck[0], map_location=device, weights_only=False)["state_dict"])
    extractor.eval()
    work = cli.work_dir or os.path.join(os.path.dirname(os.path.abspath(cli.out)), "cgan_dataset_work")
    os.makedirs(os.path.join(work, "data_check"), exist_ok=True)  # the dataset draws hysteresis plots there
    ds = PatternLoopDataset(
        root="./Data", mode=cli.split, transform=build_damage_image_transform_with_loss_mask(image_size=128, use_augmentation=False, use_clahe=True),
        extractor=extractor, extractor_args=ex_args, result_dir=work, data_path=cli.data_path,
        column_info=column_info, condition_perturb=False,
    )
    hid, dirs = [], []
    for i in range(len(ds)):
        s = ds[i]
        hid.append(torch.as_tensor(s[1]).float())
        dirs.append(torch.as_tensor(s[2]).float())
    c = torch.cat([torch.stack(hid), torch.stack(dirs)], 1)
    return c, ds.label["filename"].tolist(), ds.label["DI"].to_numpy(np.float32)


@torch.no_grad()
def _check_saved(cli, fake, filenames, G, D, c, z_dim, device):
    names = [n for n in filenames if os.path.isfile(os.path.join(cli.check_saved, os.path.splitext(n)[0] + "_test.jpg"))]
    idx = np.array([filenames.index(n) for n in names])
    saved = np.stack([
        np.array(Image.open(os.path.join(cli.check_saved, os.path.splitext(n)[0] + "_test.jpg")).convert("L").resize((128, 128), Image.BICUBIC))
        for n in names
    ])
    fx = dm.DinoFeatures(device=device)
    fs = fx(saved)[0]
    n, K = len(idx), fake.shape[1]
    ranks = []
    for k in range(K):
        fg = fx(fake[idx, k])[0]
        Dm = dm.pair_distance(fg, fs)
        ranks.append(1 + (Dm < np.diag(Dm)[:, None]).sum(1))
    ranks = np.stack(ranks, 1)
    rng = np.random.default_rng(cli.seed)
    other = (np.arange(n) + rng.integers(1, n, n)) % n
    f = fake[idx].astype(np.float32)
    same = np.abs(f - saved[:, None].astype(np.float32)).mean((1, 2, 3))
    diff = np.abs(f - saved[other][:, None].astype(np.float32)).mean((1, 2, 3))
    # Discriminator: same generated images scored with their own vs another frame's condition.
    cc = c[idx].to(device)
    z = torch.randn(n, z_dim, 1, 1, device=device, generator=torch.Generator(device).manual_seed(cli.seed + 99))
    x = G(z, cc[:, :, None, None])
    d_own = D(x, cc).cpu().numpy()
    d_other = D(x, cc[torch.as_tensor(other, device=device)]).cpu().numpy()
    out = {
        "n_frames": int(n),
        "rank_median": float(np.median(ranks)),
        "rank_top1": float((ranks == 1).mean()),
        "rank_chance": (n + 1) / 2,
        "pixel_same_frame": float(same.mean()),
        "pixel_other_frame": float(diff.mean()),
        "D_margin": float((d_own - d_other).mean()),
        "D_margin_frac_positive": float((d_own > d_other).mean()),
    }
    print(f"[cgan] check vs saved images: {out}", flush=True)
    return out


def main():
    cli = _parse_args()
    device = cli.device
    G, D, meta = load_cgan(cli.checkpoint, device)
    if cli.bn == "batch":
        _batch_stats(G)
    hp = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)["hyper_parameters"]
    if not (meta["latent_conditioning"] and meta["direction_conditioning"]) or hp.get("include_di_conditioning"):
        raise ValueError("Only the [hidden ; direction] conditioning of cGANckpt is implemented")
    c, filenames, DI = _conditions(cli, list(hp["column_info"]), device)
    if c.shape[1] != meta["condition_dim"]:
        raise ValueError(f"condition dim {c.shape[1]} != checkpoint {meta['condition_dim']}")
    print(f"[cgan] {len(filenames)} conditions, K={cli.k}, extractor {cli.extractor_path} "
          f"(checkpoint recorded {meta['extractor_path']})", flush=True)

    gen = torch.Generator(device).manual_seed(cli.seed)
    fake = np.zeros((len(c), cli.k, 128, 128), np.uint8)
    bs = cli.batch_size or len(c)
    with torch.no_grad():
        for k in range(cli.k):
            for s in range(0, len(c), bs):
                cb = c[s : s + bs].to(device)
                z = torch.randn(len(cb), meta["z_dim"], 1, 1, device=device, generator=gen)
                fake[s : s + len(cb), k] = _to_gray_u8(G(z, cb[:, :, None, None]))

    real = dm.load_real_uint8(filenames, "Data/DamagePattern/images")
    info = {
        "model": "HystereticCGAN", "checkpoint": os.path.abspath(cli.checkpoint), "split": cli.split, "k": cli.k,
        "extractor_path": os.path.abspath(cli.extractor_path), "extractor_recorded": meta["extractor_path"],
        "trained_split_file": meta["data_path"], "epoch": meta["epoch"], "seed": cli.seed, "gray": "ITU-R 601 luma",
        "bn": cli.bn, "batch_size": bs,
    }
    if cli.check_saved:
        chk = _check_saved(cli, fake, filenames, G, D, c, meta["z_dim"], device)
        info["check_saved"] = chk
        # A matching condition puts the saved image of the same frame near the top; chance is (N+1)/2.
        info["conditioning_valid"] = bool(chk["rank_median"] < 0.25 * chk["rank_chance"])
        if not info["conditioning_valid"]:
            print("[cgan] WARNING: samples do not match the original run's conditioning (wrong GRU?). "
                  "Do not use this npz as the cGAN baseline.", flush=True)
    save_samples_npz(cli.out, fake, real, c.numpy(), DI, filenames, info)
    print(f"[cgan] wrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
