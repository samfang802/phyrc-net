#!/usr/bin/env python3
"""Haar LL/LH/HL/HH (wavelet) + per-band IWT (pixel) for the 5 test samples.

Reuses test_cfm_checkpoint.main() after wrapping the panel saver.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.model.exp.wavelet_unet import haar_dwt2d, haar_iwt2d


def _to_nchw01(img):
    if torch.is_tensor(img):
        x = img.detach().float().cpu()
    else:
        x = torch.as_tensor(np.asarray(img), dtype=torch.float32)
    if x.ndim == 3 and x.shape[0] not in (1, 3):
        x = x.permute(2, 0, 1)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.max() > 1.5:
        x = x / 255.0
    return x.clamp(0.0, 1.0)


def _gray(x):
    if x.shape[0] == 1:
        return x
    return x.mean(dim=0, keepdim=True)


def _up2(band):
    return F.interpolate(band.unsqueeze(0), scale_factor=2.0, mode="nearest")[0]


def _vis_ll(ll):
    return _up2(ll).clamp(0.0, 1.0)[0].numpy()


def _vis_hf_wave(band, gain=3.0):
    vis = (0.5 + gain * _up2(band)[0]).clamp(0.0, 1.0)
    return vis.numpy()


def _band_to_pixel(ll, lh, hl, hh, which, hw):
    z = torch.zeros_like(ll)
    if which == "ll":
        rec = haar_iwt2d(ll, z, z, z, crop_hw=hw)
        return rec[0, 0].clamp(0.0, 1.0).numpy()
    if which == "lh":
        rec = haar_iwt2d(z, lh, z, z, crop_hw=hw)
    elif which == "hl":
        rec = haar_iwt2d(z, z, hl, z, crop_hw=hw)
    else:
        rec = haar_iwt2d(z, z, z, hh, crop_hw=hw)
    return (0.5 + 2.5 * rec[0, 0]).clamp(0.0, 1.0).numpy()


def _full_recon(ll, lh, hl, hh, hw):
    rec = haar_iwt2d(ll, lh, hl, hh, crop_hw=hw)
    return rec[0, 0].clamp(0.0, 1.0).numpy()


def decompose(img01):
    g = _gray(img01).unsqueeze(0)
    ll, lh, hl, hh, _pads = haar_dwt2d(g)
    hw = (int(g.shape[-2]), int(g.shape[-1]))
    return ll, lh, hl, hh, hw, g[0, 0].numpy()


def save_one_figure(gt01, inf01, di_value, out_path):
    titles_w = ["pixel", "LL (wave)", "LH (wave)", "HL (wave)", "HH (wave)"]
    titles_p = ["IWT all", "LL → pixel", "LH → pixel", "HL → pixel", "HH → pixel"]
    fig, axes = plt.subplots(4, 5, figsize=(12.5, 10.2))

    for row, img01, tag in ((0, gt01, "GT"), (2, inf01, "Inf")):
        ll, lh, hl, hh, hw, gray = decompose(img01)
        wav = [
            gray,
            _vis_ll(ll[0]),
            _vis_hf_wave(lh[0]),
            _vis_hf_wave(hl[0]),
            _vis_hf_wave(hh[0]),
        ]
        pix = [
            _full_recon(ll, lh, hl, hh, hw),
            _band_to_pixel(ll, lh, hl, hh, "ll", hw),
            _band_to_pixel(ll, lh, hl, hh, "lh", hw),
            _band_to_pixel(ll, lh, hl, hh, "hl", hw),
            _band_to_pixel(ll, lh, hl, hh, "hh", hw),
        ]
        for c, im in enumerate(wav):
            axes[row, c].imshow(im, cmap="gray", vmin=0, vmax=1)
            axes[row, c].set_title(f"{tag} {titles_w[c]}", fontsize=8)
            axes[row, c].axis("off")
        for c, im in enumerate(pix):
            axes[row + 1, c].imshow(im, cmap="gray", vmin=0, vmax=1)
            axes[row + 1, c].set_title(f"{tag} {titles_p[c]}", fontsize=8)
            axes[row + 1, c].axis("off")

    fig.suptitle(
        f"Haar DWT / IWT  ·  test CTR1  ·  DI={float(di_value):.2f}  ·  no FreeU\n"
        "LH=horizontal HF (vertical edges)  ·  HL=vertical HF (horizontal cracks)  ·  HH=diagonal",
        fontsize=11,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[haar] saved {out_path}", flush=True)


def main():
    import test_cfm_checkpoint as tcc

    orig = tcc._save_gt_inference_panel

    def wrapped(items, di_values, step, result_dir, *args, **kwargs):
        out_dir = os.path.join(result_dir, "inference_samples", "haar_dwt_iwt")
        for item, di in zip(items, di_values):
            gt = _to_nchw01(item["anchor"]["real"])
            inf = _to_nchw01(item["sampled_vis"])
            di_tag = f"{float(di):.2f}".replace(".", "p")
            name = f"haar_test_di{di_tag}.png"
            save_one_figure(gt, inf, di, os.path.join(out_dir, name))
        return orig(items, di_values, step, result_dir, *args, **kwargs)

    tcc._save_gt_inference_panel = wrapped
    sys.argv = [
        "test_cfm_checkpoint.py",
        "--checkpoint",
        os.path.join(ROOT, "results/DamagePatternGeneration/CFMADM/selfcondnoDI/checkpoint_step_014000.pt"),
        "--result_dir",
        os.path.join(ROOT, "results/DamagePatternGeneration/CFMADM/selfcondnoDI"),
        "--fixed_di_values",
        "0,0.4,0.6,0.8,1",
        "--split",
        "test",
        "--skip_metrics",
        "--num_workers",
        "1",
    ]
    tcc.main()


if __name__ == "__main__":
    main()
