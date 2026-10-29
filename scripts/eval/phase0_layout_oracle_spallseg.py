#!/usr/bin/env python3
"""Phase-0 layout-oracle: small pixel CNN that segments spall, trained on the train-split AnyLabel masks.

Used only as a measuring tool for generated images (docs/redesign/phase0_layout_oracle.md). No DINO
backbone: a 4-level U-Net from scratch, about 0.5 M parameters.

Input is exactly the eval npz representation: gray uint8 128×128 after resize + CLAHE (the generator's
transform, no augmentation; `fake` / `real` in samples/<name>.npz). Labels: the 128 masks of
Data/DamagePattern/anylabel_layout_sdf128.npz. Training uses the train split (4 columns) only, with
photometric / blur / noise / shift augmentation so it transfers to generated images. The checkpoint
is selected on R307 (val) loss; CTR1 (test) is only reported.

  python scripts/eval/phase0_layout_oracle_spallseg.py --out_dir results/phase0_layout_oracle/spallseg
Writes best.pt, metrics.json (per split: IoU, pixel F1, area bias overall / high-DI tertile).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------- model
def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
    )


class SpallSegNet(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        w = [width, width * 2, width * 4, width * 8]
        self.enc = nn.ModuleList([_block(1, w[0]), _block(w[0], w[1]), _block(w[1], w[2]), _block(w[2], w[3])])
        self.up = nn.ModuleList([nn.ConvTranspose2d(w[i + 1], w[i], 2, stride=2) for i in range(3)])
        self.dec = nn.ModuleList([_block(w[i] * 2, w[i]) for i in range(3)])
        self.head = nn.Conv2d(w[0], 1, 1)

    def forward(self, x):
        """x: (B,1,H,W) in [-1, 1] → logits (B,1,H,W)."""
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips[-1]
        for i in (2, 1, 0):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], dim=1))
        return self.head(x)


def load_segmenter(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    net = SpallSegNet(width=int(payload.get("width", 16))).to(device)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net, float(payload.get("threshold", 0.5))


@torch.no_grad()
def predict_masks(net, u8, device, threshold=0.5, batch=256):
    """u8: (N,H,W) uint8 gray → bool masks (N,H,W)."""
    out = []
    for i in range(0, len(u8), batch):
        x = torch.from_numpy(np.asarray(u8[i : i + batch])).float().to(device)[:, None] / 127.5 - 1.0
        out.append((torch.sigmoid(net(x)) >= threshold).squeeze(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,) + tuple(u8.shape[1:]), bool)


# ---------------------------------------------------------------- data
def split_images(split, data_path="id_list_val_R307.json", image_size=128):
    """GT frames of a split, through the generator's eval transform → uint8 gray (same as npz `real`)."""
    from PIL import Image

    from src.metrics.dino_mmd import to_uint8_gray
    from src.metrics.samples_io import split_rows
    from src.utils import build_damage_image_transform_with_loss_mask

    rows = split_rows(split, data_path=os.path.join("Data/DamagePattern", data_path))
    tf = build_damage_image_transform_with_loss_mask(
        image_size=image_size, use_augmentation=False, output_scale=True,
        normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5),
        use_grayscale_input=True, use_clahe=True, clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8),
    )
    imgs = []
    for fn in rows["filename"]:
        out = tf(Image.open(os.path.join("Data/DamagePattern/images", fn)))
        img = out["target_image"] if isinstance(out, dict) else out
        imgs.append(to_uint8_gray(img[None])[0])
    return rows, np.stack(imgs)


def layout_masks(filenames, cache="Data/DamagePattern/anylabel_layout_sdf128.npz"):
    z = np.load(cache)
    idx = {str(f): i for i, f in enumerate(z["filename"])}
    mask = np.asarray(z["mask"])  # read once: NpzFile re-decompresses on every z["mask"]
    return np.stack([mask[idx[str(f)]] for f in filenames]).astype(bool)


def _augment(x, y, gen):
    """x (B,1,H,W) in [-1,1], y (B,1,H,W) {0,1}. Photometric + blur + noise + flip + shift."""
    b = x.shape[0]
    dev = x.device
    flip = torch.rand(b, device=dev) < 0.5
    x = torch.where(flip.view(b, 1, 1, 1), x.flip(-1), x)
    y = torch.where(flip.view(b, 1, 1, 1), y.flip(-1), y)
    # brightness / contrast / gamma on [0,1]
    u = (x + 1) / 2
    gain = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
    bias = torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.15, 0.15)
    mean = u.mean(dim=(2, 3), keepdim=True)
    u = ((u - mean) * gain + mean + bias).clamp(0, 1)
    gamma = torch.exp(torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.3, 0.3))
    u = u.clamp_min(1e-4) ** gamma
    # blur (box-approx of Gaussian, sigma up to ~1 px) on half the batch
    blur = torch.rand(b, device=dev) < 0.5
    if blur.any():
        k = torch.tensor([1.0, 2.0, 1.0], device=dev)
        k = (k[:, None] * k[None, :]) / 16.0
        ub = F.conv2d(F.pad(u, (1, 1, 1, 1), mode="replicate"), k.view(1, 1, 3, 3))
        u = torch.where(blur.view(b, 1, 1, 1), ub, u)
    u = (u + torch.randn_like(u) * torch.empty(b, 1, 1, 1, device=dev).uniform_(0, 0.04)).clamp(0, 1)
    x = u * 2 - 1
    # integer shift up to ±8 px (reflect for the image, replicate for the mask)
    sx, sy = [int(v) for v in torch.randint(-8, 9, (2,), generator=gen)]
    if sx or sy:
        pad = (8, 8, 8, 8)
        xp = F.pad(x, pad, mode="reflect")
        yp = F.pad(y, pad, mode="replicate")
        h, w = x.shape[-2:]
        x = xp[..., 8 + sy : 8 + sy + h, 8 + sx : 8 + sx + w]
        y = yp[..., 8 + sy : 8 + sy + h, 8 + sx : 8 + sx + w]
    return x, y


def _loss(logits, y):
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter = (p * y).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + 1)
    return bce + dice.mean()


def _split_metrics(pred, gt, di):
    """pred, gt bool (N,H,W); di (N,). Area fractions, bias, IoU over frames with any spall in GT or pred."""
    a_p = pred.reshape(len(pred), -1).mean(1)
    a_g = gt.reshape(len(gt), -1).mean(1)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    hi = di >= np.quantile(di, 2 / 3)
    return {
        "n": int(len(pred)),
        "iou_pooled": float(inter / max(union, 1)),
        "area_pred_mean": float(a_p.mean()),
        "area_gt_mean": float(a_g.mean()),
        "area_bias": float((a_p - a_g).mean()),
        "area_mae": float(np.abs(a_p - a_g).mean()),
        "area_corr": float(np.corrcoef(a_p, a_g)[0, 1]) if a_g.std() > 0 and a_p.std() > 0 else float("nan"),
        "high_di_area_pred": float(a_p[hi].mean()),
        "high_di_area_gt": float(a_g[hi].mean()),
        "high_di_area_bias": float((a_p[hi] - a_g[hi]).mean()),
        "high_di_iou_pooled": float((pred[hi] & gt[hi]).sum() / max((pred[hi] | gt[hi]).sum(), 1)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/phase0_layout_oracle/spallseg")
    p.add_argument("--iters", type=int, default=4000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    cli = p.parse_args()
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cli.out_dir, exist_ok=True)

    data = {}
    for split in ("train", "val", "test"):
        rows, u8 = split_images(split)
        data[split] = (rows, u8, layout_masks(rows["filename"]))
        print(f"[spallseg] {split}: {len(rows)} frames, spall frames {(data[split][2].reshape(len(rows), -1).any(1)).sum()}", flush=True)

    def to_t(u8, m):
        x = torch.from_numpy(u8).float()[:, None] / 127.5 - 1.0
        y = torch.from_numpy(m.astype(np.float32))[:, None]
        return x.to(dev), y.to(dev)

    xtr, ytr = to_t(data["train"][1], data["train"][2])
    xva, yva = to_t(data["val"][1], data["val"][2])
    # Oversample frames with spall (158 / 533) so the positive class is seen often.
    has = data["train"][2].reshape(len(xtr), -1).any(1)
    w = np.where(has, 2.0, 1.0)
    w = torch.from_numpy(w / w.sum())

    net = SpallSegNet(width=cli.width).to(dev)
    print(f"[spallseg] params={sum(q.numel() for q in net.parameters())/1e6:.3f} M device={dev}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=cli.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cli.lr, total_steps=cli.iters, pct_start=0.1)
    gen = torch.Generator().manual_seed(cli.seed)
    best, best_it = float("inf"), -1
    log = []
    for it in range(1, cli.iters + 1):
        net.train()
        idx = torch.multinomial(w, cli.batch, replacement=True, generator=gen).to(dev)
        x, y = _augment(xtr[idx], ytr[idx], gen)
        loss = _loss(net(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % cli.eval_every == 0 or it == cli.iters:
            net.eval()
            with torch.no_grad():
                vl = float(_loss(net(xva), yva))
            log.append({"iter": it, "train_loss": float(loss), "val_loss": vl})
            print(f"[spallseg] it={it} train={float(loss):.4f} val(R307)={vl:.4f}", flush=True)
            if vl < best:
                best, best_it = vl, it
                torch.save({"state_dict": net.state_dict(), "width": cli.width, "threshold": 0.5, "iter": it}, os.path.join(cli.out_dir, "best.pt"))

    net, thr = load_segmenter(os.path.join(cli.out_dir, "best.pt"), dev)
    metrics = {"best_iter": best_it, "best_val_loss": best, "threshold": thr, "log": log, "splits": {}}
    for split in ("train", "val", "test"):
        rows, u8, m = data[split]
        pred = predict_masks(net, u8, dev, thr)
        metrics["splits"][split] = _split_metrics(pred, m, rows["DI"].to_numpy(dtype=np.float64))
        print(f"[spallseg] {split}: {json.dumps(metrics['splits'][split])}", flush=True)
    with open(os.path.join(cli.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
