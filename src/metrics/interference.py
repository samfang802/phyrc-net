"""Interference (fixtures in front of the column: posts, blocks, rebar hooks) segmentation.

DINOv3 backbone with the last few blocks fine-tuned, plus a small convolutional decoder that
outputs a 128×128 mask. Trained on the SAM masks in Data/DamagePattern/mask, which are the same
masks that produced the inpainted reference set, so a detected region on a generated image is
"what would have been inpainted on a real one".

Used for two things:
  - interference rate of generated images (a defect reported on its own)
  - masks for inpainting generated images with the same CFM inpainter as the reference

The checkpoint holds every backbone weight, so loading only needs the backbone's architecture
config: from the checkpoint ("backbone_config"), else from backbone_config.json next to it,
else from Hugging Face (gated for DINOv3). The first two need no Hugging Face access.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics.dino_mmd import DEFAULT_DINO, IMAGENET_MEAN, IMAGENET_STD


class InterferenceSegmenter(nn.Module):
    def __init__(self, model_id=DEFAULT_DINO, input_size=448, out_size=128, n_train_blocks=2, width=128,
                 backbone_config=None):
        """backbone_config: architecture dict (a config.json); given, the backbone is built with
        random weights and no download, for load_segmenter to fill from the checkpoint."""
        super().__init__()
        from transformers import AutoConfig, AutoModel

        if backbone_config is None:
            self.backbone = AutoModel.from_pretrained(model_id)
        else:
            cfg = dict(backbone_config)
            self.backbone = AutoModel.from_config(AutoConfig.for_model(cfg.pop("model_type"), **cfg))
        self.model_id = model_id
        self.input_size = int(input_size)
        self.out_size = int(out_size)
        self.n_train_blocks = int(n_train_blocks)
        cfg = self.backbone.config
        self.patch = int(cfg.patch_size)
        self.n_prefix = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)
        dim = int(cfg.hidden_size)

        # Freeze everything, then unfreeze the last n blocks and the final norm.
        for p in self.backbone.parameters():
            p.requires_grad = False
        if self.n_train_blocks > 0:
            for blk in self._layers()[-self.n_train_blocks :]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

        self.head = nn.Sequential(
            nn.Conv2d(dim, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )
        # Image skip: thin structures (hook outline) are sharper at pixel level than at 16 px tokens.
        self.skip = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 16, 3, padding=1), nn.GELU())
        self.out = nn.Sequential(nn.Conv2d(width + 16, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def _layers(self):
        # HF DINOv3ViTModel keeps its blocks at .model.layer
        return list(self.backbone.model.layer)

    def forward(self, x01):
        """x01: (B,1,H,W) gray in [0,1] at any size. Returns logits (B,1,out,out)."""
        x = F.interpolate(x01, size=(self.input_size, self.input_size), mode="bicubic", antialias=True, align_corners=False)
        x = (x.clamp(0, 1).repeat(1, 3, 1, 1) - self.mean) / self.std
        h = self.backbone(pixel_values=x).last_hidden_state[:, self.n_prefix :]
        s = self.input_size // self.patch
        h = h.transpose(1, 2).reshape(h.shape[0], -1, s, s)
        h = self.head(h)
        h = F.interpolate(h, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        img = F.interpolate(x01, size=(self.out_size, self.out_size), mode="bilinear", antialias=True, align_corners=False)
        return self.out(torch.cat([h, self.skip(img)], dim=1))

    def trainable_groups(self, lr_head, lr_backbone):
        bb = [p for p in self.backbone.parameters() if p.requires_grad]
        hd = list(self.head.parameters()) + list(self.skip.parameters()) + list(self.out.parameters())
        groups = [{"params": hd, "lr": lr_head}]
        if bb:
            groups.append({"params": bb, "lr": lr_backbone})
        return groups

    def config(self):
        return {
            "model_id": self.model_id,
            "input_size": self.input_size,
            "out_size": self.out_size,
            "n_train_blocks": self.n_train_blocks,
        }


def backbone_config_dict(model):
    return model.backbone.config.to_diff_dict()


def save_segmenter(model, path, extra=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"config": model.config(), "backbone_config": backbone_config_dict(model),
                "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_segmenter(path, device="cuda"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    bcfg = payload.get("backbone_config")
    sidecar = os.path.join(os.path.dirname(os.path.abspath(path)), "backbone_config.json")
    if bcfg is None and os.path.isfile(sidecar):
        with open(sidecar) as f:
            bcfg = json.load(f)
    model = InterferenceSegmenter(**payload["config"], backbone_config=bcfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload.get("extra", {})


@torch.no_grad()
def predict_prob(model, u8, batch_size=32, device="cuda", tta_flip=True):
    """u8 (N,H,W) → probability (N,out,out) float32. Horizontal-flip TTA averages two passes."""
    out = []
    for i in range(0, len(u8), batch_size):
        x = torch.from_numpy(np.ascontiguousarray(u8[i : i + batch_size])).to(device).float().div(255.0)[:, None]
        p = torch.sigmoid(model(x))
        if tta_flip:
            p = 0.5 * (p + torch.sigmoid(model(x.flip(-1))).flip(-1))
        out.append(p[:, 0].cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def clean_mask(prob, thr=0.5, min_area=150, border_px=3):
    """Threshold, then keep components that look like a fixture.

    min_area: at 128×128, every SAM component of val/test is ≥ 339 px (train: 95% ≥ 143 px), while
        90% of the flags on our single-frame models' samples are < 150 px dark rubble specks.
    border_px: every SAM component (874/874) touches the bottom, left or right border within 3 px;
        fixtures stand in front of the column base. None disables the rule.
    """
    import cv2

    masks = np.zeros(prob.shape, dtype=np.uint8)
    for i, p in enumerate(prob):
        m = (p > thr).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        H, W = m.shape
        for j in range(1, n):
            x, y, w, h, area = stats[j]
            if area < min_area:
                continue
            if border_px is not None and min(H - (y + h), x, W - (x + w)) > border_px:
                continue
            masks[i][lab == j] = 1
    return masks


def load_sam_masks(filenames, mask_dir="Data/DamagePattern/mask", size=128):
    """SAM interference masks resized to size×size (area-averaged, then >0.5)."""
    import cv2
    from PIL import Image

    out = np.zeros((len(filenames), size, size), dtype=np.uint8)
    for i, n in enumerate(filenames):
        stem = os.path.splitext(os.path.basename(str(n)))[0]
        m = np.array(Image.open(os.path.join(mask_dir, f"{stem}_mask.png")).convert("L"), dtype=np.float32) / 255.0
        out[i] = (cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
    return out


def mask_stats(pred, gt):
    """IoU / Dice over all pixels, plus image-level detection precision and recall."""
    p, g = pred.astype(bool), gt.astype(bool)
    inter, union = (p & g).sum(), (p | g).sum()
    tot = p.sum() + g.sum()
    det_p, det_g = p.reshape(len(p), -1).any(1), g.reshape(len(g), -1).any(1)
    tp = (det_p & det_g).sum()
    return {
        "iou": float(inter / max(union, 1)),
        "dice": float(2 * inter / max(tot, 1)),
        "image_precision": float(tp / max(det_p.sum(), 1)),
        "image_recall": float(tp / max(det_g.sum(), 1)),
        "frac_images_flagged": float(det_p.mean()),
    }
