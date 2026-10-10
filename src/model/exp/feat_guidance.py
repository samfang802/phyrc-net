"""Classifier-style feature guidance at ODE steps.

HysGRU cannot encode images. Frozen Siamese ResNet18 `_embed` (512-d) is the
visual encoder. A ridge map ``W: 512 → physics_dim`` is fit on split GT pairs
so ``f̃ = embed(x̂₁) @ W`` lives in the same space as the condition vector ``g``.
Loss is cosine or L2 vs ``g``; ``∇_{x_t} L`` is added into velocity like CG.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def siamese_embed(siamese, x_imagenet_gray: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) ImageNet-gray → (B,512) pooled embedding (trainable graph)."""
    net = siamese.net
    h = net._stem(x_imagenet_gray)
    h = net.cbam1(net.layer1(h))
    h = net.cbam2(net.layer2(h))
    h = net.cbam3(net.layer3(h))
    h = net.cbam4(net.layer4(h))
    return net._embed(h)


def target_space_to_imagenet_gray(img_m11: torch.Tensor) -> torch.Tensor:
    """CFM target space [-1,1] (1 or 3 ch) → ImageNet-normalized grayscale."""
    vis = img_m11.clamp(-1.0, 1.0) * 0.5 + 0.5
    if vis.shape[1] >= 3:
        gray = 0.299 * vis[:, 0:1] + 0.587 * vis[:, 1:2] + 0.114 * vis[:, 2:3]
    else:
        gray = vis[:, :1]
    return (gray - 0.485) / 0.229


class SiameseGruFeatGuide:
    """``L(x̂₁, g)`` for classifier-style guidance."""

    def __init__(
        self,
        siamese,
        W: Optional[torch.Tensor] = None,
        metric: str = "cosine",
    ):
        self.siamese = siamese
        self.W = W
        metric = str(metric).lower()
        if metric not in ("cosine", "l2"):
            raise ValueError(f"metric must be cosine|l2, got {metric!r}")
        self.metric = metric
        siamese.eval()
        for p in siamese.parameters():
            p.requires_grad_(False)

    def embed_image(self, img_m11: torch.Tensor) -> torch.Tensor:
        x = target_space_to_imagenet_gray(img_m11)
        size = int(getattr(self.siamese, "image_size", 128) or 128)
        if x.shape[-2] != size or x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        in_ch = int(getattr(self.siamese.net, "in_channels", 1) or 1)
        if in_ch == 3 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return siamese_embed(self.siamese, x)

    def map_to_gru(self, vis_feat: torch.Tensor) -> torch.Tensor:
        if self.W is None:
            raise RuntimeError("ridge map W is not fitted")
        return vis_feat @ self.W.to(device=vis_feat.device, dtype=vis_feat.dtype)

    def __call__(self, img_m11: torch.Tensor, gru_feat: torch.Tensor) -> torch.Tensor:
        f_tilde = self.map_to_gru(self.embed_image(img_m11))
        g = gru_feat
        while g.ndim > 2:
            g = g.squeeze(1)
        if g.ndim == 1:
            g = g.unsqueeze(0)
        if self.metric == "cosine":
            f_n = F.normalize(f_tilde, dim=-1, eps=1e-8)
            g_n = F.normalize(g, dim=-1, eps=1e-8)
            return (1.0 - (f_n * g_n).sum(dim=-1)).mean()
        return (f_tilde - g).pow(2).mean()


def fit_embed_to_gru_ridge(
    vis_feat: torch.Tensor,
    gru_feat: torch.Tensor,
    ridge: float = 10.0,
) -> torch.Tensor:
    """Least-squares ``vis @ W ≈ g`` with ridge. vis [N,512], g [N,D] → W [512,D]."""
    v = vis_feat.detach().float().cpu()
    g = gru_feat.detach().float().cpu()
    if v.ndim != 2 or g.ndim != 2 or v.shape[0] != g.shape[0]:
        raise ValueError(f"need [N,512] and [N,D], got {tuple(v.shape)} {tuple(g.shape)}")
    n, d_in = v.shape
    eye = torch.eye(d_in, dtype=v.dtype)
    a = v.t() @ v + float(ridge) * eye
    b = v.t() @ g
    w = torch.linalg.solve(a, b)
    return w


def collect_ridge_pairs(model, guide: SiameseGruFeatGuide, pool, device, ridge: float = 10.0) -> torch.Tensor:
    vis_rows = []
    g_rows = []
    with torch.no_grad():
        for item in pool:
            real01 = item["real"].unsqueeze(0).to(device=device, dtype=torch.float32)
            img_m11 = real01 * 2.0 - 1.0
            vis_rows.append(guide.embed_image(img_m11).cpu())
            hidden = item["hidden_state"].unsqueeze(0).to(device=device, dtype=torch.float32)
            g = model.process_physics_condition(hidden)
            while g.ndim > 2:
                g = g.squeeze(1)
            g_rows.append(g.detach().cpu())
    vis = torch.cat(vis_rows, dim=0)
    g = torch.cat(g_rows, dim=0)
    return fit_embed_to_gru_ridge(vis, g, ridge=ridge)
