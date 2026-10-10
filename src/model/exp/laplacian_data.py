"""128×128 load / geometric augment + Laplacian split (standalone helpers).

Training uses ``PatternLoopDataset`` (CLAHE, flips, etc.). These utilities are
the explicit Laplacian-CFM preprocessing surface from the design doc.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from src.model.exp.laplacian import decompose_laplacian


def geometric_augment_128(image, p_hflip=0.5, p_vflip=0.0, rot90=True):
    """In-batch geometric aug on ``[B,C,128,128]`` (or single ``[C,H,W]``)."""
    squeeze = image.ndim == 3
    if squeeze:
        image = image.unsqueeze(0)
    if rot90 and bool(torch.rand(()) < 0.5):
        k = int(torch.randint(0, 4, ()).item())
        image = torch.rot90(image, k, dims=(-2, -1))
    if p_hflip > 0 and bool(torch.rand(()) < p_hflip):
        image = torch.flip(image, dims=[-1])
    if p_vflip > 0 and bool(torch.rand(()) < p_vflip):
        image = torch.flip(image, dims=[-2])
    return image.squeeze(0) if squeeze else image


def default_train_transform(image_size=128):
    return v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def preprocess_to_bands(x_128):
    """``x`` [B,C,128,128] → ``(x1, x2, x3)`` at 128 / 64 / 32."""
    if x_128.shape[-1] != 128 or x_128.shape[-2] != 128:
        x_128 = F.interpolate(x_128, size=(128, 128), mode="bilinear", align_corners=False)
    return decompose_laplacian(x_128)
