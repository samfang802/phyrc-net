"""Image loading and normalization for damage quantification."""

import numpy as np
import torch
from PIL import Image

IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)


def build_quant_image_tensor(
    path,
    image_size,
    use_rgb=False,
    use_clahe=False,
    gray_mean=0.485,
    gray_std=0.229,
    rgb_mean=IMAGENET_RGB_MEAN,
    rgb_std=IMAGENET_RGB_STD,
):
    """Load a damage photo as CHW float tensor (ImageNet-normalized)."""
    from ..utils import ApplyCLAHE

    image_size = int(image_size)
    if use_rgb:
        image = Image.open(path).convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        rgb = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        if use_clahe:
            rgb = ApplyCLAHE(preserve_color=True)(rgb)
        mean = torch.tensor(rgb_mean, dtype=rgb.dtype).view(3, 1, 1)
        std = torch.tensor(rgb_std, dtype=rgb.dtype).view(3, 1, 1)
        return (rgb - mean) / std.clamp_min(1e-6)

    image = Image.open(path).convert("L")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    gray = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)
    if use_clahe:
        gray = ApplyCLAHE(preserve_color=False)(gray)
    return (gray - float(gray_mean)) / max(float(gray_std), 1e-6)
