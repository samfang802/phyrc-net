"""Independent patch-weighted Siamese DI regressor.

Shared ResNet18 (layer4 stride=1 → 8×8 at 128²), CBAM, then per-image
score/weight heads. Each image is scored on its own features (no pair concat).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from src.model.ResNet18_CBAM_Regressor import CBAM


def _adapt_conv1_to_gray(old_conv1: nn.Conv2d, pretrained: bool) -> nn.Conv2d:
    """Replace RGB stem conv with 1-channel grayscale, copying RGB-mean weights."""
    new_conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=False,
    )
    if pretrained:
        with torch.no_grad():
            new_conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
    return new_conv1


def _set_layer4_stride1(layer4: nn.Sequential) -> None:
    """Keep spatial size after layer3 (8×8 for 128² input) by not downsampling in layer4."""
    first = layer4[0]
    first.conv1.stride = (1, 1)
    if first.downsample is not None:
        first.downsample[0].stride = (1, 1)


class DualBranchPatchWeightedSiamese(nn.Module):
    """Shared-weight Siamese: independent patch DI for each image.

    Same protocol as the GAP Siamese: ``forward`` takes a batched gray tensor
    ``(B, 1, H, W)`` and returns ``(B, 1)``. Pair training concatenates A/B
    on batch then chunks.

    Shapes (H=W=128)::

        img            : (B, 1, 128, 128)
        backbone feat  : (B, 512, 8, 8)
        score          : (B, 1, 8, 8)  ReLU, unbounded above
        weight         : (B, 1, 8, 8)  Sigmoid in [0, 1]
        di             : (B, 1)  = sum(score * weight) / (sum(weight) + 1e-6)  (≥ 0, no cap)
    """

    def __init__(self, pretrained: bool = True, reduction: int = 16):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.conv1 = _adapt_conv1_to_gray(backbone.conv1, pretrained=bool(pretrained))
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        _set_layer4_stride1(self.layer4)

        self.cbam = CBAM(512, reduction=int(reduction))
        self.score_head = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.weight_head = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Shared backbone: (B, 1, H, W) → (B, 512, H/16, W/16)  (8×8 if H=W=128)."""
        if x.ndim != 4:
            raise ValueError(f"Expected (B, C, H, W), got {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(f"Expected 1-channel grayscale, got C={x.shape[1]}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.cbam(x)

    @staticmethod
    def _aggregate(score: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        weighted = score * weight
        return weighted.sum(dim=(2, 3)) / (weight.sum(dim=(2, 3)) + 1e-6)

    def forward(self, x: torch.Tensor, return_maps: bool = False):
        feat = self.extract_features(x)
        score = self.score_head(feat)
        weight = self.weight_head(feat)
        di = self._aggregate(score, weight)
        if return_maps:
            return di, score, weight
        return di
