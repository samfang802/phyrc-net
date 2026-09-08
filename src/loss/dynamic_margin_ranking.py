"""Joint absolute + pairwise gap regression loss (Focal-R) for Siamese DI learning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicMarginRankingLoss(nn.Module):
    """
    L_total = L_abs + lambda_rank * L_pair

    Before Focal-R, scale Pred/GT by ``loss_scale`` (default 100) so typical
    DI errors become |e|≫1 and (1+|e|)^γ actually amplifies gradients.

    L_abs  = FocalR(Pred_A·s, GT_A·s) + FocalR(Pred_B·s, GT_B·s)
    L_pair = FocalR((Pred_B-Pred_A)·s, (GT_B-GT_A)·s)

    Optional per-sample ``weight_a`` / ``weight_b`` reweight L_abs (and the
    pair term via their mean). Focal-R = SmoothL1 × (1 + |e|)^γ.
    """

    def __init__(
        self,
        lambda_rank: float = 0.5,
        gamma: float = 2.0,
        beta: float = 1.0,
        loss_scale: float = 100.0,
    ):
        super().__init__()
        self.lambda_rank = float(lambda_rank)
        self.loss_scale = float(loss_scale)
        self.gamma = float(gamma)
        self.beta = float(beta)

    def _focal_per(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.reshape_as(target)
        e = (pred - target).abs()
        base = F.smooth_l1_loss(pred, target, reduction="none", beta=self.beta)
        return base * (1.0 + e).pow(self.gamma)

    @staticmethod
    def _weighted_mean(per: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        if weight is None:
            return per.mean()
        w = weight.reshape_as(per)
        return (per * w).sum() / w.sum().clamp_min(1e-6)

    def forward(self, pred_a, pred_b, gt_a, gt_b, weight_a=None, weight_b=None):
        pred_a = pred_a.reshape_as(gt_a) * self.loss_scale
        pred_b = pred_b.reshape_as(gt_b) * self.loss_scale
        gt_a_s = gt_a * self.loss_scale
        gt_b_s = gt_b * self.loss_scale

        per_a = self._focal_per(pred_a, gt_a_s)
        per_b = self._focal_per(pred_b, gt_b_s)
        loss_abs = self._weighted_mean(per_a, weight_a) + self._weighted_mean(per_b, weight_b)

        per_pair = self._focal_per(pred_b - pred_a, gt_b_s - gt_a_s)
        if weight_a is None and weight_b is None:
            loss_pair = per_pair.mean()
        else:
            wa = torch.ones_like(gt_a) if weight_a is None else weight_a
            wb = torch.ones_like(gt_b) if weight_b is None else weight_b
            loss_pair = self._weighted_mean(per_pair, 0.5 * (wa + wb))

        loss_total = loss_abs + self.lambda_rank * loss_pair
        return loss_total, loss_abs.detach(), loss_pair.detach()
