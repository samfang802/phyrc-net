"""ResNet18 + CBAM + DropBlock + Manifold Mixup for DI regression.

Input:  (B, 2, H, W) = concat([Grayscale_Image, Binary_Mask])
Output: (B, 1) continuous DI (no Sigmoid; values may exceed 1.0)
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torchvision.models import resnet18, ResNet18_Weights


# ---------------------------------------------------------------------------
# Attention / regularization modules
# ---------------------------------------------------------------------------


class ChannelAttention(nn.Module):
    """CBAM channel attention: AvgPool + MaxPool → shared MLP → sigmoid gate."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    """CBAM spatial attention: channel-wise Avg/Max → concat → 7×7 conv → sigmoid."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Channel attention followed by spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class DropBlock2D(nn.Module):
    """
    DropBlock: zero contiguous spatial blocks on feature maps (train-time only).

    Reference: Ghiasi et al., "DropBlock: A regularization method for
    convolutional networks" (NeurIPS 2018).
    """

    def __init__(self, drop_prob: float = 0.1, block_size: int = 7):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.block_size = max(1, int(block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.drop_prob <= 0.0 or x.numel() == 0:
            return x

        _b, _c, height, width = x.shape
        block_size = min(self.block_size, height, width)
        if block_size <= 1:
            return F.dropout2d(x, p=self.drop_prob, training=True)

        # Effective gamma so expected dropped fraction ≈ drop_prob.
        gamma = (
            self.drop_prob
            * (height * width)
            / (block_size ** 2)
            / max((height - block_size + 1) * (width - block_size + 1), 1)
        )
        gamma = float(min(max(gamma, 0.0), 1.0))

        # Sample per-sample spatial seeds (shared across channels), expand to blocks.
        seed = torch.bernoulli(
            torch.full((_b, 1, height, width), gamma, device=x.device, dtype=x.dtype)
        )
        block_mask = F.max_pool2d(
            seed,
            kernel_size=block_size,
            stride=1,
            padding=block_size // 2,
        )
        # Even kernel may grow by 1; crop back to HxW.
        block_mask = block_mask[:, :, :height, :width]

        keep_mask = 1.0 - block_mask
        normalize = keep_mask.numel() / (keep_mask.sum() + 1e-6)
        return x * keep_mask * normalize


class FocalRLoss(nn.Module):
    """
    Focal-R regression loss: Smooth L1 (Huber) × (1 + |e|)^γ.

    Emphasizes hard / high-error samples (e.g. underestimated high-DI cases).
    """

    def __init__(self, gamma: float = 2.0, beta: float = 1.0):
        super().__init__()
        self.gamma = float(gamma)
        self.beta = float(beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.reshape_as(target)
        e = (pred - target).abs()
        base = F.smooth_l1_loss(pred, target, reduction="none", beta=self.beta)
        weighted = base * (1.0 + e).pow(self.gamma)
        return weighted.mean()


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class ResNet18_CBAM_Regressor(nn.Module):
    """
    ResNet-18 + CBAM + DropBlock(layer3/4) with optional Manifold Mixup at layer3.

    Head: Linear(512 → 1), no Sigmoid (predictions may be > 1.0).
    Optional multi-task: separate DI and R_spall heads sharing the backbone.
    """

    def __init__(
        self,
        pretrained: bool = True,
        reduction: int = 16,
        dropblock_prob: float = 0.1,
        dropblock_size_l3: int = 3,
        dropblock_size_l4: int = 2,
        mixup_alpha: float = 0.2,
        in_channels: int = 2,
        multi_task: bool = False,
        nonneg_head: bool = False,
    ):
        super().__init__()
        self.mixup_alpha = float(mixup_alpha)
        self.in_channels = int(in_channels)
        self.multi_task = bool(multi_task)
        self.nonneg_head = bool(nonneg_head)
        if self.in_channels not in (1, 2, 3):
            raise ValueError(f"in_channels must be 1, 2, or 3, got {self.in_channels}")

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        # --- stem: 1ch gray, 2ch [gray, mask], or 3ch RGB ---
        old_conv1 = backbone.conv1
        if self.in_channels == 3:
            self.conv1 = old_conv1
        else:
            new_conv1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=old_conv1.out_channels,
                kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride,
                padding=old_conv1.padding,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    rgb_mean = old_conv1.weight.mean(dim=1, keepdim=True)
                    new_conv1.weight[:, 0:1] = rgb_mean
                    if self.in_channels == 2:
                        new_conv1.weight[:, 1:2].zero_()
            self.conv1 = new_conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # 64
        self.layer2 = backbone.layer2  # 128
        self.layer3 = backbone.layer3  # 256
        self.layer4 = backbone.layer4  # 512

        self.cbam1 = CBAM(64, reduction=reduction)
        self.cbam2 = CBAM(128, reduction=reduction)
        self.cbam3 = CBAM(256, reduction=reduction)
        self.cbam4 = CBAM(512, reduction=reduction)

        # DropBlock on deep stages (layer3 / layer4 outputs).
        self.dropblock3 = DropBlock2D(drop_prob=dropblock_prob, block_size=dropblock_size_l3)
        self.dropblock4 = DropBlock2D(drop_prob=dropblock_prob, block_size=dropblock_size_l4)

        self.avgpool = backbone.avgpool
        if self.multi_task:
            self.fc_di = nn.Linear(512, 1)
            self.fc_rs = nn.Linear(512, 1)
            # Alias kept so external code that looks at ``.fc`` still sees the DI head.
            self.fc = self.fc_di
        else:
            self.fc = nn.Linear(512, 1)

    def _stem(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def _embed(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(feat)
        return torch.flatten(x, 1)

    def _head(self, feat: torch.Tensor):
        emb = self._embed(feat)
        if self.multi_task:
            return self.fc_di(emb), self.fc_rs(emb)
        pred = self.fc(emb)
        if self.nonneg_head:
            pred = F.relu(pred)
        return pred

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        mixup: bool = False,
        y_rs: torch.Tensor | None = None,
    ):
        """
        Args:
            x: (B, C, H, W) with C=1 (gray), C=2 ([gray, mask]), or C=3 (RGB)
            y: (B, 1) DI labels; required when mixup=True in training
            y_rs: (B, 1) R_spall labels; mixed with the same λ when multi_task
            mixup: enable Manifold Mixup at layer3 latent space (train only)

        Returns:
            eval / no-mixup: pred (B, 1)  or  (pred_di, pred_rs) if multi_task
            train + mixup:   (pred, y_a, y_b, lam) or
                             (pred, y_a, y_b, y_rs_a, y_rs_b, lam) if multi_task
        """
        x = self._stem(x)
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.dropblock3(x)

        do_mixup = bool(mixup) and self.training and y is not None and self.mixup_alpha > 0.0
        y_a = y_b = lam = None
        y_rs_a = y_rs_b = None
        if do_mixup:
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            lam = max(lam, 1.0 - lam)
            index = torch.randperm(x.size(0), device=x.device)
            x = lam * x + (1.0 - lam) * x[index]
            y_a = y
            y_b = y[index]
            if self.multi_task and y_rs is not None:
                y_rs_a = y_rs
                y_rs_b = y_rs[index]

        x = self.cbam4(self.layer4(x))
        x = self.dropblock4(x)
        pred = self._head(x)

        if do_mixup:
            if self.multi_task:
                return pred, y_a, y_b, y_rs_a, y_rs_b, lam
            return pred, y_a, y_b, lam
        return pred


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class ResNet18CBAMLightning(pl.LightningModule):
    """
    Lightning module for DI regression, optional dual-head R_spall.

    Batch: (image, exp, DI[, R_spall])
    Train: Manifold Mixup + Focal-R (both heads when multi_task)
    Val:   plain forward; log R² for DI and R_spall
    """

    def __init__(
        self,
        lr=1e-4,
        pretrained=True,
        reduction=16,
        result_dir="results",
        figure_last_n_epochs=100,
        dropblock_prob=0.1,
        dropblock_size_l3=3,
        dropblock_size_l4=2,
        mixup_alpha=0.2,
        mixup_prob=1.0,
        focal_gamma=2.0,
        focal_beta=1.0,
        use_mask=True,
        multi_task=False,
        rs_loss_weight=1.0,
        nonneg_head=False,
        target_mode="di",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs", "result_dir"])
        self.lr = float(lr)
        self.result_dir = result_dir
        self.figure_last_n_epochs = max(0, int(figure_last_n_epochs))
        self.mixup_prob = float(mixup_prob)
        self.multi_task = bool(multi_task)
        self.rs_loss_weight = float(rs_loss_weight)
        self.target_mode = str(target_mode)
        in_channels = 2 if bool(use_mask) else 1
        auto_nonneg = bool(nonneg_head) or self.target_mode == "rs"

        self.net = ResNet18_CBAM_Regressor(
            pretrained=bool(pretrained),
            reduction=int(reduction),
            dropblock_prob=float(dropblock_prob),
            dropblock_size_l3=int(dropblock_size_l3),
            dropblock_size_l4=int(dropblock_size_l4),
            mixup_alpha=float(mixup_alpha),
            in_channels=in_channels,
            multi_task=self.multi_task,
            nonneg_head=auto_nonneg and not self.multi_task,
        )
        self.criterion = FocalRLoss(gamma=float(focal_gamma), beta=float(focal_beta))
        empty = {
            "loss": [],
            "prediction": [],
            "gt": [],
            "prediction_rs": [],
            "gt_rs": [],
            "exp": [],
        }
        self.training_outputs = {k: list(v) for k, v in empty.items()}
        self.validation_outputs = {k: list(v) for k, v in empty.items()}

    def forward(self, image_2ch, y=None, mixup=False, y_rs=None):
        return self.net(image_2ch, y=y, mixup=mixup, y_rs=y_rs)

    def _split_pred(self, out):
        if self.multi_task:
            pred_di, pred_rs = out
            return pred_di, pred_rs
        return out, None

    def _unpack_batch(self, batch):
        if self.multi_task:
            image_2ch, exp, di, rs = batch
            return image_2ch, exp, di, rs
        image_2ch, exp, di = batch[:3]
        return image_2ch, exp, di, None

    def _task_loss(self, pred_di, di, pred_rs=None, rs=None):
        loss_di = self.criterion(pred_di, di)
        if self.multi_task and pred_rs is not None and rs is not None:
            loss_rs = self.criterion(pred_rs, rs)
            return loss_di + self.rs_loss_weight * loss_rs, loss_di, loss_rs
        return loss_di, loss_di, pred_di.new_zeros(())

    def training_step(self, batch, batch_idx):
        image_2ch, exp, di, rs = self._unpack_batch(batch)
        use_mixup = (
            self.mixup_prob > 0.0
            and self.net.mixup_alpha > 0.0
            and (torch.rand(()) < self.mixup_prob).item()
        )

        if use_mixup:
            if self.multi_task:
                pred, y_a, y_b, rs_a, rs_b, lam = self.forward(
                    image_2ch, y=di, y_rs=rs, mixup=True
                )
                pred_di, pred_rs = self._split_pred(pred)
                loss_a, _, _ = self._task_loss(pred_di, y_a, pred_rs, rs_a)
                loss_b, _, _ = self._task_loss(pred_di, y_b, pred_rs, rs_b)
                loss = lam * loss_a + (1.0 - lam) * loss_b
                target_log = lam * y_a + (1.0 - lam) * y_b
                rs_log = None if rs_a is None else lam * rs_a + (1.0 - lam) * rs_b
            else:
                pred, y_a, y_b, lam = self.forward(image_2ch, y=di, mixup=True)
                pred_di, pred_rs = self._split_pred(pred)
                loss = lam * self.criterion(pred_di, y_a) + (1.0 - lam) * self.criterion(
                    pred_di, y_b
                )
                target_log = lam * y_a + (1.0 - lam) * y_b
                rs_log = None
        else:
            pred = self.forward(image_2ch, mixup=False)
            pred_di, pred_rs = self._split_pred(pred)
            loss, _, _ = self._task_loss(pred_di, di, pred_rs, rs)
            target_log = di
            rs_log = rs

        self.training_outputs["loss"].append(loss.detach())
        self.training_outputs["prediction"].append(pred_di.detach())
        self.training_outputs["gt"].append(target_log.detach())
        if self.multi_task and pred_rs is not None and rs_log is not None:
            self.training_outputs["prediction_rs"].append(pred_rs.detach())
            self.training_outputs["gt_rs"].append(rs_log.detach())
        if isinstance(exp, (list, tuple)):
            self.training_outputs["exp"] += list(exp)
        else:
            self.training_outputs["exp"] += [str(e) for e in exp]
        return loss

    def validation_step(self, batch, batch_idx):
        image_2ch, exp, di, rs = self._unpack_batch(batch)
        pred_di, pred_rs = self._split_pred(self.forward(image_2ch, mixup=False))
        loss, _, _ = self._task_loss(pred_di, di, pred_rs, rs)
        self.validation_outputs["loss"].append(loss.detach())
        self.validation_outputs["prediction"].append(pred_di.detach())
        self.validation_outputs["gt"].append(di.detach())
        if self.multi_task and pred_rs is not None:
            self.validation_outputs["prediction_rs"].append(pred_rs.detach())
            self.validation_outputs["gt_rs"].append(rs.detach())
        if isinstance(exp, (list, tuple)):
            self.validation_outputs["exp"] += list(exp)
        else:
            self.validation_outputs["exp"] += [str(e) for e in exp]

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_train_epoch_start(self):
        self.log("Learning rate", self.trainer.optimizers[0].param_groups[0]["lr"])

    def _should_make_figure(self) -> bool:
        if self.figure_last_n_epochs <= 0:
            return False
        max_epochs = int(getattr(self.trainer, "max_epochs", 0) or 0)
        if max_epochs <= 0:
            return True
        return int(self.current_epoch) >= max(0, max_epochs - self.figure_last_n_epochs)

    def _empty_bucket(self):
        return {
            "loss": [],
            "prediction": [],
            "gt": [],
            "prediction_rs": [],
            "gt_rs": [],
            "exp": [],
        }

    def _finalize_epoch(self, bucket, split: str):
        loss = torch.stack(bucket["loss"]).mean()
        prediction = torch.cat(bucket["prediction"]).cpu().numpy().reshape(-1)
        gt = torch.cat(bucket["gt"]).cpu().numpy().reshape(-1)
        r2 = float(r2_score(gt, prediction)) if len(gt) > 1 else float("nan")

        self.log(f"{split}/loss", loss, prog_bar=True)
        self.log(f"{split}/r2", r2, prog_bar=True)

        r2_rs = float("nan")
        pred_rs = gt_rs = None
        if self.multi_task and bucket.get("prediction_rs"):
            pred_rs = torch.cat(bucket["prediction_rs"]).cpu().numpy().reshape(-1)
            gt_rs = torch.cat(bucket["gt_rs"]).cpu().numpy().reshape(-1)
            if len(gt_rs) > 1:
                r2_rs = float(r2_score(gt_rs, pred_rs))
            self.log(f"{split}/r2_rs", r2_rs, prog_bar=True)

        fig_path = None
        if self._should_make_figure():
            fig = self.draw(
                gt,
                prediction,
                bucket["exp"],
                pred_rs=pred_rs,
                gt_rs=gt_rs,
                target_name="R_spall" if str(getattr(self, "target_mode", "di")) == "rs" else "DI",
                target_lim=0.75 if str(getattr(self, "target_mode", "di")) == "rs" else 1.4,
            )
            fig_dir = os.path.join(self.result_dir, "figures")
            os.makedirs(fig_dir, exist_ok=True)
            fig_path = os.path.join(fig_dir, f"{split}_epoch_{self.current_epoch:05d}.png")
            fig.savefig(fig_path, dpi=120, bbox_inches="tight")
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log({f"figure/{split}": wandb.Image(fig)})
            except Exception:
                pass
            plt.close(fig)

        msg = (
            f"[ResNet18CBAM] epoch={self.current_epoch} "
            f"{split}/loss={float(loss):.6f} {split}/r2={r2:.4f}"
        )
        if self.multi_task and not np.isnan(r2_rs):
            msg += f" {split}/r2_rs={r2_rs:.4f}"
        if fig_path:
            msg += f" fig={fig_path}"
        print(msg, flush=True)

    def on_train_epoch_end(self):
        self._finalize_epoch(self.training_outputs, "train")
        self.training_outputs = self._empty_bucket()

    def on_validation_epoch_end(self):
        self._finalize_epoch(self.validation_outputs, "validation")
        self.validation_outputs = self._empty_bucket()

    @staticmethod
    def draw(gt, prediction, exp, pred_rs=None, gt_rs=None, target_name="DI", target_lim=1.4):
        exp = np.asarray(exp)
        dual = pred_rs is not None and gt_rs is not None
        if dual:
            fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
            panels = [
                (axes[0], gt, prediction, "DI", 1.4),
                (axes[1], gt_rs, pred_rs, "R_spall", 0.75),
            ]
        else:
            fig, ax = plt.subplots()
            panels = [(ax, gt, prediction, target_name, target_lim)]
        for ax, y_true, y_hat, name, lim in panels:
            ax.axline((0, 0), slope=1, color="black")
            for g in np.unique(exp):
                i = np.where(exp == g)[0]
                ax.scatter(y_true[i], y_hat[i], label=str(g), s=3)
            ax.set_xlabel(f"Ground Truth {name}")
            ax.set_ylabel(f"Predicted {name}")
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.legend(fontsize=7)
            ax.set_title(f"{name} regression scatter")
        fig.tight_layout()
        return fig
