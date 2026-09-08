"""Siamese ResNet18-CBAM Lightning module with pairwise Focal-R (+ optional Rs head)."""

from __future__ import annotations

import os

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

from src.loss.dynamic_margin_ranking import DynamicMarginRankingLoss
from src.model.ResNet18_CBAM_Regressor import FocalRLoss, ResNet18_CBAM_Regressor


class ResNet18SiameseLightning(pl.LightningModule):
    """
    Shared-weight Siamese DI regressor.

    Train batch:
      - single-task: (Img_A, Img_B, GT_A, GT_B, exp)
      - multi-task:  (+ Rs_A, Rs_B)

    Multi-task loss (when enabled)::

        L_DI  = Focal-R abs + λ rank   (Pred/GT × loss_scale)
        L_Rs  = weighted MSE(Pred_Rs × rs_loss_scale, GT_Rs × rs_loss_scale)
        L     = uncertainty_weight(L_DI, L_Rs)   or   L_DI + α L_Rs

    Val batch: (image, exp, DI[, Rs]) single-image regression metrics.
    """

    def __init__(
        self,
        lr=1e-4,
        pretrained=True,
        reduction=16,
        result_dir="results",
        figure_last_n_epochs=1,
        dropblock_prob=0.0,
        dropblock_size_l3=3,
        dropblock_size_l4=2,
        lambda_rank=0.5,
        focal_gamma=2.0,
        focal_beta=1.0,
        loss_scale=100.0,
        # Backward-compat alias used by older args.json / jobs.
        smooth_l1_beta=None,
        use_mask=False,
        use_rgb=False,
        multi_task=False,
        rs_loss_weight=1.0,
        rs_loss_scale=None,
        rs_pos_weight=4.0,
        uncertainty_weighting=True,
        rs_warmup_epochs=5,
        target_mode="di",
        # Emphasize hard-for-appearance samples: low DI and/or Rs=0.
        di_low_thresh=0.3,
        di_low_weight=1.0,
        rs0_weight=1.0,
        rs0_eps=1e-6,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs", "result_dir"])
        self.lr = float(lr)
        self.result_dir = result_dir
        self.figure_last_n_epochs = max(0, int(figure_last_n_epochs))
        self.multi_task = bool(multi_task)
        self.rs_loss_weight = float(rs_loss_weight)
        self.rs_loss_scale = float(loss_scale if rs_loss_scale is None else rs_loss_scale)
        self.rs_pos_weight = float(rs_pos_weight)
        self.uncertainty_weighting = bool(uncertainty_weighting) and self.multi_task
        self.rs_warmup_epochs = max(0, int(rs_warmup_epochs))
        self.target_mode = str(target_mode)
        self.di_low_thresh = float(di_low_thresh)
        self.di_low_weight = float(di_low_weight)
        self.rs0_weight = float(rs0_weight)
        self.rs0_eps = float(rs0_eps)
        if bool(use_rgb) and bool(use_mask):
            raise ValueError("use_rgb is 3-channel; do not combine with use_mask")
        if bool(use_rgb):
            in_channels = 3
        else:
            in_channels = 2 if bool(use_mask) else 1

        beta = float(focal_beta if smooth_l1_beta is None else smooth_l1_beta)
        gamma = float(focal_gamma)
        self.loss_scale = float(loss_scale)

        self.net = ResNet18_CBAM_Regressor(
            pretrained=bool(pretrained),
            reduction=int(reduction),
            dropblock_prob=float(dropblock_prob),
            dropblock_size_l3=int(dropblock_size_l3),
            dropblock_size_l4=int(dropblock_size_l4),
            mixup_alpha=0.0,  # mixup disabled for Siamese pairs
            in_channels=in_channels,
            multi_task=self.multi_task,
        )
        self.criterion = DynamicMarginRankingLoss(
            lambda_rank=float(lambda_rank),
            gamma=gamma,
            beta=beta,
            loss_scale=self.loss_scale,
        )
        self.val_criterion = FocalRLoss(gamma=gamma, beta=beta)

        # Kendall et al. learnable log-variance for task balancing.
        if self.uncertainty_weighting:
            self.log_var_di = nn.Parameter(torch.zeros(()))
            self.log_var_rs = nn.Parameter(torch.zeros(()))

        self.training_outputs = {
            "loss": [],
            "loss_abs": [],
            "loss_pair": [],
            "loss_rs": [],
            "prediction": [],
            "gt": [],
            "prediction_rs": [],
            "gt_rs": [],
            "exp": [],
        }
        self.validation_outputs = {
            "loss": [],
            "prediction": [],
            "gt": [],
            "prediction_rs": [],
            "gt_rs": [],
            "exp": [],
        }

    def forward(self, x):
        return self.net(x, mixup=False)

    def _split_pred(self, out):
        if self.multi_task:
            pred_di, pred_rs = out
            return pred_di, pred_rs
        return out, None

    def _di_sample_weight(self, gt_di: torch.Tensor, gt_rs: torch.Tensor | None) -> torch.Tensor:
        """Up-weight low-DI and Rs≈0 samples for the DI Focal-R terms."""
        w = torch.ones_like(gt_di)
        if self.di_low_weight != 1.0:
            w = torch.where(gt_di <= self.di_low_thresh, w * self.di_low_weight, w)
        if gt_rs is not None and self.rs0_weight != 1.0:
            w = torch.where(gt_rs.abs() <= self.rs0_eps, w * self.rs0_weight, w)
        return w

    def _rs_mse(self, pred_rs: torch.Tensor, gt_rs: torch.Tensor) -> torch.Tensor:
        """Scaled, positive-Rs-weighted MSE."""
        scale = self.rs_loss_scale
        err2 = (pred_rs * scale - gt_rs * scale).pow(2)
        if self.rs_pos_weight != 1.0:
            w = torch.where(
                gt_rs.reshape_as(err2) > 0,
                err2.new_full((), self.rs_pos_weight),
                err2.new_ones(()),
            )
            return (w * err2).mean()
        return err2.mean()

    def _combine_losses(self, loss_di: torch.Tensor, loss_rs: torch.Tensor) -> torch.Tensor:
        if self.uncertainty_weighting:
            # ½ e^{-s} L + ½ s   (precision weighting + regularizer)
            s_di = self.log_var_di
            s_rs = self.log_var_rs
            return (
                0.5 * torch.exp(-s_di) * loss_di
                + 0.5 * torch.exp(-s_rs) * loss_rs
                + 0.5 * (s_di + s_rs)
            )
        return loss_di + self.rs_loss_weight * loss_rs

    def training_step(self, batch, batch_idx):
        # Batch may include Rs even when multi_task is off (for DI sample weighting).
        if len(batch) >= 7:
            img_a, img_b, gt_a, gt_b, exp, rs_a, rs_b = batch
        else:
            img_a, img_b, gt_a, gt_b, exp = batch[:5]
            rs_a = rs_b = None

        assert img_a.shape == img_b.shape, (img_a.shape, img_b.shape)
        imgs = torch.cat([img_a, img_b], dim=0)
        pred_di, pred_rs = self._split_pred(self.forward(imgs))
        pred_a, pred_b = pred_di.chunk(2, dim=0)

        w_a = self._di_sample_weight(gt_a, rs_a)
        w_b = self._di_sample_weight(gt_b, rs_b)
        loss_di, loss_abs, loss_pair = self.criterion(
            pred_a, pred_b, gt_a, gt_b, weight_a=w_a, weight_b=w_b
        )
        loss_rs = pred_di.new_zeros(())
        use_rs = (
            self.multi_task
            and pred_rs is not None
            and int(self.current_epoch) >= self.rs_warmup_epochs
        )
        if self.multi_task and pred_rs is not None:
            rs_hat_a, rs_hat_b = pred_rs.chunk(2, dim=0)
            rs_pred = torch.cat([rs_hat_a, rs_hat_b], dim=0)
            rs_gt = torch.cat([rs_a, rs_b], dim=0)
            loss_rs = self._rs_mse(rs_pred, rs_gt)
            loss = self._combine_losses(loss_di, loss_rs) if use_rs else loss_di
        else:
            loss = loss_di
            rs_hat_a = rs_hat_b = None

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/loss_di", loss_di, prog_bar=False, on_step=True, on_epoch=False)
        self.log("train/loss_abs", loss_abs, prog_bar=False, on_step=True, on_epoch=False)
        self.log("train/loss_pair", loss_pair, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/di_w_mean", 0.5 * (w_a.mean() + w_b.mean()), on_step=False, on_epoch=True)
        if self.multi_task:
            self.log("train/loss_rs", loss_rs, prog_bar=True, on_step=True, on_epoch=False)
            self.log(
                "train/rs_active",
                float(use_rs),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            if self.uncertainty_weighting:
                self.log("train/log_var_di", self.log_var_di, on_step=False, on_epoch=True)
                self.log("train/log_var_rs", self.log_var_rs, on_step=False, on_epoch=True)
                self.log(
                    "train/w_di",
                    0.5 * torch.exp(-self.log_var_di),
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    "train/w_rs",
                    0.5 * torch.exp(-self.log_var_rs),
                    on_step=False,
                    on_epoch=True,
                )

        self.training_outputs["loss"].append(loss.detach())
        self.training_outputs["loss_abs"].append(loss_abs)
        self.training_outputs["loss_pair"].append(loss_pair)
        self.training_outputs["loss_rs"].append(loss_rs.detach())
        self.training_outputs["prediction"].append(
            torch.cat([pred_a.detach(), pred_b.detach()], dim=0)
        )
        self.training_outputs["gt"].append(torch.cat([gt_a.detach(), gt_b.detach()], dim=0))
        if self.multi_task:
            self.training_outputs["prediction_rs"].append(
                torch.cat([rs_hat_a.detach(), rs_hat_b.detach()], dim=0)
            )
            self.training_outputs["gt_rs"].append(torch.cat([rs_a.detach(), rs_b.detach()], dim=0))
        if isinstance(exp, (list, tuple)):
            self.training_outputs["exp"] += list(exp) + list(exp)
        else:
            self.training_outputs["exp"] += [str(e) for e in exp] * 2
        return loss

    def validation_step(self, batch, batch_idx):
        if self.multi_task:
            image, exp, di, rs = batch
        else:
            image, exp, di = batch[:3]
            rs = None

        pred_di, pred_rs = self._split_pred(self.forward(image))
        # Monitor DI quality only (don't let unstable Rs drag early-stopping).
        loss = self.val_criterion(pred_di * self.loss_scale, di * self.loss_scale)

        self.validation_outputs["loss"].append(loss.detach())
        self.validation_outputs["prediction"].append(pred_di.detach())
        self.validation_outputs["gt"].append(di.detach())
        if self.multi_task:
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

        if split == "train" and bucket.get("loss_abs"):
            self.log(
                f"{split}/loss_abs_epoch",
                torch.stack(bucket["loss_abs"]).mean(),
                prog_bar=False,
            )
            self.log(
                f"{split}/loss_pair_epoch",
                torch.stack(bucket["loss_pair"]).mean(),
                prog_bar=False,
            )
            if bucket.get("loss_rs"):
                self.log(
                    f"{split}/loss_rs_epoch",
                    torch.stack(bucket["loss_rs"]).mean(),
                    prog_bar=False,
                )

        fig_path = None
        if self._should_make_figure():
            fig = self.draw(
                gt,
                prediction,
                bucket["exp"],
                target_name=self.target_mode.upper(),
                pred_rs=pred_rs,
                gt_rs=gt_rs,
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
            f"[ResNet18Siamese] epoch={self.current_epoch} "
            f"{split}/loss={float(loss):.6f} {split}/r2={r2:.4f}"
        )
        if self.multi_task and not np.isnan(r2_rs):
            msg += f" {split}/r2_rs={r2_rs:.4f}"
        if fig_path:
            msg += f" fig={fig_path}"
        print(msg, flush=True)

    def on_train_epoch_end(self):
        self._finalize_epoch(self.training_outputs, "train")
        self.training_outputs = {
            "loss": [],
            "loss_abs": [],
            "loss_pair": [],
            "loss_rs": [],
            "prediction": [],
            "gt": [],
            "prediction_rs": [],
            "gt_rs": [],
            "exp": [],
        }

    def on_validation_epoch_end(self):
        self._finalize_epoch(self.validation_outputs, "validation")
        self.validation_outputs = {
            "loss": [],
            "prediction": [],
            "gt": [],
            "prediction_rs": [],
            "gt_rs": [],
            "exp": [],
        }

    @staticmethod
    def draw(gt, prediction, exp, target_name="DI", pred_rs=None, gt_rs=None):
        exp = np.asarray(exp)
        dual = pred_rs is not None and gt_rs is not None
        if dual:
            fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
            panels = [
                (axes[0], gt, prediction, target_name, 1.4),
                (axes[1], gt_rs, pred_rs, "R_spall", 0.75),
            ]
        else:
            fig, ax = plt.subplots()
            panels = [(ax, gt, prediction, target_name, 1.4)]
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
            ax.set_title(f"{name} regression scatter (Siamese)")
        fig.tight_layout()
        return fig
