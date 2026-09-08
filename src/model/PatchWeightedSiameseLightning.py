"""Lightning trainer: patch-weighted Siamese with original Focal-R + pairwise gap."""

from __future__ import annotations

import os

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score

from src.loss.dynamic_margin_ranking import DynamicMarginRankingLoss
from src.model.PatchWeightedSiameseRegressor import DualBranchPatchWeightedSiamese
from src.model.ResNet18_Siamese_Lightning import ResNet18SiameseLightning


class PatchWeightedSiameseLightning(pl.LightningModule):
    """Independent shared-backbone pair model; same loss as GAP Siamese."""

    def __init__(
        self,
        lr=1e-4,
        pretrained=True,
        reduction=16,
        result_dir="results",
        figure_last_n_epochs=1,
        lambda_rank=1.0,
        focal_gamma=2.0,
        focal_beta=1.0,
        loss_scale=100.0,
        smooth_l1_beta=None,
        target_mode="di",
        use_mask=False,
        **kwargs,
    ):
        super().__init__()
        if bool(use_mask):
            raise ValueError("PatchWeightedSiamese is 1-channel grayscale only; pass --no-use_mask")
        self.save_hyperparameters(ignore=["kwargs", "result_dir"])
        self.lr = float(lr)
        self.result_dir = result_dir
        self.figure_last_n_epochs = max(0, int(figure_last_n_epochs))
        self.target_mode = str(target_mode)
        self.loss_scale = float(loss_scale)
        beta = float(focal_beta if smooth_l1_beta is None else smooth_l1_beta)
        self.net = DualBranchPatchWeightedSiamese(
            pretrained=bool(pretrained),
            reduction=int(reduction),
        )
        self.criterion = DynamicMarginRankingLoss(
            lambda_rank=float(lambda_rank),
            gamma=float(focal_gamma),
            beta=beta,
            loss_scale=self.loss_scale,
        )
        self.training_outputs = {
            "loss": [],
            "loss_abs": [],
            "loss_pair": [],
            "prediction": [],
            "gt": [],
            "exp": [],
        }

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img_a, img_b, gt_a, gt_b, exp = batch[:5]
        imgs = torch.cat([img_a, img_b], dim=0)
        pred_a, pred_b = self.net(imgs).chunk(2, dim=0)
        loss, loss_abs, loss_pair = self.criterion(pred_a, pred_b, gt_a, gt_b)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/loss_abs", loss_abs, prog_bar=False, on_step=True, on_epoch=False)
        self.log("train/loss_pair", loss_pair, prog_bar=True, on_step=True, on_epoch=False)

        self.training_outputs["loss"].append(loss.detach())
        self.training_outputs["loss_abs"].append(loss_abs)
        self.training_outputs["loss_pair"].append(loss_pair)
        self.training_outputs["prediction"].append(
            torch.cat([pred_a.detach(), pred_b.detach()], dim=0)
        )
        self.training_outputs["gt"].append(torch.cat([gt_a.detach(), gt_b.detach()], dim=0))
        if isinstance(exp, (list, tuple)):
            self.training_outputs["exp"] += list(exp) + list(exp)
        else:
            self.training_outputs["exp"] += [str(e) for e in exp] * 2
        return loss

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

    def on_train_epoch_end(self):
        bucket = self.training_outputs
        loss = torch.stack(bucket["loss"]).mean()
        prediction = torch.cat(bucket["prediction"]).cpu().numpy().reshape(-1)
        gt = torch.cat(bucket["gt"]).cpu().numpy().reshape(-1)
        r2 = float(r2_score(gt, prediction)) if len(gt) > 1 else float("nan")

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/r2", r2, prog_bar=True)
        self.log("train/loss_abs_epoch", torch.stack(bucket["loss_abs"]).mean())
        self.log("train/loss_pair_epoch", torch.stack(bucket["loss_pair"]).mean())

        fig_path = None
        if self._should_make_figure():
            fig = ResNet18SiameseLightning.draw(
                gt, prediction, bucket["exp"], target_name=self.target_mode.upper()
            )
            fig_dir = os.path.join(self.result_dir, "figures")
            os.makedirs(fig_dir, exist_ok=True)
            fig_path = os.path.join(fig_dir, f"train_epoch_{self.current_epoch:05d}.png")
            fig.savefig(fig_path, dpi=120, bbox_inches="tight")
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log({"figure/train": wandb.Image(fig)})
            except Exception:
                pass
            plt.close(fig)

        msg = (
            f"[PatchWeightedSiamese] epoch={self.current_epoch} "
            f"train/loss={float(loss):.6f} train/r2={r2:.4f}"
        )
        if fig_path:
            msg += f" fig={fig_path}"
        print(msg, flush=True)

        self.training_outputs = {
            "loss": [],
            "loss_abs": [],
            "loss_pair": [],
            "prediction": [],
            "gt": [],
            "exp": [],
        }
