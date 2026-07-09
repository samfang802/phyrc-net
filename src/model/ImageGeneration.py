import wandb
import os
import numpy as np
import torch.nn as nn
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


import torch
from src.model.Component.generative import DCGAN_D, DCGAN_G, weights_init
from src.model.Component.cfm_adm import ADMUNetCFM
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchvision.transforms import v2
from sklearn.metrics import r2_score


def build_generation_condition(
    info,
    hidden_state,
    direction,
    DI,
    latent_conditioning,
    direction_conditioning,
    include_di_conditioning,
):
    if latent_conditioning:
        parts = [hidden_state]
        if direction_conditioning:
            parts.append(direction)
        if include_di_conditioning:
            parts.append(DI)
        return torch.cat(parts, dim=1)

    parts = [info]
    if direction_conditioning:
        parts.append(direction)
    parts.append(DI)
    return torch.cat(parts, dim=1)


class HystereticCGAN(pl.LightningModule):
    def __init__(self, architecture, loss_type, condition_dim, SELU, G_h_size, D_h_size, spectral_D, no_batch_norm, 
                image_size, z_dim, lr, result_dir, latent_conditioning, direction_conditioning, add_DI_loss,
                    **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs", "result_dir", "extractor"])
        self.loss_type = loss_type
        self.condition_dim = condition_dim
        self.image_size = image_size
        self.z_dim = z_dim
        self.lr = lr
        self.result_dir = result_dir
        self.latent_conditioning = latent_conditioning
        self.direction_conditioning = direction_conditioning
        self.include_di_conditioning = kwargs.get("include_di_conditioning", False)
        self.add_DI_loss = add_DI_loss
        self.lr_schedule = kwargs.get("lr_schedule", "constant")
        self.warmup_epochs = kwargs.get("warmup_epochs", 5)
        self.min_lr_ratio = kwargs.get("min_lr_ratio", 0.1)
        self.total_epochs = kwargs.get("epoch", 10) + 1
        
        
        
        if self.add_DI_loss:
            self.training_outputs = {"errD":[], "errD_real":[], "errD_fake":[], "errG":[], "errDI":[], "gt_DI":[], "pred_DI":[]}
            self.validation_outputs = {"real_image":[], "generated_data":[], "exp":[], "timestep":[], "direction":[], "gt_DI":[], "pred_DI":[]}
        else:    
            self.training_outputs = {"errD":[], "errD_real":[], "errD_fake":[], "errG":[]}
            self.validation_outputs = {"real_image":[], "generated_data":[], "exp":[], "timestep":[], "direction":[]}
            
        self.automatic_optimization = False

        self.G = DCGAN_G(
            image_size=image_size,
            z_dim=z_dim,
            condition_dim=condition_dim,
            SELU=SELU,
            G_h_size=G_h_size
        )
        
        self.D = DCGAN_D(
            image_size=image_size,
            image_dim=3,
            condition_dim=condition_dim,
            spectral_D=spectral_D,
            D_h_size=D_h_size,
            SELU=SELU,
            no_batch_norm=no_batch_norm
        )
        
        
        self.G.apply(weights_init)
        self.D.apply(weights_init)
        
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.fid = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=True) # InceptionV3
        self.bcw = nn.BCEWithLogitsLoss()
        

    def forward(self, noise, real_data, hidden_state, direction, exp, timestep, DI, info):
        condition = self.process_condition(info, hidden_state, direction, DI)
        condition = condition.view(-1, condition.size(1), 1, 1)
        generated_data = self.G(noise, condition)
        return generated_data
    
    def on_train_epoch_start(self):
        self.log('Learning rate', self.trainer.optimizers[0].param_groups[0]['lr'])

    
    def process_condition(self, info, hidden_state, direction, DI):
        return build_generation_condition(
            info,
            hidden_state,
            direction,
            DI,
            self.latent_conditioning,
            self.direction_conditioning,
            self.include_di_conditioning,
        )
        
    
    def training_step(self, batch):
        real_data, hidden_state, direction, exp, timestep, DI, info = batch
        opt_G, opt_D = self.optimizers()
        
        # GAN Training
        z = torch.randn((real_data.shape[0], self.z_dim, 1, 1)).to(self.device)
        condition = self.process_condition(info, hidden_state, direction, DI)
        condition = condition.view(-1, condition.size(1), 1, 1)

        ## Train discriminator
        for p in self.D.parameters():
            p.requires_grad = True
            
        self.toggle_optimizer(opt_D)
        self.D.zero_grad()
        opt_D.zero_grad()
        
        ## Forward all and label
        y = torch.ones(real_data.size(0)).to(self.device)
        y1 = torch.zeros(real_data.size(0)).to(self.device)
        generated_data = self.G(z, condition)
        y_pred_real = self.D(real_data, condition)
        y_pred_fake = self.D(generated_data, condition)
        
        ### Dsicriminiator Loss
        match self.loss_type:
            case "SGAN":
                errD_real = self.bcw(y_pred_real, y)
                errD_fake = self.bcw(y_pred_fake, y1)
                errD = errD_real + errD_fake
            case "RSGAN":
                errD = self.bcw(y_pred_real - y_pred_fake, y)
                errD_real = errD
                errD_fake = errD
            case "RaSGAN":
                errD = (self.bcw(y_pred_real - torch.mean(y_pred_fake), y) + self.bcw(y_pred_fake - torch.mean(y_pred_real), y1)) / 2
                errD_real = errD
                errD_fake = errD
        
        
        self.training_outputs["errD"].append(errD)
        self.training_outputs["errD_real"].append(errD_real)
        self.training_outputs["errD_fake"].append(errD_fake)
        self.manual_backward(errD)
        opt_D.step()
        self.untoggle_optimizer(opt_D)

        
        ## Train generator
        for p in self.D.parameters():
            p.requires_grad = False
            
        self.toggle_optimizer(opt_G)
        self.G.zero_grad()
        opt_G.zero_grad()
        
        ## Forward all and label
        y = torch.ones(real_data.size(0)).to(self.device)
        y1 = torch.zeros(real_data.size(0)).to(self.device)
        generated_data = self.G(z, condition)
        y_pred_fake = self.D(generated_data, condition)
        
        ### Generator Loss
        match self.loss_type:
            case "SGAN":
                errG = self.bcw(y_pred_fake, y)
            case "RSGAN":
                y_pred_real = self.D(real_data, condition)
                errG = self.bcw(y_pred_fake- y_pred_real, y)
            case "RaSGAN":
                y_pred_real = self.D(real_data, condition)
                errG = (self.bcw(y_pred_real - torch.mean(y_pred_fake), y1) + self.bcw(y_pred_fake - torch.mean(y_pred_real), y)) / 2
                
        
        
        if self.add_DI_loss:
            generated_data = generated_data * 0.5 + 0.5
            generated_data_trans = self.custom_transform(generated_data)
            pred_DI = self.regressor.forward(generated_data_trans, info)
            errDI = nn.MSELoss()(pred_DI, DI)
            errTotal = errG + errDI
            self.training_outputs["errDI"].append(errDI.detach())
            self.training_outputs["pred_DI"].append(pred_DI.detach())
            self.training_outputs["gt_DI"].append(DI.detach())
            
            

        self.training_outputs["errG"].append(errG)
        
        if self.add_DI_loss:
            self.manual_backward(errTotal)
        else:
            self.manual_backward(errG)
        opt_G.step()
        self.untoggle_optimizer(opt_G)
    
    def validation_step(self, batch, batch_id):
        real_data, hidden_state, direction, exp, timestep, DI, info = batch
        z = torch.randn((real_data.shape[0], self.z_dim, 1, 1)).to(self.device)
        condition = self.process_condition(info, hidden_state, direction, DI)
        condition = condition.view(-1, condition.size(1), 1, 1)
        generated_data = self.G(z, condition)
        
        # Regressor Loss
        generated_data = generated_data * 0.5 + 0.5
        if self.add_DI_loss:
            generated_data_trans = self.custom_transform(generated_data)
            pred_DI = self.regressor.forward(generated_data_trans, info)
            self.validation_outputs["pred_DI"].append(pred_DI)
            self.validation_outputs["gt_DI"].append(DI)
            
            
        
        # Denormalize
        real_data = real_data * 0.5 + 0.5
        
        # Save all the first results in every batch
        self.validation_outputs["real_image"].append(real_data[0, :, :, :])
        self.validation_outputs["generated_data"].append(generated_data[0, :, :, :])
        self.validation_outputs["exp"].append(exp[0])
        self.validation_outputs["timestep"].append(timestep[0])
        self.validation_outputs["direction"].append(direction[0, :])
            
        
        self.psnr.update(generated_data, real_data)
        self.ssim.update(generated_data, real_data)
        self.fid.update(generated_data, real=False)
        self.fid.update(real_data, real=True)
        

    def configure_optimizers(self):
        lr_G = self.lr
        lr_D = self.lr
        opt_G = torch.optim.Adam(self.G.parameters(), lr=lr_G)
        opt_D = torch.optim.Adam(self.D.parameters(), lr=lr_D)

        if self.lr_schedule != "warmup_cosine":
            return [opt_G, opt_D]

        def lr_lambda(epoch_idx):
            if self.warmup_epochs > 0 and epoch_idx < self.warmup_epochs:
                return float(epoch_idx + 1) / float(self.warmup_epochs)
            remain = max(self.total_epochs - self.warmup_epochs, 1)
            progress = min(max((epoch_idx - self.warmup_epochs) / remain, 0.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        sch_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lr_lambda)
        sch_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda=lr_lambda)
        return [opt_G, opt_D], [sch_G, sch_D]
    
    
    def on_train_epoch_end(self):
        errD = torch.stack(self.training_outputs["errD"]).mean()
        errD_real = torch.stack(self.training_outputs["errD_real"]).mean()
        errD_fake = torch.stack(self.training_outputs["errD_fake"]).mean()
        errG = torch.stack(self.training_outputs["errG"]).mean()
        
        self.log("train/errD", errD)
        self.log("train/errD_real", errD_real)
        self.log("train/errD_fake", errD_fake)
        self.log("train/errG", errG)
        
        if not self.add_DI_loss:
            self.training_outputs = {"errD":[], "errD_real":[], "errD_fake":[], "errG":[]}
            
        else:
            errDI = torch.stack(self.training_outputs["errDI"]).mean()
            self.log("train/errDI", errDI)
        
            # Draw Scatter Plot
            gt = torch.concatenate(self.training_outputs["gt_DI"]).cpu().detach().numpy()
            prediction = torch.concatenate(self.training_outputs["pred_DI"]).cpu().detach().numpy()
            r2 = r2_score(gt, prediction)
            self.log("train/r2", r2)
            fig, ax = plt.subplots()
            ax.axline((0, 0), slope=1, color="black")
            ax.scatter(gt, prediction, s=5)
            ax.set_xlabel("Ground Truth")
            ax.set_ylabel("Prediction")
            ax.set_xlim(0, 1.4)
            ax.set_ylim(0, 1.4)
            ax.set_title(r"$R^2$ Score: {:.4f}".format(r2))
            wandb.log({f"Plot/Scatter(train)": wandb.Image(fig)})
            plt.close("all")
            self.training_outputs = {"errD":[], "errD_real":[], "errD_fake":[], "errG":[], "errDI":[], "gt_DI":[], "pred_DI":[]}

    
    def on_validation_epoch_end(self):
        psnr = self.psnr.compute()
        ssim = self.ssim.compute()
        fid = self.fid.compute()
        self.log("val/psnr", psnr)
        self.log("val/ssim", ssim)
        self.log("val/fid", fid)

        
        self.psnr.reset()
        self.ssim.reset()
        self.fid.reset()
        
        if self.add_DI_loss:
            # Draw Scatter Plot
            gt = torch.concatenate(self.validation_outputs["gt_DI"]).cpu().numpy()
            prediction = torch.concatenate(self.validation_outputs["pred_DI"]).cpu().numpy()
            r2 = r2_score(gt, prediction)
            self.log("val/r2", r2)
            fig, ax = plt.subplots()
            ax.axline((0, 0), slope=1, color="black")
            ax.scatter(gt, prediction, s=5)
            ax.set_xlabel("Ground Truth")
            ax.set_ylabel("Prediction")
            ax.set_xlim(0, 1.4)
            ax.set_ylim(0, 1.4)
            ax.set_title(r"$R^2$ Score: {:.4f}".format(r2))
            wandb.log({f"Plot/Scatter(val)": wandb.Image(fig)})
        
        hysteresis = self.trainer.val_dataloaders.dataset.hysteresis
        print(hysteresis)
        for i in range(len(self.validation_outputs["real_image"])):
            real_image = self.validation_outputs["real_image"][i].cpu().permute(1, 2, 0).numpy()
            generated_data = self.validation_outputs["generated_data"][i].cpu().permute(1, 2, 0).numpy()
            exp = self.validation_outputs["exp"][i]
            timestep = self.validation_outputs["timestep"][i].cpu().numpy()
            direction = self.validation_outputs["direction"][i].cpu().numpy()
            decoded = np.argmax(direction)
            if decoded == 0:
                direction = "Push"
            else:
                direction = "Pull"
            
            max_force = hysteresis[exp]["max_simulated_force"]
            
            fig = plt.figure(figsize=(20, 8), constrained_layout=True)
            gs = gridspec.GridSpec(2, 3, figure=fig)
            ax0 = plt.subplot(gs[:, :2])
            ax3 = plt.subplot(gs[0, -1])
            ax4 = plt.subplot(gs[1, -1])

            line1 = ax0.plot(hysteresis[exp]['drift'], hysteresis[exp]['simulated']*max_force, linestyle="-.", color="tab:green", label="Simulated", alpha=0.5)
            line2 = ax0.plot(hysteresis[exp]['drift'], hysteresis[exp]['test']*max_force, linestyle="-", color="tab:blue", label="Experiment", alpha=0.5)
            line3 = ax0.plot(hysteresis[exp]['drift'], hysteresis[exp]['predicted']*max_force, linestyle="--", color="tab:red", label="Prediction", alpha=0.7)
            ax0.plot(hysteresis[exp]['drift'][timestep], hysteresis[exp]['simulated'][timestep]*max_force, marker="*", markeredgecolor="k", markerfacecolor="tab:green", label="Current Prediction", markersize=10)
            ax0.plot(hysteresis[exp]['drift'][timestep], hysteresis[exp]['test'][timestep]*max_force, marker="*", markeredgecolor="k", markerfacecolor="tab:blue", label="Current Experiment", markersize=10)
            ax0.plot(hysteresis[exp]['drift'][timestep], hysteresis[exp]['predicted'][timestep]*max_force, marker="*", markeredgecolor="k", markerfacecolor="tab:red", label="Current Prediction", markersize=10)
            line4 =  Line2D([0], [0], marker='*', label='Current Point', linestyle='None', markeredgecolor='k', markerfacecolor="k", markersize=10)
            ax0.legend(handles=[line1[0], line2[0], line3[0], line4])
            ax0.set_title("Hysteresis Loop")
            ax0.set_xlabel("Drift (%)")
            ax0.set_ylabel("Force (kN)")
            
            
            ax3.imshow(real_image)
            ax3.set_title("Real Image")
            ax3.axis("off")
            ax4.imshow(generated_data)
            ax4.set_title("Generated Image")
            ax4.axis("off")
            fig.suptitle(f"Experiment {exp} - {direction} - Timestep {timestep} - Epoch {self.current_epoch:05d}", fontsize=16)
            wandb.log({f"Plot/Experiment {exp} - {direction} - Timestep {timestep}": wandb.Image(fig)})
            plt.close("all")
        
        self.validation_outputs = {"real_image":[], "generated_data":[], "exp":[], "timestep":[], "direction":[], "gt_DI":[], "pred_DI":[]}

        

        
class CFMADM(pl.LightningModule):
    def __init__(
        self,
        architecture,
        condition_dim,
        image_size,
        lr,
        result_dir,
        latent_conditioning,
        direction_conditioning,
        add_DI_loss,
        cfm_model_channels,
        cfm_num_res_blocks,
        cfm_cond_dim,
        cfm_dropout,
        cfm_ode_steps,
        cfm_solver,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs", "result_dir"])
        self.condition_dim = condition_dim
        self.image_size = image_size
        self.lr = lr
        self.result_dir = result_dir
        self.latent_conditioning = latent_conditioning
        self.direction_conditioning = direction_conditioning
        self.include_di_conditioning = kwargs.get("include_di_conditioning", False)
        self.add_DI_loss = add_DI_loss
        self.cfm_ode_steps = cfm_ode_steps
        self.cfm_solver = cfm_solver.lower()
        self.lr_schedule = kwargs.get("lr_schedule", "constant")
        self.warmup_epochs = kwargs.get("warmup_epochs", 5)
        self.min_lr_ratio = kwargs.get("min_lr_ratio", 0.1)
        self.total_epochs = kwargs.get("epoch", 10) + 1
        self.inference_every_n_epochs = max(1, int(kwargs.get("inference_every_n_epochs", 10)))
        self.inference_num_images = max(1, int(kwargs.get("inference_num_images", 6)))

        self.velocity_net = ADMUNetCFM(
            in_channels=3,
            out_channels=3,
            model_channels=cfm_model_channels,
            channel_mult=(1, 2, 4, 4),
            num_res_blocks=cfm_num_res_blocks,
            cond_dim=cfm_cond_dim,
            latent_dim=condition_dim,
            dropout=cfm_dropout,
            use_mid_attention=True,
        )

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.fid = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=True)
        self.validation_outputs = {"gt_DI": [], "pred_DI": [], "samples": []}

        self.regressor = kwargs.get("regressor", None)
        if self.add_DI_loss:
            if self.regressor is None:
                raise ValueError("CFMADM with --add_DI_loss requires a loaded regressor.")
            self.custom_transform = v2.Compose([
                v2.Resize((224, 224)),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def on_train_epoch_start(self):
        self.log("Learning rate", self.trainer.optimizers[0].param_groups[0]["lr"])

    def process_condition(self, info, hidden_state, direction, DI):
        return build_generation_condition(
            info,
            hidden_state,
            direction,
            DI,
            self.latent_conditioning,
            self.direction_conditioning,
            self.include_di_conditioning,
        )

    def flow_matching_loss(self, z1, c):
        b = z1.shape[0]
        t = torch.rand(b, device=z1.device)
        z0 = torch.randn_like(z1)
        zt = (1.0 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1
        target_u = z1 - z0
        pred_v = self.velocity_net(zt, t, c)
        return torch.mean((pred_v - target_u) ** 2)

    def sample_with_ode(self, c, z0=None, num_steps=None):
        # Separate ODE solver from training objective.
        if num_steps is None:
            num_steps = self.cfm_ode_steps
        if z0 is None:
            z = torch.randn((c.shape[0], 3, self.image_size, self.image_size), device=c.device)
        else:
            z = z0
        dt = 1.0 / float(num_steps)
        for i in range(num_steps):
            t_scalar = float(i) / float(num_steps)
            t = torch.full((c.shape[0],), t_scalar, device=c.device)
            v_t = self.velocity_net(z, t, c)
            if self.cfm_solver == "heun":
                z_pred = z + dt * v_t
                t_next_scalar = float(i + 1) / float(num_steps)
                t_next = torch.full((c.shape[0],), t_next_scalar, device=c.device)
                v_next = self.velocity_net(z_pred, t_next, c)
                z = z + 0.5 * dt * (v_t + v_next)
            else:
                z = z + dt * v_t
        return z

    def training_step(self, batch, batch_idx):
        real_data, hidden_state, direction, exp, timestep, DI, info = batch
        c = self.process_condition(info, hidden_state, direction, DI)
        loss_fm = self.flow_matching_loss(real_data, c)

        if self.add_DI_loss:
            sampled = self.sample_with_ode(c)
            sampled_vis = torch.clamp(sampled * 0.5 + 0.5, 0.0, 1.0)
            sampled_trans = self.custom_transform(sampled_vis)
            pred_di = self.regressor(sampled_trans, info)
            loss_di = nn.MSELoss()(pred_di, DI)
            loss = loss_fm + loss_di
            self.log("train/errDI", loss_di, prog_bar=False, on_step=False, on_epoch=True)
        else:
            loss = loss_fm

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/fm_loss", loss_fm, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        real_data, hidden_state, direction, exp, timestep, DI, info = batch
        c = self.process_condition(info, hidden_state, direction, DI)

        val_fm_loss = self.flow_matching_loss(real_data, c)
        sampled = self.sample_with_ode(c)

        real_vis = torch.clamp(real_data * 0.5 + 0.5, 0.0, 1.0)
        sampled_vis = torch.clamp(sampled * 0.5 + 0.5, 0.0, 1.0)
        self.psnr.update(sampled_vis, real_vis)
        self.ssim.update(sampled_vis, real_vis)
        self.fid.update(sampled_vis, real=False)
        self.fid.update(real_vis, real=True)

        if len(self.validation_outputs["samples"]) < self.inference_num_images:
            max_store = self.inference_num_images - len(self.validation_outputs["samples"])
            batch_store = min(max_store, sampled_vis.shape[0])
            for i in range(batch_store):
                self.validation_outputs["samples"].append(
                    {
                        "real": real_vis[i].detach().cpu(),
                        "pred": sampled_vis[i].detach().cpu(),
                        "exp": exp[i] if isinstance(exp, (list, tuple)) else exp,
                        "timestep": int(timestep[i].item()) if torch.is_tensor(timestep[i]) else int(timestep[i]),
                    }
                )

        if self.add_DI_loss:
            sampled_trans = self.custom_transform(sampled_vis)
            pred_di = self.regressor(sampled_trans, info)
            self.validation_outputs["pred_DI"].append(pred_di.detach())
            self.validation_outputs["gt_DI"].append(DI.detach())

        self.log("val/loss", val_fm_loss, prog_bar=False, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        psnr = self.psnr.compute()
        ssim = self.ssim.compute()
        fid = self.fid.compute()
        self.log("val/psnr", psnr)
        self.log("val/ssim", ssim)
        self.log("val/fid", fid)
        self.psnr.reset()
        self.ssim.reset()
        self.fid.reset()

        if (self.current_epoch % self.inference_every_n_epochs == 0) and len(self.validation_outputs["samples"]) > 0:
            rows = len(self.validation_outputs["samples"])
            fig, axes = plt.subplots(rows, 2, figsize=(8, max(3, rows * 2.5)))
            if rows == 1:
                axes = np.array([axes])
            for i, sample in enumerate(self.validation_outputs["samples"]):
                real_img = sample["real"].permute(1, 2, 0).numpy()
                pred_img = sample["pred"].permute(1, 2, 0).numpy()
                axes[i, 0].imshow(real_img)
                axes[i, 0].set_title(f"GT | {sample['exp']} | t={sample['timestep']}")
                axes[i, 0].axis("off")
                axes[i, 1].imshow(pred_img)
                axes[i, 1].set_title("Inference")
                axes[i, 1].axis("off")
            fig.suptitle(f"CFM Inference Samples - Epoch {self.current_epoch:05d}")
            plt.tight_layout()

            save_dir = os.path.join(self.result_dir, "inference_samples")
            os.makedirs(save_dir, exist_ok=True)
            fig_path = os.path.join(save_dir, f"epoch_{self.current_epoch:05d}.png")
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"[CFM] Saved inference samples: {fig_path}")
            if wandb.run is not None:
                wandb.log({f"CFM/Inference Epoch {self.current_epoch:05d}": wandb.Image(fig)})
            plt.close(fig)

        if self.add_DI_loss and len(self.validation_outputs["gt_DI"]) > 0:
            gt = torch.concatenate(self.validation_outputs["gt_DI"]).cpu().numpy()
            prediction = torch.concatenate(self.validation_outputs["pred_DI"]).cpu().numpy()
            self.log("val/r2", r2_score(gt, prediction))
        self.validation_outputs = {"gt_DI": [], "pred_DI": [], "samples": []}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.velocity_net.parameters(), lr=self.lr)
        if self.lr_schedule != "warmup_cosine":
            return optimizer

        def lr_lambda(epoch_idx):
            if self.warmup_epochs > 0 and epoch_idx < self.warmup_epochs:
                return float(epoch_idx + 1) / float(self.warmup_epochs)
            remain = max(self.total_epochs - self.warmup_epochs, 1)
            progress = min(max((epoch_idx - self.warmup_epochs) / remain, 0.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [scheduler]


class Baseline(pl.LightningModule):
    def __init__(self, architecture, loss_type, condition_dim, G_h_size, 
                image_size, lr, result_dir, latent_conditioning, 
                    **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs", "result_dir", "extractor", "regressor"])
        self.loss_type = loss_type
        self.condition_dim = condition_dim
        self.image_size = image_size
        self.lr = lr
        self.result_dir = result_dir
        self.latent_conditioning = latent_conditioning
        self.direction_conditioning = kwargs.get("direction_conditioning", True)
        self.include_di_conditioning = kwargs.get("include_di_conditioning", False)
        self.lr_schedule = kwargs.get("lr_schedule", "constant")
        self.warmup_epochs = kwargs.get("warmup_epochs", 5)
        self.min_lr_ratio = kwargs.get("min_lr_ratio", 0.1)
        self.total_epochs = kwargs.get("epoch", 10) + 1
        
        
        self.training_outputs = {"errG":[]}
        self.validation_outputs = {"real_image":[], "generated_data":[], "exp":[], "timestep":[], "direction":[]}
            
        self.automatic_optimization = False

        self.G = DCGAN_G(
            image_size=image_size,
            z_dim=0,
            condition_dim=condition_dim,
            SELU=False,
            G_h_size=G_h_size
        )
        
        self.G.apply(weights_init)
        
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.fid = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=True) # InceptionV3
        self.bcw = nn.BCEWithLogitsLoss()
        

    def forward(self, noise, hidden_state, direction, timestep):
        condition = torch.cat([hidden_state[timestep, :], direction], dim=1)
        condition = condition.view(-1, condition.size(1), 1, 1)
        generated_data = self.G(condition)
        return generated_data
    
    def on_train_epoch_start(self):
        self.log('Learning rate', self.trainer.optimizers[0].param_groups[0]['lr'])

    
    def process_condition(self, info, hidden_state, direction, DI):
        return build_generation_condition(
            info,
            hidden_state,
            direction,
            DI,
            self.latent_conditioning,
            self.direction_conditioning,
            self.include_di_conditioning,
        )
        
    
    def training_step(self, batch):
        real_data, hidden_state, direction, exp, timestep, DI, info = batch
        opt_G = self.optimizers()
        
        # GAN Training
        z = torch.randn((real_data.shape[0], 0, 1, 1)).to(self.device) # Empty noise
        condition = self.process_condition(info, hidden_state, direction, DI)
        condition = condition.view(-1, condition.size(1), 1, 1)

        self.toggle_optimizer(opt_G)
        self.G.zero_grad()
        opt_G.zero_grad()
        
        generated_data = self.G(z, condition)
        errG = nn.MSELoss()(generated_data, real_data)
        self.training_outputs["errG"].append(errG)
        self.manual_backward(errG)
        opt_G.step()
        self.untoggle_optimizer(opt_G)
    
    def validation_step(self, batch, batch_id):
        real_data, hidden_state, direction, exp, timestep, DI, info = batch
        z = torch.randn((real_data.shape[0], 0, 1, 1)).to(self.device) # Empty noise
        condition = self.process_condition(info, hidden_state, direction, DI)
        condition = condition.view(-1, condition.size(1), 1, 1)
        generated_data = self.G(z, condition)
        
        # Denormalize
        real_data = real_data * 0.5 + 0.5
        generated_data = generated_data * 0.5 + 0.5
        
        # Save all the first results in every batch
        self.validation_outputs["real_image"].append(real_data[0, :, :, :])
        self.validation_outputs["generated_data"].append(generated_data[0, :, :, :])
        self.validation_outputs["exp"].append(exp[0])
        self.validation_outputs["timestep"].append(timestep[0])
        self.validation_outputs["direction"].append(direction[0, :])
            
        
        self.psnr.update(generated_data, real_data)
        self.ssim.update(generated_data, real_data)
        self.fid.update(generated_data, real=False)
        self.fid.update(real_data, real=True)
        

    def configure_optimizers(self):
        lr_G = self.lr
        opt_G = torch.optim.Adam(self.G.parameters(), lr=lr_G)

        if self.lr_schedule != "warmup_cosine":
            return opt_G

        def lr_lambda(epoch_idx):
            if self.warmup_epochs > 0 and epoch_idx < self.warmup_epochs:
                return float(epoch_idx + 1) / float(self.warmup_epochs)
            remain = max(self.total_epochs - self.warmup_epochs, 1)
            progress = min(max((epoch_idx - self.warmup_epochs) / remain, 0.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lr_lambda)
        return [opt_G], [scheduler]
    
    
    def on_train_epoch_end(self):
        errG = torch.stack(self.training_outputs["errG"]).mean()
        self.log("train/errG", errG)
        self.training_outputs = {"errG":[]}

    
    def on_validation_epoch_end(self):
        psnr = self.psnr.compute()
        ssim = self.ssim.compute()
        fid = self.fid.compute()
        self.log("val/psnr", psnr)
        self.log("val/ssim", ssim)
        self.log("val/fid", fid)

        
        self.psnr.reset()
        self.ssim.reset()
        self.fid.reset()
        
        
        hysteresis = self.trainer.val_dataloaders.dataset.hysteresis
        
        for i in range(len(self.validation_outputs["real_image"])):
            real_image = self.validation_outputs["real_image"][i].cpu().permute(1, 2, 0).numpy()
            generated_data = self.validation_outputs["generated_data"][i].cpu().permute(1, 2, 0).numpy()
            exp = self.validation_outputs["exp"][i]
            timestep = self.validation_outputs["timestep"][i].cpu().numpy()
            direction = self.validation_outputs["direction"][i].cpu().numpy()
            decoded = np.argmax(direction)
            if decoded == 0:
                direction = "Push"
            else:
                direction = "Pull"
            
            max_force = hysteresis[exp]["max_simulated_force"]
            
            fig = plt.figure(figsize=(20, 8), constrained_layout=True)
            gs = gridspec.GridSpec(2, 3, figure=fig)
            ax0 = plt.subplot(gs[:, :2])
            ax3 = plt.subplot(gs[0, -1])
            ax4 = plt.subplot(gs[1, -1])

            line1 = ax0.plot(hysteresis[exp]['drift'], hysteresis[exp]['simulated']*max_force, linestyle="-.", color="tab:green", label="Simulated", alpha=0.5)
            line2 = ax0.plot(hysteresis[exp]['drift'], hysteresis[exp]['test']*max_force, linestyle="-", color="tab:blue", label="Experiment", alpha=0.5)
            line3 = ax0.plot(hysteresis[exp]['drift'], hysteresis[exp]['predicted']*max_force, linestyle="--", color="tab:red", label="Prediction", alpha=0.7)
            ax0.plot(hysteresis[exp]['drift'][timestep], hysteresis[exp]['simulated'][timestep]*max_force, marker="*", markeredgecolor="k", markerfacecolor="tab:green", label="Current Prediction", markersize=10)
            ax0.plot(hysteresis[exp]['drift'][timestep], hysteresis[exp]['test'][timestep]*max_force, marker="*", markeredgecolor="k", markerfacecolor="tab:blue", label="Current Experiment", markersize=10)
            ax0.plot(hysteresis[exp]['drift'][timestep], hysteresis[exp]['predicted'][timestep]*max_force, marker="*", markeredgecolor="k", markerfacecolor="tab:red", label="Current Prediction", markersize=10)
            line4 =  Line2D([0], [0], marker='*', label='Current Point', linestyle='None', markeredgecolor='k', markerfacecolor="k", markersize=10)
            ax0.legend(handles=[line1[0], line2[0], line3[0], line4])
            ax0.set_title("Hysteresis Loop")
            ax0.set_xlabel("Drift (%)")
            ax0.set_ylabel("Force (kN)")
            
            
            ax3.imshow(real_image)
            ax3.set_title("Real Image")
            ax3.axis("off")
            ax4.imshow(generated_data)
            ax4.set_title("Generated Image")
            ax4.axis("off")
            fig.suptitle(f"Experiment {exp} - {direction} - Timestep {timestep} - Epoch {self.current_epoch:05d}", fontsize=16)
            wandb.log({f"Plot/Experiment {exp} - {direction} - Timestep {timestep}": wandb.Image(fig)})
            plt.close("all")
        
        self.validation_outputs = {"real_image":[], "generated_data":[], "exp":[], "timestep":[], "direction":[]}