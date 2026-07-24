import wandb
import os
import numpy as np
import torch.nn as nn
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


import torch
import torch.nn.functional as F
from src.model.Component.generative import DCGAN_D, DCGAN_G, weights_init
from src.model.Component.cfm_adm import ADMUNetCFM
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchvision.transforms import v2
from sklearn.metrics import r2_score
from contextlib import nullcontext

try:
    from diffusers import AutoencoderKL
except ImportError:
    AutoencoderKL = None


def build_generation_condition(
    info,
    hidden_state,
    direction,
    DI,
    latent_conditioning,
    direction_conditioning,
    include_di_conditioning,
    condition_mode="full",
):
    if condition_mode == "di_only":
        return DI

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


def _parse_fixed_di_values(raw_values, fallback_count):
    if isinstance(raw_values, (list, tuple)):
        parsed = [float(v) for v in raw_values]
    else:
        parsed = []
        for token in str(raw_values).split(","):
            token = token.strip()
            if not token:
                continue
            parsed.append(float(token))
    if not parsed:
        parsed = [0.0, 0.4, 0.75, 0.95]
    if len(parsed) < fallback_count:
        parsed.extend([parsed[-1]] * (fallback_count - len(parsed)))
    return parsed[:fallback_count]


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
        self.condition_mode = str(kwargs.get("condition_mode", "full")).lower()
        self.use_damage_mask_condition = bool(kwargs.get("use_damage_mask_condition", False))
        self.damage_mask_sigma_min = float(kwargs.get("damage_mask_sigma_min", 4.0))
        self.damage_mask_sigma_max = float(kwargs.get("damage_mask_sigma_max", 20.0))
        self.damage_mask_max_di = max(float(kwargs.get("damage_mask_max_di", 1.0)), 1e-6)
        self.add_DI_loss = add_DI_loss
        self.lr_schedule = kwargs.get("lr_schedule", "constant")
        self.warmup_epochs = kwargs.get("warmup_epochs", 5)
        self.min_lr_ratio = kwargs.get("min_lr_ratio", 0.1)
        self.total_epochs = kwargs.get("epoch", 10) + 1
        self.optimizer_name = str(kwargs.get("optimizer", "adamw")).lower()
        
        
        
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
            self.condition_mode,
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
        self.condition_mode = str(kwargs.get("condition_mode", "full")).lower()
        self.use_damage_mask_condition = bool(kwargs.get("use_damage_mask_condition", False))
        self.damage_mask_sigma_min = float(kwargs.get("damage_mask_sigma_min", 4.0))
        self.damage_mask_sigma_max = float(kwargs.get("damage_mask_sigma_max", 20.0))
        self.damage_mask_max_di = max(float(kwargs.get("damage_mask_max_di", 1.0)), 1e-6)
        self.damage_mask_split_di = float(kwargs.get("damage_mask_split_di", 0.5))
        self.damage_mask_elastic_intensity_cap = float(kwargs.get("damage_mask_elastic_intensity_cap", 0.08))
        self.damage_mask_elastic_spread_min = float(kwargs.get("damage_mask_elastic_spread_min", 0.15))
        self.damage_mask_elastic_spread_max = float(kwargs.get("damage_mask_elastic_spread_max", 0.20))
        self.damage_mask_width_scale = max(float(kwargs.get("damage_mask_width_scale", 2.0)), 1e-6)
        self.damage_mask_quadratic_scale = float(kwargs.get("damage_mask_quadratic_scale", 0.8))
        self.grad_clip_max_norm = float(kwargs.get("grad_clip_max_norm", 2.0))
        self.add_DI_loss = add_DI_loss
        self.cfm_ode_steps = cfm_ode_steps
        self.cfm_solver = cfm_solver.lower()
        self.lr_schedule = kwargs.get("lr_schedule", "constant")
        self.warmup_epochs = kwargs.get("warmup_epochs", 5)
        self.warmup_steps = int(kwargs.get("warmup_steps", 800))
        self.min_lr_ratio = kwargs.get("min_lr_ratio", 0.1)
        self.total_epochs = kwargs.get("epoch", 10) + 1
        self.total_steps = int(kwargs.get("max_steps", 15000))
        self.optimizer_name = str(kwargs.get("optimizer", "adamw")).lower()
        self.inference_every_n_epochs = max(1, int(kwargs.get("inference_every_n_epochs", 10)))
        self.inference_num_images = max(1, int(kwargs.get("inference_num_images", 6)))
        self.metrics_every_n_epochs = max(1, int(kwargs.get("metrics_every_n_epochs", 100)))
        self.cfg_scale = float(kwargs.get("cfg_scale", 1.0))
        self.use_velocity_min_snr_weighting = bool(kwargs.get("use_velocity_min_snr_weighting", True))
        self.min_snr_gamma = float(kwargs.get("min_snr_gamma", 5.0))
        self.inference_anchor_exp = str(kwargs.get("inference_anchor_exp", "")).strip()
        self.inference_anchor_timestep = int(kwargs.get("inference_anchor_timestep", -1))
        self.fixed_di_values = _parse_fixed_di_values(
            kwargs.get("fixed_di_values", "0.0,0.4,0.75,0.95"),
            self.inference_num_images,
        )
        self.use_sd3_vae_latent = kwargs.get("use_sd3_vae_latent", False)
        self.sd3_vae_model_id = kwargs.get("sd3_vae_model_id", "stabilityai/stable-diffusion-3-medium-diffusers")
        self.sd3_vae_subfolder = kwargs.get("sd3_vae_subfolder", "vae")
        self.sd3_vae_use_fp16 = kwargs.get("sd3_vae_use_fp16", True)

        self.vae = None
        self.vae_scaling_factor = 1.0
        self.vae_shift_factor = 0.0
        self.model_in_channels = 3
        self.model_spatial_size = self.image_size
        self.vae_downsample_factor = 1

        if self.use_sd3_vae_latent:
            if AutoencoderKL is None:
                raise ImportError(
                    "diffusers is required for --use_sd3_vae_latent. "
                    "Please install it with: pip install diffusers transformers"
                )

            vae_dtype = torch.float16 if (torch.cuda.is_available() and self.sd3_vae_use_fp16) else torch.float32
            self.vae = AutoencoderKL.from_pretrained(
                self.sd3_vae_model_id,
                subfolder=self.sd3_vae_subfolder,
                torch_dtype=vae_dtype,
            )
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False

            self.vae_scaling_factor = float(getattr(self.vae.config, "scaling_factor", 1.0))
            self.vae_shift_factor = float(getattr(self.vae.config, "shift_factor", 0.0))
            configured_latent_channels = int(getattr(self.vae.config, "latent_channels", 4))
            self.model_in_channels = 4
            if configured_latent_channels != self.model_in_channels:
                print(
                    f"[CFM] Overriding VAE latent_channels={configured_latent_channels} to {self.model_in_channels} "
                    f"for 4-latent + 1-mask input design.",
                    flush=True,
                )
            self.vae_downsample_factor = int(
                getattr(
                    self.vae.config,
                    "vae_scale_factor",
                    2 ** len(getattr(self.vae.config, "block_out_channels", [1, 1, 1, 1])),
                )
            )
            if self.image_size % self.vae_downsample_factor != 0:
                raise ValueError(
                    f"image_size={self.image_size} is not divisible by VAE downsample factor={self.vae_downsample_factor}"
                )
            self.model_spatial_size = self.image_size // self.vae_downsample_factor

        self.velocity_net = ADMUNetCFM(
            in_channels=self.model_in_channels,
            out_channels=self.model_in_channels,
            model_channels=cfm_model_channels,
            channel_mult=(1, 2, 4, 4),
            num_res_blocks=cfm_num_res_blocks,
            cond_dim=cfm_cond_dim,
            latent_dim=condition_dim,
            dropout=cfm_dropout,
            use_mid_attention=False,
            use_damage_mask_condition=self.use_damage_mask_condition,
        )
        cond_mlp_in = int(self.velocity_net.cond_mlp[0].in_features)
        if cond_mlp_in != int(self.condition_dim):
            raise ValueError(
                f"Condition dimension mismatch: condition_dim={self.condition_dim}, "
                f"cond_mlp.in_features={cond_mlp_in}"
            )
        print(
            f"[CFM] condition_mode={self.condition_mode} condition_dim={self.condition_dim} "
            f"-> cond_mlp.in_features={cond_mlp_in} cond_dim={cfm_cond_dim} "
            f"in_conv.in_channels={self.velocity_net.in_conv.in_channels}",
            flush=True,
        )

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.fid = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=True)
        self.validation_outputs = {"gt_DI": [], "pred_DI": [], "anchor_candidates": []}
        self._has_logged_initial_inference = False
        self._fixed_inference_anchor = None

        self.regressor = kwargs.get("regressor", None)
        if self.add_DI_loss:
            if self.regressor is None:
                raise ValueError("CFMADM with --add_DI_loss requires a loaded regressor.")
            self.custom_transform = v2.Compose([
                v2.Resize((224, 224)),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _compute_velocity_min_snr_weight(self, t):
        """Min-SNR-gamma weighting for velocity prediction loss."""
        t_clamped = t.clamp(min=1e-5, max=1.0 - 1e-5)
        snr = (t_clamped / (1.0 - t_clamped)) ** 2
        return torch.clamp(snr, max=self.min_snr_gamma) / (snr + 1.0)

    def _vae_context(self):
        if self.vae is None:
            return nullcontext()
        vae_dtype = next(self.vae.parameters()).dtype
        if self.device.type == "cuda" and vae_dtype == torch.float16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def on_fit_start(self):
        if wandb.run is not None:
            wandb.define_metric("trainer/global_step")
            wandb.define_metric("*", step_metric="trainer/global_step")
            wandb.define_metric("train/loss", step_metric="trainer/global_step")
            wandb.define_metric("train/fm_loss", step_metric="trainer/global_step")
            wandb.define_metric("train/grad_norm_L2", step_metric="trainer/global_step")
            wandb.define_metric("train/lr", step_metric="trainer/global_step")
            wandb.define_metric("val/loss", step_metric="trainer/global_step")
            wandb.define_metric("CFM/FixedDI/*", step_metric="trainer/global_step")
        self._save_damage_mask_preview()

    def _encode_to_target_space(self, image_tensor):
        if self.vae is None:
            return image_tensor
        with torch.no_grad():
            with self._vae_context():
                posterior = self.vae.encode(image_tensor).latent_dist
                latent = posterior.mean
                latent = (latent - self.vae_shift_factor) * self.vae_scaling_factor
        return latent.float()

    def _decode_from_target_space(self, target_tensor):
        if self.vae is None:
            return target_tensor
        with torch.no_grad():
            with self._vae_context():
                model_input = target_tensor / self.vae_scaling_factor + self.vae_shift_factor
                decoded = self.vae.decode(model_input.to(next(self.vae.parameters()).dtype)).sample
        return decoded.float()

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
            self.condition_mode,
        )

    def _should_log_metrics(self):
        if self.trainer is not None and self.trainer.sanity_checking:
            return False
        step_num = int(self.global_step)
        if step_num <= 0:
            return False
        return step_num % self.metrics_every_n_epochs == 0

    def _should_run_inference(self):
        if self.trainer is not None and self.trainer.sanity_checking:
            return False
        step_num = int(self.global_step)
        if not self._has_logged_initial_inference and step_num > 0:
            return True
        if step_num <= 0:
            return False
        return step_num % self.inference_every_n_epochs == 0

    def _get_inference_di_values(self):
        # Use fixed DI values for stable, comparable inference panels.
        return [float(v) for v in self.fixed_di_values[: self.inference_num_images]]

    @staticmethod
    def _bottom_band_damage_profile(di_scaled, height):
        """Bottom-initiated half-Gaussian profile active only when DI > 0.5."""
        active = di_scaled > 0.5
        normalized_di = torch.clamp((di_scaled - 0.5) / 0.5, min=0.0, max=1.0)
        intensity = 0.8 * normalized_di
        sigma_y = float(height) * 0.1 + float(height) * 0.3 * normalized_di
        return active, intensity, torch.clamp(sigma_y, min=1e-6)

    def _build_damage_mask_from_di(self, di_tensor, height, width, device, dtype):
        if not self.use_damage_mask_condition:
            return torch.zeros((di_tensor.shape[0], 1, height, width), device=device, dtype=dtype)
        di_scaled = torch.clamp(di_tensor.to(device=device, dtype=dtype) / self.damage_mask_max_di, min=0.0, max=1.0).view(-1, 1, 1)
        active, intensity, sigma_y = self._bottom_band_damage_profile(di_scaled, height)
        y = torch.arange(height, device=device, dtype=dtype).view(1, height, 1)
        gaussian_1d = torch.exp(-((y - float(height)) ** 2) / (2.0 * sigma_y ** 2))
        mask = intensity * gaussian_1d.expand(-1, -1, width)
        mask = torch.where(active, mask, torch.zeros_like(mask))
        return mask.unsqueeze(1)

    @staticmethod
    def _ensure_single_channel_mask(mask, target_shape, device, dtype):
        if mask is None:
            return torch.zeros(target_shape, device=device, dtype=dtype)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] != 1:
            mask = mask[:, :1, :, :]
        mask = mask.to(device=device, dtype=dtype)
        if mask.shape[-2:] != target_shape[-2:]:
            mask = F.interpolate(mask, size=target_shape[-2:], mode="bilinear", align_corners=False)
        return mask

    def _save_damage_mask_preview(self):
        if not self.use_damage_mask_condition:
            return
        preview_values = sorted(set([0.0, 0.5] + [float(v) for v in self.fixed_di_values]))
        di_tensor = torch.tensor(preview_values, device=self.device, dtype=torch.float32)
        masks = self._build_damage_mask_from_di(
            di_tensor,
            self.model_spatial_size,
            self.model_spatial_size,
            device=self.device,
            dtype=torch.float32,
        ).detach().cpu()

        fig, axes = plt.subplots(1, len(preview_values), figsize=(max(6, 2 * len(preview_values)), 2.5))
        if len(preview_values) == 1:
            axes = [axes]
        for idx, di_value in enumerate(preview_values):
            ax = axes[idx]
            ax.imshow(masks[idx, 0].numpy(), cmap="hot", vmin=0.0, vmax=1.0)
            ax.set_title(f"DI={di_value:.2f}")
            ax.axis("off")
        fig.suptitle("Damage Mask Preview (1-Channel)", fontsize=11)
        plt.tight_layout()
        out_dir = os.path.join(self.result_dir, "data_check")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "damage_mask_preview.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[CFM] Saved damage-mask preview: {out_path}", flush=True)

    def _select_fixed_anchor(self, candidates):
        if self._fixed_inference_anchor is not None:
            return self._fixed_inference_anchor
        pool = candidates
        if self.inference_anchor_exp:
            filtered = [c for c in candidates if str(c["exp"]) == self.inference_anchor_exp]
            if filtered:
                pool = filtered
        if not pool:
            return None
        if self.inference_anchor_timestep >= 0:
            anchor = min(pool, key=lambda c: abs(int(c["timestep"]) - self.inference_anchor_timestep))
        else:
            anchor = sorted(pool, key=lambda c: (str(c["exp"]), int(c["timestep"]), str(c["sample_uid"])))[0]
        self._fixed_inference_anchor = anchor
        return anchor

    def _select_anchors_for_di_values(self, candidates, di_values):
        """Pick one distinct GT sample nearest each target DI (4 GT + 4 inference)."""
        pool = list(candidates)
        if self.inference_anchor_exp:
            filtered = [c for c in pool if str(c["exp"]) == self.inference_anchor_exp]
            if filtered:
                pool = filtered
        if not pool:
            return []

        selected = []
        remaining = list(pool)
        for di_value in di_values:
            search = remaining if remaining else pool
            if self.inference_anchor_timestep >= 0:
                # Prefer DI match, then timestep proximity as tie-breaker.
                anchor = min(
                    search,
                    key=lambda c: (
                        abs(float(c["gt_di"]) - float(di_value)),
                        abs(int(c["timestep"]) - self.inference_anchor_timestep),
                        str(c["sample_uid"]),
                    ),
                )
            else:
                anchor = min(
                    search,
                    key=lambda c: (
                        abs(float(c["gt_di"]) - float(di_value)),
                        str(c["exp"]),
                        int(c["timestep"]),
                        str(c["sample_uid"]),
                    ),
                )
            selected.append(anchor)
            remaining = [c for c in remaining if c["sample_uid"] != anchor["sample_uid"]]
        return selected

    @staticmethod
    def _unpack_generation_batch(batch):
        if len(batch) >= 8:
            real_data, hidden_state, direction, exp, timestep, DI, info, valid_mask = batch[:8]
        else:
            real_data, hidden_state, direction, exp, timestep, DI, info = batch
            valid_mask = torch.ones(
                (real_data.shape[0], 1, real_data.shape[-2], real_data.shape[-1]),
                device=real_data.device,
                dtype=real_data.dtype,
            )
        return real_data, hidden_state, direction, exp, timestep, DI, info, valid_mask

    def flow_matching_loss(self, z1, c, valid_mask=None):
        b = z1.shape[0]
        t = torch.rand(b, device=z1.device).clamp(min=1e-5, max=1.0 - 1e-5)
        z0 = torch.randn_like(z1)
        zt = (1.0 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1
        target_u = z1 - z0
        noise_pred = self.velocity_net(zt, t, c)
        noise_true = target_u

        unreduced_loss = F.mse_loss(noise_pred, noise_true, reduction="none")
        if valid_mask is None:
            per_sample_loss = unreduced_loss.mean(dim=(1, 2, 3))
        else:
            valid_mask = valid_mask.to(device=unreduced_loss.device, dtype=unreduced_loss.dtype)
            if valid_mask.ndim == 3:
                valid_mask = valid_mask.unsqueeze(1)
            if valid_mask.shape[-2:] != unreduced_loss.shape[-2:]:
                valid_mask = F.interpolate(valid_mask, size=unreduced_loss.shape[-2:], mode="nearest")

            masked_loss = unreduced_loss * valid_mask
            # Normalize by valid pixel count and channel count per sample.
            valid_count = valid_mask.sum(dim=(1, 2, 3)).clamp_min(1e-8)
            per_sample_loss = masked_loss.sum(dim=(1, 2, 3)) / (valid_count * unreduced_loss.shape[1])

        if self.use_velocity_min_snr_weighting:
            min_snr_weight = self._compute_velocity_min_snr_weight(t).to(per_sample_loss.dtype)
            per_sample_loss = per_sample_loss * min_snr_weight

        return per_sample_loss.mean()

    def sample_with_ode(self, c, z0=None, num_steps=None):
        # Separate ODE solver from training objective. CFG and spatial damage mask are disabled.
        if num_steps is None:
            num_steps = self.cfm_ode_steps
        if z0 is None:
            z = torch.randn(
                (c.shape[0], self.model_in_channels, self.model_spatial_size, self.model_spatial_size),
                device=c.device,
            )
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
        real_data, hidden_state, direction, exp, timestep, DI, info, valid_mask = self._unpack_generation_batch(batch)
        c = self.process_condition(info, hidden_state, direction, DI)
        target_space_data = self._encode_to_target_space(real_data)
        loss_fm = self.flow_matching_loss(target_space_data, c, valid_mask=valid_mask)
        if self.trainer is not None and self.trainer.optimizers:
            current_lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])
            self.log("train/lr", current_lr, prog_bar=False, on_step=True, on_epoch=False)

        if self.add_DI_loss:
            sampled = self.sample_with_ode(c)
            sampled_image = self._decode_from_target_space(sampled)
            sampled_vis = torch.clamp(sampled_image * 0.5 + 0.5, 0.0, 1.0)
            sampled_trans = self.custom_transform(sampled_vis)
            pred_di = self.regressor(sampled_trans, info)
            loss_di = nn.MSELoss()(pred_di, DI)
            loss = loss_fm + loss_di
            self.log("train/errDI", loss_di, prog_bar=False, on_step=True, on_epoch=False)
        else:
            loss = loss_fm

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/fm_loss", loss_fm, prog_bar=False, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        real_data, hidden_state, direction, exp, timestep, DI, info, valid_mask = self._unpack_generation_batch(batch)
        c = self.process_condition(info, hidden_state, direction, DI)

        target_space_data = self._encode_to_target_space(real_data)
        val_fm_loss = self.flow_matching_loss(target_space_data, c, valid_mask=valid_mask)
        sampled = self.sample_with_ode(c)
        sampled_image = self._decode_from_target_space(sampled)

        real_vis = torch.clamp(real_data * 0.5 + 0.5, 0.0, 1.0)
        sampled_vis = torch.clamp(sampled_image * 0.5 + 0.5, 0.0, 1.0)
        if self._should_log_metrics():
            self.psnr.update(sampled_vis, real_vis)
            self.ssim.update(sampled_vis, real_vis)
            self.fid.update(sampled_vis, real=False)
            self.fid.update(real_vis, real=True)

        candidate_cap = max(32, int(self.inference_num_images) * 24)
        if len(self.validation_outputs["anchor_candidates"]) < candidate_cap:
            remaining = candidate_cap - len(self.validation_outputs["anchor_candidates"])
            take_count = min(remaining, real_vis.shape[0])
            for sample_idx in range(take_count):
                if isinstance(exp, (list, tuple)):
                    anchor_exp = exp[sample_idx]
                elif torch.is_tensor(exp) and exp.ndim > 0:
                    anchor_exp = exp[sample_idx]
                else:
                    anchor_exp = exp
                if torch.is_tensor(anchor_exp):
                    anchor_exp = anchor_exp.item()

                if torch.is_tensor(timestep):
                    anchor_timestep = int(timestep[sample_idx].item())
                elif isinstance(timestep, (list, tuple)):
                    anchor_timestep = int(timestep[sample_idx])
                else:
                    anchor_timestep = int(timestep)

                gt_di_value = float(DI[sample_idx].item()) if torch.is_tensor(DI) else float(DI)
                self.validation_outputs["anchor_candidates"].append(
                    {
                        "real": real_vis[sample_idx].detach().cpu(),
                        "hidden_state": hidden_state[sample_idx].detach().cpu(),
                        "direction": direction[sample_idx].detach().cpu(),
                        "info": info[sample_idx].detach().cpu(),
                        "exp": anchor_exp,
                        "timestep": anchor_timestep,
                        "gt_di": gt_di_value,
                        "sample_uid": f"{anchor_exp}_{anchor_timestep}_{sample_idx}_{len(self.validation_outputs['anchor_candidates'])}",
                    }
                )

        if self.add_DI_loss:
            sampled_trans = self.custom_transform(sampled_vis)
            pred_di = self.regressor(sampled_trans, info)
            self.validation_outputs["pred_DI"].append(pred_di.detach())
            self.validation_outputs["gt_DI"].append(DI.detach())

        self.log("val/loss", val_fm_loss, prog_bar=False, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        if self._should_log_metrics():
            psnr = self.psnr.compute()
            ssim = self.ssim.compute()
            fid = self.fid.compute()
            self.log("val/psnr", psnr)
            self.log("val/ssim", ssim)
            self.log("val/fid", fid)
        self.psnr.reset()
        self.ssim.reset()
        self.fid.reset()

        should_infer = self._should_run_inference()
        candidates = self.validation_outputs["anchor_candidates"]
        if should_infer and len(candidates) > 0:
            di_values = self._get_inference_di_values()
            anchors = self._select_anchors_for_di_values(candidates, di_values)
            if not anchors:
                self.validation_outputs = {"gt_DI": [], "pred_DI": [], "anchor_candidates": []}
                return
            pair_count = len(di_values)
            fig, axes = plt.subplots(pair_count, 2, figsize=(6, max(3, pair_count * 2.5)))
            if pair_count == 1:
                axes = np.array([axes], dtype=object)

            with torch.no_grad():
                for i in range(pair_count):
                    anchor = anchors[i]
                    hidden_state = anchor["hidden_state"].unsqueeze(0).to(self.device)
                    direction = anchor["direction"].unsqueeze(0).to(self.device)
                    info = anchor["info"].unsqueeze(0).to(self.device)
                    di_value = di_values[i]
                    di = torch.tensor([[di_value]], device=self.device, dtype=info.dtype)
                    c = self.process_condition(info, hidden_state, direction, di)
                    sampled = self.sample_with_ode(c)
                    sampled_image = self._decode_from_target_space(sampled)
                    sampled_vis = torch.clamp(sampled_image * 0.5 + 0.5, 0.0, 1.0)[0].detach().cpu()

                    axes[i][0].imshow(anchor["real"].permute(1, 2, 0).numpy())
                    axes[i][0].set_title(f"GT | {anchor['exp']} | t={anchor['timestep']} | GT_DI={anchor['gt_di']:.2f}")
                    axes[i][0].axis("off")
                    axes[i][1].imshow(sampled_vis.permute(1, 2, 0).numpy())
                    axes[i][1].set_title(f"Inference | DI={di_value:.2f}")
                    axes[i][1].axis("off")

            step_num = int(self.global_step)
            fig.suptitle(f"CFM Fixed-DI Inference | Step {step_num:05d}")
            plt.tight_layout()

            save_dir = os.path.join(self.result_dir, "inference_samples")
            os.makedirs(save_dir, exist_ok=True)
            fig_path = os.path.join(save_dir, f"fixed_di_step_{step_num:05d}.png")
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"[CFM] Saved fixed-DI inference samples: {fig_path}")
            if wandb.run is not None:
                # Unique key per step so each inference panel stays independent in W&B.
                wandb.log(
                    {f"CFM/FixedDI/step_{step_num:05d}": wandb.Image(fig)},
                    step=step_num,
                )
            plt.close(fig)
            self._has_logged_initial_inference = True

        if self.add_DI_loss and len(self.validation_outputs["gt_DI"]) > 0:
            gt = torch.concatenate(self.validation_outputs["gt_DI"]).cpu().numpy()
            prediction = torch.concatenate(self.validation_outputs["pred_DI"]).cpu().numpy()
            self.log("val/r2", r2_score(gt, prediction))
        self.validation_outputs = {"gt_DI": [], "pred_DI": [], "anchor_candidates": []}

    def configure_optimizers(self):
        optimizer_name = getattr(self, "optimizer_name", "adamw")
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.velocity_net.parameters(), lr=self.lr)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.velocity_net.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        if self.lr_schedule != "warmup_cosine":
            return optimizer

        def lr_lambda(step_idx):
            if self.warmup_steps > 0 and step_idx < self.warmup_steps:
                return float(step_idx + 1) / float(self.warmup_steps)
            remain = max(self.total_steps - self.warmup_steps, 1)
            progress = min(max((step_idx - self.warmup_steps) / remain, 0.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_before_optimizer_step(self, optimizer):
        # Log pre-clip L2 so the W&B y-axis shows actual magnitude (not capped at 1).
        grad_norm_sq = torch.zeros(1, device=self.device)
        for param in self.velocity_net.parameters():
            if param.grad is not None:
                grad_norm_sq += torch.sum(param.grad.detach() ** 2)
        grad_norm_l2 = torch.sqrt(grad_norm_sq).squeeze(0)
        self.log(
            "train/grad_norm_L2",
            grad_norm_l2,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        if self.grad_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_max_norm)


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
        self.condition_mode = str(kwargs.get("condition_mode", "full")).lower()
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
            self.condition_mode,
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