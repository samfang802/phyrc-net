import wandb
import numpy as np
import torch.nn as nn
import lightning.pytorch as pl



import torch
import torch.nn.functional as F

from PIL import Image

from src.model.Component.diffusionmodules import Encoder, Decoder
from src.model.Component.quantize import VectorQuantizer2 as VectorQuantizer
from src.losses.vqperceptual import VQLPIPSWithDiscriminator
from src.losses.contperceptual import LPIPSWithDiscriminator


from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchvision.transforms import v2
from sklearn.metrics import r2_score


def _unpack_reconstruction_batch(batch):
    """Accept legacy image tensors and mask-aware Dataset dictionaries."""
    if isinstance(batch, dict):
        return batch["target_image"], batch["loss_mask"]
    return batch, None


class VQGAN(pl.LightningModule):
    def __init__(self,
                 lr,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=["ddconfig", "lossconfig", "kwargs"])
        self.learning_rate = lr
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = VQLPIPSWithDiscriminator(**lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        
        self.automatic_optimization = False


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def training_step(self, batch, batch_idx):
        x, _loss_mask = _unpack_reconstruction_batch(batch)
        opt_ae, opt_disc = self.optimizers()
        xrec, qloss = self(x)

        # autoencode
        self.toggle_optimizer(opt_ae)
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.quantize.zero_grad()
        self.quant_conv.zero_grad()
        self.post_quant_conv.zero_grad()
        opt_ae.zero_grad()
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.current_epoch, optimizer_idx=0,
                                        last_layer=self.get_last_layer(), split="train")

        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        self.manual_backward(aeloss)
        opt_ae.step()
        self.untoggle_optimizer(opt_ae)

        # discriminator
        self.toggle_optimizer(opt_disc)
        self.loss.discriminator.zero_grad()
        opt_disc.zero_grad()
        discloss, log_dict_disc = self.loss(qloss, x, xrec, self.current_epoch, optimizer_idx=1,
                                        last_layer=self.get_last_layer(), split="train")
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        
        self.manual_backward(discloss)
        opt_disc.step()
        self.untoggle_optimizer(opt_disc)

    def validation_step(self, batch, batch_idx):
        x, _loss_mask = _unpack_reconstruction_batch(batch)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.current_epoch, optimizer_idx=0,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, self.current_epoch, optimizer_idx=1,
                                            last_layer=self.get_last_layer(), split="val")
        
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        if batch_idx == 0:
            self.log_images(x, xrec)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    def log_images(self, x, xrec): 
        bs = x.size(0)
        images = []
        for i in range(bs):
            image_array = np.concatenate([x[i].cpu().numpy(), xrec[i].detach().cpu().numpy()], axis=1)
            image_array = np.transpose(image_array, (1, 2, 0))
            images.append(Image.fromarray(((image_array + 1) * 127.5).astype(np.uint8)))
        
        wandb.log({"Images": [wandb.Image(image) for image in images]})
    
    def on_train_epoch_start(self):
        self.log('Learning rate', self.trainer.optimizers[0].param_groups[0]['lr'])

class VQGANInterface(VQGAN):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    
class VAE(pl.LightningModule):
    def __init__(self,
                 lr,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=["ddconfig", "lossconfig", "kwargs"])
        self.learning_rate = lr
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = LPIPSWithDiscriminator(**lossconfig)
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        
        self.automatic_optimization = False


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def training_step(self, batch, batch_idx):
        inputs, _loss_mask = _unpack_reconstruction_batch(batch)
        opt_ae, opt_disc = self.optimizers()
        reconstructions, posterior = self(inputs)

        # autoencode
        self.toggle_optimizer(opt_ae)
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.quant_conv.zero_grad()
        self.post_quant_conv.zero_grad()
        opt_ae.zero_grad()
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, self.current_epoch, optimizer_idx=0,
                                            last_layer=self.get_last_layer(), split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        self.manual_backward(aeloss)
        opt_ae.step()
        self.untoggle_optimizer(opt_ae)

        # discriminator
        self.toggle_optimizer(opt_disc)
        self.loss.discriminator.zero_grad()
        opt_disc.zero_grad()
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, self.current_epoch, optimizer_idx=1, 
                                                last_layer=self.get_last_layer(), split="train")

        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        
        self.manual_backward(discloss)
        opt_disc.step()
        self.untoggle_optimizer(opt_disc)

    def validation_step(self, batch, batch_idx):
        inputs, _loss_mask = _unpack_reconstruction_batch(batch)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, self.current_epoch, optimizer_idx=0,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, self.current_epoch, optimizer_idx=1,
                                            last_layer=self.get_last_layer(), split="val")
        
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_disc, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        if batch_idx == 0:
            self.log_images(inputs, reconstructions)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    def log_images(self, x, xrec): 
        bs = x.size(0)
        images = []
        for i in range(bs):
            image_array = np.concatenate([x[i].cpu().numpy(), xrec[i].detach().cpu().numpy()], axis=1)
            image_array = np.transpose(image_array, (1, 2, 0))
            images.append(Image.fromarray(((image_array + 1) * 127.5).astype(np.uint8)))
        
        wandb.log({"Images": [wandb.Image(image) for image in images]})
    
    def on_train_epoch_start(self):
        self.log('Learning rate', self.trainer.optimizers[0].param_groups[0]['lr'])
        
        
        
class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean