import wandb
import os
import numpy as np
import torch.nn as nn
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
from torchvision.models import *
from sklearn.metrics import r2_score

class Predictor(nn.Module):
    def __init__(self, num_layers, activation, hidden_dim=1024):
        super().__init__()
        modules = []
        match activation:
            case "relu":
                activation = nn.ReLU()
            case "softsign":
                activation = nn.Softsign()
            case "silu":
                activation = nn.SiLU()
        
        
        
        for i in range(num_layers):
            if i != (num_layers - 1):
                modules.append(nn.Linear(hidden_dim, hidden_dim // 2))
                modules.append(activation)
                hidden_dim = hidden_dim // 2
            else:
                modules.append(nn.Linear(hidden_dim, 1))
        self.fcn = nn.Sequential(*modules)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fcn(x)
        x = self.relu(x)
        return x

class DamageQuantification(pl.LightningModule):
    def __init__(self, 
            backbone,
            pretrained,
            lr=1e-3, 
            result_dir="results",
            **kwargs
        ):
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs", "result_dir"])  
        self.features_dim = kwargs["features_dim"]
        weight = None   
        if pretrained:
            weight = "IMAGENET1K_V1"
            
        match backbone:
            case "vit_b_16":
                self.backbone = vit_b_16(weights=weight)
                self.backbone.heads = nn.Identity()
                self.hidden_dim = 768
                
            case "vit_l_16":
                self.backbone = vit_l_16(weights=weight)
                self.backbone.heads = nn.Identity()
                self.hidden_dim = 1024
                
            case "vit_b_32":
                self.backbone = vit_b_32(weights=weight)
                self.backbone.heads = nn.Identity()
                self.hidden_dim = 768
            
            case "vit_l_32":
                self.backbone = vit_l_32(weights=weight)
                self.backbone.heads = nn.Identity()
                self.hidden_dim = 1024
            
            case "swin_b":
                self.backbone = swin_b(weights=weight)
                self.backbone.head = nn.Identity()
                self.hidden_dim = 1024
                
            case "resnet101":
                self.backbone = resnet101(weights=weight)
                self.backbone.fc = nn.Identity()
                self.hidden_dim = 2048
        
        self.P = Predictor(
            num_layers=3,
            activation="relu",
            hidden_dim=self.hidden_dim + self.features_dim
        )
        
        self.lr = lr
        self.result_dir = result_dir
        
        
        self.training_outputs = {"loss":[], "prediction":[], "gt":[], "exp":[]}
        self.validation_outputs = {"loss":[], "prediction":[], "gt":[], "exp":[]}

        
        self.mse_loss = nn.MSELoss()


    def forward(self, image, features):
        latent =  self.backbone(image)
        latent = torch.cat([latent, features], dim=1)
        prediction = self.P(latent)

        return prediction
    
    def on_train_epoch_start(self):
        self.log('Learning rate', self.trainer.optimizers[0].param_groups[0]['lr'])


    def training_step(self, batch, batch_id):
        image, features, exp, DI = batch
        
        prediction = self.forward(image, features)
        loss = self.mse_loss(DI, prediction)
        self.training_outputs["loss"].append(loss)
        self.training_outputs["prediction"].append(prediction)
        self.training_outputs["gt"].append(DI)
        self.training_outputs["exp"] += exp
            
        return loss
        
    
    def validation_step(self, batch, batch_id):
        image, features, exp, DI = batch
        
        prediction = self.forward(image, features)
        loss = self.mse_loss(DI, prediction)
        self.validation_outputs["loss"].append(loss)
        self.validation_outputs["prediction"].append(prediction)
        self.validation_outputs["gt"].append(DI)
        self.validation_outputs["exp"] += exp
            

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = CosineLRScheduler(opt, t_initial=self.trainer.max_epochs, \
        #                                 warmup_t=int(self.trainer.max_epochs/10), warmup_lr_init=5e-6, warmup_prefix=True)
        # scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
        return opt
        # return [opt], [{"scheduler": scheduler, "interval": "epoch"}]
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)
    
    def on_train_epoch_start(self):
        self.log('Learning rate', self.trainer.optimizers[0].param_groups[0]['lr'])
    
    def on_train_epoch_end(self):
        loss = torch.stack(self.training_outputs["loss"]).mean()
        prediction = torch.cat(self.training_outputs["prediction"]).cpu().detach().numpy()
        gt = torch.cat(self.training_outputs["gt"]).cpu().detach().numpy()
        r2 = r2_score(gt, prediction)
        
        fig = self.draw(gt, prediction, self.training_outputs["exp"])
        wandb.log({"figure/train": wandb.Image(fig)})
        
        self.log("train/loss", loss)
        self.log("train/r2", r2)
        
        self.training_outputs = {"loss":[], "prediction":[], "gt":[], "exp":[]}

    
    def on_validation_epoch_end(self):
        loss = torch.stack(self.validation_outputs["loss"]).mean()
        prediction = torch.cat(self.validation_outputs["prediction"]).cpu().numpy()
        gt = torch.cat(self.validation_outputs["gt"]).cpu().numpy()
        r2 = r2_score(gt, prediction)
        
        fig = self.draw(gt, prediction, self.validation_outputs["exp"])
        wandb.log({"figure/validation": wandb.Image(fig)})
        
        self.log("validation/loss", loss)       
        self.log("validation/r2", r2)
        
        plt.close(fig)

        self.validation_outputs = {"loss":[], "prediction":[], "gt":[], "exp":[]}

    def draw(self, gt, prediction, exp):
        exp = np.array(exp)
        fig, ax = plt.subplots()
        ax.axline((0, 0), slope=1, color="black")
        for g in np.unique(exp):
            i = np.where(exp == g)[0]
            ax.scatter(gt[i], prediction[i], label=g, s=3)
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        ax.set_xlim(0, 1.4)
        ax.set_ylim(0, 1.4)
        ax.legend()
        return fig

