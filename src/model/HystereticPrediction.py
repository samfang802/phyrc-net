import wandb
import os
import numpy as np
import torch.nn as nn
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from timm.scheduler import CosineLRScheduler

import torch
from src.model.Component.rnn import LSTM, GRU, Predictor 



class HystereticLSTM(pl.LightningModule):
    def __init__(self, 
            architecture,
            signal_feat=7,
            hidden_state_dim=1024,
            layers=4,
            MLP_activation="Softsign",
            bidirectional=False,
            lr=1e-3, 
            result_dir="results",
            **kwargs
        ):
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs", "result_dir"])     
        self.signal_feat = signal_feat
        self.hidden_state_dim = hidden_state_dim
        self.layers = layers
        self.MLP_activation = MLP_activation
        self.bidirectional = bidirectional
        
        self.simulate = kwargs["simulate"]
        
        self.lr = lr
        self.result_dir = result_dir
        
        
        self.training_outputs = {"h_loss":[], "simulated_loop":[], "predicted_force":[], "test_force":[], "max_simulate_force":[], "id_list":[]}
        self.validation_outputs = {"h_loss":[], "simulated_loop":[], "predicted_force":[], "test_force":[], "max_simulate_force":[], "id_list":[]}

        self.L = LSTM(
            input_size=self.signal_feat, 
            hidden_size=self.hidden_state_dim, 
            num_layers=self.layers, 
            bidirectional=self.bidirectional
        )
        
        self.P = Predictor(
            num_layers=3,
            activation=self.MLP_activation,
            bidirectional=self.bidirectional,
            hidden_state_dim=self.hidden_state_dim
        )
        

    def loop_prediction_loss(self, pred, gt):
        # The NRMSE is calculated as the RMSE divided by the range of the observed values, expressed as a percentage.
        return torch.sqrt(torch.mean((pred - gt) ** 2)) / (torch.max(gt) - torch.min(gt))

    def forward(self, simulated_loop):
        latent, _ = self.L(simulated_loop)
        predicted_force = self.P(latent)

        return predicted_force
    
    def on_train_epoch_start(self):
        self.log('Learning rate', self.trainer.optimizers[0].param_groups[0]['lr'])


    def training_step(self, batch, batch_id):
        simulated_loop, test_force, original_length, maximun_simulated_force, exp = batch
        
        predicted_force = self.forward(simulated_loop)
        predicted_loss = self.loop_prediction_loss(predicted_force, test_force)
        
        
        self.training_outputs["h_loss"].append(predicted_loss)
        for ncree in ["R307", "R315", "R330", "R615", "R1015", "COC", "CTR1"]:
            if ncree in exp:
                index = exp.index(ncree)
                self.training_outputs["simulated_loop"].append(simulated_loop[index, :original_length[index], :])
                self.training_outputs["predicted_force"].append(predicted_force[index, :original_length[index], :])
                self.training_outputs["test_force"].append(test_force[index, :original_length[index], :])
                self.training_outputs["max_simulate_force"].append(maximun_simulated_force[index])
                self.training_outputs["id_list"].append(exp[index])
            
            
        return predicted_loss
        
    
    def validation_step(self, batch, batch_id):
        simulated_loop, test_force, original_length, maximun_simulated_force, exp = batch
        
        predicted_force = self.forward(simulated_loop)
        predicted_loss = self.loop_prediction_loss(predicted_force, test_force)
        
        self.validation_outputs["h_loss"].append(predicted_loss)
        for ncree in ["R307", "R315", "R330", "R615", "R1015", "COC", "CTR1"]:
            if ncree in exp:
                index = exp.index(ncree)
                self.validation_outputs["simulated_loop"].append(simulated_loop[index, :original_length[index], :])
                self.validation_outputs["predicted_force"].append(predicted_force[index, :original_length[index], :])
                self.validation_outputs["test_force"].append(test_force[index, :original_length[index], :])
                self.validation_outputs["max_simulate_force"].append(maximun_simulated_force[index])
                self.validation_outputs["id_list"].append(exp[index])
            
        
        
    def configure_optimizers(self):
        opt = torch.optim.Adam(list(self.L.parameters()) + list(self.P.parameters()), lr=self.lr)
        scheduler = CosineLRScheduler(opt, t_initial=self.trainer.max_epochs, \
                                        warmup_t=int(self.trainer.max_epochs/10), warmup_lr_init=5e-6, warmup_prefix=True)
        
        return [opt], [{"scheduler": scheduler, "interval": "epoch"}]
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)
    
    def on_train_epoch_start(self):
        self.log('Learning rate', self.trainer.optimizers[0].param_groups[0]['lr'])
    
    def on_train_epoch_end(self):
        h_loss = torch.stack(self.training_outputs["h_loss"]).mean()
        self.log("train/loss", h_loss)
        self.training_outputs["h_loss"].clear()
        
        for i, exp in enumerate(self.training_outputs["id_list"]):
            predicted_force = self.training_outputs["predicted_force"][i].cpu().detach().numpy()
            simulated_loop = self.training_outputs["simulated_loop"][i].cpu().detach().numpy()
            if self.simulate:
                drift = simulated_loop[:, -2]
                simulated_force = simulated_loop[:, -1]
            else:
                drift = simulated_loop[:, -1]
                
            test_force = self.training_outputs["test_force"][i].cpu().detach().numpy()
            max_simulated_force = self.training_outputs["max_simulate_force"][i].cpu().detach().numpy()
            fig = plt.figure()
            plt.plot(drift, test_force * max_simulated_force, label="Experiment", color="tab:blue", alpha=0.7)
            plt.plot(drift, predicted_force * max_simulated_force, label="Predicted", linestyle="--", color="tab:red", alpha=0.7)
            if self.simulate:
                plt.plot(drift, simulated_force * max_simulated_force, label="Simulated", linestyle="-.", color="tab:green", alpha=0.3)
            plt.xlabel("Drift (%)")
            plt.ylabel("Force (kN)")
            plt.legend()
            plt.title(f"{exp} (Epoch={self.current_epoch})")
            plt.savefig(os.path.join(self.result_dir, "HystereticLoop", "train", f"{exp}_epoch_{self.current_epoch}.svg"))
            wandb.log({f"Hysteretic Loop (train)/Experiment {exp}": wandb.Image(fig)})
            plt.close()
        
        
        self.training_outputs = {"h_loss":[], "simulated_loop":[], "predicted_force":[], "test_force":[], "max_simulate_force":[], "id_list":[]}
    
    def on_validation_epoch_end(self):
        h_loss = torch.stack(self.validation_outputs["h_loss"]).mean()
        self.log("validation/loss", h_loss)       
        
        for i, exp in enumerate(self.validation_outputs["id_list"]):
            predicted_force = self.validation_outputs["predicted_force"][i].cpu().numpy()
            simulated_loop = self.validation_outputs["simulated_loop"][i].cpu().numpy()
            if self.simulate:
                drift = simulated_loop[:, -2]
                simulated_force = simulated_loop[:, -1]
            else:
                drift = simulated_loop[:, -1]
                
            test_force = self.validation_outputs["test_force"][i].cpu().detach().numpy()
            max_simulated_force = self.validation_outputs["max_simulate_force"][i].cpu().detach().numpy()
            fig = plt.figure()
            plt.plot(drift, test_force * max_simulated_force, label="Experiment", color="tab:blue", alpha=0.7)
            plt.plot(drift, predicted_force * max_simulated_force, label="Predicted", linestyle="--", color="tab:red", alpha=0.7)
            if self.simulate:
                plt.plot(drift, simulated_force * max_simulated_force, label="Simulated", linestyle="-.", color="tab:green", alpha=0.3)
            plt.xlabel("Drift (%)")
            plt.ylabel("Force (kN)")
            plt.legend()
            plt.title(f"{exp} (Epoch={self.current_epoch})")
            plt.savefig(os.path.join(self.result_dir, "HystereticLoop", "val", f"{exp}_epoch_{self.current_epoch}.svg"))
            wandb.log({f"Hysteretic Loop (val)/Experiment {exp}": wandb.Image(fig)})
            plt.close("all")
        
        
        self.validation_outputs = {"h_loss":[], "simulated_loop":[], "predicted_force":[], "test_force":[], "max_simulate_force":[], "id_list":[]}

        

class HystereticGRU(HystereticLSTM):
    def __init__(self, 
            architecture,
            signal_feat=7,
            hidden_state_dim=1024,
            layers=4,
            MLP_activation="Softsign",
            bidirectional=False,
            lr=1e-3, 
            result_dir="results",
            **kwargs
        ):
        super().__init__(
            architecture,
            signal_feat,
            hidden_state_dim,
            layers,
            MLP_activation,
            bidirectional,
            lr, 
            result_dir,
            **kwargs
        )
        
        self.L = GRU(
            input_size=self.signal_feat, 
            hidden_size=self.hidden_state_dim, 
            num_layers=self.layers, 
            bidirectional=self.bidirectional
        )
        
        self.P = Predictor(
            num_layers=3,
            activation=self.MLP_activation,
            bidirectional=self.bidirectional,
            hidden_state_dim=self.hidden_state_dim
        )
        
        self.loop_prediction_loss = nn.MSELoss(reduction="mean")
        
    
