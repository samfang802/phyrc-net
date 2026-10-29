"""LightningModule for ADM-UNet conditional flow matching."""

import lightning.pytorch as pl

from src.model.cfm.condition_mixin import CFMConditionMixin
from src.model.cfm.inference_mixin import CFMInferenceMixin
from src.model.cfm.lifecycle_mixin import CFMLifecycleMixin
from src.model.cfm.opa_mixin import CFMOPAMixin
from src.model.cfm.options import bind_cfm_options
from src.model.cfm.patch_mixin import CFMPatchMixin
from src.model.cfm.sampling_mixin import CFMSamplingMixin
from src.model.cfm.training_mixin import CFMTrainingMixin


class CFMADM(
    CFMPatchMixin,
    CFMSamplingMixin,
    CFMLifecycleMixin,
    CFMConditionMixin,
    CFMInferenceMixin,
    CFMTrainingMixin,
    CFMOPAMixin,
    pl.LightningModule,
):
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
        bind_cfm_options(
            self,
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
            kwargs,
        )
