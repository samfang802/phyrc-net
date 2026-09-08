"""Conditional Flow Matching (CFM) damage-pattern generation package."""

from src.model.cfm.condition import (
    build_generation_condition,
    build_physics_condition,
    build_scalar_condition,
    parse_fixed_di_values,
    parse_patch_size_probs,
)
from src.model.cfm.module import CFMADM
from src.model.cfm.laplacian_module import LaplacianBandCFM, compose_laplacian_sample
from src.model.cfm.tto import optimize_gru_features_tto, total_variation_loss
from src.model.cfm.oracle_adapter import (
    ResidualPhysicsAdapter,
    adapter_loss,
    extract_oracle_condition,
)

__all__ = [
    "CFMADM",
    "LaplacianBandCFM",
    "compose_laplacian_sample",
    "build_generation_condition",
    "build_scalar_condition",
    "build_physics_condition",
    "parse_fixed_di_values",
    "parse_patch_size_probs",
    "optimize_gru_features_tto",
    "total_variation_loss",
    "ResidualPhysicsAdapter",
    "adapter_loss",
    "extract_oracle_condition",
]
