"""Conditional flow matching. Import CFMADM from src.model.cfm.module or ImageGeneration."""

from src.model.cfm.condition import (
    build_generation_condition,
    build_physics_condition,
    build_scalar_condition,
    parse_fixed_di_values,
    parse_patch_size_probs,
    patch_size_probs_at_step,
    scalar_condition_includes_di,
)

__all__ = [
    "build_generation_condition",
    "build_scalar_condition",
    "scalar_condition_includes_di",
    "build_physics_condition",
    "parse_fixed_di_values",
    "parse_patch_size_probs",
    "patch_size_probs_at_step",
    "CFMADM",
]


def __getattr__(name):
    if name == "CFMADM":
        from src.model.cfm.module import CFMADM
        return CFMADM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
