"""Damage-pattern image generation models.

This module is a thin compatibility facade. Implementation lives in ``src.model.cfm``.
"""

from src.model.cfm import (
    CFMADM,
    LaplacianBandCFM,
    build_generation_condition,
    build_physics_condition,
    build_scalar_condition,
    parse_fixed_di_values,
    parse_patch_size_probs,
)

# Private aliases kept for any in-repo callers that used the old names.
_parse_fixed_di_values = parse_fixed_di_values
_parse_patch_size_probs = parse_patch_size_probs

__all__ = [
    "CFMADM",
    "LaplacianBandCFM",
    "build_generation_condition",
    "build_scalar_condition",
    "build_physics_condition",
    "parse_fixed_di_values",
    "parse_patch_size_probs",
]
