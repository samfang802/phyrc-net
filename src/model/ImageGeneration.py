"""Damage-pattern generators.

Attribute access loads only the requested class, so the default CFMADM
path does not import Laplacian or other experiments.
"""


def __getattr__(name):
    if name == "CFMADM":
        from src.model.cfm.module import CFMADM
        return CFMADM
    if name == "LaplacianBandCFM":
        from src.model.exp.laplacian_module import LaplacianBandCFM
        return LaplacianBandCFM
    if name in {
        "build_generation_condition",
        "build_physics_condition",
        "build_scalar_condition",
        "parse_fixed_di_values",
        "parse_patch_size_probs",
    }:
        from src.model import cfm as cfm_pkg
        return getattr(cfm_pkg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
