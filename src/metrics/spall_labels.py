"""Per-frame spalling-area labels (R_spall) used by the evaluation and analysis scripts.

Default: the hand-drawn spall polygons (Data/DamagePattern/anylabel/*.json, exported by
scripts/export/export_anylabel_r_spall.py). The older Data/DamagePattern/r_spall.csv has known
errors (spall on intact frames, missing spall on heavily spalled ones); pass ``--r_spall_csv old``
to reproduce numbers computed with it (before 2026-10-25).

The DI quantification datasets (src/data/vdi.py) keep their own default; this module is only for eval.
"""

from __future__ import annotations

R_SPALL_CSV = "Data/DamagePattern/anylabel_r_spall.csv"
R_SPALL_CSV_OLD = "Data/DamagePattern/r_spall.csv"
_ALIASES = {"anylabel": R_SPALL_CSV, "new": R_SPALL_CSV, "old": R_SPALL_CSV_OLD}


def resolve(path: str | None = None) -> str:
    """``None`` / 'anylabel' / 'new' -> anylabel csv; 'old' -> r_spall.csv; anything else is a path."""
    if not path:
        return R_SPALL_CSV
    return _ALIASES.get(path, path)


def add_arg(parser) -> None:
    parser.add_argument(
        "--r_spall_csv",
        default=R_SPALL_CSV,
        help=f"R_spall labels: a csv path, 'anylabel' (default, {R_SPALL_CSV}) or 'old' ({R_SPALL_CSV_OLD})",
    )


def load(path: str | None = None):
    """filename -> R_spall as a pandas Series."""
    import pandas as pd

    return pd.read_csv(resolve(path)).set_index("filename")["R_spall"]
