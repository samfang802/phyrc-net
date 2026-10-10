"""Backward-compatible facade for mixed training arguments."""

from .mixed import (
    DEFAULT_COLUMN_INFO,
    check_args,
    parse_args,
    save_args,
    set_dataset_args,
)

__all__ = [
    "DEFAULT_COLUMN_INFO",
    "parse_args",
    "check_args",
    "save_args",
    "set_dataset_args",
]
