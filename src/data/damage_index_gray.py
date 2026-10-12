"""Siamese grayscale damage-index dataset."""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .quant_image import build_quant_image_tensor
from .vdi import _attach_vdi_labels, _load_r_spall_map, _resolve_r_spall_csv

class DamageIndexGrayMaskDataset(Dataset):
    """
    Damage-index regression from grayscale images (± optional SAM mask).

    By default attaches empirical spall trend labels::

        Δ = R̄(DI) - R_spall
        vDI = max(0, DI + β · Δ)

    ``target_mode`` selects which regression target is returned:
      - ``\"di\"``: original OpenSees DI
      - ``\"vdi\"``: residual-corrected vDI
      - ``\"rs\"``: AnyLabel union R_spall

    Returns: image, exp, target[, R_spall]  (R_spall when ``return_rs=True``)
    """

    def __init__(
        self,
        root,
        mode,
        image_size=224,
        data_path="id_list_old.json",
        use_augmentation=False,
        gray_mean=0.485,
        gray_std=0.229,
        pattern_dirname="DamagePattern",
        use_mask=True,
        use_rgb=False,
        use_clahe=False,
        beta=0.3,
        r_spall_csv=None,
        r_bar_method="poly",
        poly_degree=2,
        ma_window=51,
        target_mode="vdi",
        return_rs=False,
        alpha=None,  # deprecated, ignored
        di_max=None,  # deprecated, ignored
        **kwargs,
    ):
        super().__init__()
        self.root = root
        self.mode = mode
        self.image_size = int(image_size)
        self.use_augmentation = bool(use_augmentation) and mode == "train"
        self.gray_mean = float(gray_mean)
        self.gray_std = float(gray_std)
        self.use_mask = bool(use_mask)
        self.use_rgb = bool(use_rgb)
        self.use_clahe = bool(use_clahe)
        if self.use_rgb and self.use_mask:
            raise ValueError("use_rgb cannot be combined with use_mask")
        self.beta = float(beta)
        self.r_bar_method = str(r_bar_method)
        self.poly_degree = int(poly_degree)
        self.ma_window = int(ma_window)
        self.target_mode = str(target_mode).lower().strip()
        if self.target_mode not in ("di", "vdi", "rs"):
            raise ValueError(f"target_mode must be 'di', 'vdi', or 'rs', got {target_mode!r}")
        self.return_rs = bool(return_rs)
        self.pattern_dirname = str(pattern_dirname)
        self.pattern_root = os.path.join(self.root, self.pattern_dirname)
        self.mask_dir = os.path.join(self.pattern_root, "mask")
        self.images_dir = os.path.join(self.pattern_root, "images")
        self._missing_mask = set()

        list_path = os.path.join(self.pattern_root, data_path)
        self.image_list = json.load(open(list_path))[mode]
        label_full = pd.read_csv(os.path.join(self.pattern_root, "label.csv"))
        self.label = label_full[label_full["filename"].isin(self.image_list)].reset_index(drop=True)

        csv_path = _resolve_r_spall_csv(self.pattern_root, r_spall_csv)
        if csv_path is None:
            alt = os.path.join(
                os.path.abspath(os.path.join(self.root, "..")),
                "results",
                "eadi_opt",
                "original_optimized",
                "predictions.csv",
            )
            if os.path.isfile(alt):
                csv_path = alt
        self.r_spall_csv = csv_path
        r_map = _load_r_spall_map(csv_path)
        if not r_map:
            print(
                f"[DamageIndexGrayMaskDataset] no R_spall csv found; using R_spall=0 "
                f"(pass r_spall_csv=... or place {self.pattern_root}/r_spall.csv)",
                flush=True,
            )

        # Fit R̄ on the full label table (stable prior across splits).
        self.label, self.r_bar, self.r_bar_info = _attach_vdi_labels(
            self.label,
            r_map,
            beta=self.beta,
            r_bar_method=self.r_bar_method,
            poly_degree=self.poly_degree,
            ma_window=self.ma_window,
            fit_label_df=label_full,
        )
        n_pos = int(np.sum(self.label["R_spall"].to_numpy(dtype=float) > 0))
        print(
            f"[DamageIndexGrayMaskDataset] mode={mode} n={len(self.label)} "
            f"target={self.target_mode} return_rs={self.return_rs} beta={self.beta} "
            f"R̄={self.r_bar_info} R_spall>0={n_pos}/{len(self.label)} csv={csv_path}",
            flush=True,
        )

    def __len__(self):
        return int(self.label.shape[0])

    def _resolve_mask_path(self, filename):
        stem, _ = os.path.splitext(str(filename))
        for path in (
            os.path.join(self.mask_dir, f"{stem}_mask.png"),
            os.path.join(self.mask_dir, f"{stem}.png"),
            os.path.join(self.mask_dir, str(filename)),
        ):
            if os.path.exists(path):
                return path
        return None

    def _load_mask_tensor(self, filename):
        mask_path = self._resolve_mask_path(filename)
        if mask_path is None:
            if filename not in self._missing_mask:
                print(
                    f"[DamageIndexGrayMaskDataset] mask missing for {filename}; using all-ones",
                    flush=True,
                )
                self._missing_mask.add(filename)
            return torch.ones(1, self.image_size, self.image_size, dtype=torch.float32)

        mask_np = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = torch.from_numpy(mask_np).unsqueeze(0)
        if mask.shape[-2:] != (self.image_size, self.image_size):
            mask = F.interpolate(
                mask.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="nearest",
            ).squeeze(0)
        # Invert white-rebar PNG → concrete=1, rebar=0
        return (1.0 - mask / 255.0).clamp_(0.0, 1.0)

    def _target_from_row(self, row) -> float:
        if self.target_mode == "di":
            if "DI" in row.index and pd.notna(row["DI"]):
                return float(row["DI"])
            return float(row["vDI_original"])
        if self.target_mode == "rs":
            return float(row["R_spall"])
        return float(row["vDI"])

    def __getitem__(self, index):
        row = self.label.iloc[index]
        filename = str(row["filename"])
        exp = str(row["exp"])
        image_ch = build_quant_image_tensor(
            os.path.join(self.images_dir, filename),
            self.image_size,
            use_rgb=self.use_rgb,
            use_clahe=self.use_clahe,
            gray_mean=self.gray_mean,
            gray_std=self.gray_std,
        )
        if self.use_mask:
            mask = self._load_mask_tensor(filename)
            if self.use_augmentation and torch.rand(()) < 0.5:
                image_ch = torch.flip(image_ch, dims=(-1,))
                mask = torch.flip(mask, dims=(-1,))
            image_ch = torch.cat([image_ch, mask], dim=0)
        elif self.use_augmentation and torch.rand(()) < 0.5:
            image_ch = torch.flip(image_ch, dims=(-1,))

        target = torch.FloatTensor([self._target_from_row(row)])
        if self.return_rs:
            rs = torch.FloatTensor([float(row["R_spall"])])
            return image_ch, exp, target, rs
        return image_ch, exp, target
