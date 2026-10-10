"""Pairwise (Siamese) sampling for same-specimen DI / vDI evolution learning."""

from __future__ import annotations

import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from src.data.quant_image import build_quant_image_tensor
from src.data.vdi import _attach_vdi_labels, _load_r_spall_map, _resolve_r_spall_csv


class DamageIndexPairwiseDataset(Dataset):
    """
    Same-specimen pairwise sampling for Siamese ranking.

    ``target_mode``:
      - ``di``: order / supervise with original OpenSees DI
      - ``vdi``: order / supervise with residual-corrected vDI

    Optionally also returns white-box ``R_spall`` for multi-task auxiliaries.

    For each index, sample two images from the same ``exp``, order so
    GT_A <= GT_B, return (Img_A, Img_B, GT_A, GT_B, exp[, Rs_A, Rs_B]).
    """

    def __init__(
        self,
        root,
        mode,
        image_size=128,
        data_path="id_list_val_R307.json",
        use_augmentation=False,
        gray_mean=0.485,
        gray_std=0.229,
        pattern_dirname="DamagePattern_inpaint_selfcondnoDI",
        use_mask=False,
        use_rgb=False,
        use_clahe=False,
        seed=None,
        beta=0.3,
        r_spall_csv=None,
        r_bar_method="poly",
        poly_degree=2,
        ma_window=51,
        target_mode="vdi",
        return_rs=False,
        modes=None,
        alpha=None,  # deprecated, ignored
        di_max=None,  # deprecated, ignored
        **kwargs,
    ):
        super().__init__()
        self.root = root
        # ``modes`` overrides single ``mode`` (e.g. ["train","val"] merges splits).
        if modes is None:
            modes = [mode]
        self.modes = [str(m) for m in modes]
        self.mode = "+".join(self.modes)
        self.image_size = int(image_size)
        self.use_augmentation = bool(use_augmentation) and ("train" in self.modes)
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
        if self.target_mode not in ("di", "vdi"):
            raise ValueError(f"target_mode must be 'di' or 'vdi', got {target_mode!r}")
        self.return_rs = bool(return_rs)
        self.pattern_dirname = str(pattern_dirname)
        self.pattern_root = os.path.join(self.root, self.pattern_dirname)
        self.mask_dir = os.path.join(self.pattern_root, "mask")
        self.images_dir = os.path.join(self.pattern_root, "images")
        self._missing_mask = set()
        self._rng = random.Random(seed)

        list_path = os.path.join(self.pattern_root, data_path)
        split_json = json.load(open(list_path))
        image_list = []
        for m in self.modes:
            if m not in split_json:
                raise KeyError(f"split {m!r} not in {list_path}; keys={list(split_json)}")
            image_list.extend(split_json[m])
        # Unique, preserve order.
        image_list = list(dict.fromkeys(image_list))
        label_full = pd.read_csv(os.path.join(self.pattern_root, "label.csv"))
        self.label = label_full[label_full["filename"].isin(image_list)].reset_index(drop=True)

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
                f"[DamageIndexPairwiseDataset] no R_spall csv found; using R_spall=0 "
                f"(pass r_spall_csv=... or place {self.pattern_root}/r_spall.csv)",
                flush=True,
            )

        self.label, self.r_bar, self.r_bar_info = _attach_vdi_labels(
            self.label,
            r_map,
            beta=self.beta,
            r_bar_method=self.r_bar_method,
            poly_degree=self.poly_degree,
            ma_window=self.ma_window,
            fit_label_df=label_full,
        )

        # groups[exp] = list of row indices with >= 1 image; keep only pairable.
        self.groups = {}
        for i, exp in enumerate(self.label["exp"].astype(str).tolist()):
            self.groups.setdefault(exp, []).append(i)
        self.pairable_exps = [e for e, idxs in self.groups.items() if len(idxs) >= 2]
        if not self.pairable_exps:
            raise RuntimeError(
                f"[DamageIndexPairwiseDataset] no specimen with >=2 images in mode={mode}"
            )

        # One entry per image that belongs to a pairable specimen → stable epoch length.
        self.anchors = []
        for exp in self.pairable_exps:
            self.anchors.extend(self.groups[exp])

        n_pos = int(np.sum(self.label["R_spall"].to_numpy(dtype=float) > 0))
        print(
            f"[DamageIndexPairwiseDataset] mode={self.mode} images={len(self.label)} "
            f"pairable_specimens={len(self.pairable_exps)} anchors={len(self.anchors)} "
            f"use_mask={self.use_mask} use_rgb={self.use_rgb} use_clahe={self.use_clahe} "
            f"beta={self.beta} R̄={self.r_bar_info} "
            f"R_spall>0={n_pos}/{len(self.label)} csv={csv_path}",
            flush=True,
        )

    def __len__(self):
        return int(len(self.anchors))

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
                    f"[DamageIndexPairwiseDataset] mask missing for {filename}; using all-ones",
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
        return (1.0 - mask / 255.0).clamp_(0.0, 1.0)

    def _load_image(self, filename):
        image = build_quant_image_tensor(
            os.path.join(self.images_dir, filename),
            self.image_size,
            use_rgb=self.use_rgb,
            use_clahe=self.use_clahe,
            gray_mean=self.gray_mean,
            gray_std=self.gray_std,
        )
        if self.use_mask:
            mask = self._load_mask_tensor(filename)
            return torch.cat([image, mask], dim=0)
        return image

    def _target_from_row(self, row) -> float:
        if self.target_mode == "di":
            if "DI" in row.index and pd.notna(row["DI"]):
                return float(row["DI"])
            return float(row["vDI_original"])
        return float(row["vDI"])

    def __getitem__(self, index):
        # Anchor tip: pick one image, then a different peer from the same specimen.
        i = int(self.anchors[int(index) % len(self.anchors)])
        row_i = self.label.iloc[i]
        exp = str(row_i["exp"])
        peers = self.groups[exp]
        j = i
        while j == i:
            j = self._rng.choice(peers)

        row_j = self.label.iloc[j]
        di_i = self._target_from_row(row_i)
        di_j = self._target_from_row(row_j)

        # Order by target: A <= B (equal allowed).
        if di_i <= di_j:
            row_a, row_b = row_i, row_j
            di_a, di_b = di_i, di_j
        else:
            row_a, row_b = row_j, row_i
            di_a, di_b = di_j, di_i

        img_a = self._load_image(str(row_a["filename"]))
        img_b = self._load_image(str(row_b["filename"]))

        if self.use_augmentation and self._rng.random() < 0.5:
            img_a = torch.flip(img_a, dims=(-1,))
            img_b = torch.flip(img_b, dims=(-1,))

        gt_a = torch.FloatTensor([di_a])
        gt_b = torch.FloatTensor([di_b])
        if self.return_rs:
            rs_a = torch.FloatTensor([float(row_a["R_spall"])])
            rs_b = torch.FloatTensor([float(row_b["R_spall"])])
            return img_a, img_b, gt_a, gt_b, exp, rs_a, rs_b
        return img_a, img_b, gt_a, gt_b, exp


def collate_pairwise(batch):
    """Stack pairwise samples into batched tensors."""
    if len(batch[0]) == 7:
        img_a, img_b, gt_a, gt_b, exp, rs_a, rs_b = zip(*batch)
        return (
            torch.stack(img_a, dim=0),
            torch.stack(img_b, dim=0),
            torch.stack(gt_a, dim=0),
            torch.stack(gt_b, dim=0),
            list(exp),
            torch.stack(rs_a, dim=0),
            torch.stack(rs_b, dim=0),
        )
    img_a, img_b, gt_a, gt_b, exp = zip(*batch)
    return (
        torch.stack(img_a, dim=0),
        torch.stack(img_b, dim=0),
        torch.stack(gt_a, dim=0),
        torch.stack(gt_b, dim=0),
        list(exp),
    )
