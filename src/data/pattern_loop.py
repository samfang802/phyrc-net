"""Damage-pattern and hysteresis-loop paired dataset."""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .hysteresis import (
    HystereticDataset,
    _prepare_hysteresis_plot_arrays,
    _to_hidden_state_matrix,
)
from .preprocessing import standardize

class PatternLoopDataset(Dataset):
    def __init__(self, root, mode, transform, extractor, extractor_args, result_dir, data_path, column_info, **kwargs):
        super().__init__()
        self.root = root
        self.mode = mode
        self.transform = transform
        self.extractor_args = extractor_args
        self.image_list = json.load(open(os.path.join(self.root, "DamagePattern", data_path)))[mode]
        self.label = pd.read_csv(os.path.join(self.root, "DamagePattern", "label.csv"))
        self.label = self.label[self.label["filename"].isin(self.image_list)].reset_index(drop=True)
        self.base_size = int(self.label.shape[0])
        train_virtual_repeat = max(1, int(kwargs.get("train_virtual_repeat", 1)))
        self.virtual_repeat_factor = train_virtual_repeat if self.mode == "train" else 1
        self.exp_list = self.label['exp'].unique()
        print("Extracting hidden state...")
        self.info = pd.read_csv(os.path.join(self.root, "Hysteresis", f"info_w_mander_big.csv")).loc[:, column_info]
        statistics = pd.read_csv(os.path.join(self.root, "Hysteresis", f"statistics_big.csv"), index_col=0)
        self.info = standardize(self.info, statistics)
        self.feature_columns = [col for col in self.info.columns if col != "id"]
        signed_keywords = (
            "disp", "drift", "deform", "rotation", "moment", "force", "shear", "axial",
            "ux", "uy", "dx", "dy", "vx", "vy", "dir", "direction"
        )
        self.signed_feature_indices = [
            i for i, name in enumerate(self.feature_columns)
            if any(token in name.lower() for token in signed_keywords)
        ]
        self.hysteresis = {}
        self.result_dir = result_dir
        self.hidden_state = self.get_hidden_state(extractor, self.exp_list, result_dir=result_dir)
        self.use_loss_mask = bool(kwargs.get("use_loss_mask", False))
        self.mask_dir = os.path.join(self.root, "DamagePattern", "mask")
        self.use_object_loss_mask = bool(kwargs.get("use_object_loss_mask", True))
        self.use_damage_mask_condition = bool(kwargs.get("use_damage_mask_condition", False))
        self.use_undamaged_ref = bool(kwargs.get("use_undamaged_ref", False))
        self.eval_real_condition_mask = bool(kwargs.get("eval_real_condition_mask", False))
        # Temporal AR pairing: same (exp, direction), previous timestep as image_prev.
        self.use_ar_temporal_pairing = bool(kwargs.get("use_ar_temporal_pairing", False)) or bool(
            kwargs.get("use_ar_self_conditioning", False)
        ) or bool(kwargs.get("use_ar_chain_ot", False)) or bool(
            kwargs.get("use_spall_growth_mask", False)
        ) or bool(kwargs.get("use_paired_bridge", False)) or bool(
            kwargs.get("use_refiner_residual", False)
        )
        # Refiner chain training also needs the frame before the previous one.
        self.use_prev2 = bool(kwargs.get("use_refiner_residual", False))
        self._prev_iloc = self._build_prev_iloc_map() if self.use_ar_temporal_pairing else None
        self._missing_object_mask = set()
        self.image_size = int(kwargs.get("image_size", 128))
        self._info_cache = {}
        self._mask_cache = {}
        self._img_cache = {}
        self._prefix_cache = {}
        self.cache_data = bool(kwargs.get("cache_data", False))
        self.condition_perturb = bool(kwargs.get("condition_perturb", False)) and self.mode == "train"
        self.condition_perturb_dist = str(kwargs.get("condition_perturb_dist", "uniform")).lower()
        perturb_min = float(kwargs.get("condition_perturb_min", 0.01))
        perturb_max = float(kwargs.get("condition_perturb_max", 0.03))
        self.condition_perturb_min = min(perturb_min, perturb_max)
        self.condition_perturb_max = max(perturb_min, perturb_max)
        self.condition_perturb_on_hidden = bool(kwargs.get("condition_perturb_on_hidden", True))
        self.condition_perturb_on_info = bool(kwargs.get("condition_perturb_on_info", True))
        self.condition_perturb_on_di = bool(kwargs.get("condition_perturb_on_di", True))
        self.g_traj_mix = bool(kwargs.get("g_traj_mix", False)) and self.mode == "train"
        self.g_traj_mix_delta = max(1, int(kwargs.get("g_traj_mix_delta", 2)))
        self.g_traj_mix_max_lambda = float(kwargs.get("g_traj_mix_max_lambda", 0.3))
        if self.g_traj_mix_max_lambda < 0.0:
            raise ValueError(f"g_traj_mix_max_lambda must be >= 0, got {self.g_traj_mix_max_lambda}")
        self._traj_neighbor_ts = self._build_traj_neighbor_timesteps() if self.g_traj_mix else {}
        self.g_pca_dim = int(kwargs.get("g_pca_dim", 0))
        if self.g_pca_dim < 0:
            raise ValueError(f"g_pca_dim must be >= 0, got {self.g_pca_dim}")
        self._g_pca_mean = None
        self._g_pca_components = None
        if self.g_pca_dim > 0:
            self._setup_g_pca()
        if self.use_ar_temporal_pairing:
            n_paired = int(np.sum(self._prev_iloc >= 0))
            print(
                f"[PatternLoopDataset] AR temporal pairing enabled: "
                f"{n_paired}/{self.base_size} samples have a previous frame",
                flush=True,
            )
        if self.g_traj_mix:
            n_mix = sum(1 for v in self._traj_neighbor_ts.values() if v)
            print(
                f"[PatternLoopDataset] g traj mix: {n_mix}/{self.base_size} frames have "
                f"neighbors (delta={self.g_traj_mix_delta}, λ≤{self.g_traj_mix_max_lambda})",
                flush=True,
            )
        if self.cache_data:
            self._preload_cache()
        print("Done!")

    def _preload_cache(self):
        """Decode every image of this split once. Random augmentation still runs per call."""
        n_cached = 0
        use_prefix = hasattr(self.transform, "enable_prefix_cache") and self.transform.enable_prefix_cache()
        for filename in self.label["filename"].tolist():
            if filename in self._img_cache or filename in self._prefix_cache:
                continue
            image = Image.open(os.path.join(self.root, "DamagePattern", "images", filename))
            if use_prefix:
                # Resize, grayscale and CLAHE are deterministic: run them once.
                self._prefix_cache[filename] = self.transform.precompute(image)
                n_cached += 1
                continue
            if image.mode not in ("RGB", "L", "RGBA"):
                continue
            image.load()
            self._img_cache[filename] = (np.array(image), image.mode)
            n_cached += 1
        for exp in self.label["exp"].unique().tolist():
            self._info_for_exp(exp)
        if self._needs_object_mask():
            for filename in self.label["filename"].tolist():
                self._load_object_mask_tensor(filename, self.image_size)
        mb = sum(a.nbytes for a, _ in self._img_cache.values()) / 1e6
        mb += sum(t.numel() * t.element_size() for t in self._prefix_cache.values()) / 1e6
        print(
            f"[PatternLoopDataset] cache_data: {n_cached} images "
            f"({'after resize+grayscale+CLAHE' if use_prefix else 'decoded'}, {mb:.0f} MB), "
            f"{len(self._mask_cache)} masks, {len(self._info_cache)} info rows",
            flush=True,
        )

    def _info_for_exp(self, exp):
        cached = self._info_cache.get(exp)
        if cached is None:
            row = self.info[self.info["id"] == exp].iloc[:, 1:].values.squeeze()
            cached = torch.FloatTensor(row)
            self._info_cache[exp] = cached
        return cached.clone()

    def _open_image(self, filename):
        cached = self._img_cache.get(filename)
        if cached is not None:
            return Image.fromarray(cached[0])
        return Image.open(os.path.join(self.root, "DamagePattern", "images", filename))

    def _build_prev_iloc_map(self):
        """For each row, iloc of previous frame in same (exp, direction), else -1."""
        prev_iloc = np.full(len(self.label), -1, dtype=np.int64)
        for _, group in self.label.groupby(["exp", "direction"], sort=False):
            ordered = group.sort_values("timestep").index.to_numpy(dtype=np.int64)
            for k in range(1, len(ordered)):
                prev_iloc[ordered[k]] = ordered[k - 1]
        return prev_iloc

    def _build_traj_neighbor_timesteps(self):
        """Same-(exp, direction) labeled-frame neighbors within ±delta along the chain."""
        neighbors = {}
        delta = int(self.g_traj_mix_delta)
        for _, group in self.label.groupby(["exp", "direction"], sort=False):
            ordered = group.sort_values("timestep")
            exps = ordered["exp"].tolist()
            dirs_ = ordered["direction"].tolist()
            ts = ordered["timestep"].to_numpy(dtype=np.int64)
            n = len(ts)
            for k in range(n):
                key = (str(exps[k]), str(dirs_[k]), int(ts[k]))
                nearby = []
                for offset in range(1, delta + 1):
                    if k - offset >= 0:
                        nearby.append(int(ts[k - offset]))
                    if k + offset < n:
                        nearby.append(int(ts[k + offset]))
                neighbors[key] = nearby
        return neighbors

    def _maybe_mix_hidden_traj(self, hidden_state, exp, direction, timestep):
        """With probability ``g_traj_mix_max_lambda``, use a neighbor g. The image stays."""
        if not self.g_traj_mix or float(self.g_traj_mix_max_lambda) <= 0.0:
            return hidden_state
        if float(np.random.uniform(0.0, 1.0)) >= float(self.g_traj_mix_max_lambda):
            return hidden_state
        key = (str(exp), str(direction), int(timestep))
        nearby = self._traj_neighbor_ts.get(key) or []
        if not nearby:
            return hidden_state
        other_ts = int(nearby[int(np.random.randint(len(nearby)))])
        other = self.hidden_state[exp][other_ts]
        if not torch.is_tensor(other):
            other = torch.as_tensor(other, dtype=hidden_state.dtype)
        return other.to(device=hidden_state.device, dtype=hidden_state.dtype)

    def _g_pca_path(self):
        return os.path.join(self.result_dir, "g_pca.npz")

    def _stack_hidden_state_matrix(self):
        mats = []
        for mat in self.hidden_state.values():
            x = mat if torch.is_tensor(mat) else torch.as_tensor(mat)
            mats.append(x.reshape(-1, x.shape[-1]).detach().cpu().float())
        if not mats:
            raise ValueError("Cannot fit g PCA: no hidden states")
        return torch.cat(mats, dim=0).numpy()

    def _setup_g_pca(self):
        """Center + SVD (PCA, no whitening). Fit on train GRU rows; val/test load the same file."""
        k = int(self.g_pca_dim)
        path = self._g_pca_path()
        if self.mode == "train":
            x = self._stack_hidden_state_matrix().astype(np.float64)
            mean = x.mean(axis=0, keepdims=True)
            xc = x - mean
            # Xc = U S Vt ; principal axes = first k rows of Vt
            _u, s, vt = np.linalg.svd(xc, full_matrices=False)
            if k > vt.shape[0]:
                raise ValueError(f"g_pca_dim={k} > rank {vt.shape[0]}")
            components = vt[:k]
            energy = float((s[:k] ** 2).sum() / max((s ** 2).sum(), 1e-12))
            os.makedirs(self.result_dir, exist_ok=True)
            np.savez(
                path,
                mean=mean.reshape(-1).astype(np.float32),
                components=components.astype(np.float32),
                singular_values=s[:k].astype(np.float32),
                explained_energy=np.float32(energy),
            )
            print(
                f"[PatternLoopDataset] g PCA fitted: {x.shape[-1]}→{k} "
                f"N={x.shape[0]} explained_energy={energy:.4f} → {path}",
                flush=True,
            )
            mean_vec = mean.reshape(-1).astype(np.float32)
            comp = components.astype(np.float32)
        else:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"g_pca.npz not found for {self.mode} split: {path} "
                    "(train must write it first)"
                )
            payload = np.load(path)
            mean_vec = np.asarray(payload["mean"], dtype=np.float32).reshape(-1)
            comp = np.asarray(payload["components"], dtype=np.float32)
            if int(comp.shape[0]) != k:
                raise ValueError(
                    f"g_pca.npz has {comp.shape[0]} components, expected g_pca_dim={k}"
                )
            energy = float(payload["explained_energy"]) if "explained_energy" in payload.files else float("nan")
            print(
                f"[PatternLoopDataset] g PCA loaded: {comp.shape[1]}→{k} "
                f"explained_energy={energy:.4f} from {path}",
                flush=True,
            )
        self._g_pca_mean = torch.from_numpy(mean_vec)
        self._g_pca_components = torch.from_numpy(comp)

    def _apply_g_pca(self, hidden_state):
        if self._g_pca_components is None:
            return hidden_state
        mean = self._g_pca_mean.to(dtype=hidden_state.dtype)
        axes = self._g_pca_components.to(dtype=hidden_state.dtype)
        return (hidden_state - mean) @ axes.t()

    def _sample_relative_noise(self, shape, device, dtype):
        if self.condition_perturb_dist == "normal":
            if self.condition_perturb_max <= 0.0:
                return torch.zeros(shape, device=device, dtype=dtype)
            std = float(np.random.uniform(self.condition_perturb_min, self.condition_perturb_max))
            return torch.randn(shape, device=device, dtype=dtype) * std
        magnitude = torch.empty(shape, device=device, dtype=dtype).uniform_(
            self.condition_perturb_min,
            self.condition_perturb_max,
        )
        sign = torch.where(
            torch.rand(shape, device=device) < 0.5,
            torch.tensor(-1.0, device=device, dtype=dtype),
            torch.tensor(1.0, device=device, dtype=dtype),
        )
        return magnitude * sign

    def _apply_condition_perturbation(self, tensor):
        noise = self._sample_relative_noise(tensor.shape, tensor.device, tensor.dtype)
        return tensor * (1.0 + noise)

    def _needs_object_mask(self):
        """Load disk masks for loss masking and/or train-time mask-as-condition."""
        if self.use_object_loss_mask:
            return True
        if self.use_damage_mask_condition and self.mode == "train":
            return True
        if self.use_damage_mask_condition and self.eval_real_condition_mask:
            return True
        return False

    def _resolve_object_mask_path(self, filename):
        stem, _ = os.path.splitext(str(filename))
        candidates = [
            os.path.join(self.mask_dir, f"{stem}_mask.png"),
            os.path.join(self.mask_dir, f"{stem}.png"),
            os.path.join(self.mask_dir, str(filename)),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _load_valid_mask(self, filename, height, width, device, dtype, flipped=False):
        if not self.use_object_loss_mask:
            return torch.ones((1, height, width), device=device, dtype=dtype)

        mask_path = self._resolve_object_mask_path(filename)
        if mask_path is None:
            if filename not in self._missing_object_mask:
                print(f"[PatternLoopDataset] object mask not found for {filename}; fallback to all-ones mask", flush=True)
                self._missing_object_mask.add(filename)
            return torch.ones((1, height, width), device=device, dtype=dtype)

        mask_np = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).to(device=device, dtype=dtype)
        if mask_tensor.shape[-2:] != (height, width):
            mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=(height, width), mode="nearest").squeeze(0)
        if flipped:
            mask_tensor = torch.flip(mask_tensor, dims=(-1,))

        valid_mask = 1.0 - (mask_tensor / 255.0)
        return valid_mask.clamp_(0.0, 1.0)

    def _load_object_mask_tensor(self, filename, image_size):
        """Load rebar object mask as (1, H, W) float tensor before geometric augmentations.

        Convention: concrete/valid=1, rebar=0 (inverted from white-rebar PNG masks).
        """
        if not self._needs_object_mask():
            return None

        cache_key = (filename, int(image_size))
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            return cached.clone()

        mask_path = self._resolve_object_mask_path(filename)
        if mask_path is None:
            if filename not in self._missing_object_mask:
                print(f"[PatternLoopDataset] object mask not found for {filename}; fallback to all-ones mask", flush=True)
                self._missing_object_mask.add(filename)
            return torch.ones((1, image_size, image_size), dtype=torch.float32)

        mask_np = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        if mask_tensor.shape[-2:] != (image_size, image_size):
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0),
                size=(image_size, image_size),
                mode="nearest",
            ).squeeze(0)
        valid_mask = (1.0 - (mask_tensor / 255.0)).clamp_(0.0, 1.0)
        if self.cache_data:
            self._mask_cache[cache_key] = valid_mask.clone()
        return valid_mask

    def _direction_to_tensor(self, direction, flipped=False):
        if direction == "Pull":
            direction_tensor = torch.FloatTensor([0.0, 1.0])
        else:
            direction_tensor = torch.FloatTensor([1.0, 0.0])
        if flipped:
            direction_tensor = torch.flip(direction_tensor, dims=(0,))
        return direction_tensor

    def _maybe_flip_info(self, info, flipped):
        if not flipped or not self.signed_feature_indices:
            return info
        info = info.clone()
        info[self.signed_feature_indices] *= -1.0
        return info

    def _maybe_perturb_conditions(self, hidden_state, info, DI):
        if not self.condition_perturb:
            return hidden_state, info, DI
        if self.condition_perturb_on_hidden:
            hidden_state = self._apply_condition_perturbation(hidden_state)
        if self.condition_perturb_on_info:
            info = self._apply_condition_perturbation(info)
        if self.condition_perturb_on_di:
            DI = self._apply_condition_perturbation(DI)
        return hidden_state, info, DI

    def _resolve_condition_mask(self, transformed_object_mask, object_mask, image):
        if not self.use_damage_mask_condition:
            return None
        if self.mode == "train" or self.eval_real_condition_mask:
            if transformed_object_mask is not None:
                return transformed_object_mask.to(dtype=image.dtype)
            if object_mask is not None:
                return object_mask.to(dtype=image.dtype)
            return torch.ones((1, image.shape[-2], image.shape[-1]), dtype=image.dtype)
        return torch.ones((1, image.shape[-2], image.shape[-1]), dtype=image.dtype)

    def _load_frame(self, filename, exp, direction, timestep, DI, force_flip=None, apply_random_erasing=True):
        """Load one damage frame (image + GRU + masks). Shared flip via force_flip."""
        prefixed = self._prefix_cache.get(filename)
        image = prefixed.clone() if prefixed is not None else self._open_image(filename)
        info = self._info_for_exp(exp)
        hidden_state = self.hidden_state[exp][int(timestep), :]
        if not torch.is_tensor(hidden_state):
            hidden_state = torch.as_tensor(hidden_state, dtype=torch.float32)
        hidden_state = self._maybe_mix_hidden_traj(hidden_state, exp, direction, timestep)
        object_mask = self._load_object_mask_tensor(filename, self.image_size)
        try:
            image = self.transform(
                image,
                object_mask=object_mask,
                force_flip=force_flip,
                apply_random_erasing=apply_random_erasing,
                **({"image_is_prefixed": True} if prefixed is not None else {}),
            )
        except TypeError:
            # Transforms without AR kwargs (legacy call sites).
            image = self.transform(image, object_mask=object_mask) if object_mask is not None else self.transform(image)

        flipped = False
        transformed_object_mask = None
        if isinstance(image, dict):
            loss_mask = image.get("loss_mask", None)
            transformed_object_mask = image.get("object_mask", None)
            flipped = bool(image.get("flipped", False))
            image = image["target_image"]
        else:
            loss_mask = None

        if loss_mask is None:
            loss_mask = torch.ones((1, image.shape[-2], image.shape[-1]), dtype=image.dtype)
        elif object_mask is None and self.use_object_loss_mask:
            valid_mask = self._load_valid_mask(
                filename,
                image.shape[-2],
                image.shape[-1],
                device=image.device,
                dtype=image.dtype,
                flipped=flipped,
            )
            loss_mask = loss_mask * valid_mask

        DI = torch.FloatTensor([float(DI)])
        direction_tensor = self._direction_to_tensor(direction, flipped=flipped)
        info = self._maybe_flip_info(info, flipped)
        hidden_state, info, DI = self._maybe_perturb_conditions(hidden_state, info, DI)
        hidden_state = self._apply_g_pca(hidden_state)
        condition_mask = self._resolve_condition_mask(transformed_object_mask, object_mask, image)

        if not self.use_object_loss_mask:
            loss_mask = torch.ones((1, image.shape[-2], image.shape[-1]), dtype=image.dtype)

        return {
            "image": image,
            "hidden_state": hidden_state,
            "direction": direction_tensor,
            "exp": exp,
            "timestep": timestep,
            "DI": DI,
            "info": info,
            "loss_mask": loss_mask,
            "condition_mask": condition_mask,
            "flipped": flipped,
        }

    def _pack_legacy_tuple(self, sample):
        image = sample["image"]
        hidden_state = sample["hidden_state"]
        direction_tensor = sample["direction"]
        exp = sample["exp"]
        timestep = sample["timestep"]
        DI = sample["DI"]
        info = sample["info"]
        loss_mask = sample["loss_mask"]
        condition_mask = sample["condition_mask"]

        if self.use_loss_mask:
            if condition_mask is not None:
                return image, hidden_state, direction_tensor, exp, timestep, DI, info, loss_mask, condition_mask
            return image, hidden_state, direction_tensor, exp, timestep, DI, info, loss_mask
        if condition_mask is not None:
            return image, hidden_state, direction_tensor, exp, timestep, DI, info, condition_mask
        return image, hidden_state, direction_tensor, exp, timestep, DI, info

    def __getitem__(self, index):
        if self.base_size == 0:
            raise IndexError("PatternLoopDataset is empty.")
        real_idx = int(index) % self.base_size
        filename, exp, direction, timestep, DI = self.label.iloc[real_idx, :].values

        if not self.use_ar_temporal_pairing:
            sample = self._load_frame(filename, exp, direction, timestep, DI)
            return self._pack_legacy_tuple(sample)

        # --- Temporal AR pack: target + previous frame (shared H-flip) ---
        target = self._load_frame(filename, exp, direction, timestep, DI)
        force_flip = bool(target["flipped"])
        prev_idx = int(self._prev_iloc[real_idx])
        has_prev = prev_idx >= 0
        if has_prev:
            p_filename, p_exp, p_direction, p_timestep, p_DI = self.label.iloc[prev_idx, :].values
            prev = self._load_frame(
                p_filename,
                p_exp,
                p_direction,
                p_timestep,
                p_DI,
                force_flip=force_flip,
                apply_random_erasing=False,
            )
        else:
            # First frame placeholder: reuse target tensors; training will force-drop self_cond.
            prev = {
                "image": target["image"],
                "hidden_state": target["hidden_state"],
                "direction": target["direction"],
                "exp": target["exp"],
                "timestep": target["timestep"],
                "DI": target["DI"],
                "info": target["info"],
                "loss_mask": target["loss_mask"],
                "condition_mask": target["condition_mask"],
                "flipped": target["flipped"],
            }

        prev2 = None
        has_prev2 = False
        if self.use_prev2:
            prev2_idx = int(self._prev_iloc[prev_idx]) if has_prev else -1
            has_prev2 = prev2_idx >= 0
            if has_prev2:
                q_filename, q_exp, q_direction, q_timestep, q_DI = self.label.iloc[prev2_idx, :].values
                prev2 = self._load_frame(
                    q_filename,
                    q_exp,
                    q_direction,
                    q_timestep,
                    q_DI,
                    force_flip=force_flip,
                    apply_random_erasing=False,
                )
            else:
                prev2 = prev

        ones = torch.ones((1, target["image"].shape[-2], target["image"].shape[-1]), dtype=target["image"].dtype)
        mask_target = target["condition_mask"] if target["condition_mask"] is not None else ones
        mask_prev = prev["condition_mask"] if prev["condition_mask"] is not None else ones
        # Unified damage / object mask for both stage forwards.
        mask_combined = torch.max(mask_prev, mask_target)

        out = {
            "image_target": target["image"],
            "image_prev": prev["image"],
            "hidden_state_target": target["hidden_state"],
            "hidden_state_prev": prev["hidden_state"],
            "direction_target": target["direction"],
            "direction_prev": prev["direction"],
            "exp": target["exp"],
            "timestep_target": target["timestep"],
            "timestep_prev": prev["timestep"],
            "DI_target": target["DI"],
            "DI_prev": prev["DI"],
            "info": target["info"],
            "loss_mask_target": target["loss_mask"],
            "loss_mask_prev": prev["loss_mask"],
            "mask_target": mask_target,
            "mask_prev": mask_prev,
            "mask_combined": mask_combined,
            "has_prev": torch.tensor(1.0 if has_prev else 0.0, dtype=torch.float32),
            # Back-compat aliases used by pack helpers / inference anchors.
            "image": target["image"],
            "hidden_state": target["hidden_state"],
            "direction": target["direction"],
            "timestep": target["timestep"],
            "DI": target["DI"],
            "loss_mask": target["loss_mask"],
            "condition_mask": mask_combined,
        }
        if self.use_prev2:
            out["image_prev2"] = prev2["image"]
            out["hidden_state_prev2"] = prev2["hidden_state"]
            out["direction_prev2"] = prev2["direction"]
            out["DI_prev2"] = prev2["DI"]
            out["has_prev2"] = torch.tensor(1.0 if has_prev2 else 0.0, dtype=torch.float32)
        return out


    def __len__(self):
        return self.base_size * self.virtual_repeat_factor

    def get_hidden_state(self, extractor, exp_list, result_dir):
        hidden_state = {}

        test_ds = HystereticDataset(root=os.path.join(self.root, "Hysteresis"), specified=exp_list, **vars(self.extractor_args))
        test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=os.cpu_count(), collate_fn=test_ds.collate_fn)
        device = extractor.device

        with torch.no_grad():
            for j, batch in enumerate(tqdm(test_dataloader)):
                simulated_loop, test_force, _, maximun_simulated_force, exp = batch
                simulated_loop = simulated_loop.to(device)
                output, _ = extractor.L(simulated_loop)
                hidden_state[exp[0]] = _to_hidden_state_matrix(output)

                predicted_force = extractor.P(output)
                self.draw(predicted_force, simulated_loop, test_force, maximun_simulated_force, exp[0], result_dir)

        return hidden_state


    def draw(self, predicted_force, simulated_loop, test_force, maximun_simulated_force, exp, result_dir):
        predicted_force, drift, simulated_force, test_force, max_simulated_force = _prepare_hysteresis_plot_arrays(
            predicted_force, simulated_loop, test_force, maximun_simulated_force
        )
        self.hysteresis[exp] = {"predicted": predicted_force, "simulated": simulated_force, "test": test_force, "drift": drift, "max_simulated_force": max_simulated_force}

        fig = plt.figure()
        plt.plot(drift, test_force * max_simulated_force, label="Experiment", color="tab:blue", alpha=0.7)
        plt.plot(drift, predicted_force * max_simulated_force, label="Predicted", linestyle="--", color="tab:red", alpha=0.7)
        plt.plot(drift, simulated_force * max_simulated_force, label="Simulated", linestyle="-.", color="tab:green", alpha=0.3)
        plt.xlabel("Drift (%)")
        plt.ylabel("Force (kN)")
        plt.legend()
        plt.title(f"{exp}")
        plt.savefig(os.path.join(result_dir, "data_check", f"{self.mode}_{exp}.svg"))
        plt.close()
