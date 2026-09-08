import os, json, torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchvision import transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt


def _to_hidden_state_matrix(output):
    hidden = output.detach().cpu()
    if hidden.dim() == 3:
        hidden = hidden[0]
    elif hidden.dim() == 1:
        hidden = hidden.unsqueeze(0)
    elif hidden.dim() != 2:
        raise ValueError(f"Unsupported hidden state shape: {tuple(hidden.shape)}")
    return hidden


def _prepare_hysteresis_plot_arrays(predicted_force, simulated_loop, test_force, maximun_simulated_force):
    predicted_force = predicted_force.detach().cpu().reshape(-1).numpy()
    loop = simulated_loop.detach().cpu().numpy()
    if loop.ndim == 3:
        loop = loop[0]
    elif loop.ndim == 1:
        loop = loop.reshape(1, -1)
    elif loop.ndim > 2:
        loop = loop.reshape(-1, loop.shape[-1])

    drift = loop[:, -2].reshape(-1)
    simulated_force = loop[:, -1].reshape(-1)
    test_force = test_force.detach().cpu().reshape(-1).numpy()
    max_simulated_force = float(maximun_simulated_force.detach().cpu().reshape(-1)[0])
    return predicted_force, drift, simulated_force, test_force, max_simulated_force


class HystereticDataset(Dataset):
    """Dataset class for the hystereticDataset dataset."""
    def __init__(self, root, mode=None, data_size="big", specified=[], simulate=True, scale=True, maximum=None, **kwargs):
        self.path = root
        
        self.simulate = simulate
        self.scale = scale
        self.maximum = maximum
        
        # Get the hysteretic loop
        self.hysteresis_loop = {}
        if len(specified) == 0:
            self.exp_list = json.load(open(os.path.join(self.path, f"id_list_{data_size}.json")))[mode]
        else:
            self.exp_list = specified
        self.columns = ["id", "concreteStrength", "transverse_yield", "transverse_strength", "longitudinal_corner_yield", "longitudinal_corner_strength", "longitudinal_intermediate_yield", 
                    "longitudinal_intermediate_strength", "lMeasured", "axialLoad", "fcc", "epscc", "fcu", "epscu", "Vn", "Vst", "reinforcementRatio", "volTransReinfRatio"]
        self.info = pd.read_csv(os.path.join(self.path, f"info_w_mander_{data_size}.csv")).loc[:, self.columns]
        statistics = pd.read_csv(os.path.join(root, f"statistics_{data_size}.csv"), index_col=0)
        # self.info = feature_encoding(root, self.info, encoding_type=encoding_type)
        self.info = standardize(self.info, statistics)
        
        for exp in self.exp_list:
            self.hysteresis_loop[exp] = {"simulate": 0, "exp": 0, "info": 0}
            
        for exp in self.exp_list:
            df_test = pd.read_csv(os.path.join(self.path, "Experiment", f"{exp}.csv"))
            df_simulate = pd.read_csv(os.path.join(self.path, "Simulate", f"{exp}.csv"))
            info = self.info[self.info["id"]==exp].iloc[:, 1:].values
            # Repeat the information for the length of the signal
            simulation_values = df_simulate.loc[:, ["Drift(%)", "Force(kN)"]].values
            test_values = df_test.loc[:, ["Force(kN)"]].values
            
            self.hysteresis_loop[exp]["simulate"] = simulation_values
            self.hysteresis_loop[exp]["exp"] = test_values
            self.hysteresis_loop[exp]["info"] = info

    def __getitem__(self, index):
        
        exp = self.exp_list[index]
        simulated_loop = torch.FloatTensor(self.hysteresis_loop[exp]["simulate"])
        test_force = torch.FloatTensor(self.hysteresis_loop[exp]["exp"])
        info = torch.FloatTensor(self.hysteresis_loop[exp]["info"])
        if self.simulate and self.scale:
            maximun_simulated_force = simulated_loop[:, 1].max()
        else:
            maximun_simulated_force = torch.FloatTensor([self.maximum]) # Normalize to MN 
        
        # Normalize the force with the maximum value from simulated loop
        simulated_loop[:, 1] = simulated_loop[:, 1] / maximun_simulated_force
        # Concate info to simulated loop
        if self.simulate:
            input = torch.cat([info.repeat(simulated_loop.shape[0], 1), simulated_loop], dim=1)
        else:
            input = torch.cat([info.repeat(simulated_loop.shape[0], 1), simulated_loop[:, 0:1]], dim=1)
            
        test_force[:, 0] = test_force[:, 0] / maximun_simulated_force
        
        original_length = simulated_loop.shape[0]
        
        return input, test_force, original_length, maximun_simulated_force, exp
    
    def __len__(self):
        """Return the number of images."""
        return len(self.hysteresis_loop)

    def collate_fn(self, batch):
        simulated_loop, test_force, original_length, maximun_simulated_force, exp = zip(*batch)
        simulated_loop = torch.nn.utils.rnn.pad_sequence(simulated_loop, batch_first=True, padding_value=0)
        test_force = torch.nn.utils.rnn.pad_sequence(test_force, batch_first=True, padding_value=0)
        original_length = torch.tensor(original_length)
        maximun_simulated_force = torch.stack(maximun_simulated_force)
        exp = list(exp)
        return simulated_loop, test_force, original_length, maximun_simulated_force, exp

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
        ) or bool(kwargs.get("use_ar_chain_ot", False))
        self._prev_iloc = self._build_prev_iloc_map() if self.use_ar_temporal_pairing else None
        self._missing_object_mask = set()
        self.image_size = int(kwargs.get("image_size", 128))
        self.condition_perturb = bool(kwargs.get("condition_perturb", False)) and self.mode == "train"
        self.condition_perturb_dist = str(kwargs.get("condition_perturb_dist", "uniform")).lower()
        perturb_min = float(kwargs.get("condition_perturb_min", 0.01))
        perturb_max = float(kwargs.get("condition_perturb_max", 0.03))
        self.condition_perturb_min = min(perturb_min, perturb_max)
        self.condition_perturb_max = max(perturb_min, perturb_max)
        self.condition_perturb_on_hidden = bool(kwargs.get("condition_perturb_on_hidden", True))
        self.condition_perturb_on_info = bool(kwargs.get("condition_perturb_on_info", True))
        self.condition_perturb_on_di = bool(kwargs.get("condition_perturb_on_di", True))
        if self.use_ar_temporal_pairing:
            n_paired = int(np.sum(self._prev_iloc >= 0))
            print(
                f"[PatternLoopDataset] AR temporal pairing enabled: "
                f"{n_paired}/{self.base_size} samples have a previous frame",
                flush=True,
            )
        print("Done!")

    def _build_prev_iloc_map(self):
        """For each row, iloc of previous frame in same (exp, direction), else -1."""
        prev_iloc = np.full(len(self.label), -1, dtype=np.int64)
        for _, group in self.label.groupby(["exp", "direction"], sort=False):
            ordered = group.sort_values("timestep").index.to_numpy(dtype=np.int64)
            for k in range(1, len(ordered)):
                prev_iloc[ordered[k]] = ordered[k - 1]
        return prev_iloc
        
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
        valid_mask = 1.0 - (mask_tensor / 255.0)
        return valid_mask.clamp_(0.0, 1.0)

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
        image = Image.open(os.path.join(self.root, "DamagePattern", "images", filename))
        info = self.info[self.info["id"] == exp].iloc[:, 1:].values.squeeze()
        info = torch.FloatTensor(info)
        hidden_state = self.hidden_state[exp][int(timestep), :]
        object_mask = self._load_object_mask_tensor(filename, self.image_size)
        try:
            image = self.transform(
                image,
                object_mask=object_mask,
                force_flip=force_flip,
                apply_random_erasing=apply_random_erasing,
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

        ones = torch.ones((1, target["image"].shape[-2], target["image"].shape[-1]), dtype=target["image"].dtype)
        mask_target = target["condition_mask"] if target["condition_mask"] is not None else ones
        mask_prev = prev["condition_mask"] if prev["condition_mask"] is not None else ones
        # Unified damage / object mask for both stage forwards.
        mask_combined = torch.max(mask_prev, mask_target)

        return {
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
        

class DamageIndexDataset(Dataset):
    def __init__(self, root, mode, transform, column_info, **kwargs) -> None:
        super().__init__()
        self.root = root
        self.mode = mode
        self.transform = transform
        self.image_list = json.load(open(os.path.join(self.root, "DamagePattern", f"id_list_old.json")))[mode]
        self.label = pd.read_csv(os.path.join(self.root, "DamagePattern", "label.csv"))
        self.label = self.label[self.label["filename"].isin(self.image_list)]
        self.info = pd.read_csv(os.path.join(self.root, "Hysteresis", f"info_w_mander_big.csv")).loc[:, column_info]
        statistics = pd.read_csv(os.path.join(self.root, "Hysteresis", f"statistics_big.csv"), index_col=0)
        self.info = standardize(self.info, statistics)
        
    def __getitem__(self, index):
        filename, exp, direction, timestep, DI = self.label.iloc[index, :].values
        features = self.info[self.info["id"]==exp].iloc[:, 1:].values.squeeze()
        features = torch.FloatTensor(features)
        DI = torch.FloatTensor([DI])
        image = Image.open(os.path.join(self.root, "DamagePattern", "images", filename))
        image = self.transform(image)
        return image, features, exp, DI
    
    def __len__(self):
        return self.label.shape[0]


def _resolve_r_spall_csv(pattern_root, r_spall_csv=None):
    """Prefer explicit path, then ``<pattern_root>/r_spall.csv``."""
    if r_spall_csv:
        return str(r_spall_csv)
    candidate = os.path.join(pattern_root, "r_spall.csv")
    if os.path.isfile(candidate):
        return candidate
    return None


def _load_r_spall_map(r_spall_csv):
    """Load ``filename -> R_spall`` (also index by stem for .jpg/.png mismatch)."""
    if not r_spall_csv or not os.path.isfile(r_spall_csv):
        return {}
    df = pd.read_csv(r_spall_csv)
    col = "R_spall" if "R_spall" in df.columns else None
    if col is None or "filename" not in df.columns:
        raise ValueError(
            f"r_spall_csv must contain columns ['filename', 'R_spall'], got {list(df.columns)}"
        )
    mapping = {}
    for _, row in df.iterrows():
        name = str(row["filename"])
        val = float(np.clip(float(row[col]), 0.0, 1.0))
        mapping[name] = val
        stem, _ = os.path.splitext(name)
        mapping[stem] = val
    return mapping


def _lookup_r_spall(mapping: dict, filename: str, default: float = 0.0) -> float:
    name = str(filename)
    if name in mapping:
        return float(mapping[name])
    stem, _ = os.path.splitext(name)
    if stem in mapping:
        return float(mapping[stem])
    return float(default)


def fit_r_bar(
    vdi,
    r_spall,
    method: str = "poly",
    poly_degree: int = 2,
    ma_window: int = 51,
):
    """Fit expected spall trend ``R̄(vDI)`` from (vDI, R_spall) samples.

    Returns
    -------
    r_bar : callable
        ``r_bar(x) -> float`` or array, clipped to [0, 1].
    info : dict
        Fit metadata for logging.
    """
    x = np.asarray(vdi, dtype=np.float64).reshape(-1)
    y = np.asarray(r_spall, dtype=np.float64).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    method = str(method).lower().strip()
    if len(x) < 2:
        # Degenerate: constant 0 trend.
        def r_bar_const(q):
            q = np.asarray(q, dtype=np.float64)
            out = np.zeros_like(q, dtype=np.float64)
            return float(out) if np.ndim(out) == 0 else out

        return r_bar_const, {"method": "const0", "n": int(len(x))}

    if method in ("ma", "moving_average", "moving-avg"):
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        w = max(1, int(ma_window))
        if w % 2 == 0:
            w += 1
        # Odd window centered rolling mean; edges use expanding partial window.
        kernel = np.ones(w, dtype=np.float64) / float(w)
        pad = w // 2
        ys_pad = np.pad(ys, (pad, pad), mode="edge")
        ys_ma = np.convolve(ys_pad, kernel, mode="valid")
        # Unique x for interpolation (average y at duplicate vDI).
        uniq_x, inv = np.unique(xs, return_inverse=True)
        uniq_y = np.zeros_like(uniq_x)
        counts = np.zeros_like(uniq_x)
        np.add.at(uniq_y, inv, ys_ma)
        np.add.at(counts, inv, 1.0)
        uniq_y = uniq_y / np.maximum(counts, 1.0)

        def r_bar_ma(q):
            q_arr = np.asarray(q, dtype=np.float64)
            yq = np.interp(q_arr, uniq_x, uniq_y, left=uniq_y[0], right=uniq_y[-1])
            yq = np.clip(yq, 0.0, 1.0)
            return float(yq) if np.ndim(yq) == 0 else yq

        return r_bar_ma, {
            "method": "ma",
            "ma_window": int(w),
            "n": int(len(x)),
            "x_min": float(uniq_x.min()),
            "x_max": float(uniq_x.max()),
        }

    # Default: polynomial regression.
    deg = int(max(1, min(int(poly_degree), max(1, len(x) - 1))))
    coeffs = np.polyfit(x, y, deg=deg)

    def r_bar_poly(q):
        q_arr = np.asarray(q, dtype=np.float64)
        yq = np.polyval(coeffs, q_arr)
        yq = np.clip(yq, 0.0, 1.0)
        return float(yq) if np.ndim(yq) == 0 else yq

    return r_bar_poly, {
        "method": "poly",
        "poly_degree": int(deg),
        "coeffs": [float(c) for c in coeffs],
        "n": int(len(x)),
    }


def compute_vdi(vdi_original: float, r_spall: float, r_bar, beta: float = 0.3) -> float:
    """vDI = max(0, vDI_original + β · (R̄(vDI_original) - R_spall)).

    No upper clip — vDI may exceed 1 when the original DI is already large
    or the residual correction pushes it above 1.
    """
    vo = float(vdi_original)
    rs = float(np.clip(float(r_spall), 0.0, 1.0))
    r_hat = float(r_bar(vo))
    delta = r_hat - rs
    return float(max(0.0, vo + float(beta) * delta))


def compute_di_new(di_original: float, r_spall: float, beta: float = 0.3, r_bar=None, **_kwargs) -> float:
    """Deprecated alias of :func:`compute_vdi` (needs ``r_bar``)."""
    if r_bar is None:
        # Fallback: treat expected R as 0 → Δ = -R_spall
        r_bar = lambda _x: 0.0  # noqa: E731
    return compute_vdi(di_original, r_spall, r_bar=r_bar, beta=beta)


def robust_linear_di(di_original: float, r_spall: float, beta: float = 0.3, r_bar=None, **_kwargs) -> float:
    """Deprecated alias of :func:`compute_vdi`."""
    return compute_di_new(di_original, r_spall, beta=beta, r_bar=r_bar)


def _attach_vdi_labels(
    label_df: pd.DataFrame,
    r_map: dict,
    beta: float,
    r_bar_method: str = "poly",
    poly_degree: int = 2,
    ma_window: int = 51,
    fit_label_df: pd.DataFrame = None,
):
    """Add R_spall / R_bar / delta / vDI columns. Fit R̄ on ``fit_label_df`` (default: same df)."""
    out = label_df.copy()
    r_vals = [_lookup_r_spall(r_map, fn) for fn in out["filename"].astype(str)]
    out["R_spall"] = r_vals

    fit_df = label_df if fit_label_df is None else fit_label_df
    fit_r = np.asarray(
        [_lookup_r_spall(r_map, fn) for fn in fit_df["filename"].astype(str)],
        dtype=np.float64,
    )
    fit_vdi = fit_df["DI"].to_numpy(dtype=np.float64)
    r_bar, fit_info = fit_r_bar(
        fit_vdi,
        fit_r,
        method=r_bar_method,
        poly_degree=poly_degree,
        ma_window=ma_window,
    )

    vdi_orig = out["DI"].to_numpy(dtype=np.float64)
    r_hat = np.asarray([float(r_bar(v)) for v in vdi_orig], dtype=np.float64)
    delta = r_hat - np.asarray(r_vals, dtype=np.float64)
    out["vDI_original"] = vdi_orig
    out["R_bar"] = r_hat
    out["delta"] = delta
    out["vDI"] = [
        compute_vdi(vo, rs, r_bar=r_bar, beta=beta)
        for vo, rs in zip(vdi_orig.tolist(), r_vals)
    ]
    # Keep legacy column for any external tooling.
    out["DI_new"] = out["vDI"]
    return out, r_bar, fit_info


IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)


def build_quant_image_tensor(
    path,
    image_size,
    use_rgb=False,
    use_clahe=False,
    gray_mean=0.485,
    gray_std=0.229,
    rgb_mean=IMAGENET_RGB_MEAN,
    rgb_std=IMAGENET_RGB_STD,
):
    """Load a damage photo as CHW float tensor (ImageNet-normalized)."""
    from src.utils import ApplyCLAHE

    image_size = int(image_size)
    if use_rgb:
        image = Image.open(path).convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        rgb = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        if use_clahe:
            rgb = ApplyCLAHE(preserve_color=True)(rgb)
        mean = torch.tensor(rgb_mean, dtype=rgb.dtype).view(3, 1, 1)
        std = torch.tensor(rgb_std, dtype=rgb.dtype).view(3, 1, 1)
        return (rgb - mean) / std.clamp_min(1e-6)

    image = Image.open(path).convert("L")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    gray = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)
    if use_clahe:
        gray = ApplyCLAHE(preserve_color=False)(gray)
    return (gray - float(gray_mean)) / max(float(gray_std), 1e-6)


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

def min_max_scaling(df, statistics):
    for column in df.columns:
        # Avoid the columns which are object type
        if column in statistics.columns:
            df[column] = (df[column] - statistics[column]["min"]) / (statistics[column]["max"] - statistics[column]["min"])   
    return df

def standardize(df, statistics):
    for column in df.columns:
        # Avoid the columns which are object type
        if column in statistics.columns:
            df[column] = (df[column] - statistics[column]["mean"]) / (statistics[column]["std"])   
    return df

def feature_encoding(root, df, encoding_type):
    match encoding_type:
        case "one-hot":
            encode = json.load(open(os.path.join(root, "encoding/one_hot.json")))
            for key in encode.keys():
                for category in encode[key]:
                    df[f"{key}_{category}"] = [1 if i == category else 0 for i in df[key]]
                df.drop(columns=key, inplace=True)
                    
        case "label":
            encode = json.load(open(os.path.join(root, "encoding/label.json")))
            for key in encode.keys():
                for category in encode[key].keys():
                    mask = df[key] == category
                    df.loc[mask, key] = encode[key][category]
                    
        case "frequency":
            encode = json.load(open(os.path.join(root, "encoding/freq.json")))
            for key in encode.keys():
                for category in encode[key].keys():
                    mask = df[key] == category
                    df.loc[mask, key] = encode[key][category]
        case _:
            raise ValueError("Encoding type should be one of ['one-hot', 'label', 'frequency']")
        
    return df
    