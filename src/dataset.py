import os, json, torch, glob, argparse, importlib
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchvision import transforms
import hysteresis as hys

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

class InteriorLoopDataset(Dataset):
    def __init__(self, root, exp, extractor_args, extractor, column_info, data_path, result_dir, **kwargs):
        super().__init__()
        self.root = root
        self.exp = exp
        self.extractor_args = extractor_args
        self.info = pd.read_csv(os.path.join(self.root, "Hysteresis", f"info_w_mander_big.csv")).loc[:, column_info]
        statistics = pd.read_csv(os.path.join(self.root, "Hysteresis", f"statistics_big.csv"), index_col=0)
        self.info = standardize(self.info, statistics)
        
        if self.exp in ["R307", "R315", "R330", "R615", "R1015"]:
            self.training_data = True
            # Classfy the time step
            self.label = pd.read_csv(os.path.join(self.root, "DamagePattern", "label.csv"))
            self.label = self.label[self.label["exp"]==exp]
            self.image_list = json.load(open(os.path.join(self.root, "DamagePattern", f"{data_path}")))
            self.image_list["train"] = [i for i in self.image_list["train"] if i in self.label["filename"].values]
            self.image_list["val"] = [i for i in self.image_list["val"] if i in self.label["filename"].values]
        else:
            self.training_data = False

        df = pd.read_csv(os.path.join(self.root, "Hysteresis", "Experiment", f"{exp}.csv"))
        self.reversalIndex = hys.Hysteresis(np.column_stack([df["Drift(%)"], df["Force(kN)"]])).reversalIndexes
        print("Extracting hidden state...")
        self.hysteresis = {}
        self.hidden_state = self.get_hidden_state(extractor, [exp], result_dir=result_dir)
        # Reversal index
        
        self.time_step = np.append(self.reversalIndex, self.reversalIndex)
        print("Done!")
        
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

    def identify_mode(self, dir_, timestep):
        temp_df = self.label[(self.label["direction"]==dir_) & (self.label["timestep"]==timestep)]
        # Check if dir_ and timestep in the label
        if temp_df.shape[0] != 0:
            # Check the label is in train or validation
            filename = temp_df["filename"].values
            if filename in self.image_list["train"]:
                mode = "train"
            else:
                mode = "val"
        else:
            mode = "inter"
        return mode
        
    def __getitem__(self, index):
        ts = self.time_step[index]
        info = self.info[self.info["id"]==self.exp].iloc[:, 1:].values.squeeze()
        info = torch.FloatTensor(info)
        
        if self.training_data:
            if index < (self.time_step.shape[0] // 2):
                dir_ = "Pull"
                mode = self.identify_mode(dir_, ts)
                hidden_state = self.hidden_state[self.exp][ts, :]
                direction = torch.tensor([0.0, 1.0])       
            else:
                dir_ = "Push"
                mode = self.identify_mode(dir_, ts)
                hidden_state = self.hidden_state[self.exp][ts, :]
                direction = torch.tensor([1.0, 0.0])
            
        else:
            if index < (self.time_step.shape[0] // 2):
                dir_ = "Pull"
                hidden_state = self.hidden_state[self.exp][ts, :]
                mode = "inter"
                direction = torch.tensor([0.0, 1.0])       
            else:
                dir_ = "Push"
                hidden_state = self.hidden_state[self.exp][ts, :]
                mode = "inter"
                direction = torch.tensor([1.0, 0.0])
                
        return hidden_state, direction, ts, mode, info
        
        
    def __len__(self):
        return self.time_step.shape[0]

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
        plt.savefig(os.path.join(result_dir, "data_check", f"{exp}.svg"))
        plt.close()

class PatternLoopDataset(Dataset):
    def __init__(self, root, mode, transform, extractor, extractor_args, result_dir, data_path, column_info, **kwargs):
        super().__init__()
        self.root = root
        self.mode = mode
        self.transform = transform
        self.extractor_args = extractor_args
        self.image_list = json.load(open(os.path.join(self.root, "DamagePattern", data_path)))[mode]
        self.label = pd.read_csv(os.path.join(self.root, "DamagePattern", "label.csv"))
        self.label = self.label[self.label["filename"].isin(self.image_list)]
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
        self.image_size = int(kwargs.get("image_size", 128))
        self._missing_object_mask = set()
        self.use_damage_mask_condition = bool(kwargs.get("use_damage_mask_condition", False))
        self.damage_mask_sigma_min = float(kwargs.get("damage_mask_sigma_min", 4.0))
        self.damage_mask_sigma_max = float(kwargs.get("damage_mask_sigma_max", 20.0))
        self.damage_mask_max_di = max(float(kwargs.get("damage_mask_max_di", 1.0)), 1e-6)
        self.damage_mask_split_di = float(kwargs.get("damage_mask_split_di", 0.5))
        self.damage_mask_elastic_intensity_cap = float(kwargs.get("damage_mask_elastic_intensity_cap", 0.08))
        self.damage_mask_elastic_spread_min = float(kwargs.get("damage_mask_elastic_spread_min", 0.15))
        self.damage_mask_elastic_spread_max = float(kwargs.get("damage_mask_elastic_spread_max", 0.20))
        self.damage_mask_width_scale = max(float(kwargs.get("damage_mask_width_scale", 2.0)), 1e-6)
        self.damage_mask_quadratic_scale = float(kwargs.get("damage_mask_quadratic_scale", 0.8))
        self.condition_perturb = bool(kwargs.get("condition_perturb", False)) and self.mode == "train"
        self.condition_perturb_dist = str(kwargs.get("condition_perturb_dist", "uniform")).lower()
        perturb_min = float(kwargs.get("condition_perturb_min", 0.01))
        perturb_max = float(kwargs.get("condition_perturb_max", 0.03))
        self.condition_perturb_min = min(perturb_min, perturb_max)
        self.condition_perturb_max = max(perturb_min, perturb_max)
        self.condition_perturb_on_hidden = bool(kwargs.get("condition_perturb_on_hidden", True))
        self.condition_perturb_on_info = bool(kwargs.get("condition_perturb_on_info", True))
        self.condition_perturb_on_di = bool(kwargs.get("condition_perturb_on_di", True))
        print("Done!")
        
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

    @staticmethod
    def _bottom_band_damage_profile(di_scaled, height):
        """Bottom-initiated half-Gaussian profile active only when DI > 0.5."""
        if di_scaled <= 0.5:
            return False, 0.0, 0.0
        normalized_di = (di_scaled - 0.5) / 0.5
        normalized_di = max(0.0, min(float(normalized_di), 1.0))
        intensity = 0.8 * normalized_di
        sigma_y = float(height) * 0.1 + float(height) * 0.2 * normalized_di
        return True, intensity, max(sigma_y, 1e-6)

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
        """Load rebar object mask as (1, H, W) float tensor before geometric augmentations."""
        if not self.use_object_loss_mask:
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

    def _build_damage_mask(self, di_value, height, width, device, dtype):
        di_scaled = max(0.0, min(float(di_value) / self.damage_mask_max_di, 1.0))
        is_active, intensity_scale, sigma_y = self._bottom_band_damage_profile(di_scaled, height)
        if not is_active:
            return torch.zeros((1, height, width), device=device, dtype=dtype)

        y = torch.arange(height, device=device, dtype=dtype).view(height, 1)
        gaussian_1d = torch.exp(-((y - float(height)) ** 2) / (2.0 * sigma_y * sigma_y))
        mask = intensity_scale * gaussian_1d.expand(height, width)
        return mask.unsqueeze(0)

    def __getitem__(self, index):
        if self.base_size == 0:
            raise IndexError("PatternLoopDataset is empty.")
        real_idx = int(index) % self.base_size
        filename, exp, direction, timestep, DI = self.label.iloc[real_idx, :].values
        image = Image.open(os.path.join(self.root, "DamagePattern", "images", filename))
        info = self.info[self.info["id"]==exp].iloc[:, 1:].values.squeeze()
        info = torch.FloatTensor(info)
        hidden_state = self.hidden_state[exp][int(timestep), :]
        object_mask = self._load_object_mask_tensor(filename, self.image_size)
        try:
            image = self.transform(image, object_mask=object_mask)
        except TypeError:
            image = self.transform(image)
        flipped = False
        if isinstance(image, dict):
            loss_mask = image.get("loss_mask", None)
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
        DI = torch.FloatTensor([DI])
        if direction == "Pull":
            direction_tensor = torch.FloatTensor([0.0, 1.0])
        else:
            direction_tensor = torch.FloatTensor([1.0, 0.0])

        if flipped:
            direction_tensor = torch.flip(direction_tensor, dims=(0,))
            if self.signed_feature_indices:
                info = info.clone()
                info[self.signed_feature_indices] *= -1.0
        if self.condition_perturb:
            # Apply small multiplicative perturbation on train conditions only.
            if self.condition_perturb_on_hidden:
                hidden_state = self._apply_condition_perturbation(hidden_state)
            if self.condition_perturb_on_info:
                info = self._apply_condition_perturbation(info)
            if self.condition_perturb_on_di:
                DI = self._apply_condition_perturbation(DI)
        damage_mask = None
        # Spatial damage mask injection disabled: only hysteretic scalar conditions are used.
        if self.use_damage_mask_condition:
            damage_mask = self._build_damage_mask(
                DI.item(),
                image.shape[-2],
                image.shape[-1],
                device=image.device,
                dtype=image.dtype,
            )
        if self.use_loss_mask:
            if damage_mask is not None:
                return image, hidden_state, direction_tensor, exp, timestep, DI, info, loss_mask, damage_mask
            return image, hidden_state, direction_tensor, exp, timestep, DI, info, loss_mask
        if damage_mask is not None:
            return image, hidden_state, direction_tensor, exp, timestep, DI, info, damage_mask
        return image, hidden_state, direction_tensor, exp, timestep, DI, info
        

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
    
class DamagePatternDataset(Dataset):
    def __init__(self, root, transform) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        json_list = json.load(open(os.path.join(self.root, "DamagePattern", f"id_list_old.json")))
        self.image_list = json_list["train"] + json_list["val"]

        
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, "DamagePattern", "images", self.image_list[index]))
        transformed = self.transform(image)
        if isinstance(transformed, dict):
            target_image = transformed["target_image"] / 127.5 - 1.0
            return {
                "target_image": target_image,
                "loss_mask": transformed["loss_mask"],
            }

        return transformed / 127.5 - 1.0
    
    def __len__(self):
        return len(self.image_list)

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


if __name__ == "__main__":
    # # HystereticDataset
    # ds = HystereticDataset(root="./Data/Hysteresis", mode="train")
    
    # PatternLoopDataset
    extractor_path="./results/HystereticPrediction/HystereticGRU/2024-06-12_18-13-47"
    extractor_args = json.load(open(glob.glob(os.path.join(extractor_path, "**/args.json"), recursive=True)[0], "r"))
    extractor_args = argparse.Namespace(**extractor_args)
    extractor = importlib.import_module(f"src.model.HystereticPrediction").__dict__[extractor_args.architecture](**vars(extractor_args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = extractor.to(device)
    state_dict = torch.load(glob.glob(os.path.join(extractor_path, "**/best.ckpt"), recursive=True)[0])["state_dict"]
    status = extractor.load_state_dict(state_dict)
    print(status)
    
    
    ds = PatternLoopDataset(root="./Data", mode="train", extractor=extractor)
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=ds.collate_fn)
    print(next(iter(dl)))
    
    # # # DamageIndexDataset
    # transforms = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])
    # ds = DamageIndexDataset(root="./Data", mode="train", transform=transforms)
    # dl = DataLoader(ds, batch_size=8, shuffle=True)
    # sample = next(iter(dl))
    # print(sample)
    # for i in sample:
    #     print(i.shape)
    
    