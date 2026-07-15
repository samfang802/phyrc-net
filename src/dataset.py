import os, json, torch, glob, argparse, importlib
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
        self.exp_list = self.label['exp'].unique()
        print("Extracting hidden state...")
        self.info = pd.read_csv(os.path.join(self.root, "Hysteresis", f"info_w_mander_big.csv")).loc[:, column_info]
        statistics = pd.read_csv(os.path.join(self.root, "Hysteresis", f"statistics_big.csv"), index_col=0)
        self.info = standardize(self.info, statistics)
        self.hysteresis = {}
        self.hidden_state = self.get_hidden_state(extractor, self.exp_list, result_dir=result_dir)
        print("Done!")
        

    def __getitem__(self, index):
        filename, exp, direction, timestep, DI = self.label.iloc[index, :].values
        image = Image.open(os.path.join(self.root, "DamagePattern", "images", filename))
        info = self.info[self.info["id"]==exp].iloc[:, 1:].values.squeeze()
        info = torch.FloatTensor(info)
        hidden_state = self.hidden_state[exp][int(timestep), :]
        image = self.transform(image)
        DI = torch.FloatTensor([DI])
        if direction == "Pull":
            direction = torch.FloatTensor([0.0, 1.0])
        else:
            direction = torch.FloatTensor([1.0, 0.0])
        return image, hidden_state, direction, exp, timestep, DI, info
        

    def __len__(self):
        return self.label.shape[0]
    
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
    
    