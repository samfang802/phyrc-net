"""Hysteresis helpers and dataset."""

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import standardize

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
