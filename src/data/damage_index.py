"""Legacy damage-index image dataset."""

import json
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .preprocessing import standardize

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
