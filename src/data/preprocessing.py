"""Tabular preprocessing helpers for dataset construction."""

import json
import os

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
