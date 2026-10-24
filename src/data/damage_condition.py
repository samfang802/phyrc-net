"""v11 generator condition from the damage-state encoder (docs/ARCHITECTURE_UPDATE_V11.md).

For every specimen, a matrix indexed by loop timestep t (the same indexing as the GRU hidden state):
  [ z (encoder state at the latest load peak <= t) | 4 damage scalars ]
The damage scalars are the encoder's predicted strength ratio, stiffness ratio (>= 0) and cumulative
energy, in the encoder's standardized target units, plus log1p(largest |drift| so far, %).
Before the first peak (t < 30) z is the zero GRU state and the scalars are those of an undamaged
column (strength 1, stiffness 1, energy 0, drift 0).
"""

import json
import os

import numpy as np
import pandas as pd
import torch

from .damage_state import SEG, TARGETS, TOKEN_FEATURES, DamageStateDataset

DAMAGE_SCALARS = ("strength_ratio", "stiffness_ratio", "cum_energy", "log1p_max_drift")
N_DAMAGE_SCALARS = len(DAMAGE_SCALARS)


def build_frame_conditions(encoder_run, exps, root="Data/Hysteresis"):
    """{exp: torch.FloatTensor [n_points, z_dim + 4]} for the given specimens."""
    from src.model.damage_state_encoder import DamageStateEncoder

    stats = {k: np.asarray(v, dtype=np.float32) for k, v in json.load(open(os.path.join(encoder_run, "stats.json"))).items()}
    model = DamageStateEncoder.load_from_checkpoint(os.path.join(encoder_run, "best.ckpt"), map_location="cpu").eval()
    ds = DamageStateDataset(list(exps), root=root, stats=stats)
    t_mean, t_std = stats["tgt_mean"], stats["tgt_std"]
    j = {k: TARGETS.index(k) for k in ("strength_ratio", "stiffness_ratio", "cum_energy")}
    undamaged = np.array([(1.0 - t_mean[j["strength_ratio"]]) / t_std[j["strength_ratio"]],
                          (1.0 - t_mean[j["stiffness_ratio"]]) / t_std[j["stiffness_ratio"]],
                          (0.0 - t_mean[j["cum_energy"]]) / t_std[j["cum_energy"]],
                          0.0], dtype=np.float32)
    out = {}
    with torch.no_grad():
        for it, item in zip(ds.items, ds):
            b = DamageStateDataset.collate([item])
            z, s, _ = model.peak_states(b)
            h = torch.cat([z, s[:, None].expand(-1, z.shape[1], -1)], dim=2)
            reg = model.reg_head(h)[0].numpy()  # standardized target units
            stiff_raw = np.maximum(reg[:, j["stiffness_ratio"]] * t_std[j["stiffness_ratio"]] + t_mean[j["stiffness_ratio"]], 0.0)
            scal = np.stack([
                reg[:, j["strength_ratio"]],
                (stiff_raw - t_mean[j["stiffness_ratio"]]) / t_std[j["stiffness_ratio"]],
                reg[:, j["cum_energy"]],
                np.log1p(it["tokens"][:, TOKEN_FEATURES.index("max_drift_so_far")]),
            ], axis=1).astype(np.float32)
            per_peak = np.concatenate([z[0].numpy(), scal], axis=1)                       # [m, z+4]
            before = np.concatenate([np.zeros(z.shape[2], np.float32), undamaged])[None]  # t < 30
            table = np.concatenate([before, per_peak], axis=0)                            # index k = t // 30
            n_points = SEG * per_peak.shape[0] + 1
            k = np.arange(n_points) // SEG
            out[it["id"]] = torch.from_numpy(table[k])
    return out


def load_signs(exps, root="Data/Hysteresis"):
    """{exp: np.ndarray [n_points]} sign of the simulated drift (0 at zero drift)."""
    return {e: np.sign(pd.read_csv(os.path.join(root, "Simulate", f"{e}.csv"))["Drift(%)"].to_numpy()) for e in exps}
