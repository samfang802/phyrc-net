"""Half-cycle tokens and damage-state targets for the damage-state encoder (docs/DAMAGE_STATE_ENCODER.md).

Every loop in Data/Hysteresis is resampled to 30 points per half-cycle: drift reverses at indices
30k, and the damage images are taken at those peaks (label timesteps are multiples of 30). Token k
(k = 1..m) is the segment [30(k-1), 30k] and describes the state at the peak 30k.

Inputs come from the simulated loop (known at inference); targets from the experimental loop.
Forces are divided by the specimen's maximum |simulated force|, as in HystereticDataset.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SEG = 30
IMAGE_SPECIMENS = ("R315", "R330", "R615", "R1015", "R307", "CTR1")
# Signed token features: these flip with the loading direction (mirror augmentation).
TOKEN_FEATURES = ("peak_drift", "abs_peak_drift", "peak_force", "abs_peak_force", "seg_energy", "secant_stiffness",
                  "max_drift_so_far", "cum_energy", "sign")
SIGNED_TOKEN_FEATURES = ("peak_drift", "peak_force", "sign")
TARGETS = ("abs_peak_force", "strength_ratio", "stiffness_ratio", "cum_energy", "abs_residual_drift")
FAILURE_TYPES = ("Flexure", "Flexure-Shear", "Shear")
STATIC_FEATURES = ("n_axial", "L_over_h", "rho_l", "rho_t", "fc", "mech_rho_l", "confinement")
N_INFO = 17                      # column-info features of the GRU input (HystereticDataset.columns minus id)
# The per-point input always ends with drift and simulated force: the two signed columns.


def _segment_tokens(drift, force):
    """Per-token features of one loop. ``force`` is already normalized."""
    m = (len(drift) - 1) // SEG
    peaks = np.arange(1, m + 1) * SEG
    pd_ = drift[peaks]
    pf = force[peaks]
    seg_e = np.array([np.trapz(force[SEG * (k - 1): SEG * k + 1], drift[SEG * (k - 1): SEG * k + 1]) for k in range(1, m + 1)])
    seg_e = np.abs(seg_e)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_sec = np.where(valid_tokens(drift), pf / pd_, 0.0)
    return {
        "peak_drift": pd_,
        "abs_peak_drift": np.abs(pd_),
        "peak_force": pf,
        "abs_peak_force": np.abs(pf),
        "seg_energy": seg_e,
        "secant_stiffness": k_sec,
        "max_drift_so_far": np.maximum.accumulate(np.abs(pd_)),
        "cum_energy": np.cumsum(seg_e),
        "sign": np.sign(pd_),
    }


def _residual_drift(drift, force):
    """Drift where the force crosses zero inside each half-cycle (unloading), 0 if it does not."""
    m = (len(drift) - 1) // SEG
    out = np.zeros(m)
    for k in range(1, m + 1):
        d = drift[SEG * (k - 1): SEG * k + 1]
        f = force[SEG * (k - 1): SEG * k + 1]
        s = np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
        if len(s):
            i = s[0]
            out[k - 1] = d[i] - f[i] * (d[i + 1] - d[i]) / (f[i + 1] - f[i])
    return out


def valid_tokens(drift):
    """False for tokens whose peak is not a real excursion (the final return to zero): |peak drift|
    below 10% of the largest drift so far. Their secant stiffness is ill-defined."""
    pd_ = drift[np.arange(1, (len(drift) - 1) // SEG + 1) * SEG]
    hist = np.maximum.accumulate(np.abs(pd_))
    return np.abs(pd_) >= 0.1 * np.maximum(hist, 1e-6)


def damage_targets(drift, force_exp):
    """Damage-state targets per token from the experimental loop (normalized force)."""
    tok = _segment_tokens(drift, force_exp)
    f = tok["abs_peak_force"]
    hist_max = np.maximum.accumulate(f)
    strength_ratio = np.where(hist_max > 1e-6, f / np.maximum(hist_max, 1e-6), 1.0)
    valid = valid_tokens(drift)
    k = np.where(valid, np.abs(tok["secant_stiffness"]), 0.0)
    k0 = k[np.argmax(valid)] if valid.any() else 1.0
    stiffness_ratio = k / max(k0, 1e-6)
    return {
        "abs_peak_force": f,
        "strength_ratio": strength_ratio,
        "stiffness_ratio": stiffness_ratio,
        "cum_energy": tok["cum_energy"],
        "abs_residual_drift": np.abs(_residual_drift(drift, force_exp)),
        "failed": strength_ratio < 0.8,
    }


def static_descriptors(info_row):
    """Dimensionless specimen descriptors. ``depth`` is taken as the loading direction (not verified)."""
    fc = float(info_row["concreteStrength"])
    b, h = float(info_row["width"]), float(info_row["depth"])
    rho_l, rho_t = float(info_row["reinforcementRatio"]), float(info_row["volTransReinfRatio"])
    return {
        "n_axial": float(info_row["axialLoad"]) * 1e3 / (fc * b * h),
        "L_over_h": float(info_row["lMeasured"]) / h,
        "rho_l": rho_l,
        "rho_t": rho_t,
        "fc": fc,
        "mech_rho_l": rho_l * float(info_row["longitudinal_corner_yield"]) / fc,
        "confinement": rho_t * float(info_row["transverse_yield"]) / fc,
    }


def load_specimen(root, spec_id, info):
    sim = pd.read_csv(os.path.join(root, "Simulate", f"{spec_id}.csv"))
    exp = pd.read_csv(os.path.join(root, "Experiment", f"{spec_id}.csv"))
    drift = sim["Drift(%)"].to_numpy(dtype=np.float64)
    scale = float(np.abs(sim["Force(kN)"]).max())
    f_sim = sim["Force(kN)"].to_numpy(dtype=np.float64) / scale
    f_exp = exp["Force(kN)"].to_numpy(dtype=np.float64)[: len(drift)] / scale
    tok = _segment_tokens(drift, f_sim)
    tgt = damage_targets(drift, f_exp)
    row = info.loc[spec_id]
    st = static_descriptors(row)
    ft = str(row.get("failureType", "Flexure"))
    return {
        "id": spec_id,
        "tokens": np.stack([tok[k] for k in TOKEN_FEATURES], axis=1).astype(np.float32),
        "targets": np.stack([tgt[k] for k in TARGETS], axis=1).astype(np.float32),
        "failed": tgt["failed"].astype(np.float32),
        "valid": valid_tokens(drift).astype(np.float32),
        "static": np.array([st[k] for k in STATIC_FEATURES], dtype=np.float32),
        "failure_type": FAILURE_TYPES.index(ft) if ft in FAILURE_TYPES else 0,
    }


def specimen_split(root="Data/Hysteresis"):
    """GRU split of id_list_big.json with the six imaged specimens removed from both sides."""
    split = json.load(open(os.path.join(root, "id_list_big.json")))
    keep = lambda ids: [i for i in ids if i not in IMAGE_SPECIMENS]
    return keep(split["train"]), keep(split["val"])


def gru_inputs(ids, root="Data/Hysteresis", extractor_args="phase1_extractor/args.json"):
    """The GRU extractor's own per-point input for each specimen: 17 standardized column-info
    features repeated over time, drift (%), simulated force / max simulated force [n, 19]; and the
    experimental force on the same scale [n]. Built by HystereticDataset with the extractor's args."""
    from .hysteresis import HystereticDataset

    a = json.load(open(extractor_args))
    ds = HystereticDataset(root=root, specified=list(ids), **a)
    out = {}
    for i, e in enumerate(ds.exp_list):
        inp, tf, _, _, _ = ds[i]
        out[e] = (inp.numpy().astype(np.float32), tf.numpy().reshape(-1).astype(np.float32))
    return out


class DamageStateDataset(Dataset):
    """One item per specimen. Token / static normalisation statistics come from the train split.

    Items carry both the half-cycle tokens and the GRU's per-point input (``seq``, see gru_inputs)
    with the per-point experimental force. ``static`` is the standardized dimensionless descriptors
    followed by the 17 column-info features (already standardized by HystereticDataset)."""

    def __init__(self, ids, root="Data/Hysteresis", stats=None, seq_info="raw"):
        # seq_info="dimless": the per-point input carries the 7 standardized dimensionless descriptors
        # instead of the 17 column-info features (drift and simulated force unchanged).
        self.seq_info = seq_info
        info = pd.read_csv(os.path.join(root, "info_w_mander_big.csv"), dtype={"id": str}).set_index("id")
        self.items = [load_specimen(root, i, info) for i in ids]
        seqs = gru_inputs(ids, root)
        for it in self.items:
            it["seq"], it["force_exp"] = seqs[it["id"]]
            assert it["seq"].shape[0] == SEG * it["tokens"].shape[0] + 1, it["id"]
        if stats is None:
            tok = np.concatenate([it["tokens"] for it in self.items])
            tgt = np.concatenate([it["targets"] for it in self.items])
            stt = np.stack([it["static"] for it in self.items])
            signed = [TOKEN_FEATURES.index(k) for k in SIGNED_TOKEN_FEATURES]
            tok_mean = tok.mean(0)
            tok_mean[signed] = 0.0  # keep signed features symmetric so mirroring stays exact
            stats = {
                "tok_mean": tok_mean, "tok_std": tok.std(0) + 1e-6,
                "tgt_mean": tgt.mean(0), "tgt_std": tgt.std(0) + 1e-6,
                "st_mean": stt.mean(0), "st_std": stt.std(0) + 1e-6,
            }
        self.stats = {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}

    def __len__(self):
        return len(self.items)

    def _seq(self, it):
        if self.seq_info == "raw":
            return it["seq"]
        st = (it["static"] - self.stats["st_mean"]) / self.stats["st_std"]
        return np.concatenate([np.repeat(st[None], it["seq"].shape[0], 0), it["seq"][:, -2:]], axis=1).astype(np.float32)

    def __getitem__(self, i):
        it = self.items[i]
        s = self.stats
        return {
            "id": it["id"],
            "tokens": torch.from_numpy((it["tokens"] - s["tok_mean"]) / s["tok_std"]),
            "targets": torch.from_numpy((it["targets"] - s["tgt_mean"]) / s["tgt_std"]),
            "failed": torch.from_numpy(it["failed"]),
            "valid": torch.from_numpy(it["valid"]),
            "static": torch.from_numpy(np.concatenate([(it["static"] - s["st_mean"]) / s["st_std"], it["seq"][0, :N_INFO]])),
            "failure_type": int(it["failure_type"]),
            "seq": torch.from_numpy(self._seq(it)),
            "force_exp": torch.from_numpy(it["force_exp"]),
        }

    @staticmethod
    def collate(batch):
        lengths = torch.tensor([b["tokens"].shape[0] for b in batch])
        pad = lambda key: torch.nn.utils.rnn.pad_sequence([b[key] for b in batch], batch_first=True)
        return {
            "id": [b["id"] for b in batch],
            "tokens": pad("tokens"),
            "targets": pad("targets"),
            "failed": pad("failed"),
            "valid": pad("valid"),
            "static": torch.stack([b["static"] for b in batch]),
            "failure_type": torch.tensor([b["failure_type"] for b in batch]),
            "lengths": lengths,
            "seq": pad("seq"),
            "force_exp": pad("force_exp"),
            "point_lengths": torch.tensor([b["seq"].shape[0] for b in batch]),
        }
