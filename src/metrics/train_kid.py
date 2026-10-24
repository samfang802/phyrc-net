"""Generation of a split for evaluation, and the in-training periodic KID.

``refiner_sequence_samples`` walks every face of every column of a split in timestep order with
``sample_refiner_sequence`` (the sampler of generate_eval_samples.py for refiners).
``PeriodicKID`` scores those samples during training with the same DINOv3 kid_global as
scripts/eval/eval_dinov3_kid.py (block U-statistic, one group per condition), against the same
reference images, so the numbers line up with docs/eval/results.md (val, K = 8). It also logs the
exposure numbers the v10.5 changes target: mean gray, share of frames with mean > 200,
saturated frames and tone_jump along trajectories.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.metrics import dino_mmd as dm
from src.metrics.temporal import trajectories


def is_sequence_refiner(cfm):
    return bool(getattr(cfm, "use_refiner_residual", False))


@torch.no_grad()
def refiner_sequence_samples(cfm, dl, ds, K, seed, device):
    """Walk every (column, direction) in timestep order; K trajectories per walk.

    Returns fake uint8 (N, K, H, W), real list of uint8 batches, cond list, DI list, row-aligned
    with ``ds.label``.
    """
    if len(ds) != len(ds.label):
        raise ValueError("refiner path needs one dataset item per label row")
    cs, gs, real, cond, DI = [], [], [], [], []
    for batch in dl:
        real_data, hidden_state, direction, _exp, _t, di, info, *_ = cfm._unpack_generation_batch(batch)
        c, g = cfm._unpack_conditions(info.to(device), hidden_state.to(device), direction.to(device), di.to(device))
        cs.append(c)
        gs.append(g)
        real.append(dm.to_uint8_gray(real_data))
        cond.append(hidden_state.reshape(hidden_state.shape[0], -1).float().cpu().numpy())
        DI.append(di.view(-1).float().cpu().numpy())
    c_all = torch.cat(cs)
    g_all = None if gs[0] is None else torch.cat(gs)
    lab = ds.label
    fake = np.empty((len(lab), K, *real[0].shape[1:]), dtype=np.uint8)
    for (exp, direction), rows in lab.groupby(["exp", "direction"], sort=True):
        order = rows.sort_values("timestep", kind="stable").index.to_numpy()
        torch.manual_seed(seed + hash((str(exp), str(direction))) % 10007)
        conds = [c_all[i : i + 1].repeat(K, 1) for i in order]
        feats = [None if g_all is None else g_all[i : i + 1].repeat(K, *[1] * (g_all.dim() - 1)) for i in order]
        print(f"[gen] refiner {exp}/{direction}: {len(order)} frames x K={K}", flush=True)
        for i, z in zip(order, cfm.sample_refiner_sequence(conds, feats)):
            fake[i] = dm.to_uint8_gray(cfm._decode_from_target_space(z))
    return fake, real, cond, DI


@torch.no_grad()
def single_frame_samples(cfm, dl, ds, K, seed, device):
    """K independent single-frame samples per row, row-aligned with ``ds.label``."""
    fake = None
    for k in range(K):
        torch.manual_seed(seed + k)
        row = 0
        for batch in dl:
            _real, hidden_state, direction, _exp, _t, di, info, _vm, mask, *_ = cfm._unpack_generation_batch(batch)
            c, g = cfm._unpack_conditions(info.to(device), hidden_state.to(device), direction.to(device), di.to(device))
            mask = None if mask is None else mask.to(device)
            u8 = dm.to_uint8_gray(cfm._decode_from_target_space(cfm.sample_with_ode(c, gru_feat=g, damage_mask=mask)))
            if fake is None:
                fake = np.empty((len(ds.label), K, *u8.shape[1:]), dtype=np.uint8)
            fake[row : row + len(u8), k] = u8
            row += len(u8)
    return fake


def exposure_stats(fake, label):
    """Mean gray, bright / saturated frame shares and tone_jump (gray levels) of (N, K, H, W) uint8."""
    frame_mean = fake.reshape(*fake.shape[:2], -1).mean(-1)
    sat = ((fake <= 5) | (fake >= 250)).reshape(*fake.shape[:2], -1).mean(-1) > 0.01
    jumps = []
    for rows in trajectories(label).values():
        jumps.append(np.abs(np.diff(frame_mean[rows], axis=0)).ravel())
    return {
        "mean_gray": float(frame_mean.mean()),
        "bright_frac": float((frame_mean > 200).mean()),
        "saturated_frac": float(sat.mean()),
        "tone_jump": float(np.concatenate(jumps).mean()) if jumps else float("nan"),
    }


class PeriodicKIDRunner:
    """Holds DINOv3 and the reference features between calls."""

    def __init__(self, dataset, k, image_dir, seed=0, batch_size=64):
        self.dataset = dataset
        self.k = int(k)
        self.image_dir = image_dir
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.fx = None
        self.real_g = None

    def _features(self, device):
        if self.fx is None:
            self.fx = dm.DinoFeatures(dm.DEFAULT_DINO, device=str(device))
            names = self.dataset.label["filename"].tolist()
            real_u8 = dm.load_real_uint8(names, self.image_dir, int(self.dataset.image_size))
            self.real_g, _ = self.fx(real_u8)
        return self.fx

    def __call__(self, cfm):
        device = cfm.device
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        was_training = cfm.training
        cfm.eval()
        try:
            dl = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            if is_sequence_refiner(cfm):
                fake = refiner_sequence_samples(cfm, dl, self.dataset, self.k, self.seed, device)[0]
            else:
                fake = single_frame_samples(cfm, dl, self.dataset, self.k, self.seed, device)
            fx = self._features(device)
            n = fake.shape[0]
            fake_g, _ = fx(fake.reshape(-1, *fake.shape[2:]))
            gf = np.repeat(np.arange(n), self.k)
            gr = np.arange(n)
            dev = str(device)
            kid = dm.GroupMMD(
                dm.group_sums(fake_g, gf, fake_g, gf, n, dm.poly_kernel, dev),
                dm.group_sums(self.real_g, gr, self.real_g, gr, n, dm.poly_kernel, dev),
                dm.group_sums(fake_g, gf, self.real_g, gr, n, dm.poly_kernel, dev),
                dm.counts(gf, n),
                dm.counts(gr, n),
            ).value()
            out = {"kid_global": float(kid), **exposure_stats(fake, self.dataset.label)}
        finally:
            cfm.train(was_training)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        return out


def append_log(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[kid] {json.dumps(record)}", flush=True)


def kid_log_path(result_dir):
    return os.path.join(result_dir, "kid_log.jsonl")
