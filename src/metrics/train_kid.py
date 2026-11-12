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
import zlib

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.metrics import dino_mmd as dm
from src.metrics.temporal import trajectories


def is_sequence_refiner(cfm):
    return bool(getattr(cfm, "use_refiner_residual", False))


@torch.no_grad()
def refiner_sequence_samples(cfm, dl, ds, K, seed, device, layouts=None, layout_sdf=None):
    """Walk every (column, direction) in timestep order; K trajectories per walk.

    ``layouts`` (layout_cond models only): filename -> (L, S, S) spall coverage; each frame's condition
    mask is then [object mask 1 | layout SDF] (src/data/layout_io.coverage_to_sdf128), with sample k
    of every frame reading layout k % L (L = 1: a deterministic layout, e.g. the gravity template).
    ``layout_sdf``: the same as ready SDFs, filename -> (L, H, W) or (H, W) (the dataset's layout cache,
    used by the periodic KID so it reads the layouts the model trains on). Without either the layout
    input is 0 (the dropped value).

    Returns fake uint8 (N, K, H, W), real list of uint8 batches, cond list, DI list, row-aligned
    with ``ds.label``.
    """
    if (layouts is not None or layout_sdf is not None) and not bool(getattr(cfm, "layout_cond", False)):
        raise ValueError("layouts given but the model has no layout channel (layout_cond=False)")
    if layouts is not None and layout_sdf is not None:
        raise ValueError("pass layouts or layout_sdf, not both")
    if len(ds) != len(ds.label):
        raise ValueError("refiner path needs one dataset item per label row")
    # ILF-lite memory at sampling (diagnostic, --set refiner_memory_source=...): self = accumulated from the
    # generated frames; oracle = from the real frames, same rule (upper bound); zero = always empty;
    # swap = from the real frames of the next walk of the split (the other face of the same column: same
    # loading history, wrong crack positions), aligned by sequence progress.
    mem_src = str(getattr(cfm, "refiner_memory_source", "self"))
    if mem_src not in ("self", "oracle", "zero", "swap"):
        raise ValueError(f"refiner_memory_source must be self|oracle|zero|swap, got {mem_src!r}")
    mem_given = bool(getattr(cfm, "refiner_memory", False)) and mem_src != "self"
    # Frame 0 source (--set refiner_frame0_source=...): base = the frozen single-frame model draws it; real = the
    # walk's real first photo (given, as a pre-test photo would be).
    f0_src = str(getattr(cfm, "refiner_frame0_source", "base"))
    if f0_src not in ("base", "real"):
        raise ValueError(f"refiner_frame0_source must be base|real, got {f0_src!r}")
    keep_real = mem_given or f0_src == "real"
    cs, gs, real, real_t, cond, DI = [], [], [], [], [], []
    for batch in dl:
        real_data, hidden_state, direction, _exp, _t, di, info, *_ = cfm._unpack_generation_batch(batch)
        c, g = cfm._unpack_conditions(info.to(device), hidden_state.to(device), direction.to(device), di.to(device))
        cs.append(c)
        gs.append(g)
        real.append(dm.to_uint8_gray(real_data))
        if keep_real:
            real_t.append(real_data.float().cpu())
        cond.append(hidden_state.reshape(hidden_state.shape[0], -1).float().cpu().numpy())
        DI.append(di.view(-1).float().cpu().numpy())
    c_all = torch.cat(cs)
    g_all = None if gs[0] is None else torch.cat(gs)
    real_all = torch.cat(real_t) if keep_real else None
    lab = ds.label
    fake = np.empty((len(lab), K, *real[0].shape[1:]), dtype=np.uint8)
    walks = {key: rows.sort_values("timestep", kind="stable").index.to_numpy()
             for key, rows in lab.groupby(["exp", "direction"], sort=True)}
    walk_keys = list(walks)
    if mem_given and mem_src == "swap" and len(walk_keys) < 2:
        raise ValueError("refiner_memory_source=swap needs at least two walks in the split")
    gate_mode = getattr(cfm, "refiner_memory_gate", "none") if getattr(cfm, "refiner_memory", False) else None

    def walk_inputs(exp, direction, order):
        conds = [c_all[i : i + 1].repeat(K, 1) for i in order]
        feats = [None if g_all is None else g_all[i : i + 1].repeat(K, *[1] * (g_all.dim() - 1)) for i in order]
        masks = None
        if layouts is not None or layout_sdf is not None:
            from src.data import layout_io

            size = real[0].shape[-1]
            masks = []
            for i in order:
                f = str(lab.loc[i, "filename"])
                if layouts is not None:
                    cov = np.asarray(layouts[f], dtype=np.float32)
                    if cov.shape[0] not in (1, K) and cov.shape[0] < K:
                        raise ValueError(f"layout for {f} has {cov.shape[0]} samples, need 1 or >= K={K}")
                    sdf = torch.from_numpy(layout_io.coverage_to_sdf128(cov[[k % cov.shape[0] for k in range(K)]], size=size))
                else:
                    if f not in layout_sdf:
                        raise ValueError(f"no cached layout for {f}")
                    a = np.asarray(layout_sdf[f], dtype=np.float32)
                    a = a[None] if a.ndim == 2 else a
                    sdf = torch.from_numpy(a[[k % a.shape[0] for k in range(K)]])
                    if sdf.shape[-1] != size:
                        sdf = torch.nn.functional.interpolate(sdf[:, None], size=(size, size), mode="bilinear",
                                                              align_corners=False)[:, 0]
                sdf = sdf.unsqueeze(1)
                masks.append(torch.cat([torch.ones_like(sdf), sdf], dim=1).to(device=c_all.device, dtype=c_all.dtype))
        memory, gates = None, None
        if gate_mode is not None:
            from src.model.cfm.refiner_memory import write_gates

            gates = write_gates([str(lab.loc[i, "filename"]) for i in order], gate_mode)
        if mem_given:
            track = cfm.refiner_memory_track()
            n_mem = int(cfm.refiner_memory_hint_channels)
            memory = []
            src, src_gates = order, gates
            if mem_src == "swap":
                donor = walks[walk_keys[(walk_keys.index((exp, direction)) + 1) % len(walk_keys)]]
                src = donor[np.round(np.linspace(0, len(donor) - 1, len(order))).astype(int)]
                src_gates = write_gates([str(lab.loc[i, "filename"]) for i in src], gate_mode)
            for i, gate in zip(src, src_gates):
                x = cfm._encode_to_target_space(real_all[i : i + 1].to(device)).to(c_all.dtype)
                prev = track.value
                if prev is None or mem_src == "zero":
                    prev = torch.zeros_like(x[:, :1]).expand(-1, n_mem, -1, -1)
                memory.append(prev.repeat(K, 1, 1, 1))
                track.update(x, gate)
        frame0 = None
        if f0_src == "real":
            frame0 = cfm._encode_to_target_space(real_all[order[0] : order[0] + 1].to(device)).to(c_all.dtype)
            frame0 = frame0.repeat(K, 1, 1, 1)
        print(f"[gen] refiner {exp}/{direction}: {len(order)} frames x K={K}{' + layouts' if masks else ''}"
              f"{f' + {mem_src} memory' if mem_given else ''}{' + real frame 0' if frame0 is not None else ''}",
              flush=True)
        return conds, feats, masks, memory, gates, frame0

    # Walk seed: crc32 of the walk name, the same in every process (Python's hash() of a str is salted per
    # process, so the old hash() seed gave every run different noise). Each walk draws from its own generator,
    # so batching walks does not change any walk's noise.
    seeds = {key: int(seed) + zlib.crc32(f"{key[0]}/{key[1]}".encode()) % 10007 for key in walk_keys}
    items = [(key, order, walk_inputs(*key, order)) for key, order in walks.items()]
    # All walks of the split in one batch (padded to the longest walk), unless a per-frame flag differs between
    # walks or rows interact: given memory / write gates, the state-roll diagnostic, the joint sampler.
    batch_walks = (not mem_given and int(getattr(cfm, "refiner_state_roll", 0) or 0) == 0
                   and not getattr(cfm, "refiner_neighbors", False)
                   and all(it[2][4] is None or all(it[2][4]) for it in items)
                   and int(getattr(cfm, "refiner_walk_batch", 1)) != 0)
    groups = [items] if batch_walks else [[it] for it in items]
    for grp in groups:
        n_max = max(len(order) for _, order, _ in grp)

        def cat_frames(j):  # per frame: the walks' lists padded with their last entry, rows walk-major
            lists = [inp[j] for _, _, inp in grp]
            if lists[0] is None or lists[0][0] is None:
                return None
            return [torch.cat([lst[min(f, len(lst) - 1)] for lst in lists]) for f in range(n_max)]

        conds, feats = cat_frames(0), cat_frames(1)
        if feats is None:
            feats = [None] * n_max
        masks = cat_frames(2)
        memory = grp[0][2][3] if len(grp) == 1 else None
        gates = grp[0][2][4] if len(grp) == 1 else None
        f0 = [inp[5] for _, _, inp in grp]
        frame0 = None if f0[0] is None else torch.cat(f0)
        if len(grp) == 1:
            torch.manual_seed(seeds[grp[0][0]])  # samplers that draw outside _refiner_start_noise (joint)
        with cfm.refiner_noise_streams([seeds[key] for key, _, _ in grp], [K] * len(grp), c_all.device):
            zs = cfm.sample_refiner_sequence(conds, feats, damage_mask=masks, memory=memory, memory_gate=gates,
                                             frame0=frame0)
        for w, (_, order, _) in enumerate(grp):
            for i, z in zip(order, zs):
                fake[i] = dm.to_uint8_gray(cfm._decode_from_target_space(z[w * K : (w + 1) * K]))
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
                # layout_cond: the val frames' cached layouts (the predicted kind, as at inference), not 0.
                lay = getattr(self.dataset, "_layout", None) if getattr(cfm, "layout_cond", False) else None
                fake = refiner_sequence_samples(cfm, dl, self.dataset, self.k, self.seed, device,
                                                layout_sdf=lay or None)[0]
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
