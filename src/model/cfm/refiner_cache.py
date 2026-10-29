"""Offline base states for the refiner state track (v10.5, docs/ARCHITECTURE_UPDATE_V10_5.md).

For every frame of a split, ``n`` full-ODE samples of the frozen single-frame base, conditioned
exactly as at inference (clean conditions, no flip, the frame's own object mask). The state track
then reads these instead of the 1-step / 5-step samples it draws on the fly, so the base hop
carries the base's real full-ODE error, tone spikes included.

The file is keyed by base checkpoint, split frames, ``n``, ODE steps and solver, and built once
under a file lock so two jobs starting together do not both build it.
"""

import fcntl
import hashlib
import os

import torch

CACHE_DIR = os.path.join("results", "DamagePatternGeneration", "state_cache")


def _cache_path(model, dataset, n):
    base_ckpt = os.path.abspath(str(model.refiner_base_checkpoint))
    frames = "\n".join(sorted(dataset.label["filename"].tolist()))
    key = hashlib.md5(f"{base_ckpt}\n{frames}".encode()).hexdigest()[:12]
    base = model._refiner_base_ref[0]
    name = (
        f"{dataset.mode}_n{int(n)}_{base.cfm_solver}{int(base.cfm_ode_steps)}_"
        f"s{int(dataset.image_size)}_{key}.pt"
    )
    return os.path.join(CACHE_DIR, name)


def _frame_conditions(dataset, rows):
    """Clean conditions of label rows: what the base sees at inference for these frames."""
    info, hidden, direction, di, masks = [], [], [], [], []
    size = int(dataset.image_size)
    for _, row in rows.iterrows():
        hs = dataset.hidden_state[row["exp"]][int(row["timestep"]), :]
        hs = dataset._apply_g_pca(torch.as_tensor(hs, dtype=torch.float32))
        if getattr(dataset, "app_code_dim", 0) > 0:  # trailing appearance-code column, as in __getitem__
            hs = torch.cat([hs, hs.new_tensor([float(dataset.app_index(row["exp"], row["direction"]))])])
        hidden.append(hs)
        info.append(dataset._info_for_exp(row["exp"]))
        direction.append(dataset._direction_to_tensor(row["direction"], flipped=False, exp=row["exp"], timestep=row["timestep"]))
        di.append(torch.FloatTensor([float(row["DI"])]))
        mask = dataset._load_object_mask_tensor(row["filename"], size)
        masks.append(torch.ones(1, size, size) if mask is None else mask.float())
    return (
        torch.stack(info),
        torch.stack(hidden),
        torch.stack(direction),
        torch.stack(di),
        torch.stack(masks),
    )


@torch.no_grad()
def _build(model, dataset, n, device, rows_per_batch):
    base = model._refiner_base_ref[0]
    base.to(device).eval()
    states = {}
    label = dataset.label
    for start in range(0, len(label), rows_per_batch):
        rows = label.iloc[start : start + rows_per_batch]
        info, hs, dr, di, mask = (t.to(device) for t in _frame_conditions(dataset, rows))
        rep = lambda x: x.repeat_interleave(int(n), dim=0)
        c, g = base._unpack_conditions(rep(info), rep(hs), rep(dr), rep(di))
        x = base.sample_with_ode(c, gru_feat=g, damage_mask=rep(mask))
        x = x.mean(dim=1, keepdim=True).to(torch.float16).cpu()
        for j, name in enumerate(rows["filename"].tolist()):
            states[name] = x[j * int(n) : (j + 1) * int(n)].clone()
        print(f"[state_cache] {min(start + rows_per_batch, len(label))}/{len(label)} frames", flush=True)
    return states


def load_or_build_state_cache(model, dataset, n, device, rows_per_batch=64):
    """{filename: (n, 1, H, W) float16} for every frame of ``dataset``; built once, then loaded."""
    model._attach_refiner_base()
    # Attached before fit, so it sits on the CPU; validation still samples it on the fly.
    model._refiner_base_ref[0].to(device).eval()
    path = _cache_path(model, dataset, n)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if os.path.isfile(path):
                states = torch.load(path, map_location="cpu", weights_only=False)
                print(f"[state_cache] loaded {len(states)} frames from {path}", flush=True)
                return states
            print(f"[state_cache] building {path} ({len(dataset.label)} frames x {n})", flush=True)
            states = _build(model, dataset, n, device, rows_per_batch)
            tmp = path + ".tmp"
            torch.save(states, tmp)
            os.replace(tmp, path)
            print(f"[state_cache] wrote {path}", flush=True)
            return states
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
