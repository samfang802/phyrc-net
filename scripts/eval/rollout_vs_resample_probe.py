#!/usr/bin/env python3
"""Is the error of an h-step self-rollout the error of one Resampling-Forcing hop at some lower t_s? (No training.)

For target frames t of the split (real walks), S_{t-1} is built two ways, both at the honest time t-1:
  roll_h    inference: start from the real frame t-1-h (and the real memory up to it), generate frames
            t-h .. t-1 with the full sampler (cfm_ode_steps, frame conditions, Δg, memory accumulated from the
            generated frames, S confidence = refiner_state_conf_value), h = 1, 2, 4, 8
  rf_ts     training: the real frame t-1 noised to t_s and integrated back (Euler, refiner_resample_steps) reading
            the real frame t-2 and the real memory before t-1, as _refiner_resampled_prev; t_s = 0 is a fresh draw
            (= "parallel resampling" of Resampling Forcing, arXiv 2512.15702, Tab. 2)
  ar<ts>_h  the paper's autoregressive resampling: from the real anchor t-1-h, each of frames t-h .. t-1 is its real
            frame noised to t_s and integrated back reading the previous resampled frame (S confidence 1 for the
            real anchor, t_s after), real memory up to the frame before; one t_s per chain, h = 2, 4, 8
Each S is scored against the real frame t-1 (uint8 gray):
  rmse_lo / mid / hi   error per band (Gaussian σ 4 / 1 split)
  d_mean, std_ratio, hp_ratio   tone offset, global contrast and fine texture relative to the real frame
  keep, false          real cracks found in S (2 px) / cracks of S away from real cracks, per real crack pixel
and by what the refiner does with it on frame t (real memory up to t-1, same noise / flow times per variant):
  loss                 CFM loss of the real frame t
  dmem                 loss increase when the memory is zeroed (how much the model leans on memory given this S)
Per h, the closest t_s on the standardised S statistics is reported (the calibration t_s(h), if any fits).
The resampling hops (rf / ar) integrate with the model's training solver (refiner_resample_solver); the ar chain
writes its memory from the resampled frames for window-trained models (refiner_window > 0, as rfmv2 trains) and from
the real frames otherwise (--ar_mem overrides). Target positions are batched --chunk at a time; every position
draws its noise from its own generator (seed + 100003 * walk + t), so the batching does not change the result.

  python3 scripts/eval/rollout_vs_resample_probe.py --model rfm=<run>@<ckpt> --out results/rollout_probe/rfm.json
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import ndimage as ndi
from torch.utils.data import DataLoader

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_spec = importlib.util.spec_from_file_location(
    "_s_probe", os.path.join(_ROOT, "scripts", "eval", "refiner_s_probe.py"))
_sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sp)

from src.metrics import damage_texture as dt  # noqa: E402
from src.metrics import dino_mmd as dm  # noqa: E402
from src.model.cfm.refiner_memory import memory_feature  # noqa: E402

HOPS = (1, 2, 4, 8)
TS = (0.9, 0.7, 0.5, 0.3, 0.1, 0.0)
AR_TS = (0.5, 0.3, 0.1, 0.0)
AR_HOPS = (2, 4, 8)
STATS = ("rmse_lo", "rmse_mid", "rmse_hi", "d_mean", "std_ratio", "hp_ratio", "keep", "false")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, help="name=result_dir@checkpoint")
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--n", type=int, default=8, help="draws per target frame")
    p.add_argument("--t_first", type=int, default=10)
    p.add_argument("--t_every", type=int, default=3)
    p.add_argument("--flow_draws", type=int, default=4, help="flow-time draws per row for the loss on frame t")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--app_mean", action="store_true",
                   help="train split: give every frame the mean appearance code (as unseen columns get)")
    p.add_argument("--chunk", type=int, default=12, help="target positions per batch (rows = chunk * n)")
    p.add_argument("--ar_mem", default="auto", choices=("auto", "real", "resampled"),
                   help="memory written along the ar chain; auto = resampled for window-trained models")
    p.add_argument("--out", required=True)
    return p.parse_args()


@torch.no_grad()
def _load_split(cfm, ds, device):
    """Per dataset row (label order): target-space frame, c, g, condition mask."""
    xs, cs, gs, ms = [], [], [], []
    for batch in DataLoader(ds, batch_size=32, shuffle=False, num_workers=1):
        real, hs, dr, _exp, _ts, di, info, _valid, cmask, _ar = cfm._unpack_generation_batch(batch)
        c, g = cfm._unpack_conditions(info.to(device), hs.to(device), dr.to(device), di.to(device))
        xs.append(cfm._encode_to_target_space(real.to(device)))
        cs.append(c)
        gs.append(g)
        ms.append(None if cmask is None else cfm._align_spatial_to_target(cmask.to(device)))
    cat = lambda v: None if v[0] is None else torch.cat(v)
    return torch.cat(xs), torch.cat(cs), cat(gs), cat(ms)


def _walks(ds):
    out = []
    for _, rows in ds.label.groupby(["exp", "direction"], sort=True):
        out.append(rows.sort_values("timestep", kind="stable").index.to_numpy())
    return out


class _BatchTrack:
    """Rule memory (MemoryTrack, gate none, ttl 0) of many probe positions side by side, each with its own
    history: the window is NaN-padded per row and written with nanmedian, as the rfmv2 training windows do."""

    def __init__(self, recent, value, sig, tau):
        self.recent, self.value, self.sig, self.tau = recent, value, sig, tau

    def update(self, frame):
        f = memory_feature(frame, self.sig, self.tau).float()
        self.recent = self.recent[1:] + [f]
        self.value = torch.minimum(self.value, torch.nanmedian(torch.stack(self.recent), dim=0).values)


class _Ctx:
    def __init__(self, cfm, X, C, G, M, n):
        self.cfm, self.X, self.C, self.G, self.M, self.n = cfm, X, C, G, M, n
        self.has_mem = bool(getattr(cfm, "refiner_memory", False))
        self.conf_inf = float(getattr(cfm, "refiner_state_conf_value", 1.0))
        self.solver = str(getattr(cfm, "refiner_resample_solver", "euler"))
        if self.has_mem:
            if getattr(cfm, "refiner_memory_encoder", False) or int(getattr(cfm, "refiner_memory_ttl", 0)) > 0 \
                    or getattr(cfm, "refiner_memory_gate", "none") not in ("none",):
                raise ValueError("the batched probe supports the plain rule memory only (no encoder, gate none, ttl 0)")
            self.sig, self.tau = float(cfm.refiner_memory_sigma), float(cfm.refiner_memory_tau)
            self.win = int(cfm.refiner_memory_window)
            self.F = memory_feature(X, self.sig, self.tau).float()  # every frame's memory feature, once
            self._mem_cache = {}

    def R(self, v, idx):
        """Rows ``idx`` (one per position), each repeated n times: [len(idx) * n, ...]."""
        if v is None:
            return None
        return v[torch.as_tensor([int(i) for i in idx], device=v.device)].repeat_interleave(self.n, dim=0)

    def dg(self, i, j):
        if not getattr(self.cfm, "use_delta_g", False) or self.G is None:
            return None
        return self.R(self.G, i) - self.R(self.G, j)

    def mask(self, i):
        return _sp._eval_r2._eval_damage_mask(self.cfm, self.R(self.C, i), self.R(self.M, i))

    def _memory_after(self, hist):
        """(value, last-window features) after the real frames ``hist`` (one position, 1 row)."""
        key = tuple(int(r) for r in hist)
        if key not in self._mem_cache:
            value = torch.zeros_like(self.F[:1])
            for q in range(len(key)):
                w = self.F[list(key[max(0, q - self.win + 1): q + 1])]
                value = torch.minimum(value, w.median(dim=0, keepdim=True).values)
            recent = [self.F[k:k + 1] for k in key[-self.win:]]
            pad = [torch.full_like(self.F[:1], float("nan"))] * (self.win - len(recent))
            self._mem_cache[key] = (value, pad + recent)
        return self._mem_cache[key]

    def track(self, hists):
        """Memory tracks fed with the real frames ``hists[b]`` (in order), one per position, n rows each."""
        if not self.has_mem:
            return None
        got = [self._memory_after(h) for h in hists]
        rep = lambda t: t.repeat_interleave(self.n, dim=0)
        value = rep(torch.cat([v for v, _ in got]))
        recent = [rep(torch.cat([r[k] for _, r in got])) for k in range(self.win)]
        return _BatchTrack(recent, value, self.sig, self.tau)

    def coords(self, x):
        """Full-image coordinate maps (patch-diffusion models), as the joint sampler builds them."""
        if not getattr(self.cfm, "use_patch_diffusion", False):
            return None
        size = x.shape[-1]
        gx, gy = self.cfm._build_full_coordinate_maps(size, size, device=x.device, dtype=torch.float32)
        return torch.stack([gx, gy], dim=0).unsqueeze(0).expand(x.shape[0], -1, -1, -1).contiguous()

    def hint(self, state, tr, conf, hops=None):
        """``hops``: self-generated frames behind S (rfmv2 chain-length channel); None = the model's default
        (1 hop when conf < 1, else 0), which is right for the single-hop rf_ts variants."""
        if self.has_mem:
            mem = None if tr is None else tr.value
            if mem is None:
                mem = torch.zeros_like(state[:, :1]).expand(-1, self.cfm.refiner_memory_hint_channels, -1, -1)
            state = torch.cat([state, mem.to(device=state.device, dtype=state.dtype)], dim=1)
        return self.cfm._refiner_conf_hint(state, conf, hops=hops)

    @contextlib.contextmanager
    def hop_solver(self):
        solver = self.cfm.cfm_solver
        try:
            self.cfm.cfm_solver = self.solver
            yield
        finally:
            self.cfm.cfm_solver = solver


@torch.no_grad()
def _rollout(ctx, walk, ps, h, seeds):
    """S_{t-1} after h generated frames from the real frame t-1-h, for every position t in ``ps``."""
    cfm, n = ctx.cfm, ctx.n
    starts = [p - 1 - h for p in ps]
    tr = ctx.track([walk[: s + 1] for s in starts])
    state = ctx.R(ctx.X, [walk[s] for s in starts])
    B = state.shape[0]
    with cfm.refiner_noise_streams(seeds, [n] * len(ps), state.device):
        shared = cfm._refiner_start_noise(None, B, state.device) if cfm.refiner_shared_lowfreq_noise else None
        for q in range(1, h + 1):
            i, j = [walk[s + q] for s in starts], [walk[s + q - 1] for s in starts]
            z0 = cfm._refiner_start_noise(shared, B, state.device)
            state = cfm.sample_with_ode(ctx.R(ctx.C, i), z0=z0, gru_feat=ctx.R(ctx.G, i), damage_mask=ctx.mask(i),
                                        lr_cond=ctx.hint(state, tr, ctx.conf_inf, hops=q - 1),
                                        gru_delta=ctx.dg(i, j))
            if tr is not None:
                tr.update(state)
    return state


@torch.no_grad()
def _resample(ctx, walk, ps, t_s, seeds):
    """Training's S_{t-1}: the real frame t-1 noised to t_s, integrated back reading the real t-2 (+ memory)."""
    cfm, n = ctx.cfm, ctx.n
    i1, i2 = [walk[p - 1] for p in ps], [walk[p - 2] for p in ps]
    tr = ctx.track([walk[: p - 1] for p in ps])  # memory before t-1 (real frames <= t-2)
    x1 = ctx.R(ctx.X, i1)
    with cfm.refiner_noise_streams(seeds, [n] * len(ps), x1.device):
        eps = cfm._refiner_start_noise(None, x1.shape[0], x1.device)
    with ctx.hop_solver():
        return cfm.sample_with_ode(ctx.R(ctx.C, i1), z0=eps, z_init=(1.0 - t_s) * eps + t_s * x1, t_start=t_s,
                                   num_steps=int(cfm.refiner_resample_steps), gru_feat=ctx.R(ctx.G, i1),
                                   damage_mask=ctx.mask(i1), lr_cond=ctx.hint(ctx.R(ctx.X, i2), tr, 1.0),
                                   gru_delta=ctx.dg(i1, i2))


@torch.no_grad()
def _ar_resample(ctx, walk, ps, t_s, h, seeds, mem_resampled):
    """Autoregressive self-resampling over the h frames before t (the training window's chain)."""
    cfm, n = ctx.cfm, ctx.n
    starts = [p - 1 - h for p in ps]
    tr = ctx.track([walk[: s + 1] for s in starts])
    state, conf = ctx.R(ctx.X, [walk[s] for s in starts]), 1.0
    B = state.shape[0]
    with cfm.refiner_noise_streams(seeds, [n] * len(ps), state.device), ctx.hop_solver():
        for q in range(1, h + 1):
            i, j = [walk[s + q] for s in starts], [walk[s + q - 1] for s in starts]
            xq = ctx.R(ctx.X, i)
            eps = cfm._refiner_start_noise(None, B, xq.device)
            state = cfm.sample_with_ode(ctx.R(ctx.C, i), z0=eps, z_init=(1.0 - t_s) * eps + t_s * xq, t_start=t_s,
                                        num_steps=int(cfm.refiner_resample_steps), gru_feat=ctx.R(ctx.G, i),
                                        damage_mask=ctx.mask(i), lr_cond=ctx.hint(state, tr, conf, hops=q - 1),
                                        gru_delta=ctx.dg(i, j))
            conf = t_s
            if tr is not None:
                tr.update(state if mem_resampled else xq)
    return state


@torch.no_grad()
def _frame_t_loss(ctx, walk, ps, S, conf, seeds, draws, hops=None):
    """Per position: CFM loss of the real frame t with S and the real memory up to t-1; and with the memory zeroed."""
    cfm, n, P = ctx.cfm, ctx.n, len(ps)
    i, j = [walk[p] for p in ps], [walk[p - 1] for p in ps]
    tr = ctx.track([walk[:p] for p in ps])
    x1 = ctx.R(ctx.X, i)
    c, g, m, dg, coords = ctx.R(ctx.C, i), ctx.R(ctx.G, i), ctx.mask(i), ctx.dg(i, j), ctx.coords(x1)
    shape = tuple(x1.shape[1:])
    out = {}
    for zero in (False, True):
        gens = [torch.Generator(device=x1.device).manual_seed(int(sd)) for sd in seeds]
        tot = torch.zeros(P, device=x1.device)
        for _ in range(draws):
            z0s, taus = [], []
            for gen in gens:  # per position: z0 then tau, the order of the unbatched probe
                z0s.append(torch.randn((n, *shape), generator=gen, device=x1.device))
                taus.append(torch.rand(n, generator=gen, device=x1.device))
            z0 = torch.cat(z0s)
            tau = torch.cat(taus).clamp(1e-5, 1 - 1e-5)
            zt = (1 - tau.view(-1, 1, 1, 1)) * z0 + tau.view(-1, 1, 1, 1) * x1
            hint = ctx.hint(S, tr, conf, hops)
            if ctx.has_mem and zero:
                c0, nm = int(cfm.model_in_channels), int(cfm.refiner_memory_hint_channels)
                hint = torch.cat([hint[:, :c0], torch.zeros_like(hint[:, c0:c0 + nm]), hint[:, c0 + nm:]], 1)
            if ctx.has_mem:
                hint = cfm._refiner_memory_time_gate(hint, tau)
            v = cfm._predict_velocity(zt, tau, c, coords=coords, damage_mask=m, gru_feat=g, lr_cond=hint,
                                      gru_delta=dg)
            tot += cfm._cfm_mse(v, x1 - z0).view(P, n).mean(dim=1)
        out["loss_nomem" if zero else "loss"] = (tot / draws).tolist()
        if not ctx.has_mem:
            out["loss_nomem"] = out["loss"]
            break
    return [{"loss": a, "loss_nomem": b, "dmem": b - a} for a, b in zip(out["loss"], out["loss_nomem"])]


def _bands(u):
    u = u.astype(np.float32)
    b4, b1 = ndi.gaussian_filter(u, (0, 4, 4)), ndi.gaussian_filter(u, (0, 1, 1))
    return b4, b1 - b4, u - b1


def _stats(S_u8, R_u8):
    """S [n, H, W] against the real frame R [H, W] (uint8 gray)."""
    R = np.broadcast_to(R_u8, S_u8.shape)
    bs, br = _bands(S_u8), _bands(np.ascontiguousarray(R))
    out = {f"rmse_{k}": float(np.sqrt(((a - b) ** 2).mean())) for k, a, b in zip(("lo", "mid", "hi"), bs, br)}
    s, r = S_u8.astype(np.float32), R.astype(np.float32)
    out["d_mean"] = float(s.mean() - r.mean())
    out["std_ratio"] = float(s.std(axis=(1, 2)).mean() / max(r[0].std(), 1e-6))
    out["hp_ratio"] = float(np.abs(bs[2]).mean() / max(np.abs(br[2]).mean(), 1e-6))
    cr = dt.crack_map(R_u8[None].astype(np.uint8))[0]
    cs = dt.crack_map(S_u8.astype(np.uint8))
    st = np.ones((5, 5), bool)
    crd = ndi.binary_dilation(cr, st)
    n_r = max(int(cr.sum()), 1)
    out["keep"] = float(np.mean([(cr & ndi.binary_dilation(c, st)).sum() / n_r for c in cs]))
    out["false"] = float(np.mean([(c & ~crd).sum() / n_r for c in cs]))
    return out


def main():
    cli = _parse_args()
    device = torch.device("cuda")
    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    name, cfm, ds, _dl, _lay, step = _sp._build(cli.model, cli.split, "mean", device)
    if cli.split == "train":
        ds.layout_drop_prob = 0.0  # every frame keeps its layout, as at sampling
    if cli.app_mean:
        ds.app_seq_index = {}  # no training sequence matches -> app_unseen_code (-2 = mean of the codes)
    X, C, G, M = _load_split(cfm, ds, device)
    ctx = _Ctx(cfm, X, C, G, M, cli.n)
    walks = _walks(ds)
    variants = ([f"roll_{h}" for h in HOPS] + [f"rf_{t:g}" for t in TS]
                + [f"ar{t:g}_{h}" for h in AR_HOPS for t in AR_TS])
    rows = {v: [] for v in variants}
    panel = []
    print(f"[rollout] {name} step {step} {cli.split}: {len(walks)} walks, memory {ctx.has_mem}, "
          f"conf at sampling {ctx.conf_inf}, ode steps {cfm.cfm_ode_steps}, resample steps {cfm.refiner_resample_steps}",
          flush=True)
    mem_resampled = cli.ar_mem == "resampled" or (cli.ar_mem == "auto" and int(getattr(cfm, "refiner_window", 0)) > 0)
    print(f"[rollout] resample solver {ctx.solver}, ar memory from {'resampled' if mem_resampled else 'real'} frames, "
          f"{cli.chunk} positions per batch", flush=True)
    n = cli.n
    t0 = time.time()
    for wi, walk in enumerate(walks):
        pos = list(range(max(cli.t_first, max(HOPS) + 1), len(walk), cli.t_every))
        for c0 in range(0, len(pos), max(1, cli.chunk)):
            ps = pos[c0:c0 + max(1, cli.chunk)]
            seeds = [cli.seed + 100003 * wi + p for p in ps]
            R_u8 = dm.to_uint8_gray(cfm._decode_from_target_space(ctx.X[torch.as_tensor(
                [int(walk[p - 1]) for p in ps], device=ctx.X.device)]))
            S_all = {}
            for v in variants:
                hops = None
                if v.startswith("roll_"):
                    hops = int(v.split("_")[1])
                    S, conf = _rollout(ctx, walk, ps, hops, seeds), ctx.conf_inf
                elif v.startswith("ar"):
                    t_s, h = float(v[2:].split("_")[0]), int(v.split("_")[1])
                    S, conf, hops = _ar_resample(ctx, walk, ps, t_s, h, seeds, mem_resampled), t_s, h
                else:
                    t_s = float(v.split("_")[1])
                    S, conf = _resample(ctx, walk, ps, t_s, seeds), t_s
                S_u8 = dm.to_uint8_gray(cfm._decode_from_target_space(S))
                losses = _frame_t_loss(ctx, walk, ps, S, conf, seeds, cli.flow_draws, hops)
                for k, p in enumerate(ps):
                    S_all[(v, p)] = S_u8[k * n]
                    rows[v].append({"walk": wi, "pos": int(p), **_stats(S_u8[k * n:(k + 1) * n], R_u8[k]),
                                    **losses[k]})
            for k, p in enumerate(ps):
                if len(panel) < 6 and p % 12 == 1:
                    panel.append(np.concatenate([R_u8[k]] + [S_all[(v, p)] for v in variants
                                                             if not v.startswith("ar") or v.endswith("_8")], 1))
                r = {v: rows[v][-len(ps) + k] for v in variants}
                print(f"[rollout] walk {wi} t={p}: " + " ".join(
                    f"{v} lo {r[v]['rmse_lo']:.1f} keep {r[v]['keep']:.2f} dmem {r[v]['dmem']:+.4f}"
                    for v in ("roll_1", "roll_8", "rf_0", "ar0.3_8", "ar0_8")), flush=True)
    print(f"[rollout] sampling done in {time.time() - t0:.0f} s", flush=True)

    summary = {v: {k: float(np.mean([r[k] for r in rows[v]])) for k in STATS + ("loss", "dmem")} for v in variants}
    # Calibration: per h, the t_s whose S statistics are closest (each statistic standardised over all variants).
    mat = np.array([[summary[v][k] for k in STATS] for v in variants])
    z = (mat - mat.mean(0)) / (mat.std(0) + 1e-9)
    zi = {v: z[i] for i, v in enumerate(variants)}
    match = {}
    for h in HOPS:
        d = {f"{t:g}": float(np.linalg.norm(zi[f"roll_{h}"] - zi[f"rf_{t:g}"])) for t in TS}
        best = min(d, key=d.get)
        # a fit is only meaningful if the nearest t_s is about as close as neighbouring t_s are to each other
        match[f"roll_{h}"] = {"nearest_ts": best, "dist": d}
    rf_spacing = float(np.mean([np.linalg.norm(zi[f"rf_{a:g}"] - zi[f"rf_{b:g}"]) for a, b in zip(TS, TS[1:])]))
    print("\n[rollout] variant   " + " ".join(f"{k:>9}" for k in STATS + ("loss", "dmem")))
    for v in variants:
        print(f"[rollout] {v:9s} " + " ".join(f"{summary[v][k]:9.4f}" for k in STATS + ("loss", "dmem")))
    print(f"[rollout] mean distance between neighbouring t_s: {rf_spacing:.2f}")
    for h in HOPS:
        m = match[f"roll_{h}"]
        print(f"[rollout] roll_{h}: nearest t_s {m['nearest_ts']} (dist {m['dist'][m['nearest_ts']]:.2f}); "
              + " ".join(f"{t}:{d:.2f}" for t, d in m["dist"].items()))
    match_ar = {}
    for h in AR_HOPS:
        d = {f"{t:g}": float(np.linalg.norm(zi[f"roll_{h}"] - zi[f"ar{t:g}_{h}"])) for t in AR_TS}
        best = min(d, key=d.get)
        match_ar[f"roll_{h}"] = {"nearest_ts": best, "dist": d}
        print(f"[rollout] roll_{h} vs autoregressive resampling ({h} hops): nearest t_s {best} "
              f"(dist {d[best]:.2f}); " + " ".join(f"{t}:{x:.2f}" for t, x in d.items()))
    with open(cli.out, "w") as fh:
        json.dump({"model": cli.model, "step": step, "split": cli.split, "n": cli.n, "summary": summary,
                   "match": match, "match_ar": match_ar, "resample_solver": ctx.solver,
                   "ar_mem_resampled": mem_resampled, "rf_spacing": rf_spacing, "rows": rows}, fh, indent=2)
    if panel:
        from PIL import Image

        Image.fromarray(np.concatenate(panel, 0).astype(np.uint8)).save(os.path.splitext(cli.out)[0] + "_panel.png")
    cols = [v for v in variants if not v.startswith("ar") or v.endswith("_8")]
    print(f"[rollout] wrote {cli.out}  (panel columns: real t-1, {', '.join(cols)})")


if __name__ == "__main__":
    main()
