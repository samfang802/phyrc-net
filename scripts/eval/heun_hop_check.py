#!/usr/bin/env python3
"""Does a Heun resampling hop leave start noise in S? And what does TF32 do to sampling time? (No training.)

The training hop of rfmv2 (refiner_resample_solver heun, refiner_resample_steps 8): the real frame t-1 noised to
t_s, z = (1 - t_s) eps + t_s x1, integrated to 1 reading the real t-2 and the real memory. Heun's last stage reads
v(z, t = 1), where the start noise is no longer visible, so the result should keep about dt / 2 * eps. Per t_s
and solver (euler | heun | heun_last_euler = Heun with an Euler last step), against the real frame x1:
  corr_eps   correlation of S - x1 with eps (0 = no leftover start noise)
  res_std    std of S - x1 in gray levels; res_hp: its high-pass part (Gaussian σ 1)
  keep/false/hp_ratio   crack and texture statistics of rollout_vs_resample_probe.py
Also the full sampler (t_s = 0, cfm_ode_steps) heun vs heun_last_euler (corr_eps, hp_ratio against the real
frame t), and the wall time of one full-sampler batch with TF32 matmuls off / on (and the pixel change).

  python3 scripts/eval/heun_hop_check.py --model rfmv2=<run>@<ckpt> --out x.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_spec = importlib.util.spec_from_file_location(
    "_rp", os.path.join(_ROOT, "scripts", "eval", "rollout_vs_resample_probe.py"))
_rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rp)
_sp = _rp._sp

from src.metrics import dino_mmd as dm  # noqa: E402
from src.model.cfm.refiner_memory import _gauss_1d  # noqa: E402

TS = (0.9, 0.7, 0.5, 0.3, 0.1)
SOLVERS = ("euler", "heun", "heun_last_euler")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, help="name=result_dir@checkpoint")
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--t_every", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def _hp(x):
    k, r = _gauss_1d(1.0, x.device, x.dtype)
    c = x.shape[1]
    b = torch.nn.functional.conv2d(torch.nn.functional.pad(x, (r, r, 0, 0), mode="reflect"),
                                   k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    b = torch.nn.functional.conv2d(torch.nn.functional.pad(b, (0, 0, r, r), mode="reflect"),
                                   k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x - b


def _set_solver(cfm, mode):
    cfm.cfm_solver = "euler" if mode == "euler" else "heun"
    cfm.cfm_heun_last_euler = mode == "heun_last_euler"


def _residual_stats(S, x1, eps, n, P):
    r = (S - x1).float()
    e = eps.float()
    corr = torch.nn.functional.cosine_similarity(r.flatten(1), e.flatten(1), dim=1).mean()
    g = 127.5  # target space [-1, 1] -> gray levels
    return {"corr_eps": float(corr), "res_std": float(r.std() * g), "res_hp": float(_hp(r).std() * g)}


@torch.no_grad()
def main():
    cli = _parse_args()
    device = torch.device("cuda")
    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    name, cfm, ds, _dl, _lay, step = _sp._build(cli.model, cli.split, "mean", device)
    X, C, G, M = _rp._load_split(cfm, ds, device)
    ctx = _rp._Ctx(cfm, X, C, G, M, cli.n)
    walks = _rp._walks(ds)
    n = cli.n
    solver0 = cfm.cfm_solver
    res = {"model": cli.model, "step": step, "hop": {}, "full": {}, "tf32": {}}
    print(f"[heun] {name} step {step}: resample steps {cfm.refiner_resample_steps}, ode steps {cfm.cfm_ode_steps}",
          flush=True)
    for wi, walk in enumerate(walks):
        ps = list(range(10, len(walk), cli.t_every))
        P = len(ps)
        seeds = [cli.seed + 100003 * wi + p for p in ps]
        i1, i2 = [walk[p - 1] for p in ps], [walk[p - 2] for p in ps]
        tr = ctx.track([walk[: p - 1] for p in ps])
        x1 = ctx.R(X, i1)
        R_u8 = dm.to_uint8_gray(cfm._decode_from_target_space(X[torch.as_tensor([int(x) for x in i1], device=device)]))
        with cfm.refiner_noise_streams(seeds, [n] * P, device):
            eps = cfm._refiner_start_noise(None, P * n, device)
        hint = ctx.hint(ctx.R(X, i2), tr, 1.0)
        for t_s in TS:
            for mode in SOLVERS:
                _set_solver(cfm, mode)
                S = cfm.sample_with_ode(ctx.R(C, i1), z0=eps, z_init=(1.0 - t_s) * eps + t_s * x1, t_start=t_s,
                                        num_steps=int(cfm.refiner_resample_steps), gru_feat=ctx.R(G, i1),
                                        damage_mask=ctx.mask(i1), lr_cond=hint, gru_delta=ctx.dg(i1, i2))
                st = _residual_stats(S, x1, eps, n, P)
                S_u8 = dm.to_uint8_gray(cfm._decode_from_target_space(S))
                cr = [_rp._stats(S_u8[b * n:(b + 1) * n], R_u8[b]) for b in range(P)]
                for k in ("keep", "false", "hp_ratio"):
                    st[k] = float(np.mean([c[k] for c in cr]))
                res["hop"].setdefault(f"{t_s:g}", {}).setdefault(mode, []).append(st)
        # full sampler: frame t-1 from noise reading the real t-2 (as a sampling step from a real frame)
        for mode in ("heun", "heun_last_euler"):
            _set_solver(cfm, mode)
            S = cfm.sample_with_ode(ctx.R(C, i1), z0=eps, gru_feat=ctx.R(G, i1), damage_mask=ctx.mask(i1),
                                    lr_cond=ctx.hint(ctx.R(X, i2), tr, ctx.conf_inf, hops=8),
                                    gru_delta=ctx.dg(i1, i2))
            st = _residual_stats(S, x1, eps, n, P)
            S_u8 = dm.to_uint8_gray(cfm._decode_from_target_space(S))
            cr = [_rp._stats(S_u8[b * n:(b + 1) * n], R_u8[b]) for b in range(P)]
            for k in ("keep", "false", "hp_ratio"):
                st[k] = float(np.mean([c[k] for c in cr]))
            res["full"].setdefault(mode, []).append(st)
        if wi == 0:  # TF32 timing on this batch, the eval sampler's solver
            cfm.cfm_solver, cfm.cfm_heun_last_euler = solver0, False
            outs = {}
            for tf32 in (False, True, False):
                torch.backends.cuda.matmul.allow_tf32 = tf32  # convs already use TF32 (cudnn default)
                torch.cuda.synchronize()
                t0 = time.time()
                outs[tf32] = cfm.sample_with_ode(ctx.R(C, i1), z0=eps, gru_feat=ctx.R(G, i1), damage_mask=ctx.mask(i1),
                                                 lr_cond=ctx.hint(ctx.R(X, i2), tr, ctx.conf_inf, hops=8),
                                                 gru_delta=ctx.dg(i1, i2))
                torch.cuda.synchronize()
                res["tf32"][f"time_{'on' if tf32 else 'off'}"] = time.time() - t0  # the second "off" wins (warm)
            torch.backends.cuda.matmul.allow_tf32 = False
            d = np.abs(dm.to_uint8_gray(cfm._decode_from_target_space(outs[True])).astype(np.int16)
                       - dm.to_uint8_gray(cfm._decode_from_target_space(outs[False])).astype(np.int16))
            res["tf32"].update(rows=int(P * n), mean_abs_diff=float(d.mean()), max_abs_diff=int(d.max()))
    cfm.cfm_solver, cfm.cfm_heun_last_euler = solver0, False

    avg = lambda lst: {k: float(np.mean([x[k] for x in lst])) for k in lst[0]}
    keys = ("corr_eps", "res_std", "res_hp", "keep", "false", "hp_ratio")
    print("\n[heun] resampling hop of the real frame t-1 (reading the real t-2)")
    print("[heun]   t_s  solver           " + " ".join(f"{k:>9}" for k in keys))
    for t_s in TS:
        for mode in SOLVERS:
            a = avg(res["hop"][f"{t_s:g}"][mode])
            res["hop"][f"{t_s:g}"][mode] = a
            print(f"[heun]   {t_s:<4g} {mode:16s} " + " ".join(f"{a[k]:9.3f}" for k in keys))
    print("[heun] full sampler from noise (frame t-1 given the real t-2)")
    for mode in ("heun", "heun_last_euler"):
        a = avg(res["full"][mode])
        res["full"][mode] = a
        print(f"[heun]        {mode:16s} " + " ".join(f"{a[k]:9.3f}" for k in keys))
    t = res["tf32"]
    print(f"[heun] TF32 on one batch of {t['rows']} rows x {cfm.cfm_ode_steps} steps: off {t['time_off']:.1f} s, "
          f"on {t['time_on']:.1f} s; pixels mean |diff| {t['mean_abs_diff']:.3f}, max {t['max_abs_diff']}", flush=True)
    with open(cli.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[heun] wrote {cli.out}")


if __name__ == "__main__":
    main()
