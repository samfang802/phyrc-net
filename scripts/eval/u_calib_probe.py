#!/usr/bin/env python3
"""Can the spread of two draws stand in for the real error? (Pre-check for the U channel, no training.)

For every real pair (t-1, t) of the split, frame t is drawn n=3 times (x_a, x_b, x_c: same conditions, same S,
independent start noise). Per pixel, in [0,1] gray, smoothed by a Gaussian G_σ:
  D = G_σ * |x_a - x_b|      the predicted U before scaling (what inference can see)
  E = G_σ * |x_a - GT_t|     the real error of the frame that is kept (x_a)
  C = G_σ * |x_a - x_c|      a second spread: if the model were calibrated, GT_t would be one more draw, so
                             corr(D, C) is the ceiling that corr(D, E) can reach
Two settings:
  tf     S = GT_{t-1} (teacher forcing): only the error of this hop exists.
  chain  S = y_a, one draw of t-1 from GT_{t-2} (with its own spread D_prev = G_σ*|y_a - y_b|): the error
         already written into S is inherited by every draw of t, so D cannot see it. Measures how much of E
         that is, and whether a carried U, max(D, γ·D_prev), recovers it.
Reported per σ: k = mean E / mean D (fit on the pooled split; also per column and per DI tercile, the
"area bias" of U), pixel Pearson / Spearman of D vs E (object pixels), the same for D vs C (ceiling), image-level
corr of the mean D and mean E, AUROC of D for the top-20% E pixels, and E percentiles (to pick u0).

  python3 scripts/eval/u_calib_probe.py --split val --out_dir results/u_calib_probe/val \
      --model v12gtr_6k=<result_dir>@<checkpoint>[@<layouts npz>]
Without a layouts npz the dataset's own layout cache is used (train: GT iso32, val: grav), as in training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from scipy import ndimage as ndi
from scipy import stats

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.metrics import dino_mmd as dm  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_s_probe", os.path.join(_ROOT, "scripts", "eval", "refiner_s_probe.py"))
_sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sp)

SIGMAS = (0.0, 1.0, 2.0, 4.0, 8.0)
GAMMAS = (0.5, 0.8, 1.0)
PIX_PER_IMG = 1024  # object pixels kept per image for the pooled rank statistics


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, help="name=result_dir@checkpoint[@layouts npz]")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split", default="val", choices=("test", "val", "train"))
    p.add_argument("--batch", type=int, default=16, help="Pairs per batch (x8 ODE solves)")
    p.add_argument("--max_pairs", type=int, default=0, help="Evenly subsample to this many pairs (0 = all)")
    p.add_argument("--no_chain", action="store_true", help="Teacher forcing only")
    p.add_argument("--app_code", default="mean")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _gray01(cfm, x):
    return dm.to_uint8_gray(cfm._decode_from_target_space(x)).astype(np.float32) / 255.0


def _smooth(a, s):
    return a if s <= 0 else ndi.gaussian_filter(a, (0, s, s))


def _auroc(score, pos):
    n1 = int(pos.sum())
    n0 = pos.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class _Acc:
    """Per σ: pooled sums for k and Pearson, a pixel subsample for ranks, per-image means."""

    def __init__(self, keys):
        self.keys = keys
        self.px = {s: {k: [] for k in keys} for s in SIGMAS}
        self.img = {s: {k: [] for k in keys} for s in SIGMAS}
        self.meta = []

    def add(self, maps, valid, meta, rng):
        """maps: {key: (N,H,W) unsmoothed abs-diff}; valid: (N,H,W) bool."""
        n = valid.shape[0]
        pick = []
        for i in range(n):
            idx = np.flatnonzero(valid[i])
            pick.append(rng.choice(idx, size=min(PIX_PER_IMG, idx.size), replace=False))
        for s in SIGMAS:
            for k in self.keys:
                m = _smooth(maps[k], s)
                for i in range(n):
                    flat = m[i].reshape(-1)
                    self.px[s][k].append(flat[pick[i]])
                    self.img[s][k].append(float(flat[valid[i].reshape(-1)].mean()))
        self.meta += meta

    def report(self, pairs_of):
        """pairs_of: [(name, a_key, b_key)] → correlations of b against a."""
        out = {}
        exps = np.array([m["exp"] for m in self.meta])
        di = np.array([m["DI"] for m in self.meta])
        terc = np.digitize(di, np.quantile(di, [1 / 3, 2 / 3])) if len(di) > 2 else np.zeros(len(di), int)
        for s in SIGMAS:
            px = {k: np.concatenate(v) for k, v in self.px[s].items()}
            im = {k: np.asarray(v) for k, v in self.img[s].items()}
            r = {"mean": {k: float(px[k].mean()) for k in self.keys}}
            if "E" in px:
                r["E_pct"] = {f"p{q}": float(np.percentile(px["E"], q)) for q in (50, 90, 95, 99)}
                r["k"] = float(im["E"].mean() / max(im["D"].mean(), 1e-9))
                r["k_by_exp"] = {e: float(im["E"][exps == e].mean() / max(im["D"][exps == e].mean(), 1e-9))
                                 for e in sorted(set(exps))}
                r["k_by_DI_tercile"] = [float(im["E"][terc == j].mean() / max(im["D"][terc == j].mean(), 1e-9))
                                        for j in range(3) if (terc == j).any()]
            for name, a, b in pairs_of:
                if a not in px or b not in px:
                    continue
                top = px[b] >= np.percentile(px[b], 80)
                r[name] = {
                    "pearson": float(np.corrcoef(px[a], px[b])[0, 1]),
                    "spearman": float(stats.spearmanr(px[a], px[b])[0]),
                    "auroc_top20": _auroc(px[a], top),
                    "img_pearson": float(np.corrcoef(im[a], im[b])[0, 1]) if len(im[a]) > 2 else float("nan"),
                    "per_img_pearson": float(np.nanmean([
                        np.corrcoef(x, y)[0, 1] for x, y in zip(self.px[s][a], self.px[s][b])
                        if x.std() > 0 and y.std() > 0])),
                }
            out[f"sigma_{s:g}"] = r
        return out


@torch.no_grad()
def _run(name, cfm, ds, dl, lay, cli, device, panel):
    if getattr(cfm, "refiner_memory", False):
        raise SystemExit("refiner_memory models are not supported by this probe")
    real, cs, gs, vms, cms = [], [], [], [], []
    for batch in dl:
        real_data, hs, dr, _e, _t, di, info, vm, cm, *_ = cfm._unpack_generation_batch(batch)
        c, g = cfm._unpack_conditions(info.to(device), hs.to(device), dr.to(device), di.to(device))
        real.append(cfm._encode_to_target_space(real_data.to(device)))
        cs.append(c)
        gs.append(g)
        vms.append(None if vm is None else vm.cpu())
        cms.append(None if cm is None else cm.cpu())
    real, c_all = torch.cat(real), torch.cat(cs)
    g_all = None if gs[0] is None else torch.cat(gs)
    valid_all = None if vms[0] is None else (torch.cat(vms)[:, 0] > 0.5).numpy()
    cmask_all = None if cms[0] is None else torch.cat(cms)
    lab = ds.label
    if len(lab) != len(real):
        raise SystemExit("needs one dataset item per label row")

    pairs = []  # (t-2 or -1, t-1, t)
    for _, rows in lab.groupby(["exp", "direction"], sort=True):
        order = rows.sort_values("timestep", kind="stable").index.to_numpy()
        pairs += [(int(order[k - 2]) if k >= 2 else -1, int(order[k - 1]), int(order[k])) for k in range(1, len(order))]
    if cli.max_pairs and len(pairs) > cli.max_pairs:
        pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, cli.max_pairs).round().astype(int)]
    P = len(pairs)
    print(f"[u_calib] {name}: {len(lab)} frames, {P} pairs ({sum(p[0] >= 0 for p in pairs)} with t-2)", flush=True)

    size = real.shape[-1]
    from src.data import layout_io

    def _cond_mask(rows):
        """[object | layout of each row's own frame]; dataset cache unless a layouts npz was given."""
        if not getattr(cfm, "layout_cond", False):
            return None if cmask_all is None else cmask_all[rows].to(device)
        out = []
        for i in rows:
            f = str(lab.loc[i, "filename"])
            if lay is not None:
                cov = np.asarray(lay[f], dtype=np.float32)[:1]
                sdf = torch.from_numpy(layout_io.coverage_to_sdf128(cov, size=size))[:, None]
            else:
                s = ds._layout.get(f) if getattr(ds, "_layout", None) is not None else None
                if s is None:
                    sdf = torch.zeros(1, 1, size, size)
                else:
                    s = np.asarray(s, dtype=np.float32)
                    sdf = torch.from_numpy(s[0] if s.ndim == 3 else s)[None, None]
            out.append(torch.cat([torch.ones_like(sdf), sdf], 1))
        return torch.cat(out).to(device)

    conf = float(getattr(cfm, "refiner_state_conf_value", 1.0))
    hint = (lambda s: cfm._refiner_conf_hint(s, conf)) if hasattr(cfm, "_refiner_conf_hint") else (lambda s: s)
    use_dg = bool(getattr(cfm, "use_delta_g", False))
    torch.manual_seed(cli.seed)
    rng = np.random.default_rng(cli.seed)

    def draw(S, rows_t, rows_p, m):
        rep = lambda x: x.repeat(m, *([1] * (x.dim() - 1)))  # [draw 0 rows | draw 1 rows | ...]
        c = rep(c_all[rows_t])
        g = None if g_all is None else rep(g_all[rows_t])
        dg = None if (not use_dg or g_all is None) else g - rep(g_all[rows_p])
        mk = _cond_mask(rows_t)
        mk = None if mk is None else rep(mk)
        z0 = cfm._refiner_start_noise(None, c.shape[0], device).to(S.dtype)
        x = cfm.sample_with_ode(c, z0=z0, gru_feat=g, damage_mask=mk, lr_cond=hint(rep(S)), gru_delta=dg)
        return x, _gray01(cfm, x).reshape(m, len(rows_t), *x.shape[-2:])

    tf = _Acc(("D", "E", "C"))
    ch = _Acc(("D", "E", "C", "I", "Dp") + tuple(f"Dmax{g:g}" for g in GAMMAS))
    panel_pick = set(np.linspace(0, P - 1, 6).round().astype(int).tolist())
    for b0 in range(0, P, cli.batch):
        idx = list(range(b0, min(P, b0 + cli.batch)))
        tp = [pairs[j][1] for j in idx]
        tt = [pairs[j][2] for j in idx]
        gt_t = _gray01(cfm, real[tt])
        valid = np.ones(gt_t.shape, bool) if valid_all is None else valid_all[tt]
        meta = [{"exp": str(lab.loc[i, "exp"]), "DI": float(lab.loc[i, "DI"]) if "DI" in lab else 0.0} for i in tt]

        _, x = draw(real[tp], tt, tp, 3)
        tf.add({"D": np.abs(x[0] - x[1]), "E": np.abs(x[0] - gt_t), "C": np.abs(x[0] - x[2])}, valid, meta, rng)
        for k, j in enumerate(idx):
            if j in panel_pick:
                gp = _gray01(cfm, real[[tp[k]]])[0]
                d, e = np.abs(x[0, k] - x[1, k]), np.abs(x[0, k] - gt_t[k])
                sc = lambda a: np.clip(a / 0.25, 0, 1)
                panel.append((np.concatenate([gp, x[0, k], x[1, k], sc(d), sc(e), gt_t[k]], 1) * 255).astype(np.uint8))

        sub = [k for k, j in enumerate(idx) if pairs[j][0] >= 0]
        if not cli.no_chain and sub:
            t2 = [pairs[idx[k]][0] for k in sub]
            t1 = [tp[k] for k in sub]
            t0 = [tt[k] for k in sub]
            ym, y = draw(real[t2], t1, t2, 2)  # two draws of t-1 from the real t-2
            S = ym[: len(t1)]  # draw a, kept in model space, is the next S
            ya, gt_p = y[0], _gray01(cfm, real[t1])
            Dp = np.abs(ya - y[1])  # spread of the previous hop, for the carried U
            _, x2 = draw(S, t0, t1, 3)
            gt0 = gt_t[sub]
            D, E = np.abs(x2[0] - x2[1]), np.abs(x2[0] - gt0)
            maps = {"D": D, "E": E, "C": np.abs(x2[0] - x2[2]), "I": np.abs(ya - gt_p), "Dp": Dp}
            for gm in GAMMAS:
                maps[f"Dmax{gm:g}"] = np.maximum(D, gm * Dp)
            ch.add(maps, valid[sub], [meta[k] for k in sub], rng)
        print(f"[u_calib] {name}: {idx[-1] + 1}/{P}", flush=True)

    rep = {"pairs": P, "tf": tf.report([("D_vs_E", "D", "E"), ("D_vs_C_ceiling", "D", "C")])}
    if not cli.no_chain:
        rep["chain"] = ch.report(
            [("D_vs_E", "D", "E"), ("D_vs_C_ceiling", "D", "C"), ("I_vs_E", "I", "E"), ("Dp_vs_I", "Dp", "I")]
            + [(f"Dmax{g:g}_vs_E", f"Dmax{g:g}", "E") for g in GAMMAS])
    return rep


def _print(split, name, rep):
    for setting in ("tf", "chain"):
        if setting not in rep:
            continue
        print(f"\n[{split}] {name} / {setting} (pairs {rep['pairs']})")
        cols = ["σ", "k=E/D", "mean E", "D~E pearson", "D~E spearman", "AUROC top20", "img r", "ceiling D~C pearson",
                "ceiling spearman", "ceiling AUROC"]
        if setting == "chain":
            cols += ["I~E pearson", "Dmax0.5~E", "Dmax0.8~E", "Dmax1~E"]
        print("| " + " | ".join(cols) + " |")
        print("|" + "---|" * len(cols))
        for key, r in rep[setting].items():
            a, c = r["D_vs_E"], r["D_vs_C_ceiling"]
            row = [key.split("_")[1], f"{r['k']:.2f}", f"{r['mean']['E']:.4f}", f"{a['pearson']:.3f}",
                   f"{a['spearman']:.3f}", f"{a['auroc_top20']:.3f}", f"{a['img_pearson']:.3f}",
                   f"{c['pearson']:.3f}", f"{c['spearman']:.3f}", f"{c['auroc_top20']:.3f}"]
            if setting == "chain":
                row += [f"{r['I_vs_E']['pearson']:.3f}"] + [f"{r[f'Dmax{g:g}_vs_E']['pearson']:.3f}" for g in GAMMAS]
            print("| " + " | ".join(row) + " |")
        r4 = rep[setting]["sigma_4"]
        print(f"σ=4: k by column {json.dumps({k: round(v, 2) for k, v in r4['k_by_exp'].items()})}, "
              f"by DI tercile {[round(v, 2) for v in r4['k_by_DI_tercile']]}, E pct {r4['E_pct']}")


def main():
    cli = _parse_args()
    device = torch.device("cuda")
    os.makedirs(cli.out_dir, exist_ok=True)
    name, cfm, ds, dl, lay, step = _sp._build(cli.model, cli.split, cli.app_code, device)
    panel = []
    rep = {"step": step, "spec": cli.model, "split": cli.split, **_run(name, cfm, ds, dl, lay, cli, device, panel)}
    with open(os.path.join(cli.out_dir, f"report_{name}.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    _print(cli.split, name, rep)
    from PIL import Image

    Image.fromarray(np.concatenate(panel, 0)).save(os.path.join(cli.out_dir, f"panel_{name}.png"))
    with open(os.path.join(cli.out_dir, "panel_columns.txt"), "w") as fh:
        fh.write("GT_{t-1} | x_a | x_b | |x_a-x_b| (x4) | |x_a-GT_t| (x4) | GT_t   (teacher forcing, σ=0)\n")
    print(f"[u_calib] wrote {cli.out_dir}/report_{name}.json and panel_{name}.png")


if __name__ == "__main__":
    main()
