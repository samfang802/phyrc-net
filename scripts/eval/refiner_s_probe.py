#!/usr/bin/env python3
"""What does a refiner take from S_{t-1}? Same conditions and start noise for frame t, only S changes.

For every pair (t-1, t) of the split, frame t is drawn with S_{t-1} set to
  gt        the real previous photo GT_{t-1}
  other     GT_{t-1} of another pair (half the list away: other face or other time)
  swap_lo   GT_{t-1} with its low band (Gaussian σ=4) replaced by the other photo's
  swap_mid  ... its mid band (σ=1 minus σ=4) replaced
  swap_hi   ... its high band (x minus σ=1) replaced
Per band, gain = <band(ΔO), band(ΔS)> / |band(ΔS)|², summed over all samples: the share of a
change in S that reaches the output (1 = copied, 0 = ignored). The swap_* rows change S in one
band only. The real-data reference is the same regression of GT_t − GT'_t on GT_{t-1} − GT'_{t-1}.
Crack keep (damage_texture.crack_map, 2 px) from S into the output is reported as a side number,
against the chance level of the output drawn with the other S.

  python3 scripts/eval/refiner_s_probe.py --out_dir results/refiner_s_probe \
      --model full_6k=<result_dir>@<checkpoint>[@<layouts npz>]
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from scipy import ndimage as ndi
from torch.utils.data import DataLoader

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from argument.argument_mixed import set_dataset_args  # noqa: E402
from src.dataset import PatternLoopDataset  # noqa: E402
from src.metrics import damage_texture as dt  # noqa: E402
from src.metrics import dino_mmd as dm  # noqa: E402
from src.utils import build_damage_image_transform_with_loss_mask  # noqa: E402
from train_mixed import _load_args_from_result_dir, _resolve_required_match  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_eval_r2", os.path.join(_ROOT, "scripts", "eval", "eval_cfm_generated_r2_siamese.py")
)
_eval_r2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_r2)

VARIANTS = ("gt", "other", "swap_lo", "swap_mid", "swap_hi")
BANDS = ("lo", "mid", "hi")
SIG_LO, SIG_HI = 4.0, 1.0


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", action="append", required=True,
                   help="name=result_dir@checkpoint[@layouts npz]; repeatable")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split", default="test", choices=("test", "val", "train"))
    p.add_argument("--n", type=int, default=2, help="Start-noise draws per pair")
    p.add_argument("--batch", type=int, default=32, help="Pairs per batch")
    p.add_argument("--app_code", default="mean", help="Appearance-code models: null | mean")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _bands(x):
    """(N,H,W) float → dict of the three bands; they sum to x."""
    g4 = ndi.gaussian_filter(x, (0, SIG_LO, SIG_LO))
    g1 = ndi.gaussian_filter(x, (0, SIG_HI, SIG_HI))
    return {"lo": g4, "mid": g1 - g4, "hi": x - g1}


def _swap(a, b, band):
    """a with one band taken from b (model space, (N,C,H,W) tensors)."""
    an, bn = a.cpu().numpy(), b.cpu().numpy()
    shp = an.shape
    ba, bb = _bands(an.reshape(-1, *shp[-2:])), _bands(bn.reshape(-1, *shp[-2:]))
    out = an.reshape(-1, *shp[-2:]) - ba[band] + bb[band]
    return torch.from_numpy(out.reshape(shp)).to(a.device, a.dtype)


class _Gain:
    def __init__(self):
        self.num = {b: 0.0 for b in BANDS}
        self.den = {b: 0.0 for b in BANDS}
        self.oo = {b: 0.0 for b in BANDS}

    def add(self, dO, dS):
        bo, bs = _bands(dO), _bands(dS)
        for b in BANDS:
            self.num[b] += float((bo[b] * bs[b]).sum())
            self.den[b] += float((bs[b] ** 2).sum())
            self.oo[b] += float((bo[b] ** 2).sum())

    def out(self):
        r = {}
        for b in BANDS:
            r[f"gain_{b}"] = self.num[b] / max(self.den[b], 1e-9)
            r[f"corr_{b}"] = self.num[b] / max(np.sqrt(self.den[b] * self.oo[b]), 1e-9)
        return r


def _dil(c, r=2):
    return ndi.binary_dilation(c, np.ones((1, 2 * r + 1, 2 * r + 1), bool))


def _build(spec, split, app_code, device):
    name, rest = spec.split("=", 1)
    parts = rest.split("@")
    rdir, ckpt = os.path.abspath(parts[0]), os.path.abspath(parts[1])
    layouts = parts[2] if len(parts) > 2 and parts[2] else ""
    args = _load_args_from_result_dir(rdir)
    args.result_dir = rdir
    args.train_virtual_repeat = 1
    os.makedirs(os.path.join(rdir, "data_check"), exist_ok=True)
    ex_args = _load_args_from_result_dir(args.extractor_path)
    extractor = getattr(importlib.import_module("src.model.HystereticPrediction"), ex_args.architecture)(
        **vars(ex_args)).to(device)
    ex_ckpt = _resolve_required_match(os.path.join(args.extractor_path, "**/best.ckpt"), "extractor")
    extractor.load_state_dict(torch.load(ex_ckpt, map_location=device, weights_only=False)["state_dict"])
    extractor.eval()
    transform = build_damage_image_transform_with_loss_mask(
        image_size=args.image_size, use_augmentation=False, output_scale=True,
        normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5),
        use_grayscale_input=args.use_grayscale_input, use_clahe=args.use_clahe,
        clahe_clip_limit=args.clahe_clip_limit, clahe_tile_grid_size=args.clahe_tile_grid_size,
    )
    kw = dict(vars(args))
    kw["use_loss_mask"] = True
    kw["use_damage_mask_condition"] = args.architecture == "CFMADM" and bool(args.use_damage_mask_condition)
    if kw["use_damage_mask_condition"]:
        kw["use_object_loss_mask"] = False
    if int(getattr(args, "app_code_dim", 0)) > 0:
        kw["app_unseen_code"] = -1 if app_code == "null" else -2
    if split == "train":
        kw.update(condition_perturb=False, g_traj_mix=False, share_photometric_aug=False,
                  opa_lambda_ord=0.0, opa_lambda_mag=0.0)
    ds = PatternLoopDataset(root="./Data", mode=split, transform=transform, extractor=extractor,
                            extractor_args=ex_args, **kw)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=1)
    set_dataset_args(args, dl)
    cfm = getattr(importlib.import_module("src.model.ImageGeneration"), args.architecture)(
        regressor=None, **vars(args)).to(device)
    step, _ = _eval_r2._load_cfm_checkpoint(cfm, ckpt, device)
    cfm.eval()
    for p in cfm.parameters():
        p.requires_grad = False
    lay = None
    if layouts:
        if not getattr(cfm, "layout_cond", False):
            raise SystemExit(f"{name}: layouts given but the model has no layout channel")
        from src.data import layout_io

        d = layout_io.load_layout_npz(layouts)
        lay = {f: d["layout"][i] for i, f in enumerate(d["filename"])}
    return name, cfm, ds, dl, lay, step


@torch.no_grad()
def _probe(name, cfm, ds, dl, lay, cli, device, panel_rows):
    real, cs, gs = [], [], []
    for batch in dl:
        real_data, hs, dr, _exp, _t, di, info, *_ = cfm._unpack_generation_batch(batch)
        c, g = cfm._unpack_conditions(info.to(device), hs.to(device), dr.to(device), di.to(device))
        real.append(cfm._encode_to_target_space(real_data.to(device)))
        cs.append(c)
        gs.append(g)
    real, c_all = torch.cat(real), torch.cat(cs)
    g_all = None if gs[0] is None else torch.cat(gs)
    lab = ds.label
    if len(lab) != len(real):
        raise SystemExit("needs one dataset item per label row")
    pairs = []
    for _, rows in lab.groupby(["exp", "direction"], sort=True):
        order = rows.sort_values("timestep", kind="stable").index.to_numpy()
        pairs += [(int(order[k - 1]), int(order[k])) for k in range(1, len(order))]
    P = len(pairs)
    partner = [(j + P // 2) % P for j in range(P)]
    print(f"[s_probe] {name}: {len(lab)} frames, {P} pairs, n={cli.n}", flush=True)

    size = real.shape[-1]
    from src.data import layout_io

    def _mask(i, m):
        if lay is None:
            return None
        cov = np.asarray(lay[str(lab.loc[i, "filename"])], dtype=np.float32)[:1]
        sdf = torch.from_numpy(layout_io.coverage_to_sdf128(cov, size=size))[:, None]
        return torch.cat([torch.ones_like(sdf), sdf], 1).to(device).repeat(m, 1, 1, 1)

    gains = {v: _Gain() for v in VARIANTS if v != "gt"}
    ref_gain = _Gain()
    keep = {"s_in_o": 0, "s_in_o_other": 0, "n_s": 0, "gt_keep": 0}
    E = {v: 0.0 for v in VARIANTS}
    E_gt, npx = 0.0, 0
    use_dg = bool(getattr(cfm, "use_delta_g", False))
    gen = torch.Generator(device=device).manual_seed(cli.seed)
    panel_pick = set(np.linspace(0, P - 1, 4).round().astype(int).tolist())
    n = int(cli.n)
    for b0 in range(0, P, cli.batch):
        idx = list(range(b0, min(P, b0 + cli.batch)))
        tp = [pairs[j][0] for j in idx]
        tt = [pairs[j][1] for j in idx]
        op = [pairs[partner[j]][0] for j in idx]
        ot = [pairs[partner[j]][1] for j in idx]
        rep = lambda x: x.repeat_interleave(n, dim=0)
        s_gt, s_ot = rep(real[tp]), rep(real[op])
        S = {"gt": s_gt, "other": s_ot, "swap_lo": _swap(s_gt, s_ot, "lo"),
             "swap_mid": _swap(s_gt, s_ot, "mid"), "swap_hi": _swap(s_gt, s_ot, "hi")}
        c, g = rep(c_all[tt]), None if g_all is None else rep(g_all[tt])
        dg = None if (not use_dg or g_all is None) else g - rep(g_all[tp])
        masks = None
        if lay is not None:
            masks = torch.cat([_mask(i, n) for i in tt])
        z0 = torch.randn(s_gt.shape, generator=gen, device=device)
        O = {}
        for v in VARIANTS:
            x = cfm.sample_with_ode(c, z0=z0, gru_feat=g, damage_mask=masks, lr_cond=S[v], gru_delta=dg)
            O[v] = dm.to_uint8_gray(cfm._decode_from_target_space(x)).astype(np.float32)
        Su = {v: dm.to_uint8_gray(cfm._decode_from_target_space(S[v])).astype(np.float32) for v in VARIANTS}
        for v in gains:
            gains[v].add(O["gt"] - O[v], Su["gt"] - Su[v])
        gtt = dm.to_uint8_gray(cfm._decode_from_target_space(real[tt])).astype(np.float32)
        gtt_o = dm.to_uint8_gray(cfm._decode_from_target_space(real[ot])).astype(np.float32)
        gtp = Su["gt"][::n]
        ref_gain.add(gtt - gtt_o, gtp - Su["other"][::n])

        cs_ = dt.crack_map(Su["gt"].astype(np.uint8))
        keep["n_s"] += int(cs_.sum())
        keep["s_in_o"] += int((cs_ & _dil(dt.crack_map(O["gt"].astype(np.uint8)))).sum())
        keep["s_in_o_other"] += int((cs_ & _dil(dt.crack_map(O["other"].astype(np.uint8)))).sum())
        keep["gt_keep"] += int((cs_[::n] & _dil(dt.crack_map(gtt.astype(np.uint8)))).sum())
        for v in VARIANTS:
            E[v] += float(dt.local_maps(O[v].astype(np.uint8))[1].sum())
        E_gt += float(dt.local_maps(gtt.astype(np.uint8))[1].sum()) * n
        npx += O["gt"].size
        for k, j in enumerate(idx):
            if j in panel_pick:
                r = k * n
                panel_rows.append(np.concatenate(
                    [Su["gt"][r], O["gt"][r], Su["other"][r], O["other"][r], Su["swap_hi"][r], O["swap_hi"][r],
                     gtt[k]], 1).astype(np.uint8))
        print(f"[s_probe] {name}: {idx[-1] + 1}/{P}", flush=True)

    rep_ = {"pairs": P, "n": n, "gain": {v: gains[v].out() for v in gains}, "real_gain": ref_gain.out(),
            "crack_keep_from_S": keep["s_in_o"] / max(keep["n_s"], 1),
            "crack_keep_chance": keep["s_in_o_other"] / max(keep["n_s"], 1),
            "gt_crack_keep": keep["gt_keep"] / max(keep["n_s"] // n, 1),
            "texture_E": {v: E[v] / npx for v in VARIANTS}, "texture_E_gt": E_gt / npx}
    return rep_


def main():
    cli = _parse_args()
    device = torch.device("cuda")
    os.makedirs(cli.out_dir, exist_ok=True)
    report, panels = {}, {}
    for spec in cli.model:
        name, cfm, ds, dl, lay, step = _build(spec, cli.split, cli.app_code, device)
        rows = []
        report[name] = {"step": step, "spec": spec, **_probe(name, cfm, ds, dl, lay, cli, device, rows)}
        panels[name] = np.concatenate(rows, 0)
        del cfm
        torch.cuda.empty_cache()
        with open(os.path.join(cli.out_dir, "report.json"), "w") as fh:
            json.dump(report, fh, indent=2)

    first = next(iter(report.values()))
    print(f"\n[{cli.split}] gain = share of a change in S_(t-1) that reaches frame t, per band "
          f"(lo σ>{SIG_LO:g}, mid, hi σ<{SIG_HI:g}); real data reference = GT_t vs GT_(t-1)")
    print("| model | other lo | other mid | other hi | swap_lo lo | swap_mid mid | swap_hi hi | "
          "crack keep from S | chance | E(gt S) | E(other S) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    rg = first["real_gain"]
    print(f"| *real* | {rg['gain_lo']:.2f} | {rg['gain_mid']:.2f} | {rg['gain_hi']:.2f} | | | | "
          f"{first['gt_crack_keep']:.2f} | | {first['texture_E_gt']:.2f} | |")
    for name, r in report.items():
        g = r["gain"]
        print(f"| {name} | {g['other']['gain_lo']:.2f} | {g['other']['gain_mid']:.2f} | {g['other']['gain_hi']:.2f} | "
              f"{g['swap_lo']['gain_lo']:.2f} | {g['swap_mid']['gain_mid']:.2f} | {g['swap_hi']['gain_hi']:.2f} | "
              f"{r['crack_keep_from_S']:.2f} | {r['crack_keep_chance']:.2f} | "
              f"{r['texture_E']['gt']:.2f} | {r['texture_E']['other']:.2f} |")

    from PIL import Image

    for name, im in panels.items():
        Image.fromarray(im).save(os.path.join(cli.out_dir, f"panel_{name}.png"))
    with open(os.path.join(cli.out_dir, "panel_columns.txt"), "w") as fh:
        fh.write("S=GT_{t-1} | out | S=other | out | S=swap_hi | out | GT_t\n")
    print(f"[s_probe] wrote {cli.out_dir}/report.json and panel_*.png")


if __name__ == "__main__":
    main()
