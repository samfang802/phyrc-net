#!/usr/bin/env python3
"""Predicted (hspP) spall layouts for the TRAINING frames, cross-fitted, as a layout_cond SDF cache.

Train/test consistency for v12 (docs/redesign/phase1_fiber_clock.md §6): at test the layout is
OFP c_iou shape + hsp_P area (build_clock_layouts.py), both fitted on the 4 training columns, so CTR1 / R307 are
out of sample. Training on the ground-truth AnyLabel layouts instead shows the model only smooth exact
polygons. Here every training column h gets the same kind of layout, fitted without h:

  shape  OFP variant c (ks 3, hidden 8, wd 1e-4 = the c_iou setting, 400 steps) on the 3 other columns,
         K rollouts of h's sequences (same seeding as scripts/train_ofp.py)
  area   R_spall = f(hsp_P) with f the model class phase0_clock_check picks for hsp_P (LOCO-best on all 4),
         fitted on the 3 other columns' frames
  layout build_clock_layouts.apply_area (same priority order and tie-breaking as at test)

--shape real_iso32 (v12gt): the training frames' REAL monotone 32x32 coverage (onset_field.npz iso32, the
eval's iso32 ceiling layout), so the layout is the most reliable spall signal in training and the renderer
learns to follow it (docs/redesign/phase1_fiber_clock.md §8: with predicted training layouts v12 / v12basis
ignored the layout). Same 32x32 coverage -> coverage_to_sdf128 path as the test-time layouts. Val frames
still get --val_layouts (a prediction); test frames never enter.

--shape template: the gravity template (build_clock_layouts.gravity_template, from the 4 training columns)
replaces the OFP rollout as the shape, K = 1; with --fit all the area is the same f(hsp_P) as at test
(scripts/eval/build_clock_layouts.py --shape template), so train and test layouts are made identically.

Outputs (--out_dir):
  hspP_loco_train.npz       layout npz (src/data/layout_io.py), split "train", (N, K, 32, 32)
  hspP_loco_sdf128.npz      cache for --layout_cache: filename, sdf float16 (N, K, 128, 128); the dataset
                            draws one of the K per item. Only training frames: val / test frames are absent,
                            so they read layout 0 (no ground truth anywhere).
  report.json               per held-out column: area bias, IoU with the real iso32 layout

  OMP_NUM_THREADS=2 python scripts/eval/build_train_layout_cache.py
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.data import layout_io  # noqa: E402
from src.model import ofp as O  # noqa: E402

_spec = importlib.util.spec_from_file_location("bcl", os.path.join(ROOT, "scripts", "eval", "build_clock_layouts.py"))
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
P0 = B.P0
TRAIN = list(P0.TRAIN)


def rollouts(model, seqs, K, seed):
    """As scripts/train_ofp.py: one generator per sequence, seed + sum of the name's code points."""
    return {n: model.rollout(torch.from_numpy(s["tau"]), K, torch.Generator().manual_seed(seed + sum(map(ord, n))))
            for n, s in seqs.items()}


def real_iso32_cache(a):
    split = json.load(open("Data/DamagePattern/id_list_val_R307.json"))
    fn = list(split["train"])
    lay = layout_io.onset_layout(fn, "iso32")[:, None]                            # (N, 1, 32, 32)
    stem = os.path.splitext(a.cache_name)[0]
    layout_io.save_layout_npz(os.path.join(a.out_dir, f"{stem}_train_layouts.npz"), fn, lay, "real_iso32", "train")
    cache_fn, cache_lay = list(fn), lay
    report = {"shape": "real_iso32", "n_train": len(fn), "train_area_mean": float(lay.mean())}
    if a.val_layouts:
        v = layout_io.load_layout_npz(a.val_layouts)
        assert v["split"] == "val", f"--val_layouts must be a val layout npz, got split {v['split']!r}"
        assert set(v["filename"]) == set(split["val"]), "val layout npz does not cover exactly the val frames"
        cache_fn += list(v["filename"])
        cache_lay = np.concatenate([lay, v["layout"][:, :1]])
        report["val_layouts"] = os.path.abspath(a.val_layouts)
    assert not set(cache_fn) & set(split["test"]), "test frame in the cache"
    sdf = layout_io.coverage_to_sdf128(cache_lay).astype(np.float16)
    np.savez_compressed(os.path.join(a.out_dir, a.cache_name), filename=np.asarray(cache_fn), sdf=sdf,
                        meta=np.asarray(json.dumps({"source": "build_train_layout_cache.py", **vars(a)})))
    json.dump(report, open(os.path.join(a.out_dir, f"{stem}_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="results/phase1_fiber_clock/frames_with_clocks.csv")
    ap.add_argument("--clock", default="hsp_P")
    ap.add_argument("--out_dir", default="results/phase1_fiber_layout/train_cache")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ofp_steps", type=int, default=400)
    ap.add_argument("--val_layouts", default="",
                    help="also put these val (R307) layouts in the cache, e.g. results/phase1_fiber_layout/layouts/"
                         "ofp_hspP_val.npz (OFP + hsp_P fitted on the 4 training columns: a prediction, not ground "
                         "truth). The periodic val KID and the val panel then read the test-time kind of layout. "
                         "Test (CTR1) frames are never added.")
    ap.add_argument("--cache_name", default="hspP_loco_sdf128.npz")
    ap.add_argument("--shape", choices=("ofp", "template", "real_iso32"), default="ofp",
                    help="ofp = OFP c rollouts (K of them); template = the gravity template, K = 1; "
                         "real_iso32 = the training frames' real iso32 coverage, K = 1")
    ap.add_argument("--fit", choices=("loco", "all"), default="loco",
                    help="loco = OFP and area fitted without the frame's column (test-like error); all = the very "
                         "OFP c_iou model (results/phase1_ofp/models/ofp_c_iou.pt) and area function used at test, "
                         "fitted on all 4 columns (in-sample on training frames)")
    ap.add_argument("--area_source", choices=("clock", "true"), default="clock",
                    help="training-frame area: clock = cross-fitted f(clock) (test-like error); true = the frame's own "
                         "R_spall label (training data), so layout area always matches the photo and the model learns "
                         "to follow the layout's area; the shape stays the cross-fitted OFP rollout either way")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    if a.shape == "real_iso32":
        return real_iso32_cache(a)
    frames = pd.read_csv(a.frames)
    best = P0.run_clock(frames, [a.clock])["best_model"]
    Model = next(m for m in P0.MODELS_1D if m.name == best)
    allseq = O.load_sequences()
    train_seq = O.sequences_of(allseq, TRAIN)

    rows = []  # (filename, layout (K,S,S), area target, seq, timestep)
    report = {"clock": a.clock, "area_model": best, "area_source": a.area_source, "fit": a.fit, "shape": a.shape,
              "ofp": "c ks3 hidden8 wd1e-4" if a.shape == "ofp" else None, "per_column": {}}
    prio = None
    if a.shape == "template":
        assert a.fit == "all", "the template is fitted on all 4 training columns: use --fit all"
        T = B.gravity_template()
        prio = B.template_priority(T, 1, a.seed)
        np.save(os.path.join(a.out_dir, f"{os.path.splitext(a.cache_name)[0]}_template.npy"), T)
    for h in TRAIN:
        others = {n: s for n, s in train_seq.items() if n.split("_")[0] != h}
        mine = {n: s for n, s in train_seq.items() if n.split("_")[0] == h}
        if a.shape == "template":
            fit_cols = TRAIN
        elif a.fit == "all":
            ck = torch.load("results/phase1_ofp/models/ofp_c_iou.pt", map_location="cpu", weights_only=False)
            ofp = O.OFP(ck["cfg"]["variant"], ck["cfg"]["ks"], ck["cfg"]["hidden"])
            ofp.load_state_dict(ck["state"])
            ofp.eval()
            fit_cols = TRAIN
        else:
            ofp = O.fit("c", others, ks=3, hidden=8, wd=1e-4, steps=a.ofp_steps)
            fit_cols = [c for c in TRAIN if c != h]
        rolls = rollouts(ofp, mine, a.K, a.seed) if a.shape == "ofp" else None
        fit_rows = frames[frames["exp"].isin(fit_cols)]
        area_model = Model().fit(fit_rows[[a.clock]].to_numpy(float), fit_rows["R_spall"].to_numpy(float))
        fh = frames[frames["exp"] == h].set_index("filename")
        ious, n_h = [], 0
        for n, s in mine.items():
            if rolls is not None:
                lay = O.layout_for_frames(rolls[n], s["frames"])                   # (n_frames, K, 32, 32)
            else:                                                                  # shape from prio, not lay
                lay = np.zeros((len(s["frames"]), 1) + prio.shape[1:], np.float32)
            for j, f in enumerate(s["frames"]["filename"]):
                if f not in fh.index:
                    continue
                r = fh.loc[f]
                A = float(np.clip(area_model.predict(np.array([[r[a.clock]]], float))[0], 0, 1))
                rows.append((f, lay[j], A, n, int(r["timestep"]), float(r["R_spall"]), s["frames"]["gt"].iloc[j]))
                n_h += 1
        report["per_column"][h] = {"n_frames": n_h}
        print(f"[train_cache] {h}: OFP / area fitted on {sorted(fit_cols)}, {n_h} frames", flush=True)

    fn = [r[0] for r in rows]
    label = pd.read_csv("Data/DamagePattern/label.csv")
    split = json.load(open("Data/DamagePattern/id_list_val_R307.json"))
    want = set(split["train"])
    missing = sorted(want - set(fn))
    assert not missing, f"{len(missing)} training frames without a layout, e.g. {missing[:3]}"
    assert set(fn) <= want, "non-training frame in the cache"
    lay = np.stack([r[1] for r in rows]).astype(np.float32)
    A = np.array([r[2] for r in rows]) if a.area_source == "clock" else np.array([r[5] for r in rows])
    seq = np.array([r[3] for r in rows])
    ts = np.array([r[4] for r in rows])
    out = B.apply_area(lay, fn, A, seq, ts, a.seed, prio)

    true = np.array([r[5] for r in rows])
    gt = np.stack([r[6] for r in rows]).astype(bool)                               # iso32 > 0.5
    area = out.mean(axis=(1, 2, 3))
    exp = np.array([f.split("_")[0] for f in fn])
    for h in TRAIN:
        m = exp == h
        big = m & (gt.reshape(len(gt), -1).mean(1) > 0.10)
        inter = (out[big] > 0.5) & gt[big][:, None]
        union = (out[big] > 0.5) | gt[big][:, None]
        iou = float((inter.sum((2, 3)) / np.maximum(union.sum((2, 3)), 1)).mean()) if big.any() else float("nan")
        report["per_column"][h].update(area_bias=float((area[m] - true[m]).mean()),
                                       area_rmse=float(np.sqrt(((area[m] - true[m]) ** 2).mean())),
                                       iou_vs_iso32_big=iou)
    report["all"] = {"area_bias": float((area - true).mean()), "area_rmse": float(np.sqrt(((area - true) ** 2).mean()))}

    stem = os.path.splitext(a.cache_name)[0]
    layout_io.save_layout_npz(os.path.join(a.out_dir, f"{stem}_train_layouts.npz"), fn, out, f"{a.shape}_{a.fit}+{a.area_source}_area", "train")
    cache_fn, cache_lay = list(fn), out
    if a.val_layouts:
        v = layout_io.load_layout_npz(a.val_layouts)
        assert v["split"] == "val", f"--val_layouts must be a val layout npz, got split {v['split']!r}"
        assert set(v["filename"]) == set(split["val"]), "val layout npz does not cover exactly the val frames"
        assert v["layout"].shape[1:] == out.shape[1:], (v["layout"].shape, out.shape)
        cache_fn += list(v["filename"])
        cache_lay = np.concatenate([out, v["layout"]])
        report["val_layouts"] = os.path.abspath(a.val_layouts)
    assert not set(cache_fn) & set(split["test"]), "test frame in the cache"
    sdf = layout_io.coverage_to_sdf128(cache_lay).astype(np.float16)                # (N, K, 128, 128)
    np.savez_compressed(os.path.join(a.out_dir, a.cache_name), filename=np.asarray(cache_fn), sdf=sdf,
                        meta=np.asarray(json.dumps({"source": "build_train_layout_cache.py", **vars(a)})))
    json.dump(report, open(os.path.join(a.out_dir, f"{stem}_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
