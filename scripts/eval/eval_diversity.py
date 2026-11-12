#!/usr/bin/env python3
"""Diversity of generated damage images: copying, coverage, and within-condition spread.

Every image set is first brought into the reference domain, and that one copy feeds every metric (DINOv3,
spall segmenter, crack maps). The reals are photo → inpaint → CLAHE → JPEG q75 files, so: our models draw in
the CLAHE domain → `--jpeg NAME=75`; a model trained on the raw photos (cGAN) → `--clahe NAME` (the
generator's own resize/gray/CLAHE transform) then `--jpeg NAME=75`. The train reference is the train list
of --data_path for every model. SE = block jackknife over the test conditions (block = face x drift level).

Four groups, all on the eval-suite sample files (fake (G,K,H,W), one row per test condition):

  copy     crack copy rate. On the conditions whose real frame has no spall (segmenter area <= --spall_min),
           crack maps (damage_texture.crack_map) of every sample are compared with every train real by
           tolerance F1 and the best F1 is kept. The threshold is the --copy_q quantile of the same best F1
           of the test reals: an unseen column overlaps a training one by chance up to there. Copy rate =
           share of samples above it; a model that does not copy sits at 1 − copy_q (the other real split
           is reported as a check of that). The train column of the best match is listed.
           Spall content copy rate. Spall regions (segmenter) of every image with spall area > --spall_min
           are cut into 16 x 16 patches (stride 8, >= 60 % spall, not flat); each patch's best normalized
           cross-correlation with any train spall patch (any position) is kept. Threshold = the
           --spall_copy_q quantile of the test reals' patches; rate = share of a model's spall patches above.
           Spall shape is not scored: the gravity layout itself overlaps the train masks more than any sample.
  prdc     density / coverage (Naeem et al., ICML 2020) against the test reals and against the train reals.
           One column of near-duplicate frames makes plain k-NN radii degenerate (eval_dinov3_kid.py), so the
           radius of a real is its k-th neighbour outside its own block. `outside` = share of fakes in no ball
           of train or test reals (drawn something neither set looks like). One fake per condition, mean over K.
  spread   is the amount of variation right? Spall area of every sample by the spall segmenter
           (phase0_layout_oracle_spallseg.py), truth = the same segmenter on the real frame. Spread/skill
           ratio sqrt((K+1)/K) * spread / RMSE of the ensemble mean (GenCast, Price et al., Nature 2024):
           1 = calibrated, < 1 = too little variation for its error. Rank histogram of the truth among the K
           samples: flat = calibrated, mass in the two end bins = under-dispersed (expected 2 / (K+1)).
           Only conditions with real spall area > --spall_min.
  within   variation inside one condition: mean pairwise cosine distance of the K samples (DINOv3) and the
           Vendi score (Friedman & Dieng 2023; per condition, as Stein et al. NeurIPS 2023 recommend for
           conditional models), and 1 − pairwise IoU of the K spall masks. Real references at the same
           drift level and cycle: the other face of the column (an upper reference: also differs in face
           appearance) and the same face one cycle away (a near-duplicate lower reference).

  python scripts/eval/eval_diversity.py --split test --no_clahe \
      --image_dir Data/DamagePattern_inpaint_selfcondnoDI/images \
      --model cGAN=results/eval_suite/v106/test/progression_sel/cGAN_eq.npz \
      --model grav=results/eval_suite/v12gtrefiner/test/samples/v12gtrefiner_006000_grav.npz \
      --clahe cGAN --jpeg cGAN=75 grav=75 --out_dir results/eval_suite/diversity
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.metrics import damage_texture as dtx  # noqa: E402
from src.metrics import dino_mmd as dm  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"_{name}", os.path.join(_ROOT, "scripts", "eval", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_kid = _load_script("eval_dinov3_kid")
_seg = _load_script("phase0_layout_oracle_spallseg")


def _parts(f):
    p = os.path.basename(str(f)).replace(".jpg", "").split("_")
    return p[0], p[1], p[2], int(p[3])


def _jk(stat, blocks):
    """stat(mask) on all conditions, SE by leave-one-block-out."""
    labels = np.unique(blocks)
    full = stat(np.ones(len(blocks), bool))
    loo = np.array([stat(blocks != b) for b in labels])
    loo = loo[np.isfinite(loo)]
    n = len(loo)
    se = float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2))) if n > 1 else float("nan")
    return {"value": float(full), "se": se}


def _best_crack_f1(maps, train_c, tol, chunk=256):
    """Best tolerance F1 of every map against the train maps, and the index of that train map."""
    best, arg = [], []
    for s in range(0, len(maps), chunk):
        F = dtx.crack_f1(maps[s:s + chunk], train_c, tol)
        best.append(F.max(1))
        arg.append(F.argmax(1))
    return np.concatenate(best), np.concatenate(arg)


def _spall_patches(u8, masks, size=16, stride=8, min_cover=0.6, min_std=6.0):
    """Unit-norm z-scored patches inside the spall masks, and the image index of each."""
    import torch.nn.functional as F

    x = torch.from_numpy(np.asarray(u8)).float()[:, None]
    m = torch.from_numpy(np.asarray(masks)).float()[:, None]
    P = F.unfold(x, size, stride=stride).transpose(1, 2)
    cover = F.unfold(m, size, stride=stride).transpose(1, 2).mean(-1)
    sd = P.std(-1)
    ok = (cover >= min_cover) & (sd > min_std)
    P = (P - P.mean(-1, keepdim=True)) / (sd[..., None] + 1e-6) / size
    img = torch.arange(len(x))[:, None].expand_as(ok)
    return P[ok], img[ok].numpy()


def _best_ncc(P, train_P, device, chunk=2048):
    if len(P) == 0:
        return np.zeros(0)
    T = train_P.to(device)
    return torch.cat([(P[s:s + chunk].to(device) @ T.T).max(1).values.cpu() for s in range(0, len(P), chunk)]).numpy()


def _blocked_radii(x, blocks, k):
    """k-th nearest-neighbour distance of every row of x, neighbours from other blocks only."""
    d = torch.cdist(torch.from_numpy(x).double(), torch.from_numpy(x).double())
    b = np.asarray(blocks)
    same = torch.from_numpy(b[:, None] == b[None, :])
    d[same] = float("inf")
    return d.kthvalue(k, dim=1).values.numpy()


def _dc(real, rad, fake, k):
    d = torch.cdist(torch.from_numpy(real).double(), torch.from_numpy(fake).double()).numpy()
    inside = d < rad[:, None]
    return {
        "precision": float(inside.any(0).mean()),
        "density": float(inside.sum(0).mean() / k),
        "coverage": float((d.min(1) < rad).mean()),
    }, inside.any(0)


def _vendi(x):
    """Vendi score of rows of x with the cosine kernel."""
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    w = np.clip(np.linalg.eigvalsh(xn @ xn.T / len(x)), 0, None)
    w = w[w > 0]
    return float(np.exp(-np.sum(w * np.log(w))))


def _cos_pairs(x):
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    s = xn @ xn.T
    iu = np.triu_indices(len(x), 1)
    return float(np.mean(1 - s[iu]))


def _iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", action="append", required=True, metavar="NAME=NPZ")
    p.add_argument("--jpeg", nargs="*", default=[], metavar="NAME=QUALITY")
    p.add_argument("--clahe", nargs="*", default=[], metavar="NAME", help="models drawn in the raw-photo domain (applied before --jpeg)")
    p.add_argument("--split", default="test")
    p.add_argument("--other_split", default="val")
    p.add_argument("--data_path", default="Data/DamagePattern/id_list_val_R307.json")
    p.add_argument("--image_dir", default="Data/DamagePattern_inpaint_selfcondnoDI/images")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--no_clahe", action="store_true")
    p.add_argument("--dino_model_id", default=dm.DEFAULT_DINO)
    p.add_argument("--segmenter", default="results/phase0_layout_oracle/spallseg/best.pt")
    p.add_argument("--knn", type=int, default=5)
    p.add_argument("--spall_min", type=float, default=0.02,
                   help="spread / rank where the real spall area exceeds this, crack copy where it does not")
    p.add_argument("--crack_tol", type=int, default=3)
    p.add_argument("--copy_q", type=float, default=0.95, help="copy threshold = this quantile of the test reals' best F1")
    p.add_argument("--spall_copy_q", type=float, default=0.99, help="spall copy threshold = this quantile of the test reals' patch NCC")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", required=True)
    cli = p.parse_args()
    os.makedirs(cli.out_dir, exist_ok=True)
    rng = np.random.default_rng(cli.seed)
    clahe = not cli.no_clahe
    split_ids = json.load(open(cli.data_path))

    runs = {}
    for item in cli.model:
        name, path = item.split("=", 1)
        runs[name] = _kid._load_npz(path)
    jpeg_q = {n: int(q) for n, q in (j.split("=", 1) for j in cli.jpeg)}
    for n in list(jpeg_q) + list(cli.clahe):
        if n not in runs:
            raise ValueError(f"--jpeg / --clahe name {n} is not a --model")
    names = list(next(iter(runs.values()))["filename"])
    for name, r in runs.items():
        if list(r["filename"]) != names:
            raise ValueError(f"{name}: rows differ from the first npz")
    G = len(names)
    blocks = np.array([dm.frame_block(f) for f in names])

    feats = _kid._Cache(dm.DinoFeatures(cli.dino_model_id, device=cli.device), cli.out_dir, 0, cli.seed,
                        cache_dir=_kid.SHARED_CACHE)
    load = lambda fns: dm.load_real_uint8(list(fns), cli.image_dir, cli.image_size, clahe)
    test_u8 = load(names)
    test_g = feats(test_u8)[0]
    train_names = split_ids["train"]
    train_u8 = load(train_names)
    train_g = feats(train_u8)[0]
    train_blocks = np.array([dm.frame_block(f) for f in train_names])
    train_col = np.array([_parts(f)[0] for f in train_names])
    other_u8 = load(split_ids[cli.other_split]) if cli.other_split else None
    other_g = feats(other_u8)[0] if other_u8 is not None else None

    # ---- references shared by every model
    rad_test = _blocked_radii(test_g, blocks, cli.knn)
    rad_train = _blocked_radii(train_g, train_blocks, cli.knn)

    net, thr = _seg.load_segmenter(cli.segmenter, cli.device)
    seg_real = _seg.predict_masks(net, test_u8, cli.device, thr)
    truth = seg_real.reshape(G, -1).mean(1)

    no_spall = truth <= cli.spall_min
    train_c = dtx.crack_map(train_u8)
    real_f1, _ = _best_crack_f1(dtx.crack_map(test_u8[no_spall]), train_c, cli.crack_tol)
    copy_thr = float(np.quantile(real_f1, cli.copy_q))

    train_seg = _seg.predict_masks(net, train_u8, cli.device, thr)
    has = train_seg.reshape(len(train_u8), -1).mean(1) > cli.spall_min
    train_P, _ = _spall_patches(train_u8[has], train_seg[has])
    real_sp = truth > cli.spall_min
    real_P, _ = _spall_patches(test_u8[real_sp], seg_real[real_sp])
    real_ncc = _best_ncc(real_P, train_P, cli.device)
    spall_thr = float(np.quantile(real_ncc, cli.spall_copy_q))

    # within-condition real references: other face, same drift and cycle; same face, adjacent cycle
    key = {(_parts(f)[1], _parts(f)[2], _parts(f)[3]): i for i, f in enumerate(names)}
    faces = sorted({_parts(f)[1] for f in names})
    pair_face, pair_cycle = [], []
    for i, f in enumerate(names):
        _, fa, dr, cy = _parts(f)
        other_face = [fb for fb in faces if fb != fa]
        if other_face and (other_face[0], dr, cy) in key:
            pair_face.append((i, key[(other_face[0], dr, cy)]))
        if (fa, dr, cy + 1) in key:
            pair_cycle.append((i, key[(fa, dr, cy + 1)]))
    pair_face = [(a, b) for a, b in pair_face if a < b]

    def real_pair_stats(pairs):
        if not pairs:
            return None
        cos = [_cos_pairs(test_g[[a, b]]) for a, b in pairs]
        ious = [_iou(seg_real[a], seg_real[b]) for a, b in pairs if (seg_real[a] | seg_real[b]).sum() >= 50]
        ious_flip = [max(_iou(seg_real[a], seg_real[b]), _iou(seg_real[a], seg_real[b][:, ::-1]))
                     for a, b in pairs if (seg_real[a] | seg_real[b]).sum() >= 50]
        return {"n": len(pairs), "cos_dist": float(np.mean(cos)), "vendi_2": float(np.mean([_vendi(test_g[[a, b]]) for a, b in pairs])),
                "mask_1_minus_iou": float(1 - np.nanmean(ious)) if ious else None,
                "mask_1_minus_iou_hflip_best": float(1 - np.nanmean(ious_flip)) if ious_flip else None}

    report = {
        "split": cli.split, "n_conditions": G, "dino": cli.dino_model_id, "knn": cli.knn, "segmenter": cli.segmenter,
        "image_dir": cli.image_dir, "train_list": cli.data_path, "jpeg": jpeg_q, "clahe": list(cli.clahe),
        "reference": {
            "copy": {"n_no_spall_conditions": int(no_spall.sum()), "q": cli.copy_q, "threshold_f1": copy_thr,
                     "test_real_best_f1_mean": float(real_f1.mean())},
            "spall_copy": {"n_train_patches": int(len(train_P)), "n_test_real_patches": int(len(real_P)),
                           "q": cli.spall_copy_q, "threshold_ncc": spall_thr},
            "real_pairs_other_face": real_pair_stats(pair_face),
            "real_pairs_next_cycle": real_pair_stats(pair_cycle),
        },
        "models": {},
    }
    if other_g is not None:
        dc_t, in_t = _dc(test_g, rad_test, other_g, cli.knn)
        dc_tr, in_tr = _dc(train_g, rad_train, other_g, cli.knn)
        omask = _seg.predict_masks(net, other_u8, cli.device, thr)
        oseg = omask.reshape(len(other_u8), -1).mean(1)
        of1, _ = _best_crack_f1(dtx.crack_map(other_u8[oseg <= cli.spall_min]), train_c, cli.crack_tol)
        oP, _ = _spall_patches(other_u8[oseg > cli.spall_min], omask[oseg > cli.spall_min])
        report["reference"][f"other_real_{cli.other_split}"] = {
            "copy_rate": float((of1 > copy_thr).mean()), "n_no_spall": int(len(of1)),
            "spall_copy_rate": float((_best_ncc(oP, train_P, cli.device) > spall_thr).mean()), "n_spall_patches": int(len(oP)),
            "prdc_vs_test": dc_t, "prdc_vs_train": dc_tr,
            "outside_train_and_test": float((~in_t & ~in_tr).mean()), "n": int(len(other_u8))}

    for name, r in runs.items():
        fake = r["fake"]
        K = fake.shape[1]
        u8 = fake.reshape(-1, *fake.shape[2:])
        if name in cli.clahe:
            u8 = dm.clahe_uint8(u8, cli.image_size)
        if name in jpeg_q:
            u8 = _kid._jpeg_u8(u8, jpeg_q[name])
        fg = feats(u8)[0].reshape(G, K, -1)
        u8 = u8.reshape(G, K, *fake.shape[2:])
        out = {"k": int(K), "npz_meta": r["meta"]}

        # crack copy rate
        f1, arg = _best_crack_f1(dtx.crack_map(u8[no_spall].reshape(-1, *fake.shape[2:])), train_c, cli.crack_tol)
        f1 = f1.reshape(-1, K)
        above = f1 > copy_thr
        idx = np.flatnonzero(no_spall)
        sub = lambda m: np.isin(idx, np.flatnonzero(m))
        cols, cnt = np.unique(train_col[arg], return_counts=True)
        out["copy"] = {
            "copy_rate": _jk(lambda m: above[sub(m)].mean(), blocks),
            "best_f1_mean": float(f1.mean()),
            "best_match_column": {str(c): round(float(n) / len(arg), 3) for c, n in sorted(zip(cols, cnt), key=lambda t: -t[1])},
        }

        # density / coverage
        rows_t, rows_tr, outside = [], [], []
        for k in range(K):
            dt, it = _dc(test_g, rad_test, fg[:, k], cli.knn)
            dtr, itr = _dc(train_g, rad_train, fg[:, k], cli.knn)
            rows_t.append(dt)
            rows_tr.append(dtr)
            outside.append(~it & ~itr)
        outside = np.stack(outside, 1)
        hi_sp = truth > 0.3
        out["prdc"] = {
            "vs_test": {m: float(np.mean([x[m] for x in rows_t])) for m in rows_t[0]},
            "vs_train": {m: float(np.mean([x[m] for x in rows_tr])) for m in rows_tr[0]},
            "outside_train_and_test": _jk(lambda m: outside[m].mean(), blocks),
            "outside_train_and_test_real_spall_gt0.3": float(outside[hi_sp].mean()) if hi_sp.any() else None,
        }

        # spread / skill on spall area
        seg = _seg.predict_masks(net, u8.reshape(-1, *fake.shape[2:]), cli.device, thr).reshape(G, K, *fake.shape[2:])
        area = seg.reshape(G, K, -1).mean(-1)

        # spall content copy rate
        flat_area, flat_cond = area.ravel(), np.repeat(np.arange(G), K)
        hs = flat_area > cli.spall_min
        P, img = _spall_patches(u8.reshape(-1, *fake.shape[2:])[hs], seg.reshape(-1, *fake.shape[2:])[hs])
        p_above = _best_ncc(P, train_P, cli.device) > spall_thr
        p_cond = flat_cond[hs][img]
        out["copy"]["spall_copy_rate"] = _jk(lambda m: p_above[m[p_cond]].mean() if m[p_cond].any() else float("nan"), blocks)
        out["copy"]["n_spall_patches"] = int(len(P))

        sp = truth > cli.spall_min
        if K > 1:
            def ssr(mask):
                mm = mask & sp
                if mm.sum() < 2:
                    return float("nan")
                spread = np.sqrt(np.mean(area[mm].var(1, ddof=1)))
                skill = np.sqrt(np.mean((area[mm].mean(1) - truth[mm]) ** 2))
                return float(np.sqrt((K + 1) / K) * spread / skill)

            jitter = rng.uniform(-1e-9, 1e-9, size=area.shape)
            rank = ((area + jitter) < truth[:, None]).sum(1)[sp]
            hist = np.bincount(rank, minlength=K + 1)
            out["spread"] = {
                "n": int(sp.sum()),
                "spread_skill_ratio": _jk(ssr, blocks),
                "spread": float(np.sqrt(np.mean(area[sp].var(1, ddof=1)))),
                "rmse_ensemble_mean": float(np.sqrt(np.mean((area[sp].mean(1) - truth[sp]) ** 2))),
                "rank_hist": hist.tolist(),
                "rank_end_bins_frac": float((hist[0] + hist[-1]) / hist.sum()),
                "rank_end_bins_expected": 2 / (K + 1),
            }
            # within-condition variation
            cos = np.array([_cos_pairs(fg[i]) for i in range(G)])
            ven = np.array([_vendi(fg[i]) for i in range(G)])
            miou = np.full(G, np.nan)
            for i in range(G):
                v = [_iou(seg[i, a], seg[i, b]) for a in range(K) for b in range(a + 1, K) if (seg[i, a] | seg[i, b]).sum() >= 50]
                if v:
                    miou[i] = np.nanmean(v)
            sel = [a for a, _ in pair_face]
            out["within"] = {
                "cos_dist": _jk(lambda m: cos[m].mean(), blocks),
                "vendi_k": float(ven.mean()),
                "vendi_2_first_pair": float(np.mean([_vendi(fg[i, :2]) for i in range(G)])),
                "mask_1_minus_iou": _jk(lambda m: 1 - np.nanmean(miou[m]), blocks),
                "cos_dist_on_face_pair_conditions": float(cos[sel].mean()) if sel else None,
            }
        else:
            out["spread"] = {"n": int(sp.sum()), "note": "K = 1: no ensemble, spread is 0 by construction"}
            out["within"] = {"note": "K = 1"}
        out["spall_area_bias_on_spall_conditions"] = float((area[sp].mean(1) - truth[sp]).mean())
        report["models"][name] = out
        print(f"[div] {name}: K={K} copy_rate={out['copy']['copy_rate']['value']:.3f} "
              f"spall_copy_rate={out['copy']['spall_copy_rate']['value']:.3f} "
              f"outside={out['prdc']['outside_train_and_test']['value']:.3f} "
              f"SSR={out['spread'].get('spread_skill_ratio', {}).get('value', float('nan')):.2f}", flush=True)

    with open(os.path.join(cli.out_dir, f"diversity_{cli.split}.json"), "w") as f:
        json.dump(report, f, indent=1)
    _write_md(report, os.path.join(cli.out_dir, f"diversity_{cli.split}.md"))


def _f(v, nd=3):
    if isinstance(v, dict):
        se = v.get("se")
        return f"{v['value']:.{nd}f} ± {se:.{nd}f}" if se == se and se is not None else f"{v['value']:.{nd}f}"
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:.{nd}f}"


def _write_md(rep, path):
    L = [f"# 多樣性評估：{rep['split']}（{rep['n_conditions']} 個條件）", ""]
    ref = rep["reference"]
    c = ref["copy"]
    L += [f"前處理：CLAHE {rep['clahe'] or '無'}，JPEG {rep['jpeg'] or '無'}；訓練參考集 {rep['train_list']} 的 train。", "",
          "## 參考值（真實圖）", "",
          f"- 裂縫抄襲門檻：test 真實圖（無剝落 {c['n_no_spall_conditions']} 個條件）對訓練柱最佳裂縫 F1 的第 "
          f"{c['q'] * 100:.0f} 百分位 = {c['threshold_f1']:.3f}（平均 {c['test_real_best_f1_mean']:.3f}）；CTR1 本身是 {1 - c['q']:.2f}",
          f"- 剝落內容抄襲門檻：test 真實圖剝落小塊（{ref['spall_copy']['n_test_real_patches']} 塊）對訓練剝落小塊"
          f"（{ref['spall_copy']['n_train_patches']} 塊）最佳 NCC 的第 {ref['spall_copy']['q'] * 100:.0f} 百分位 = "
          f"{ref['spall_copy']['threshold_ncc']:.3f}；CTR1 本身是 {1 - ref['spall_copy']['q']:.2f}"]
    for k, v in ref.items():
        if k.startswith("other_real_"):
            L.append(f"- {k}：裂縫抄襲率 {v['copy_rate']:.3f}（n = {v['n_no_spall']}），剝落內容抄襲率 {v['spall_copy_rate']:.3f}，"
                     f"coverage vs test {v['prdc_vs_test']['coverage']:.3f}，落在兩者之外 {v['outside_train_and_test']:.3f}")
    for k in ("real_pairs_other_face", "real_pairs_next_cycle"):
        v = ref.get(k)
        if v:
            L.append(f"- {k}（n = {v['n']}）：cos 距離 {v['cos_dist']:.3f}，1 − mask IoU {_f(v['mask_1_minus_iou'])}"
                     f"（水平翻轉取較佳 {_f(v['mask_1_minus_iou_hflip_best'])}）")
    L += ["", "## 抄襲率（超過真實柱門檻的比例）", "",
          "| 模型 | K | 裂縫抄襲率 | 剝落內容抄襲率 | 裂縫最佳 F1 平均 | 裂縫最佳配對的訓練柱 |", "|---|---|---|---|---|---|"]
    for n, m in rep["models"].items():
        c = m["copy"]
        cols = "、".join(f"{k} {v:.2f}" for k, v in c["best_match_column"].items())
        L.append(f"| {n} | {m['k']} | {_f(c['copy_rate'])} | {_f(c['spall_copy_rate'])} | {c['best_f1_mean']:.3f} | {cols} |")
    L += ["", "## density / coverage（半徑排除同一區塊）", "",
          "| 模型 | density vs test | coverage vs test | density vs train | coverage vs train | 落在兩者之外 | 之外（真實剝落 > 0.3） |",
          "|---|---|---|---|---|---|---|"]
    for n, m in rep["models"].items():
        d = m["prdc"]
        L.append(f"| {n} | {d['vs_test']['density']:.3f} | {d['vs_test']['coverage']:.3f} | {d['vs_train']['density']:.3f} | "
                 f"{d['vs_train']['coverage']:.3f} | {_f(d['outside_train_and_test'])} | {_f(d['outside_train_and_test_real_spall_gt0.3'])} |")
    L += ["", "## 剝落面積的 spread / skill（真實剝落 > spall_min 的條件）", "",
          "| 模型 | n | spread/skill ↔1 | spread | ensemble mean RMSE | rank 兩端比例（期望） | rank histogram |", "|---|---|---|---|---|---|---|"]
    for n, m in rep["models"].items():
        s = m["spread"]
        if "spread_skill_ratio" in s:
            L.append(f"| {n} | {s['n']} | {_f(s['spread_skill_ratio'], 2)} | {s['spread']:.3f} | {s['rmse_ensemble_mean']:.3f} | "
                     f"{s['rank_end_bins_frac']:.2f}（{s['rank_end_bins_expected']:.2f}） | {s['rank_hist']} |")
        else:
            L.append(f"| {n} | {s['n']} | 0（K = 1） | 0 | — | — | — |")
    L += ["", "## 同條件內的變化", "",
          "| 模型 | cos 距離 | Vendi（K 張） | Vendi（前 2 張） | 1 − mask IoU |", "|---|---|---|---|---|"]
    for n, m in rep["models"].items():
        w = m["within"]
        if "cos_dist" in w:
            L.append(f"| {n} | {_f(w['cos_dist'])} | {w['vendi_k']:.2f} | {w['vendi_2_first_pair']:.3f} | {_f(w['mask_1_minus_iou'])} |")
        else:
            L.append(f"| {n} | 0（K = 1） | 1 | 1 | 0 |")
    for k, lab in (("real_pairs_other_face", "真實：另一面"), ("real_pairs_next_cycle", "真實：同面下一圈")):
        v = ref.get(k)
        if v:
            L.append(f"| *{lab}* | {v['cos_dist']:.3f} | — | {v['vendi_2']:.3f} | {_f(v['mask_1_minus_iou'])} |")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[div] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
