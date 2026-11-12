#!/usr/bin/env python3
"""Which per-frame damage rate explains when real persistent structure is born? (docs/redesign/phase1_ilf.md §13)

No training, CPU only. Target per frame of each walk (column face): real step births (irreversibility_curve.step_births:
dark at t, present at t+2 and t+4, absent at t-2, t-4, t-6; same-sign frames only), divided by the walk mean so the
fit is about timing, not crack density. Real frames go through the eval transform (128 px, gray, CLAHE), identical to
the `real` array of the eval npz.

Candidate rates per frame (peak k = timestep // 30), Park-Ang increment split in its two terms:
  c1 / s2        binary: first cycle of the drift level (suffix 1-2) / its first - peak (suffix 2)
  env            increase of max |drift| at this peak, / max |drift| of the walk (envelope term)
  env_pn         the same per sign: (+ envelope increase, - envelope increase)
  E_<src>        energy of the half cycle ending at this peak, int F d(disp) over the last 30 points,
                 / (max |F| * max |disp|) (energy term). src: sim (OpenSees), gru (phase1_extractor, the GRU whose
                 state g the refiner reads; force from the simulated loop), exp (measured; only an upper reference)
  env+E_<src>, env_pn+E_<src>
  dg_norm        |g_t - g_{t-1}| (the refiner's delta g)
  dg_ridge       ridge on the first 16 PCs of g_t - g_{t-1} (PCA and alpha chosen on the train columns)
Each physical feature is divided by its walk mean as well (scale-free). OLS with intercept, fitted on the 4 train
columns (8 walks); reported: R^2 and Pearson r on test (CTR1) and val (R307), and leave-one-column-out R^2 on train.
GRU caveat: R307 was in the GRU's training set (its GRU force is ~the measured one); CTR1 was in its val set.

  python scripts/eval/phase1_damage_rate.py --out results/phase1_damage_rate/rates.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "eval"))
import irreversibility_curve as IR  # noqa: E402
from src.data.hysteresis import HystereticDataset  # noqa: E402
from src.metrics.dino_mmd import to_uint8_gray  # noqa: E402
from src.model.HystereticPrediction import HystereticGRU  # noqa: E402
from src.utils import build_damage_image_transform_with_loss_mask  # noqa: E402

SEG = 30
SPLIT = {"R315": "train", "R330": "train", "R615": "train", "R1015": "train", "R307": "val", "CTR1": "test"}


def real_frames(files):
    tf = build_damage_image_transform_with_loss_mask(
        image_size=128, use_augmentation=False, output_scale=True, normalize_mean=(0.5,) * 3,
        normalize_std=(0.5,) * 3, use_grayscale_input=True, use_clahe=True, clahe_clip_limit=2.0,
        clahe_tile_grid_size="8,8")
    out = []
    for f in files:
        x = tf(Image.open(os.path.join(ROOT, "Data/DamagePattern/images", f)))["target_image"]
        out.append(to_uint8_gray(x[None])[0])
    return np.stack(out)


def loops(exps, extractor="phase1_extractor"):
    """{exp: dict(drift, disp, F_sim, F_gru, F_exp)} on the full simulated time axis."""
    a = json.load(open(os.path.join(ROOT, extractor, "args.json")))
    a["result_dir"] = "/tmp"
    m = HystereticGRU(**a)
    m.load_state_dict(torch.load(os.path.join(ROOT, extractor, "best.ckpt"), map_location="cpu",
                                 weights_only=False)["state_dict"])
    m.eval()
    ds = HystereticDataset(root=os.path.join(ROOT, "Data/Hysteresis"), specified=list(exps), data_size="big",
                           simulate=True, scale=True, maximum=1.0)
    info = pd.read_csv(os.path.join(ROOT, "Data/Hysteresis/info_w_mander_big.csv"), dtype={"id": str}).set_index("id")
    out = {}
    for i, e in enumerate(ds.exp_list):
        x, y, _, mx, _ = ds[i]
        with torch.no_grad():
            gru = m(x[None])[0, :, 0].numpy() * float(mx)
        drift = x[:, -2].numpy()
        out[e] = dict(drift=drift, disp=drift / 100 * float(info.loc[e, "lMeasured"]), F_sim=x[:, -1].numpy() * float(mx),
                      F_gru=gru, F_exp=y[:, 0].numpy() * float(mx))
    return out


def features(lp, ts, files, g_of):
    """Per-frame candidate features of one walk (dict name -> (T, d))."""
    k = ts // SEG
    d = lp["drift"]
    peaks = d[np.arange(int(k.max()) + 1) * SEG]
    mx = np.maximum.accumulate(np.abs(peaks))
    mxp = np.maximum.accumulate(np.maximum(peaks, 0))
    mxn = np.maximum.accumulate(np.maximum(-peaks, 0))
    inc = lambda a: np.diff(np.concatenate([[0.0], a]))
    du = np.abs(lp["disp"]).max()
    f = {"env": (inc(mx) / mx[-1])[k][:, None],
         "env_pn": np.stack([inc(mxp) / max(mxp[-1], 1e-9), inc(mxn) / max(mxn[-1], 1e-9)], 1)[k]}
    for src in ("sim", "gru", "exp"):
        F = lp[f"F_{src}"]
        E = np.zeros(len(peaks))
        for j in range(1, len(peaks)):
            s = slice((j - 1) * SEG, j * SEG + 1)
            E[j] = np.trapezoid(F[s], lp["disp"][s]) if hasattr(np, "trapezoid") else np.trapz(F[s], lp["disp"][s])
        f[f"E_{src}"] = (np.abs(E) / (np.abs(F).max() * du))[k][:, None]
    suf = np.array([int(x.rsplit("_", 1)[1].split(".")[0]) for x in files])
    f["c1"] = ((suf == 1) | (suf == 2)).astype(float)[:, None]
    f["s2"] = (suf == 2).astype(float)[:, None]
    g = np.stack([g_of[x] for x in files])
    dg = np.vstack([np.zeros((1, g.shape[1])), np.diff(g, axis=0)])
    f["dg"] = dg
    f["dg_norm"] = np.linalg.norm(dg, axis=1)[:, None]
    return f


def scale_free(x, valid):
    m = np.abs(x[valid]).mean(0)
    return x / np.where(m > 0, m, 1.0)


MODELS = {
    "c1": ["c1"], "s2": ["s2"], "env": ["env"], "env_pn": ["env_pn"],
    "E_sim": ["E_sim"], "E_gru": ["E_gru"], "E_exp": ["E_exp"],
    "env+E_sim": ["env", "E_sim"], "env+E_gru": ["env", "E_gru"], "env+E_exp": ["env", "E_exp"],
    "env_pn+E_sim": ["env_pn", "E_sim"], "env_pn+E_gru": ["env_pn", "E_gru"], "env_pn+E_exp": ["env_pn", "E_exp"],
    "c1+E_sim": ["c1", "E_sim"], "c1+E_gru": ["c1", "E_gru"], "c1+E_exp": ["c1", "E_exp"],
    "s2+E_sim": ["s2", "E_sim"], "s2+E_gru": ["s2", "E_gru"], "s2+E_exp": ["s2", "E_exp"],
    "dg_norm": ["dg_norm"], "c1+dg_norm": ["c1", "dg_norm"],
}


def design(w, names):
    cols = []
    for n in names:
        x = w["f"][n]
        cols.append(x if n in ("c1", "s2") else scale_free(x, w["valid"]))
    X = np.hstack(cols)[w["valid"]]
    return np.hstack([X, np.ones((len(X), 1))])


def fit_eval(train, tests, make):
    Xtr = np.vstack([make(w) for w in train])
    ytr = np.concatenate([w["y"][w["valid"]] for w in train])
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    out = {}
    for name, ws in tests.items():
        X = np.vstack([make(w) for w in ws])
        y = np.concatenate([w["y"][w["valid"]] for w in ws])
        p = X @ beta
        out[name] = dict(r2=float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                         r=float(np.corrcoef(p, y)[0, 1]) if p.std() > 0 else 0.0)
    return beta, out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="Data/DamagePattern/label.csv")
    p.add_argument("--g", default="Data/DamagePattern/derived/gru_g.npz", help="per-frame GRU state g (dump_gru_g.py)")
    p.add_argument("--thr", type=float, default=0.05)
    p.add_argument("--n_pc", type=int, default=16)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    lab = pd.read_csv(os.path.join(ROOT, a.label))
    lab = lab[lab.exp.isin(SPLIT)]
    gz = np.load(os.path.join(ROOT, a.g))
    g_of = dict(zip(gz["filename"].tolist(), gz["g"]))
    lp = loops(SPLIT)
    walks = []
    for (exp, direction), g in lab.groupby(["exp", "direction"], sort=True):
        g = g.sort_values("timestep", kind="stable")
        files = g.filename.tolist()
        dark = IR.dark_masks(real_frames(files), a.thr, 0.5, 2.0)
        n, valid = IR.step_births(dark, 1)
        y = n / max(n[valid].mean(), 1e-9)
        walks.append(dict(exp=exp, direction=direction, split=SPLIT[exp], y=y, valid=valid, births=n.tolist(),
                          f=features(lp[exp], g.timestep.to_numpy(), files, g_of)))
        print(f"[rate] {exp}/{direction}: {len(files)} frames, step births/frame {n[valid].mean():.1f}", flush=True)

    train = [w for w in walks if w["split"] == "train"]
    tests = {s: [w for w in walks if w["split"] == s] for s in ("test", "val")}
    cols_train = sorted({w["exp"] for w in train})
    report = {"models": {}, "walks": [{k: w[k] for k in ("exp", "direction", "split", "births")} for w in walks]}

    def run(name, make_factory):
        beta, res = fit_eval(train, tests, make_factory(train))
        loco = []
        for c in cols_train:
            tr = [w for w in train if w["exp"] != c]
            _, r = fit_eval(tr, {"held": [w for w in train if w["exp"] == c]}, make_factory(tr))
            loco.append(r["held"]["r2"])
        report["models"][name] = dict(beta=beta.tolist(), loco_r2=float(np.mean(loco)), **res)

    for name, feats in MODELS.items():
        run(name, lambda tr, feats=feats: (lambda w: design(w, feats)))

    # ridge probe on PCs of delta g (PCA and alpha from the training walks of each fit)
    def dg_factory(tr):
        D = np.vstack([w["f"]["dg"][w["valid"]] for w in tr])
        mu = D.mean(0)
        _, _, Vt = np.linalg.svd(D - mu, full_matrices=False)
        V = Vt[: a.n_pc].T
        Z = (D - mu) @ V
        sd = Z.std(0) + 1e-9
        return lambda w: np.hstack([((w["f"]["dg"] - mu) @ V / sd)[w["valid"]], np.ones((w["valid"].sum(), 1))])

    for alpha in (1.0, 10.0, 100.0):
        # ridge via augmented least squares (intercept not penalised)
        def fit_ridge(trw, testws, alpha=alpha):
            mk = dg_factory(trw)
            X = np.vstack([mk(w) for w in trw])
            y = np.concatenate([w["y"][w["valid"]] for w in trw])
            reg = np.sqrt(alpha) * np.eye(X.shape[1])
            reg[-1, -1] = 0.0
            beta, *_ = np.linalg.lstsq(np.vstack([X, reg]), np.concatenate([y, np.zeros(X.shape[1])]), rcond=None)
            out = {}
            for nm, ws in testws.items():
                Xt = np.vstack([mk(w) for w in ws])
                yt = np.concatenate([w["y"][w["valid"]] for w in ws])
                pt = Xt @ beta
                out[nm] = dict(r2=float(1 - ((yt - pt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()),
                               r=float(np.corrcoef(pt, yt)[0, 1]))
            return out
        res = fit_ridge(train, tests)
        loco = [fit_ridge([w for w in train if w["exp"] != c], {"h": [w for w in train if w["exp"] == c]})["h"]["r2"]
                for c in cols_train]
        report["models"][f"dg_ridge_a{alpha:g}"] = dict(loco_r2=float(np.mean(loco)), **res)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"\n{'model':16s} | {'LOCO R2':>7s} | {'test R2':>7s} {'r':>5s} | {'val R2':>7s} {'r':>5s} | beta")
    for name, r in report["models"].items():
        beta = " ".join(f"{b:+.2f}" for b in r.get("beta", []))
        print(f"{name:16s} | {r['loco_r2']:7.3f} | {r['test']['r2']:7.3f} {r['test']['r']:5.2f} | "
              f"{r['val']['r2']:7.3f} {r['val']['r']:5.2f} | {beta}")


if __name__ == "__main__":
    main()
