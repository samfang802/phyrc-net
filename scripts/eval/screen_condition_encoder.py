#!/usr/bin/env python3
"""Screen a damage-state encoder against the GRU extractor before retraining any generator.

Same frames and probes as scripts/eval/analyze_condition_g.py (docs/eval/condition_g.md §6-§10), on the
raw representations (z-scored per dimension): GRU g, encoder z, encoder [z, s].
  1. variance by specimen / drift level / drift sign
  2. ridge probes -> R_spall, DI (train specimens leave-one-specimen-out; R307 / CTR1 from all four)
  3. change between consecutive labelled frames of one face (opposite loading direction)
  4. peak-force error on the encoder's val specimens, encoder head vs GRU

  python scripts/eval/screen_condition_encoder.py results/DamageStateEncoder/<run> [--out <dir>]
"""

import argparse, glob, importlib, json, os, sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV

sys.path.insert(0, os.getcwd())
from src.data.damage_state import SEG, TARGETS, DamageStateDataset, IMAGE_SPECIMENS
from src.data.hysteresis import HystereticDataset
from src.model.damage_state_encoder import DamageStateEncoder

TR = ["R315", "R330", "R615", "R1015"]


def gru_states(ids):
    a = argparse.Namespace(**json.load(open("phase1_extractor/args.json")))
    ext = getattr(importlib.import_module("src.model.HystereticPrediction"), a.architecture)(**vars(a))
    ck = glob.glob("phase1_extractor/**/best.ckpt", recursive=True)[0]
    ext.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]); ext.eval()
    ds = HystereticDataset(root="Data/Hysteresis", specified=list(ids), **vars(a))
    H, P, T = {}, {}, {}
    with torch.no_grad():
        for i, e in enumerate(ds.exp_list):
            inp, tf, _, _, _ = ds[i]
            out, _ = ext.L(inp[None])
            H[e], P[e], T[e] = out[0].numpy(), ext.P(out).reshape(-1).numpy(), tf.numpy().reshape(-1)
    return H, P, T


def encoder_states(run, ids):
    model = DamageStateEncoder.load_from_checkpoint(os.path.join(run, "best.ckpt"), map_location="cpu").eval()
    stats = {k: np.asarray(v, dtype=np.float32) for k, v in json.load(open(os.path.join(run, "stats.json"))).items()}
    seq_info = json.load(open(os.path.join(run, "args.json"))).get("seq_info", "raw")
    ds = DamageStateDataset(list(ids), stats=stats, seq_info=seq_info)
    Z, S, R, Zinv = {}, {}, {}, {}
    with torch.no_grad():
        for item in ds:
            b = DamageStateDataset.collate([item])
            z, s, z_pts = model.peak_states(b)
            h = torch.cat([z, s[:, None].expand(-1, z.shape[1], -1)], dim=2)
            reg = model.reg_head(h)[0].numpy() * stats["tgt_std"] + stats["tgt_mean"]
            if z_pts is not None:
                Z[item["id"]] = z_pts[0].numpy()  # one state per point, indexed by timestep
                # mirror-invariant part: average with the state of the mirrored loop
                bm = dict(b); bm["seq"] = b["seq"].clone(); bm["seq"][..., -2:] *= -1
                Zinv[item["id"]] = 0.5 * (z_pts[0].numpy() + model.peak_states(bm)[2][0].numpy())
            else:
                # one state per peak; before the first peak (timestep 0) the zero GRU state
                Z[item["id"]] = np.concatenate([np.zeros((1, z.shape[2]), np.float32), z[0].numpy()])
            S[item["id"]], R[item["id"]] = s[0].numpy(), reg
    return Z, S, R, ds, Zinv


def decompose(X, F, mask):
    X = X[mask]; f = F[mask].reset_index(drop=True); m = X.mean(0); tot = ((X - m) ** 2).sum()
    return {k: float(sum(len(i) * ((X[i].mean(0) - m) ** 2).sum() for i in f.groupby(k).indices.values()) / tot)
            for k in ("exp", "level", "sign")}


def probe(X, F, y, trm):
    out = np.full(len(F), np.nan)
    ok = F[y].notna().values
    for e in IMAGE_SPECIMENS:
        tr = ok & trm & (F.exp.values != e)
        out[F.exp.values == e] = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(X[tr], F.loc[tr, y]).predict(X[F.exp.values == e])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()
    out = cli.out or os.path.join(cli.run, "screen")
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(2)
    val_ids = json.load(open(os.path.join(cli.run, "args.json")))["val_ids"]

    H, P, T = gru_states(list(IMAGE_SPECIMENS) + val_ids)
    Z, S, R, eds, Zinv = encoder_states(cli.run, list(IMAGE_SPECIMENS) + val_ids)
    points = json.load(open(os.path.join(cli.run, "args.json"))).get("input_mode", "tokens") == "points"

    lab = pd.read_csv("Data/DamagePattern/label.csv")
    F = lab[lab.exp.isin(IMAGE_SPECIMENS)].reset_index(drop=True)
    F["level"] = F.filename.str.split("_").str[2].astype(int)
    rs = pd.read_csv("Data/DamagePattern/r_spall.csv").set_index("filename")["R_spall"]
    F["R_spall"] = F.filename.map(rs)
    drift = {e: pd.read_csv(f"Data/Hysteresis/Simulate/{e}.csv")["Drift(%)"].values for e in IMAGE_SPECIMENS}
    F["sign"] = [int(np.sign(drift[r.exp][int(r.timestep)])) for _, r in F.iterrows()]
    k = (F.timestep if points else F.timestep // SEG).astype(int).values
    reps = {
        "GRU g": np.stack([H[r.exp][int(r.timestep)] for _, r in F.iterrows()]),
        "encoder z": np.stack([Z[r.exp][kk] for (_, r), kk in zip(F.iterrows(), k)]),
    }
    reps["encoder [z, s]"] = np.concatenate([reps["encoder z"], np.stack([S[e] for e in F.exp])], axis=1)
    if Zinv:
        reps["encoder z_inv"] = np.stack([Zinv[r.exp][kk] for (_, r), kk in zip(F.iterrows(), k)])
        reps["encoder [z_inv, s]"] = np.concatenate([reps["encoder z_inv"], np.stack([S[e] for e in F.exp])], axis=1)
    trm = F.exp.isin(TR).values
    for name in list(reps):
        X = reps[name]; mu, sd = X[trm].mean(0), X[trm].std(0) + 1e-6; reps[name] = (X - mu) / sd

    lines = []
    def emit(title, obj):
        s = f"\n### {title}\n{obj.to_string() if hasattr(obj, 'to_string') else obj}\n"
        print(s); lines.append(s)

    emit("variance fraction (6 specimens)", pd.DataFrame({n: decompose(X, F, np.ones(len(F), bool)) for n, X in reps.items()}).round(3))

    rows = []
    for name, X in reps.items():
        r = {"rep": name}
        for y in ("R_spall", "DI"):
            F[f"{y}_hat"] = probe(X, F, y, trm)
            g = F.groupby(["exp", "level"])[[y, f"{y}_hat"]].mean()
            err = (g[f"{y}_hat"] - g[y])
            for e, lv in (("CTR1", 6000), ("CTR1", 8000), ("R307", 6000), ("R307", 8000)):
                r[f"{y} {e} {lv // 1000}%"] = err.get((e, lv), np.nan)
            if y == "DI":
                r["DI R1015 <=1% (LOSO)"] = err.loc["R1015"][err.loc["R1015"].index <= 1000].mean()
            loso = g[g.index.get_level_values(0).isin(TR)]
            r[f"{y} LOSO RMSE (train 4)"] = float(np.sqrt(((loso[f'{y}_hat'] - loso[y]) ** 2).mean()))
        rows.append(r)
    probe_tab = pd.DataFrame(rows).set_index("rep")
    emit("probe error = predicted - actual (mean over frames of the level)", probe_tab.T.round(3))

    cs = []
    for name, X in reps.items():
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        for (e, d), f in F.sort_values("timestep").groupby(["exp", "direction"]):
            ii = f.index.values
            for a_, b_ in zip(ii[:-1], ii[1:]):
                if F.sign[a_] != 0 and F.sign[b_] != 0:
                    cs.append(dict(rep=name, same_sign=F.sign[a_] == F.sign[b_], cos=float(Xn[a_] @ Xn[b_]),
                                   rel=float(np.linalg.norm(X[b_] - X[a_]) / np.linalg.norm(X[a_]))))
    emit("consecutive labelled frames of one face", pd.DataFrame(cs).groupby(["rep", "same_sign"])[["cos", "rel"]].median().round(3))

    # peak force: GRU prediction at the peak indices vs encoder head, on the encoder's val specimens
    fi = TARGETS.index("abs_peak_force")
    err_g, err_e = [], []
    for e in val_ids:
        it = next(x for x in eds.items if x["id"] == e)
        n = it["targets"].shape[0]; pk = np.arange(1, n + 1) * SEG
        tgt = it["targets"][:, fi]; v = it["valid"] > 0
        err_g.append(np.sqrt(np.mean((np.abs(P[e][pk]) - np.abs(T[e][pk]))[v] ** 2)))
        err_e.append(np.sqrt(np.mean((R[e][:, fi] - tgt)[v] ** 2)))
    emit("peak-force RMSE on encoder val specimens (normalized force)",
         pd.Series({"GRU (median)": np.median(err_g), "encoder (median)": np.median(err_e),
                    "GRU (mean)": np.mean(err_g), "encoder (mean)": np.mean(err_e)}).round(4))
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.writelines(lines)
    # one row per representation for sweeps (scripts/eval/collect_encoder_screens.py)
    var = {n: decompose(X, F, np.ones(len(F), bool)) for n, X in reps.items()}
    summary = {n: {**{k: float(v) for k, v in probe_tab.loc[n].items()}, "sign_var": var[n]["sign"]} for n in reps}
    summary["_force_rmse_median"] = {"GRU": float(np.median(err_g)), "encoder": float(np.median(err_e))}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
