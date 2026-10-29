"""Can PEER SPD damage observations predict when the 6 image columns start spalling?

1. Parse the PEER SPD rectangular-column XML (per-specimen ``<damage>`` block) into
   ``Data/Hysteresis/peer_spd_damage.csv`` and match it to ``info_w_mander_big.csv``.
2. Fit onset drift (% of lMeasured) of cover crushing / significant spalling from column
   parameters: a Berry-Eberhard-form log-linear fit, the same plus confinement / reinforcement
   terms, and a bootstrap MLP ensemble (GPU when available).
3. Predict the 6 image columns and compare with the first drift level at which the hand
   spall polygons (anylabel) exceed a threshold.

The image columns never enter any fit. CTR1 is reported, not used for selection.
"""

import argparse
import glob
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch

INFO = "Data/Hysteresis/info_w_mander_big.csv"
FRAMES = "results/phase0/clock/frames_with_clocks.csv"
IMAGE_COLS = ["R315", "R330", "R615", "R1015", "R307", "CTR1"]
DAMAGE = ["crushing", "spalling", "buckling", "longBarFracture", "spiralFracture", "lossAxial"]


def _num(s):
    try:
        return float(str(s).replace(",", ""))
    except (TypeError, ValueError):
        return float("nan")


def parse_xml(xml_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(xml_dir, "*.xml"))):
        t = ET.parse(f).getroot().find("columnTest")
        if t is None or t.findtext("specimen/type", "").strip() != "Rectangular":
            continue
        g = lambda p: _num(t.findtext(p))
        r = dict(
            spd_id=t.get("id"), name=t.findtext("specimen/name", "").strip(),
            fc=g("materialProperties/concreteStrength"), fyt=g("materialProperties/transverseSteel/yieldStress"),
            width=g("geometry/width"), depth=g("geometry/depth"), lMeasured=g("geometry/lMeasured"),
            axialLoad=g("loading/axialLoad"), rho_l=g("longitudinalReinforcement/reinforcementRatio"),
            rho_t=g("transverseReinforcement/volTransReinfRatio"),
            cover=g("longitudinalReinforcement/parallelToLoad/clearCover"),
            failure=t.findtext("failureType", "").strip(),
        )
        for k in DAMAGE:
            r[k + "_mm"] = g("damage/" + k)
        rows.append(r)
    return pd.DataFrame(rows)


def match_ids(spd, info):
    key = lambda fc, b, h, p, l: (round(fc, 1), round(b), round(h), round(p), round(l))
    lut = {key(r.fc, r.width, r.depth, r.axialLoad, r.lMeasured): r.spd_id for r in spd.itertuples()}
    m = {}
    for r in info.itertuples():
        k = key(_num(r.concreteStrength), _num(r.width), _num(r.depth), _num(r.axialLoad), _num(r.lMeasured))
        if k in lut:
            m[lut[k]] = str(r.id)
    spd["info_id"] = spd["spd_id"].map(m)
    return spd


def features(df):
    n = df["axialLoad"] * 1e3 / (df["fc"] * df["width"] * df["depth"])
    l_d = df["lMeasured"] / df["depth"]
    rho_eff = df["rho_t"] * df["fyt"] / df["fc"]
    return pd.DataFrame({
        "log1m_n": np.log(np.clip(1 - n, 1e-3, None)),
        "log_ld": np.log(1 + l_d / 10),
        "rho_eff": rho_eff,
        "rho_l": df["rho_l"],
        "cover_d": df["cover"] / df["depth"],
        "log_depth": np.log(df["depth"]),
        "n": n,
    })


SETS = {
    "BE": ["log1m_n", "log_ld"],
    "BE+conf": ["log1m_n", "log_ld", "rho_eff"],
    "BE+conf+rhol+size": ["log1m_n", "log_ld", "rho_eff", "rho_l", "cover_d", "log_depth"],
}


def ridge(x, y, lam=1e-3):
    X = np.c_[np.ones(len(x)), x]
    A = X.T @ X + lam * np.eye(X.shape[1])
    A[0, 0] -= lam
    return np.linalg.solve(A, X.T @ y)


def loo_linear(x, y):
    pred = np.empty_like(y)
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        mu, sd = x[keep].mean(0), x[keep].std(0) + 1e-9
        w = ridge((x[keep] - mu) / sd, y[keep])
        pred[i] = np.r_[1, (x[i] - mu) / sd] @ w
    return pred


def mlp_ensemble(x, y, xq, device, n_models=64, hidden=16, epochs=1500, wd=1e-2, seed=0):
    """Bootstrap ensemble of tiny MLPs trained in parallel (one batched weight tensor)."""
    g = torch.Generator().manual_seed(seed)
    mu, sd = x.mean(0), x.std(0) + 1e-9
    X = torch.tensor((x - mu) / sd, dtype=torch.float32, device=device)
    Y = torch.tensor(y, dtype=torch.float32, device=device)
    Q = torch.tensor((xq - mu) / sd, dtype=torch.float32, device=device)
    n, d = X.shape
    idx = torch.randint(0, n, (n_models, n), generator=g).to(device)
    Xb, Yb = X[idx], Y[idx]
    W1 = (torch.randn(n_models, d, hidden, generator=g) / d ** 0.5).to(device).requires_grad_()
    b1 = torch.zeros(n_models, 1, hidden, device=device, requires_grad=True)
    W2 = (torch.randn(n_models, hidden, 1, generator=g) / hidden ** 0.5).to(device).requires_grad_()
    b2 = torch.full((n_models, 1, 1), float(y.mean()), device=device, requires_grad=True)
    opt = torch.optim.AdamW([W1, b1, W2, b2], lr=1e-2, weight_decay=wd)
    f = lambda z: (torch.tanh(z @ W1 + b1) @ W2 + b2).squeeze(-1)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((f(Xb) - Yb) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return f(Q.unsqueeze(0).expand(n_models, -1, -1)).cpu().numpy()


def observed_onsets(thr):
    fr = pd.read_csv(FRAMES)
    out = {}
    for col in IMAGE_COLS:
        g = fr[fr["exp"] == col]
        per_face = {}
        for face, h in g.groupby("face"):
            hit = h[h["R_spall"] > thr]
            per_face[face] = float(hit["max_drift"].min()) if len(hit) else float("nan")
        out[col] = dict(first=float(np.nanmin(list(per_face.values()))), **{f"face_{k}": v for k, v in per_face.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml_dir", required=True)
    ap.add_argument("--out", default="results/phase0/spd_onset")
    ap.add_argument("--csv", default="Data/Hysteresis/peer_spd_damage.csv")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    info = pd.read_csv(INFO)
    spd = match_ids(parse_xml(args.xml_dir), info)
    spd.to_csv(args.csv, index=False)
    print(f"[spd] {len(spd)} rectangular specimens, {spd['info_id'].notna().sum()} matched to {INFO}; device={device}")

    img = info[info["id"].isin(IMAGE_COLS)].copy()
    img = img.rename(columns={"concreteStrength": "fc", "reinforcementRatio": "rho_l", "volTransReinfRatio": "rho_t",
                              "transverse_yield": "fyt", "parallel_clearCover": "cover"})
    for c in ["fc", "width", "depth", "lMeasured", "axialLoad", "rho_l", "rho_t", "fyt", "cover"]:
        img[c] = img[c].map(_num)
    img = img.set_index("id").loc[IMAGE_COLS].reset_index()
    fq = features(img)

    report = {"device": device, "n_specimens": int(len(spd)), "n_matched": int(spd["info_id"].notna().sum()), "targets": {}}
    lines = []
    for target in ["crushing", "spalling"]:
        d = spd[spd[target + "_mm"] > 0].copy()
        y = np.log(100 * d[target + "_mm"] / d["lMeasured"]).to_numpy()
        fx = features(d)
        ok = np.isfinite(fx.to_numpy()).all(1) & np.isfinite(y)
        fx, y, d = fx[ok], y[ok], d[ok]
        lines.append(f"\n== target: {target} onset drift (%), n = {len(y)}; n range {fx['n'].min():.2f}-{fx['n'].max():.2f}, "
                     f"rho_eff range {fx['rho_eff'].min():.3f}-{fx['rho_eff'].max():.3f}")
        res = {"n": int(len(y)), "models": {}}
        for name, cols in SETS.items():
            x, xq = fx[cols].to_numpy(), fq[cols].to_numpy()
            loo = loo_linear(x, y)
            mu, sd = x.mean(0), x.std(0) + 1e-9
            w = ridge((x - mu) / sd, y)
            pq = np.exp(np.c_[np.ones(len(xq)), (xq - mu) / sd] @ w)
            ens = np.exp(mlp_ensemble(x, y, xq, device))
            res["models"][name] = dict(
                loo_rmse_log=float(np.sqrt(np.mean((loo - y) ** 2))),
                loo_cov=float(np.std(np.exp(loo) / np.exp(y))),
                coef_std=dict(zip(["intercept"] + cols, map(float, w))),
                linear_pred=dict(zip(IMAGE_COLS, map(float, pq))),
                mlp_median=dict(zip(IMAGE_COLS, map(float, np.median(ens, 0)))),
                mlp_p10=dict(zip(IMAGE_COLS, map(float, np.percentile(ens, 10, 0)))),
                mlp_p90=dict(zip(IMAGE_COLS, map(float, np.percentile(ens, 90, 0)))),
            )
            lines.append(f"  [{name}] LOO RMSE(log) {res['models'][name]['loo_rmse_log']:.3f}  coef "
                         + " ".join(f"{k}={v:+.3f}" for k, v in res['models'][name]['coef_std'].items()))
            lines.append("    " + "  ".join(f"{c}: lin {pq[i]:.2f} | mlp {np.median(ens[:, i]):.2f} "
                                             f"[{np.percentile(ens[:, i], 10):.2f},{np.percentile(ens[:, i], 90):.2f}]"
                                             for i, c in enumerate(IMAGE_COLS)))
        report["targets"][target] = res

    report["observed"] = {}
    for thr in [0.02, 0.05]:
        obs = observed_onsets(thr)
        report["observed"][str(thr)] = obs
        lines.append(f"\n== observed first spall drift (%) from anylabel, R_spall > {thr}: "
                     + "  ".join(f"{c} {obs[c]['first']:.2f}" for c in IMAGE_COLS))
    lines.append("\n== image-column parameters: " + "  ".join(
        f"{c}: n={fq['n'][i]:.3f} rho_eff={fq['rho_eff'][i]:.3f} L/D={img['lMeasured'][i] / img['depth'][i]:.1f}"
        for i, c in enumerate(IMAGE_COLS)))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(args.out, "report.txt"), "w").write(txt + "\n")
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
