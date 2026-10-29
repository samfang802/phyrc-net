#!/usr/bin/env python3
"""Items 1 (spall geometry) and 3 (monotone onset field T*, damage-age map) of docs/redesign/dataset_audit.md.

From the hand-drawn spall polygons (Data/DamagePattern/anylabel/*.json, 750x800):
  * per frame: spall fraction R, area in px, rows covered (top / bottom), area inside the bottom-D band;
  * per frame: 16x16 and 32x32 cell coverage (exact area average of the full-resolution mask);
  * per sequence and cell: isotonic regression (non-decreasing in time) of the coverage, onset = first frame whose
    fitted coverage >= 0.5, onset drift = max |drift| reached at that frame (frames_with_clocks.csv max_drift);
  * damage age per frame and cell = max(0, max_drift_f - onset_drift) (drift-% units) and frames since onset.
Optional --align frame_audit.csv: shift each mask by minus the grid-phase shift vs the sequence's first frame
(only where the grid fit is trustworthy) before the cell averages.
Writes into Data/DamagePattern/derived/: spall_frame_geometry.csv, onset_field{,_aligned}.npz, onset_summary{,_aligned}.csv
"""
import argparse, json, os
import numpy as np, pandas as pd, cv2
from sklearn.isotonic import IsotonicRegression

ROOT = "Data/DamagePattern"; OUT = f"{ROOT}/derived"; H, W = 800, 750
C = pd.read_csv("results/phase0/clock/frames_with_clocks.csv")[["filename", "level", "cycle", "max_drift"]]


def spall_mask(fn, dx=0.0, dy=0.0):
    d = json.load(open(f"{ROOT}/anylabel/{fn[:-4]}.json"))
    m = np.zeros((H, W), np.uint8)
    for s in d["shapes"]:
        if s["label"] == "spall":
            pts = np.round(np.array(s["points"]) - np.array([dx, dy])).astype(np.int32)
            cv2.fillPoly(m, [pts], 1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--align", default="")
    a = ap.parse_args()
    suf = "_aligned" if a.align else ""
    L = pd.read_csv(f"{ROOT}/label.csv").merge(C, on="filename")
    L["seq"] = L.filename.str.extract(r"^(\w+?_\w+?)_")[0]
    L["idx"] = L.filename.str.extract(r"_(\d+)\.jpg")[0].astype(int)
    L = L.sort_values(["seq", "timestep", "idx"]).reset_index(drop=True)
    sh = {}
    if a.align:
        A = pd.read_csv(a.align).set_index("filename")
        ref = A.groupby("seq")[["pitch_x", "pitch_y"]].transform(lambda x: x.iloc[:10].median())
        ok = ((A.pitch_x - ref.pitch_x).abs() < 4) & ((A.pitch_y - ref.pitch_y).abs() < 4) & (A.gdx_ref.abs() < 30) & (A.gdy_ref.abs() < 30)
        for fn in L.filename:
            sh[fn] = (A.at[fn, "gdx_ref"], A.at[fn, "gdy_ref"]) if ok[fn] else (0.0, 0.0)
        print("aligned frames used:", int(ok.sum()), "of", len(ok))
    N = len(L)
    cov = {16: np.zeros((N, 16, 16), np.float32), 32: np.zeros((N, 32, 32), np.float32)}
    geo = []
    for i, r in enumerate(L.itertuples()):
        dx, dy = sh.get(r.filename, (0.0, 0.0))
        m = spall_mask(r.filename, dx, dy)
        for g in cov:
            cov[g][i] = cv2.resize(m.astype(np.float32), (g, g), interpolation=cv2.INTER_AREA)
        rows = m.sum(1) > 0
        ys = np.where(rows)[0]
        # row coverage >= 10% of width -> robust top of the spalled zone
        rc = np.where(m.sum(1) >= 0.10 * W)[0]
        geo.append(dict(filename=r.filename, area_px=int(m.sum()), R=m.mean(),
                        y_top=int(ys.min()) if len(ys) else -1, y_bot=int(ys.max()) if len(ys) else -1,
                        y_top10=int(rc.min()) if len(rc) else -1,
                        **{f"band{b}": float(m[H - b:].mean()) for b in (400, 500, 600)}))
    G = pd.concat([L[["filename", "seq", "timestep", "level", "cycle", "max_drift", "exp"]], pd.DataFrame(geo).drop(columns="filename")], axis=1)
    G.to_csv(f"{OUT}/spall_frame_geometry{suf}.csv", index=False)
    seqs = sorted(L.seq.unique())
    res = {"filenames": L.filename.values.astype(str), "seq": L.seq.values.astype(str), "max_drift": L.max_drift.values.astype(np.float32),
           "timestep": L.timestep.values}
    summ = []
    for g in (16, 32):
        iso = np.zeros_like(cov[g]); onset = np.full((len(seqs), g, g), np.nan, np.float32)
        onset_t = np.full((len(seqs), g, g), -1, np.int32)
        age = np.zeros_like(cov[g]); agef = np.zeros_like(cov[g])
        for si, s in enumerate(seqs):
            ix = np.where(L.seq.values == s)[0]; md = L.max_drift.values[ix]; x = np.arange(len(ix))
            for u in range(g):
                for v in range(g):
                    y = cov[g][ix, u, v]
                    if y.max() < 0.05:
                        continue
                    f = IsotonicRegression(increasing=True, y_min=0, y_max=1).fit_transform(x, y)
                    iso[ix, u, v] = f
                    k = np.where(f >= 0.5)[0]
                    if len(k):
                        onset[si, u, v] = md[k[0]]; onset_t[si, u, v] = k[0]
            for u in range(g):
                for v in range(g):
                    if not np.isnan(onset[si, u, v]):
                        age[ix, u, v] = np.maximum(0, md - onset[si, u, v]) * (x >= onset_t[si, u, v])
                        agef[ix, u, v] = np.maximum(0, x - onset_t[si, u, v]) * (x >= onset_t[si, u, v])
            o = onset[si]; frac_cells = (~np.isnan(o)).mean()
            # drift at which 5% / 25% of cells have iso >= .5
            fr = (iso[ix] >= 0.5).reshape(len(ix), -1).mean(1)
            def first(th):
                k = np.where(fr >= th)[0]; return float(md[k[0]]) if len(k) else np.nan
            summ.append(dict(grid=g, seq=s, first_cell_onset_drift=float(np.nanmin(o)) if frac_cells else np.nan,
                             median_onset_drift=float(np.nanmedian(o)) if frac_cells else np.nan,
                             frac_cells_ever=float(frac_cells), drift_at_5pct_cells=first(0.05), drift_at_25pct_cells=first(0.25),
                             final_cell_frac=float(fr[-1])))
        res.update({f"cov{g}": cov[g], f"iso{g}": iso, f"onset{g}": onset, f"onset_frame{g}": onset_t,
                    f"age_drift{g}": age, f"age_frames{g}": agef})
    res["seq_names"] = np.array(seqs)
    np.savez_compressed(f"{OUT}/onset_field{suf}.npz", **res)
    pd.DataFrame(summ).to_csv(f"{OUT}/onset_summary{suf}.csv", index=False)
    print(pd.DataFrame(summ).round(2).to_string())


if __name__ == "__main__":
    main()
