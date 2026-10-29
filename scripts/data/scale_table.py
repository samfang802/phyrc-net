#!/usr/bin/env python3
"""Item 1 of docs/redesign/dataset_audit.md: physical scale, photo coverage, spall in cm^2 and in the hinge band.

Assumptions (flagged in the doc):
  * the drawn grid is square with a 10 cm pitch (A). Evidence: for the 750 mm wide R columns the photo width is exactly
    7.5 pitches. Alternative (B) for CTR1: photo = whole 870 mm face, so pitch = 11.6 cm.
  * the column base is at (or just below) the bottom image row (spall polygons always reach row ~799).
  * the photographed face is as wide as `width` of info_w_mander_big.csv; hinge band height = `depth` (D) above the base.
Outputs (Data/DamagePattern/derived/): scale_per_sequence.csv, spall_scaled_frames.csv, spall_compare_levels.csv,
spall_rowprofile.npz (per frame, 800-row spall coverage).
"""
import json, numpy as np, pandas as pd, cv2
ROOT = "Data/DamagePattern"; OUT = f"{ROOT}/derived"; H, W = 800, 750
A = pd.read_csv(f"{OUT}/frame_audit.csv"); G = pd.read_csv(f"{OUT}/spall_frame_geometry.csv")
info = pd.read_csv("Data/Hysteresis/info_w_mander_big.csv", dtype={"id": str}).set_index("id")
A["exp"] = A.seq.str.split("_").str[0]
early = A[A.timestep <= 1500].groupby("seq")  # drift <= ~1%
rows = []
for s, g in A.groupby("seq", sort=False):
    g10 = g.sort_values("timestep").iloc[:10]
    px, py = g10.pitch_x.median(), g10.pitch_y.median()
    exp = s.split("_")[0]; b, D, Lc = info.loc[exp, ["width", "depth", "lMeasured"]]
    rows.append(dict(seq=s, exp=exp, pitch_x_px=px, pitch_y_px=py, anisotropy=py / px,
                     pxcm_x=px / 10, pxcm_y=py / 10, cover_w_cm=W / (px / 10), cover_h_cm=H / (py / 10),
                     sect_width_mm=b, sect_depth_mm=D, col_len_mm=Lc,
                     cover_w_over_width=W / (px / 10) / (b / 10), cover_h_over_D=H / (py / 10) / (D / 10)))
S = pd.DataFrame(rows); S.to_csv(f"{OUT}/scale_per_sequence.csv", index=False)

# per-frame spall in physical units, scenario A (10 cm) and B (CTR1: 8.62 px/cm both axes)
sc = S.set_index("seq")
F = G.copy(); F["exp"] = F.seq.str.split("_").str[0]
prof = np.zeros((len(F), H), np.float32)
for i, r in enumerate(F.itertuples()):
    d = json.load(open(f"{ROOT}/anylabel/{r.filename[:-4]}.json")); m = np.zeros((H, W), np.uint8)
    for sh in d["shapes"]:
        if sh["label"] == "spall":
            cv2.fillPoly(m, [np.round(np.array(sh["points"])).astype(np.int32)], 1)
    prof[i] = m.mean(1)
np.savez_compressed(f"{OUT}/spall_rowprofile.npz", filenames=F.filename.values.astype(str), row_cov=prof)
out = []
for scen in ("A", "B"):
    for i, r in enumerate(F.itertuples()):
        s0 = sc.loc[r.seq]
        if scen == "B" and r.exp == "CTR1":
            pxx = pyy = W / (s0.sect_width_mm / 10)
        else:
            pxx, pyy = s0.pxcm_x, s0.pxcm_y
        Dpx = int(round(s0.sect_depth_mm / 10 * pyy))
        band = prof[i, H - min(Dpx, H):]
        area = prof[i].sum() * W / (pxx * pyy)
        ytop = r.y_top10
        out.append(dict(scenario=scen, filename=r.filename, seq=r.seq, exp=r.exp, level=r.level, max_drift=r.max_drift, R=r.R,
                        area_cm2=area, hinge_Dpx=Dpx, hinge_band_cm2=band.sum() * W / (pxx * pyy),
                        hinge_frac=band.mean(), hinge_photo_frac=band.sum() / (len(band)),
                        top_height_cm=(H - ytop) / pyy if ytop >= 0 else 0.0,
                        top_height_over_D=((H - ytop) / pyy) / (s0.sect_depth_mm / 10) if ytop >= 0 else 0.0))
O = pd.DataFrame(out); O.to_csv(f"{OUT}/spall_scaled_frames.csv", index=False)
# compare by drift level: mean over frames of that level, then over the two faces
O["lv"] = O.level.round(2)
cmp_ = (O[O.lv.isin([3, 4, 5, 6, 8])].groupby(["scenario", "exp", "lv"])
        [["R", "area_cm2", "hinge_frac", "top_height_cm", "top_height_over_D"]].mean().reset_index())
# last frame of the level only (final cycle of that drift level, same loading history position for every column)
T = pd.read_csv(f"{ROOT}/label.csv")[["filename", "timestep"]]
O2 = O.merge(T, on="filename")
last = O2[O2.lv.isin([3, 4, 5, 6, 8])].sort_values("timestep").groupby(["scenario", "seq", "lv"]).tail(1)
cl = last.groupby(["scenario", "exp", "lv"])[["R", "area_cm2", "hinge_frac", "top_height_cm", "top_height_over_D"]].mean().reset_index()
cmp_ = cmp_.merge(cl, on=["scenario", "exp", "lv"], suffixes=("", "_lastcycle"))
cmp_.to_csv(f"{OUT}/spall_compare_levels.csv", index=False)
pd.set_option("display.width", 250)
print(S.round(2).to_string())
for scen in ("A", "B"):
    print(scen); print(cmp_[(cmp_.scenario == scen) & cmp_.exp.isin(["CTR1", "R315", "R330"])][["exp","lv","R","R_lastcycle","area_cm2_lastcycle","hinge_frac_lastcycle","top_height_over_D_lastcycle"]].round(3).to_string())
