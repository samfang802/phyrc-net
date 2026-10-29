#!/usr/bin/env python3
"""Item 4 of docs/redesign/dataset_audit.md: feasibility of a 'drawn line' mask (grid + pen marks + interference).

Per sample frame (full resolution 800x750), code bits in the saved uint8 mask:
  1 = grid line   black-hat(V) > thr within +-4 px of the fitted grid lines (positions from frame_audit.csv, or the
                  sequence's early-frame median when the per-frame fit is unreliable), outside spall polygons
  2 = pen         ink: Lab (a,b) distance from the image median > 9 AND black-hat(L, 21x21) > 8, outside spall polygons
  4 = interference  hand-drawn 'interference' polygons (Data/DamagePattern/anylabel)
Metrics: how much of the dark thin structure (black-hat(V) > thr, outside spall) is explained by the mask (3 px dilation),
overlap of the pen mask with the interference polygons. Writes derived/grid_pen_mask_sample.npz and
derived/grid_pen_mask_metrics.csv and results/dataset_audit/grid_pen_masks.png.
"""
import json, numpy as np, pandas as pd, cv2
ROOT = "Data/DamagePattern"; OUT = f"{ROOT}/derived"; H, W = 800, 750; THR = 25
A = pd.read_csv(f"{OUT}/frame_audit.csv").set_index("filename")
C = pd.read_csv("results/phase0/clock/frames_with_clocks.csv").set_index("filename")


def polys(fn):
    d = json.load(open(f"{ROOT}/anylabel/{fn[:-4]}.json"))
    sp = np.zeros((H, W), np.uint8); it = np.zeros((H, W), np.uint8)
    for s in d["shapes"]:
        cv2.fillPoly(sp if s["label"] == "spall" else it, [np.round(np.array(s["points"])).astype(np.int32)], 1)
    return sp > 0, it > 0


def masks(fn, seqref):
    bgr = cv2.imread(f"{ROOT}/images/{fn}"); b, g, r = [bgr[:, :, i].astype(np.int16) for i in range(3)]
    v = bgr.max(2); mx, mn = bgr.max(2).astype(np.int16), bgr.min(2).astype(np.int16)
    sat = (mx - mn) / np.maximum(mx, 1)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    bh = cv2.morphologyEx(v, cv2.MORPH_BLACKHAT, k)
    sp, it = polys(fn)
    a = A.loc[fn]; ok = abs(a.pitch_x - seqref["px"]) < 4 and abs(a.pitch_y - seqref["py"]) < 4
    x0, y0, px, py = (a.x0, a.y0, a.pitch_x, a.pitch_y) if ok else (seqref["x0"], seqref["y0"], seqref["px"], seqref["py"])
    xs = np.arange(W); ys = np.arange(H)
    dx = np.abs((xs - x0 + px / 2) % px - px / 2); dy = np.abs((ys - y0 + py / 2) % py - py / 2)
    near = (dx[None, :] < 4) | (dy[:, None] < 4)
    grid = (bh > THR) & near & ~sp
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab = lab[:, :, 1:] - np.median(lab[:, :, 1:].reshape(-1, 2), 0)
    dist = np.sqrt((ab ** 2).sum(2))
    bhl = cv2.morphologyEx(lab[:, :, 0].astype(np.uint8), cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)))
    broad = cv2.blur(dist, (41, 41)) > 11       # large tinted areas (rust-coloured hoop, painted rebar)
    pen = (dist > 9) & (bhl > 8) & ~sp & ~broad          # tinted AND thin-dark: ink strokes, not the broad orange hoop
    pen = cv2.morphologyEx(pen.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    return bgr, bh, sp, it, grid, pen, ok


def main():
    L = pd.read_csv(f"{ROOT}/label.csv"); L["seq"] = L.filename.str.extract(r"^(\w+?_\w+?)_")[0]
    sample = []
    for s, gp in L.groupby("seq"):
        gp = gp.sort_values("timestep"); n = len(gp)
        for q in (0.1, 0.5, 0.95):
            sample.append(gp.filename.iloc[int(q * (n - 1))])
    codes, rows = [], []
    for fn in sample:
        seq = fn.rsplit("_", 2)[0]
        e = A[A.seq == seq].iloc[:10]
        ref = dict(px=e.pitch_x.median(), py=e.pitch_y.median(), x0=e.x0.median(), y0=e.y0.median())
        bgr, bh, sp, it, grid, pen, ok = masks(fn, ref)
        code = grid.astype(np.uint8) | (pen.astype(np.uint8) << 1) | (it.astype(np.uint8) << 2)
        codes.append(code)
        dark = (bh > THR) & ~sp
        expl = cv2.dilate(((code > 0)).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        pen_in_it = (pen & it).sum()
        rows.append(dict(filename=fn, seq=seq, level=C.loc[fn, "level"], fit_ok=ok, grid_px=int(grid.sum()), pen_px=int(pen.sum()), interf_px=int(it.sum()),
                         spall_px=int(sp.sum()), dark_px=int(dark.sum()), dark_explained=float((dark & expl).sum() / max(dark.sum(), 1)),
                         dark_unexplained_frac_of_image=float((dark & ~expl).mean()),
                         pen_in_interf=float(pen_in_it / max(pen.sum(), 1)), interf_with_pen=float(pen_in_it / max(it.sum(), 1)),
                         pen_outside_interf_px=int((pen & ~it).sum())))
    M = pd.DataFrame(rows); M.to_csv(f"{OUT}/grid_pen_mask_metrics.csv", index=False)
    np.savez_compressed(f"{OUT}/grid_pen_mask_sample.npz", filenames=np.array(sample), code=np.stack(codes))
    pd.set_option("display.width", 250)
    print(M.round(3).to_string())
    print(M.groupby(M.level.round(0) >= 3)[["dark_explained", "pen_in_interf", "interf_with_pen"]].mean())
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pick = ["CTR1_NE", "R315_NW", "R615_SE", "R307_NW"]
    fig, ax = plt.subplots(len(pick), 4, figsize=(14, 3.6 * len(pick)))
    for r_, s in enumerate(pick):
        for c_, fn in enumerate([f for f in sample if f.startswith(s)][1:]):
            i = sample.index(fn); im = cv2.imread(f"{ROOT}/images/{fn}")[:, :, ::-1]; cd = codes[i]
            ov = im.copy(); ov[(cd & 1) > 0] = (255, 0, 255); ov[(cd & 2) > 0] = (0, 200, 0); 
            edge = (cv2.dilate((cd & 4).astype(np.uint8), np.ones((3, 3), np.uint8)) - (cd & 4).astype(np.uint8)) > 0; ov[edge] = (255, 160, 0)
            if c_ == 0: ax[r_, 0].imshow(im); ax[r_, 0].set_title(fn, fontsize=8)
            ax[r_, c_ + 1].imshow(ov); ax[r_, c_ + 1].set_title(fn + " overlay", fontsize=8)
        for a_ in ax[r_]: a_.axis("off")
    ax[0, 3].text(0.0, -0.08, "magenta = grid, green = pen, orange outline = interference polygon", transform=ax[0, 3].transAxes, fontsize=8)
    plt.tight_layout(); plt.savefig("results/dataset_audit/grid_pen_masks.png", dpi=70)


if __name__ == "__main__":
    main()
