#!/usr/bin/env python3
"""Per-frame measurements for docs/redesign/dataset_audit.md (items 1, 2, 5).

For every frame of Data/DamagePattern/images (785):
  grid fit   : pitch (px) and phase of the drawn grid, x and y. Black lines = black-hat of V=max(R,G,B)
               (pen is saturated, so one channel stays bright; black ink is dark in all channels), profile averaged over
               valid pixels only (outside spall / interference polygons), then a weighted DFT over pitch 85..120 px.
  alignment  : grid phase shift vs the sequence's first frame and vs the previous frame (wrapped to +-pitch/2);
               template matching (NCC) of the bottom strip (foundation stub, rod, rebar hoop) vs the first / previous frame.
  photometry : grey statistics on the concrete region (not spall, not interference, not grid line, not pen),
               full resolution and after the generator's resize-128 + CLAHE.
Writes Data/DamagePattern/derived/frame_audit.csv. Does not modify existing files. torch on cuda if available.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, cv2, torch
import torch.nn.functional as F

ROOT = "Data/DamagePattern"
OUT = f"{ROOT}/derived"
H, W = 800, 750
BY0, BY1 = 660, 800   # bottom strip: foundation stub / threaded rod / rebar hoop, assumed static in the world


def poly_masks(fn):
    d = json.load(open(f"{ROOT}/anylabel/{fn[:-4]}.json"))
    sp = np.zeros((H, W), np.uint8); it = np.zeros((H, W), np.uint8)
    for s in d["shapes"]:
        pts = np.round(np.array(s["points"])).astype(np.int32)
        cv2.fillPoly(sp if s["label"] == "spall" else it, [pts], 1)
    return sp, it


def load(fn):
    bgr = cv2.imread(f"{ROOT}/images/{fn}")
    rgb = bgr[:, :, ::-1].copy()
    assert rgb.shape[:2] == (H, W), (fn, rgb.shape)
    sp, it = poly_masks(fn)
    g8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g128 = cv2.resize(g8, (128, 128), interpolation=cv2.INTER_AREA)
    g128 = cv2.createCLAHE(2.0, (8, 8)).apply(g128)
    return rgb, sp, it, g128, g8[BY0:BY1].copy()


def blackhat(v, k=31):
    p = k // 2
    dil = F.max_pool2d(v, k, 1, p)
    ero = -F.max_pool2d(-dil, k, 1, p)
    return (ero - v).clamp(min=0)


PITCHES = torch.arange(85.0, 120.0, 0.1)


def dft_fit(sig, w, dev):
    """sig, w: (B, n). Weighted DFT over candidate pitches. Returns pitch, line offset x0 in [0,pitch), strength."""
    n = sig.shape[1]
    x = torch.arange(n, device=dev, dtype=torch.float32)
    s = (sig - (sig * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1e-6)) * w
    f = (1.0 / PITCHES.to(dev))  # (P,)
    ph = 2 * np.pi * f[:, None] * x[None]  # (P, n)
    c = torch.complex(torch.cos(ph), -torch.sin(ph))  # e^{-i ph}
    coef = s.to(torch.complex64) @ c.T  # (B, P)
    mag = coef.abs()
    k = mag.argmax(1)
    cb = coef[torch.arange(len(k)), k]
    p = PITCHES.to(dev)[k]
    x0 = (-torch.angle(cb) / (2 * np.pi) * p) % p
    strength = mag[torch.arange(len(k)), k] / (s.abs().sum(1) + 1e-6)
    return p, x0, strength


def wrap(d, p):
    return (d + p / 2) % p - p / 2


def strip_shift(ref, cur, pad=40):
    """Shift of the bottom strip `cur` relative to `ref` by normalised cross-correlation of the central part."""
    t = ref[20:-8, 60:-60]
    r = cv2.matchTemplate(cur, t, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(r)
    return loc[0] - 60, loc[1] - 20, mx


def phase_corr(a, b):
    """Translation of b relative to a (B,H,W): returns (dy, dx) so that b(x) ~ a(x - d)."""
    A = torch.fft.rfft2(a); B = torch.fft.rfft2(b)
    R = B * A.conj(); R = R / (R.abs() + 1e-6)
    r = torch.fft.irfft2(R, s=a.shape[-2:])
    idx = r.flatten(1).argmax(1)
    hh, ww = a.shape[-2:]
    dy = (idx // ww).float(); dx = (idx % ww).float()
    dy = torch.where(dy > hh / 2, dy - hh, dy); dx = torch.where(dx > ww / 2, dx - ww, dx)
    peak = r.flatten(1).max(1).values
    return dy, dx, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=f"{OUT}/frame_audit.csv")
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L = pd.read_csv(f"{ROOT}/label.csv")
    L["seq"] = L.filename.str.extract(r"^(\w+?_\w+?)_")[0]
    L["idx"] = L.filename.str.extract(r"_(\d+)\.jpg")[0].astype(int)
    L = L.sort_values(["seq", "timestep", "idx"]).reset_index(drop=True)
    if a.only:
        L = L[L.seq.isin(a.only.split(","))].reset_index(drop=True)
    rows = []
    first = {}      # seq -> (grid map of first frame band, rod patch)
    prev = {}
    for b0 in range(0, len(L), a.bs):
        chunk = L.iloc[b0:b0 + a.bs]
        data = [load(f) for f in chunk.filename]
        rgb = torch.from_numpy(np.stack([d[0] for d in data])).to(dev).permute(0, 3, 1, 2).float()
        sp = torch.from_numpy(np.stack([d[1] for d in data])).to(dev).bool()
        it = torch.from_numpy(np.stack([d[2] for d in data])).to(dev).bool()
        g128 = torch.from_numpy(np.stack([d[3] for d in data])).to(dev).float()
        v = rgb.max(1).values[:, None]
        gray = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2])
        sat = (rgb.max(1).values - rgb.min(1).values)
        gm = blackhat(v)[:, 0]
        bad = sp | it
        # dilate bad by 9 px so polygon borders do not leak
        bad = F.max_pool2d(bad[:, None].float(), 9, 1, 4)[:, 0] > 0
        good = (~bad).float()
        # edges (left/right 6%) excluded: dark background
        good[:, :, :40] = 0; good[:, :, -40:] = 0
        # x profile (use rows 0..H), y profile
        gx = (gm * good).sum(1) / good.sum(1).clamp(min=1)
        wx = (good.sum(1) > 0.3 * H).float()
        gy = (gm * good).sum(2) / good.sum(2).clamp(min=1)
        wy = (good.sum(2) > 0.3 * W).float()
        px, x0, sx = dft_fit(gx, wx, dev)
        py, y0, sy = dft_fit(gy, wy, dev)
        # grid line mask: black-hat strong and near fitted lines
        thr = 25.0
        xs = torch.arange(W, device=dev).float()[None]; ys = torch.arange(H, device=dev).float()[None]
        dxl = wrap(xs - x0[:, None], px[:, None]).abs(); dyl = wrap(ys - y0[:, None], py[:, None]).abs()
        near = ((dxl[:, None, :] < 4) | (dyl[:, :, None] < 4))
        gridline = (gm > thr) & near
        penmask = (sat > 60) & ~gridline
        conc = (~bad) & (~(gm > thr)) & (sat < 45)
        conc[:, :, :40] = False; conc[:, :, -40:] = False
        # grid band map for alignment (upper band, good only)
        for i, r in enumerate(chunk.itertuples()):
            s = r.seq
            c = conc[i]
            vals = gray[i][c]
            q = torch.quantile(vals[::7], torch.tensor([0.1, 0.5, 0.7, 0.9], device=dev)) if vals.numel() > 1000 else torch.full((4,), float("nan"), device=dev)
            rgbm = [rgb[i, k][c].mean().item() if vals.numel() > 1000 else float("nan") for k in range(3)]
            gl = gm[i][gridline[i]]
            gc = g128[i]
            cm = F.interpolate(c[None, None].float(), size=(128, 128), mode="area")[0, 0] > 0.9
            cv = gc[cm]
            q128 = torch.quantile(cv, torch.tensor([0.1, 0.5, 0.7, 0.9], device=dev)) if cv.numel() > 100 else torch.full((4,), float("nan"), device=dev)
            row = dict(filename=r.filename, seq=s, timestep=r.timestep,
                       pitch_x=px[i].item(), x0=x0[i].item(), str_x=sx[i].item(),
                       pitch_y=py[i].item(), y0=y0[i].item(), str_y=sy[i].item(),
                       valid_frac=good[i].mean().item(), conc_frac=c.float().mean().item(),
                       g_p10=q[0].item(), g_p50=q[1].item(), g_p70=q[2].item(), g_p90=q[3].item(),
                       g_std=vals.std().item() if vals.numel() > 1000 else float("nan"),
                       mean_r=rgbm[0], mean_g=rgbm[1], mean_b=rgbm[2],
                       grid_dark=gl.mean().item() if gl.numel() else float("nan"),
                       grid_px=int(gridline[i].sum().item()), pen_px=int(penmask[i].sum().item()),
                       c128_p10=q128[0].item(), c128_p50=q128[1].item(), c128_p70=q128[2].item(), c128_p90=q128[3].item())
            strip = data[i][4]
            if s not in first:
                first[s] = dict(strip=strip, x0=row["x0"], y0=row["y0"], px=row["pitch_x"], py=row["pitch_y"])
                prev[s] = dict(strip=strip, x0=row["x0"], y0=row["y0"])
            f0, pv = first[s], prev[s]
            row["gdx_ref"] = wrap(torch.tensor(row["x0"] - f0["x0"]), torch.tensor(f0["px"])).item()
            row["gdy_ref"] = wrap(torch.tensor(row["y0"] - f0["y0"]), torch.tensor(f0["py"])).item()
            row["gdx_prev"] = wrap(torch.tensor(row["x0"] - pv["x0"]), torch.tensor(f0["px"])).item()
            row["gdy_prev"] = wrap(torch.tensor(row["y0"] - pv["y0"]), torch.tensor(f0["py"])).item()
            for nm, ref in (("st_ref", f0["strip"]), ("st_prev", pv["strip"])):
                dx_, dy_, mx = strip_shift(ref, strip)
                row[nm + "_dx"] = dx_; row[nm + "_dy"] = dy_; row[nm + "_ncc"] = mx
            prev[s] = dict(strip=strip, x0=row["x0"], y0=row["y0"])
            rows.append(row)
        print(b0, len(L), flush=True)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
