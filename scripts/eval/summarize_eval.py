#!/usr/bin/env python3
"""One Markdown report from the three evaluation outputs of one split.

Reads (any of them may be missing; its columns are then left out):
  <kid_dir>/dinov3_kid.json          eval_dinov3_kid.py (main backbone)
  <kid_extra_dir>/dinov3_kid.json    the same with other backbones → robustness table
  <texture_dir>/damage_texture.json  eval_damage_texture.py
  <temporal_dir>/temporal.json       eval_temporal.py

Writes one table per metric family, the reference rows next to them, and every model's paired
difference against --baseline with its z. Models whose condition is the DINOv3 feature
(oracle_dino) get their DINOv3-based numbers in brackets: those are circular (docs/eval/oracle.md);
with another backbone they are shown plainly.
How to read every column: docs/eval/README.md.

Example:
  python scripts/eval/summarize_eval.py --kid_dir out/test/kid --texture_dir out/test/texture \
      --temporal_dir out/test/temporal --baseline gn --title "test (CTR1)" --out out/test/summary.md
"""

from __future__ import annotations

import argparse
import json
import os


def _parse_args():
    p = argparse.ArgumentParser(description="Markdown summary of one split")
    p.add_argument("--kid_dir", default="")
    p.add_argument("--kid_extra_dir", nargs="*", default=[], help="eval_dinov3_kid.py outputs of other backbones")
    p.add_argument("--texture_dir", default="")
    p.add_argument("--temporal_dir", default="")
    p.add_argument("--baseline", default="", help="Model every other model is compared with (default: first)")
    p.add_argument("--title", default="")
    p.add_argument("--out", required=True)
    return p.parse_args()


def _load(d, name):
    path = os.path.join(d, name) if d else ""
    if path and os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def _v(d, digits=3, se=True):
    """{value, se} → '0.123 ± 0.010'; plain numbers pass through; missing → '—'."""
    if d is None:
        return "—"
    if isinstance(d, dict):
        val = d.get("value")
        if val is None:
            return "—"
        s = d.get("se")
        return f"{val:.{digits}f} ± {s:.{digits}f}" if (se and s is not None) else f"{val:.{digits}f}"
    return f"{d:.{digits}f}"


def _pct(x):
    return "—" if x is None else f"{100 * x:.0f}%"


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _cmp(comparisons, a, b, key):
    """Paired difference a − b from a comparisons dict keyed 'a_vs_b' (either order)."""
    if not comparisons:
        return "—"
    if f"{a}_vs_{b}" in comparisons and key in comparisons[f"{a}_vs_{b}"]:
        c, sign = comparisons[f"{a}_vs_{b}"][key], 1
    elif f"{b}_vs_{a}" in comparisons and key in comparisons[f"{b}_vs_{a}"]:
        c, sign = comparisons[f"{b}_vs_{a}"][key], -1
    else:
        return "—"
    return f"{sign * c['diff']:+.3f} (z = {sign * c['z']:+.1f})"


def _backbone(kid):
    return kid.get("dino_model_id", "").split("/")[-1] or "DINOv3"


def _robustness(extras, names, base):
    """Main realism / condition numbers under every other backbone. KID is divided by the other real
    column's KID, since raw KID scales differ by orders of magnitude between backbones."""
    md = ["", "### 換 backbone 的穩健性檢查", "",
          "同一批樣本換特徵空間重算。KID 的原始尺度在不同 backbone 間差很多，這裡除以「另一根真實柱」的 KID"
          "（test 才有；val 顯示原始值）。排名和配對 z 跟主表一致，結論才不依賴 backbone。", ""]
    for kid in extras:
        other = (kid.get("reference_rows", {}).get("other_real_val_kid") or {}).get("value")
        comp = kid.get("comparisons", {})
        rows = []
        for n in names:
            m = kid["models"].get(n)
            if m is None:
                continue
            br = (lambda x: f"({x})") if m.get("circular_eval_warning") else (lambda x: x)
            k = m["kid_global"]["value"]
            pg = m.get("progression_global", {})
            rows.append([n, f"{k / other:.2f}" if other else f"{k:.3f}",
                         br(_pct(m.get("nbr_global", {}).get("frac_of_ceiling"))),
                         br(_v(pg.get("ratio"), 2, se=False)), br(_v(pg.get("corr"), 2, se=False)),
                         "—" if n == base else _cmp(comp, n, base, "kid_global").split("z = ")[-1].rstrip(")"),
                         "—" if n == base else _cmp(comp, n, base, "nbr_global").split("z = ")[-1].rstrip(")")])
        md += [f"**{_backbone(kid)}**", ""]
        md.append(_table(["模型", "KID ÷ 另一根柱 ↓" if other else "KID ↓", "GT 鄰居對比（達上限）↑",
                          "演進 ratio →1", "演進 corr ↑", f"KID 和 {base} 的 z", f"鄰居對比和 {base} 的 z"], rows))
        md.append("")
    return md


def main():
    cli = _parse_args()
    kid = _load(cli.kid_dir, "dinov3_kid.json")
    extras = [e for e in (_load(d, "dinov3_kid.json") for d in cli.kid_extra_dir) if e]
    tex = _load(cli.texture_dir, "damage_texture.json")
    tmp = _load(cli.temporal_dir, "temporal.json")
    names = []
    for src in (kid and kid["models"], tex and [k for k in tex["texture"] if k not in ("gt", "train_real", "other_real")],
                tmp and tmp["models"]):
        for n in src or []:
            if n not in names:
                names.append(n)
    if not names:
        raise SystemExit("no evaluation output found")
    base = cli.baseline or names[0]
    md = [f"# 評估摘要{'：' + cli.title if cli.title else ''}", ""]
    md.append(f"配對比較的基準模型：**{base}**。箭頭：↓ 越低越好，↑ 越高越好，→1 / →GT 越接近越好。"
              "各欄定義見 `docs/eval/README.md`。")
    circular = {n for n in names if kid and kid["models"].get(n, {}).get("circular_eval_warning")}
    if circular:
        md.append(f"\n**循環評估**（條件就是 DINOv3 特徵，括號內的數字不能拿來比，見 `docs/eval/oracle.md`）：{', '.join(sorted(circular))}。")

    if kid:
        md += ["", f"## 真實感與條件一致性（{_backbone(kid)}）", ""]
        rows = []
        for n in names:
            m = kid["models"].get(n)
            if m is None:
                continue
            br = (lambda s: f"({s})") if n in circular else (lambda s: s)
            pg = m.get("progression_global", {})
            top = (m.get("probe_spall_top") or {}).get("probe_on_fake")
            rows.append([
                n, _v(m["kid_global"]), _v(m["kid_patch"]),
                br(_v(pg.get("ratio"), 2)), br(_v(pg.get("corr"), 2)),
                br(_pct(m.get("nbr_global", {}).get("frac_of_ceiling"))),
                br(f"{m['probe']['R_spall']['r2']:.2f}") if "probe" in m else "—",
                br("—" if top is None else f"{top:.2f}"),
                _pct(m.get("saturated_frames", {}).get("fake_frac")),
            ])
        ref = kid.get("reference_rows", {})
        po = ref.get("progression_global_other_specimens", {})
        rows.append(["*另一根真實柱*", _v(ref.get("other_real_val_kid"), se=False), "", _v(po.get("ratio"), 2),
                     _v(po.get("corr"), 2), "100%", "", "", "0%"])
        md.append(_table(["模型", "KID ↓", "patch KID ↓", "演進 ratio →1", "演進 corr ↑",
                          "GT 鄰居對比（達上限）↑", "R_spall probe R² ↑", "高剝落條件的剝落讀數 ↑", "飽和幀 ↓"], rows))
        hs = ref.get("probe_on_heldout_reals", {})
        if hs:
            md.append(f"\nprobe 用在真實圖（上限）：R_spall R² = {hs['R_spall']['r2']:.2f}。")
        comp = kid.get("comparisons", {})
        rows = [[f"{n} − {base}", _cmp(comp, n, base, "kid_global"), _cmp(comp, n, base, "nbr_global"),
                 _cmp(comp, n, base, "progression_global"), _cmp(comp, n, base, "progression_global_corr")]
                for n in names if n != base and n in kid["models"]]
        if rows:
            md += ["", f"和 {base} 的配對差（區塊 jackknife；|z| ≥ 2 約為顯著；演進指標只有約 7 個單位，要更保守）：", ""]
            md.append(_table(["比較", "KID", "GT 鄰居對比", "演進 ratio", "演進 corr"], rows))
        if extras:
            md += _robustness(extras, names, base)

    if tex:
        md += ["", "## 損傷型態（像素統計，高 DI 三分位）", ""]
        t = tex["texture"]
        keys = ("damage_area", "fog_share", "white_share", "black_share", "E_damage")
        bl = tex["cracks"].get("before_loading", {})

        def row(n, src, cell):
            return [n] + [cell(src["high"][k]) for k in keys]

        rows = [row("*GT*", t["gt"], lambda d: _v(d, se=False))]
        for r in ("train_real", "other_real"):
            if r in t and t[r]:
                rows.append(row({"train_real": "*訓練柱*", "other_real": "*另一根真實柱*"}[r], t[r], lambda d: _v(d, se=False)))
        for n in names:
            if n in t:
                rows.append(row(n, t[n], lambda d: _v(d["model"])))
        hdr = ["模型", "損傷面積 →GT", "灰霧比例 ↓", "白塊比例 →GT", "極暗比例 →GT", "細紋理能量 E →GT"]
        md.append(_table(hdr, rows))
        cr = tex["cracks"]
        rows = [["*GT*", "—", _v(cr["gt"]["f1_train_max"], se=False), "—", f"{bl.get('gt', float('nan')):.4f}"]]
        for n in names:
            if n in cr:
                c = cr[n]
                rows.append([n, _v(c["f1_gt"]), _v(c["f1_train_max"]), _v(c["copy_excess"]),
                             f"{bl.get(n, float('nan')):.4f}"])
        md += ["", f"裂縫（低、中 DI 共 {cr['n_conditions']} 個條件）：", ""]
        md.append(_table(["模型", "和自己 GT 的裂縫 F1 ↑", "和最像訓練圖的 F1", "沿用訓練柱程度 copy excess ↓",
                          "加載前的裂縫面積 →GT"], rows))
        comp = tex.get("composite")
        if comp:
            rows = [[n, _v(comp["models"][n]["score"], 2)] for n in names if n in comp["models"]]
            md += ["", f"損傷型態綜合偏差（相對 GT 的 RMS z，↓；真實訓練柱的中位數是 "
                       f"{comp['real_column_median_score']:.2f}）：", ""]
            md.append(_table(["模型", "分數"], rows))

    if tmp:
        md += ["", "## 時序一致性（同一面的相鄰幀）", ""]
        keys = ("flicker", "tone_jump", "struct_flicker", "persistence", "crack_keep", "regress")
        g = tmp["gt"]
        rows = [["*GT*"] + [_v(g[k], 2, se=False) for k in keys] + ["1"]]
        for n in names:
            if n in tmp["models"]:
                m = tmp["models"][n]
                rows.append([n] + [_v(m[k], 2) for k in keys] + [_v(m["flicker_ratio"], 2, se=False)])
        md.append(_table(["模型", "相鄰幀變化 →GT", "色調跳動 →GT", "結構閃爍 →GT", "損傷保留 ↑",
                          "裂縫保留 ↑", "損傷倒退 ↓", "變化量 / GT →1"], rows))
        md.append("\n完全不動的序列也會得到很高的保留率，所以要和上面的演進 ratio 一起看。")

    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    with open(cli.out, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[summary] wrote {cli.out}")


if __name__ == "__main__":
    main()
