// Progress deck 2026-11-10 -> results/deck_20261110/progress_20261110.pptx
// NODE_PATH=<dir with pptxgenjs> node scripts/viz/build_deck_20261110.js
// Figures: scripts/viz/deck_figures.py, scripts/viz/deck_copy_example.py, drift panels in results/eval_suite/meeting/.
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..");
const FIG = path.join(ROOT, "results/deck_20261110/figures");
const PAN = path.join(ROOT, "results/eval_suite/meeting/drift_panels");
const OUT = path.join(ROOT, "results/deck_20261110/progress_20261110.pptx");
const EXTRA = path.join(ROOT, "results/deck_20261110/extra_rows.json"); // optional V12 jitter rows

const INK = "1F2430", INK2 = "52514E", MUTED = "8A8984", BLUE = "2A78D6", ORANGE = "EB6834";
const TINT = "EAF2FC", CARD = "F3F5F8", LINE = "C9CDD4", WHITE = "FFFFFF";
const FONT = "Calibri";
const W = 13.333, X0 = 0.5, CW = 12.333;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Progress update 2026-11-10";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: WHITE },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: X0, y: 0.3, w: CW, h: 0.8, fontFace: FONT, fontSize: 30,
      bold: true, color: INK, valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.4, y: 7.0, w: 0.5, h: 0.3, fontFace: FONT, fontSize: 12, color: MUTED, align: "right" },
});

const aspect = {
  "app_code_fog.png": 2.956, "copy_crack_example.png": 2.729, "copy_rates.png": 2.972, "copy_spall_example.png": 1.698,
  "encoder_vs_gru.png": 2.06, "gravity_jitter.png": 2.956, "gravity_template.png": 4.253, "hspP_clock.png": 2.834,
  "refiner_s_use.png": 3.087, panel: 1.054,
};

function newSlide(title) {
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText(title, { placeholder: "title" });
  return s;
}

function img(s, file, x, y, w, key, alt) {
  const a = aspect[key || path.basename(file)];
  s.addImage({ path: file, x, y, w, h: w / a, altText: alt || path.basename(file) });
  return y + w / a;
}

// items: [["keyword", " rest of the sentence."], ...]
function bullets(s, items, x, y, w, h, size = 18) {
  const runs = [];
  items.forEach((it, i) => {
    const last = i === items.length - 1;
    runs.push({ text: it[0], options: { bold: true, color: INK, bullet: { indent: 18 }, paraSpaceAfter: 6 } });
    runs.push({ text: it[1], options: { color: INK, breakLine: !last } });
  });
  s.addText(runs, { x, y, w, h, fontFace: FONT, fontSize: size, valign: "top", margin: 0, isTextBox: true });
}

function takeaway(s, text, y = 6.5) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: X0, y, w: CW, h: 0.62, fill: { color: TINT }, line: { color: TINT },
    rectRadius: 0.08, objectName: "takeaway" });
  s.addText(text, { x: X0 + 0.25, y, w: CW - 0.5, h: 0.62, fontFace: FONT, fontSize: 18, bold: true, color: INK,
    valign: "middle", margin: 0, isTextBox: true });
}

function box(s, text, x, y, w, h, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: o.fill || CARD }, line: { color: o.line || LINE, width: 1 },
    rectRadius: 0.06 });
  s.addText(text, { x: x + 0.06, y, w: w - 0.12, h, fontFace: FONT, fontSize: o.size || 15, color: o.color || INK,
    bold: !!o.bold, align: "center", valign: "middle", margin: 0, isTextBox: true });
}

function arrow(s, x1, y1, x2, y2, color = INK2) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001,
    h: Math.abs(y2 - y1) || 0.001, flipH: x2 < x1, flipV: y2 < y1,
    line: { color, width: 1.75, endArrowType: "triangle" } });
}

function label(s, text, x, y, w, h, o = {}) {
  s.addText(text, { x, y, w, h, fontFace: FONT, fontSize: o.size || 18, color: o.color || INK, bold: !!o.bold,
    italic: !!o.italic, align: o.align || "left", valign: o.valign || "top", margin: 0, isTextBox: true });
}

// table: header row + rows; ref rows in italics; bold the per-column best given by `best` (row index per col)
function table(s, header, rows, x, y, w, colW, o = {}) {
  const head = header.map((t) => ({ text: t, options: { bold: true, color: INK, fill: { color: TINT }, align: "center",
    valign: "middle" } }));
  const body = rows.map((r) => r.cells.map((c, j) => ({ text: String(c), options: {
    color: r.ref ? INK2 : INK, italic: !!r.ref, bold: (o.best && o.best[j] === r.name) || (j === 0 && !r.ref),
    align: j === 0 ? "left" : "center", valign: "middle", fill: { color: r.hl ? "FDF0EA" : WHITE } } })));
  s.addTable([head, ...body], { x, y, w, colW, fontFace: FONT, fontSize: o.size || 18, border: { type: "solid", pt: 0.75, color: LINE },
    rowH: o.rowH || 0.42, margin: [0.03, 0.06, 0.03, 0.06] });
}

// ======================================================================= 1. copy metrics
{
  const s = newSlide("Copy Metrics: Is the Generator Copying Training Columns?");
  img(s, path.join(FIG, "copy_crack_example.png"), X0, 1.25, 7.3);
  img(s, path.join(FIG, "copy_rates.png"), X0, 4.0, 7.3);
  img(s, path.join(FIG, "copy_spall_example.png"), 8.1, 1.25, 4.73);
  bullets(s, [
    ["Crack copy:", " best F1 vs 533 train images."],
    ["Spall copy:", " best NCC of 16×16 spall patches vs ~10k train patches."],
    ["Thresholds:", " from unseen CTR1; R307 = the not-copying range."],
  ], 8.1, 4.15, 4.73, 2.3);
  takeaway(s, "cGAN copies cracks (0.87) and spall texture (0.26); ours copy only cracks, and mildly.", 6.55);
}

// ======================================================================= 2. protocol
{
  const s = newSlide("Evaluation Protocol: What Each Metric Answers");
  const cards = [
    ["Realism", ["KID ↓ (DINOv3)", "FID ↓ (Inception)"]],
    ["Condition", ["Progression ratio → 1", "Progression corr ↑", "GT-neighbour contrast ↑"]],
    ["Damage content", ["Damage area → GT", "Texture deviation ↓"]],
    ["Copy", ["Crack copy rate", "Spall-content copy rate", "Both within the real range"]],
    ["Diversity & temporal", ["Spread / skill → 1", "1 − mask IoU", "Damage regress → GT", "Change ratio → 1"]],
  ];
  const cw = 2.4, gap = (CW - 5 * cw) / 4;
  cards.forEach(([h, ms], i) => {
    const x = X0 + i * (cw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.3, w: cw, h: 3.55, fill: { color: CARD }, line: { color: CARD }, rectRadius: 0.08 });
    s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: 1.55, w: 0.48, h: 0.48, fill: { color: i === 0 ? ORANGE : BLUE }, line: { color: WHITE } });
    label(s, String(i + 1), x + 0.2, 1.55, 0.48, 0.48, { size: 18, bold: true, color: WHITE, align: "center", valign: "middle" });
    label(s, h, x + 0.2, 2.1, cw - 0.3, 0.8, { size: 20, bold: true, valign: "middle" });
    s.addText(ms.map((m, k) => ({ text: m, options: { breakLine: k < ms.length - 1, paraSpaceAfter: 8 } })),
      { x: x + 0.2, y: 2.95, w: cw - 0.25, h: 1.85, fontFace: FONT, fontSize: 18, color: INK, valign: "top", margin: 0, isTextBox: true });
  });
  const nw = (CW - 0.3) / 2;
  [["Realism ≠ condition.", " KID and FID do not see the condition: moving samples to random conditions leaves them unchanged, while progression and neighbour contrast fall to ≈ 0."],
   ["One image domain.", " Every model goes through CLAHE → interference removal → JPEG q75, and every number is read against a real reference: another column, GT, or the training columns."]]
    .forEach(([k, t], i) => {
      const x = X0 + i * (nw + 0.3);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.05, w: nw, h: 1.7, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
      s.addText([{ text: k, options: { bold: true } }, { text: t }], { x: x + 0.25, y: 5.05, w: nw - 0.5, h: 1.7, fontFace: FONT,
        fontSize: 18, color: INK, valign: "middle", margin: 0, isTextBox: true });
    });
}

// ======================================================================= 3. V11 encoder
{
  const s = newSlide("V11: A Damage-State Encoder Replaces the GRU State");
  box(s, "Simulated loop →\nhalf-cycle tokens (9)", 0.5, 1.45, 2.15, 0.85);
  box(s, "Causal 2-layer GRU", 3.05, 1.45, 1.7, 0.85, { fill: TINT, line: BLUE, bold: true });
  box(s, "z_k (256)", 5.15, 1.45, 1.1, 0.85, { fill: TINT, line: BLUE, bold: true });
  box(s, "Static descriptors\n+ column info", 0.5, 2.75, 2.15, 0.85);
  box(s, "MLP → s (64)", 3.05, 2.75, 1.7, 0.85);
  arrow(s, 2.65, 1.875, 3.05, 1.875); arrow(s, 4.75, 1.875, 5.15, 1.875);
  arrow(s, 2.65, 3.175, 3.05, 3.175); arrow(s, 3.9, 2.75, 3.9, 2.3);
  box(s, "Heads: strength, stiffness, energy, peak force, residual drift", 0.5, 3.95, 5.75, 0.6, { size: 14 });
  arrow(s, 5.7, 2.3, 5.7, 3.95);
  label(s, "h = e_c(t, face, sign) + α·e_g(z) + α_d·e_d(scalars)", 0.5, 4.7, 6.1, 0.4, { size: 18, italic: true, color: INK2 });
  img(s, path.join(FIG, "encoder_vs_gru.png"), 6.75, 1.3, 6.08);
  bullets(s, [
    ["Old g:", " 37% of its variance is loading direction, and the bidirectional pass leaks future cycles."],
    ["New z:", " causal and sign-free (2%); R307 spall error drops from +0.28 to +0.17 at 8% drift."],
  ], X0, 5.25, CW, 1.15);
  takeaway(s, "z removes direction noise and fixes R307, but CTR1 spalling and peak force do not improve.", 6.55);
}

// ======================================================================= 4. V11-App
{
  const s = newSlide("V11-App: Separating Appearance from Damage");
  box(s, "8 training sequences\n(column × face)", 0.5, 1.45, 2.0, 0.95);
  box(s, "8-d code each,\nown zero-init gate", 2.85, 1.45, 2.1, 0.95, { fill: TINT, line: BLUE, bold: true });
  box(s, "Unseen column", 0.5, 2.85, 2.0, 0.95);
  box(s, "Mean of 8 codes\n(no test photo)", 2.85, 2.85, 2.1, 0.95, { fill: TINT, line: BLUE, bold: true });
  arrow(s, 2.5, 1.925, 2.85, 1.925); arrow(s, 2.5, 3.325, 2.85, 3.325);
  img(s, path.join(FIG, "app_code_fog.png"), 5.15, 1.3, 7.68);
  bullets(s, [
    ["Fog came from the condition:", " z and the damage scalars also carried sequence texture, turning high-DI damage grey."],
    ["A per-sequence appearance code", " absorbs that texture during training, so the shared condition only has to carry damage."],
    ["Fog returns to the GT level", " (0.08 vs 0.086) and fine texture recovers; temporal metrics reach GT level after the refiner."],
  ], X0, 4.2, CW, 2.1);
  takeaway(s, "The appearance code fixes the fog, but the amount of spalling on CTR1 is still too small.", 6.55);
}

// ======================================================================= 5. hsp_P
{
  const s = newSlide("hsp_P: A Fiber-Section Clock for the Amount of Spalling");
  const steps = ["Column parameters + simulated cycles", "Fiber section (Mander concrete, bilinear steel)",
    "Plastic-hinge curvature φ_b", "Spall height hsp_P (mm)", "Isotonic f → spall area A"];
  steps.forEach((t, i) => {
    box(s, t, 0.5, 1.35 + i * 0.6, 5.2, 0.45, { fill: i === 3 ? TINT : CARD, line: i === 3 ? BLUE : LINE, bold: i === 3, size: 15 });
    if (i < steps.length - 1) arrow(s, 3.1, 1.8 + i * 0.6, 3.1, 1.95 + i * 0.6);
  });
  label(s, "hsp_P = L_p · (φ_b − φ_sp) / (φ_b − φ_y)", 0.5, 4.4, 5.2, 0.4, { size: 18, italic: true, color: INK2 });
  img(s, path.join(FIG, "hspP_clock.png"), 6.0, 1.25, 6.83);
  table(s, ["Clock: area bias", "CTR1 6%", "CTR1 8%", "R307 6%", "R307 8%"], [
    { name: "max_drift", cells: ["max drift", "−0.134", "−0.088", "+0.140", "+0.303"] },
    { name: "eps_max", cells: ["outer strain", "−0.030", "+0.002", "+0.152", "+0.293"] },
    { name: "hsp_P", cells: ["hsp_P", "+0.034", "−0.002", "−0.036", "−0.101"], hl: true },
  ], 6.0, 3.8, 6.83, [2.03, 1.2, 1.2, 1.2, 1.2], { size: 18, rowH: 0.37 });
  bullets(s, [
    ["Not learnable from images:", " the 4 training columns share axial load, hoops and section."],
    ["R307:", " more strength loss than CTR1 (0.43 vs 0.66 at 8%) but far less spalling."],
    ["Fix:", " the plastic-hinge term L_sp = 0.022 f_y d_b (thinner bars) corrects R307 in 27/27 settings."],
  ], X0, 5.35, CW, 1.15);
  takeaway(s, "Mechanics decides how much spalls; R307 is fixed by the plastic-hinge length, not the strain level.", 6.6);
}

// ======================================================================= 6. gravity basis
{
  const s = newSlide("Gravity Basis: Where the Spalling Goes");
  img(s, path.join(FIG, "gravity_template.png"), X0, 1.3, 9.0);
  table(s, ["High-DI IoU", "CTR1", "R307"], [
    { name: "t", cells: ["Template", "0.565", "0.359"], hl: true },
    { name: "o", cells: ["OFP shape", "0.528", "0.196"] },
  ], 9.85, 1.6, 2.98, [1.4, 0.79, 0.79]);
  bullets(s, [
    ["One fill order for every column and face:", " cells spall in order of the training spall frequency T, bottom first."],
    ["Area from the clock:", " each frame takes the first round(A · 32²) cells, turned into a truncated SDF as an extra U-Net input."],
    ["Train on exact layouts:", " with predicted layouts the model ignores the channel; with true layouts it follows the clock at inference."],
    ["Result:", " high-DI spall bias −0.001 on both held-out columns (CTR1 and R307)."],
  ], X0, 3.75, CW, 2.6);
  takeaway(s, "The clock decides how much, the template decides where; training must use the true layout.", 6.55);
}

// ======================================================================= 7. panels (figures only)
for (const [face, faceName] of [["NE", "NE face"], ["SW", "SW face"]]) {
  const s = newSlide(`Drift × DI Panels: V11 vs V12 (CTR1, ${faceName})`);
  const pw = 3.95, gap = (CW - 3 * pw) / 2;
  [["GT", `drift_di_gt_${face}.png`], ["V11 (app_r_6k)", `drift_di_v11_app_r_6k_${face}.png`],
   ["V12 (template + jitter)", `drift_di_v12gtrefiner_gravnoise_k0_${face}.png`]].forEach(([t, f], i) => {
    const x = X0 + i * (pw + gap);
    label(s, t, x, 1.2, pw, 0.4, { size: 20, bold: true, align: "center" });
    img(s, path.join(PAN, f), x, 1.65, pw, "panel", `${t} drift x DI panel ${face}`);
  });
  label(s, "V12 layout: gravity template + boundary jitter (same hsp_P area, a different boundary per sample).",
    X0, 5.6 + 0.15, CW, 0.4, { size: 18, color: INK2 });
}

// ======================================================================= 8. tables (tables only)
let extra = {};
if (fs.existsSync(EXTRA)) extra = JSON.parse(fs.readFileSync(EXTRA, "utf8"));
{
  const s = newSlide("Results (test, CTR1): Realism and Condition");
  const rows = [
    { name: "V10", cells: ["V10 refiner", "0.113", "128.5", "0.43", "0.75", "87%", "0.108", "1.06"] },
    { name: "V10.6", cells: ["V10.6C", "0.094", "117.9", "0.44", "0.88", "93%", "0.131", "1.09"] },
    { name: "V11", cells: ["V11", "0.090", "117.7", "0.53", "0.90", "116%", "0.097", "1.06"] },
    { name: "V12", cells: ["V12", "0.087", "110.0", "0.61", "0.91", "133%", "0.146", "1.19"] },
  ];
  if (extra.test_a) rows.push({ name: "V12j", cells: extra.test_a });
  rows.push({ name: "cGAN", cells: ["cGAN", "0.121", "133.1", "0.40", "0.56", "59%", "0.057", "2.49"] });
  rows.push({ name: "ref", ref: true, cells: ["Real reference", "0.072", "115.4", "0.53", "0.83", "100%", "0.164", "0 (train 1.20)"] });
  table(s, ["Model", "KID ↓", "FID ↓", "Progression ratio → 1", "Progression corr ↑", "GT-neighbour ↑", "Damage area → GT", "Texture deviation ↓"],
    rows, X0, 1.4, CW, [2.1, 1.25, 1.25, 1.6, 1.55, 1.5, 1.5, 1.578],
    { best: [null, "V12", "V12", "V12", "V12", "V12", "V12", "V10"], rowH: 0.5 });
  label(s, "FID: mean of 8 single-draw scores (comparable to cGAN's K = 1). Real reference = another real column (R307) for KID, FID, progression and neighbour contrast; GT for damage area.",
    X0, 6.15, CW, 0.8, { size: 18, color: INK2 });
}
{
  const s = newSlide("Results (test, CTR1): Copy, Diversity, Temporal");
  const rows = [
    { name: "V10", cells: ["V10 refiner", "0.37", "0.09", "0.28", "0.77", "0.33", "2.45"] },
    { name: "V10.6", cells: ["V10.6C", "0.42", "0.03", "0.25", "0.81", "0.26", "1.13"] },
    { name: "V11", cells: ["V11", "0.26", "0.11", "0.27", "0.68", "0.21", "1.08"] },
    { name: "V12", cells: ["V12", "0.52", "0.06", "0.11", "0.70", "0.13", "1.05"] },
  ];
  if (extra.test_b) rows.push({ name: "V12j", cells: extra.test_b });
  rows.push({ name: "cGAN", cells: ["cGAN", "0.87", "0.26", "— (K = 1)", "— (K = 1)", "0.39", "1.99"] });
  rows.push({ name: "ref", ref: true, cells: ["Real reference", "0.22 (R307)", "0.19 (R307)", "1", "0.72 – 0.89", "0.12", "1"] });
  table(s, ["Model", "Crack copy", "Spall-content copy", "Spall spread / skill → 1", "1 − mask IoU", "Damage regress → GT", "Change ratio → 1"],
    rows, X0, 1.4, CW, [2.2, 1.6, 1.8, 1.9, 1.6, 1.65, 1.583], { best: [null, "V11", "V10.6", "V10", null, "V12", "V12"], rowH: 0.5 });
  label(s, "1 − mask IoU range: same face next cycle (0.72, near duplicate) to the other face (0.89).", X0, 6.2, CW, 0.5, { size: 18, color: INK2 });
}
{
  const s = newSlide("Results (val, R307)");
  const rows = [
    { name: "V10", cells: ["V10 refiner", "0.049", "73.3", "0.75", "0.89", "82%", "0.142", "2.24", "0.32", "3.19"] },
    { name: "V10.6", cells: ["V10.6C", "0.033", "71.9", "0.76", "0.88", "79%", "0.141", "2.17", "0.21", "1.39"] },
    { name: "V11", cells: ["V11", "0.025", "63.6", "0.76", "0.91", "116%", "0.087", "1.44", "0.22", "1.09"] },
    { name: "V12", cells: ["V12", "0.028", "65.7", "0.89", "0.95", "114%", "0.053", "1.99", "0.17", "1.14"] },
  ];
  if (extra.val) rows.push({ name: "V12j", cells: extra.val });
  rows.push({ name: "ref", ref: true, cells: ["Real (R307 / others)", "—", "—", "0.96", "0.90", "100%", "0.079", "1.60", "0.29", "1"] });
  table(s, ["Model", "KID ↓", "FID ↓", "Prog. ratio → 1", "Prog. corr ↑", "GT-nbr ↑", "Damage area → GT", "Texture dev. ↓", "Regress → GT", "Change ratio → 1"],
    rows, X0, 1.4, CW, [2.05, 1.05, 1.0, 1.25, 1.2, 1.15, 1.25, 1.2, 1.15, 1.033],
    { best: [null, "V11", "V11", "V12", "V12", "V11", "V11", "V11", "V12", "V11"], rowH: 0.55 });
  label(s, "cGAN is not listed on val: R307 is in its training set.", X0, 6.2, CW, 0.5, { size: 18, color: INK2 });
}

// ======================================================================= 9. refiner
{
  const s = newSlide("Rethinking the Refiner: What Does It Take from S?");
  img(s, path.join(FIG, "refiner_s_use.png"), X0, 1.3, 8.2);
  box(s, "Intended S:\ncached base sample (30 steps)", 9.05, 1.4, 3.78, 0.95, { size: 16 });
  box(s, "Actual S (cache bug):\non-the-fly base sample,\n70% 1-step Euler", 9.05, 2.75, 3.78, 1.05,
    { size: 15, fill: "FDF0EA", line: ORANGE, bold: true });
  arrow(s, 10.94, 2.35, 10.94, 2.75, ORANGE);
  bullets(s, [
    ["Single frame:", " swapping S barely moves the output (gain < 0.07; real data 0.30–0.83)."],
    ["Rollout:", " S from another trajectory drops keep@1 from 0.88 to 0.75, so persistence comes from S."],
    ["Blurry training S:", " the refiner took only coarse structure; with sharp S it relied on S and turned grey."],
    ["Consequence:", " fine structure under-accumulates (lag-24 born/died 2.41 vs 3.46); the probe gap is open."],
  ], X0, 4.1, CW, 2.3);
  takeaway(s, "The refiner does read the previous frame, but only its coarse structure.", 6.55);
}

// ======================================================================= 10. resampling forcing (method)
{
  const s = newSlide("Resampling Forcing: Train on Self-Denoised History");
  box(s, "Real x_{t−1}\nnoised to t_s", 0.5, 1.45, 1.9, 0.95);
  box(s, "Real x_{t−2}\n(as its S)", 0.5, 2.65, 1.9, 0.95);
  box(s, "Model, no gradient\n(few ODE steps)", 2.85, 2.0, 2.1, 1.05, { fill: TINT, line: BLUE, bold: true });
  box(s, "Ŝ_{t−1}: real above t_s,\nown error below", 5.4, 2.0, 2.3, 1.05);
  box(s, "Condition c_t +\nconfidence t_s", 5.4, 3.35, 2.3, 0.8);
  box(s, "Model", 8.15, 2.0, 1.35, 1.05, { fill: TINT, line: BLUE, bold: true });
  box(s, "pred.\nx_t", 9.95, 2.0, 0.9, 1.05);
  box(s, "Loss vs real x_t", 11.3, 2.0, 1.53, 1.05, { fill: "FDF0EA", line: ORANGE });
  arrow(s, 2.4, 1.925, 2.85, 2.35); arrow(s, 2.4, 3.125, 2.85, 2.75);
  arrow(s, 4.95, 2.525, 5.4, 2.525); arrow(s, 7.7, 2.525, 8.15, 2.525); arrow(s, 7.7, 3.75, 8.4, 3.05);
  arrow(s, 9.5, 2.525, 9.95, 2.525); arrow(s, 10.85, 2.525, 11.3, 2.525);
  label(s, "Parallel: every resampled frame reads real t−2 (one-step error)    →    Autoregressive window: each resampled frame reads the previous resampled one (errors accumulate)",
    X0, 4.3, CW, 0.75, { size: 18, color: INK2, italic: true });
  bullets(s, [
    ["Goal:", " the history seen in training should look like the history produced at inference."],
    ["Degrade, then restore:", " structure above t_s stays real; detail below carries the model's own error."],
    ["Schedule:", " teacher-forcing warm-up, then ramp in resampled history; t_s is also a confidence input."],
  ], X0, 5.15, CW, 1.4);
  label(s, "Based on Resampling Forcing (arXiv 2512.15702) and Diffusion Forcing (arXiv 2407.01392).", X0, 6.6, CW, 0.4, { size: 18, color: INK2 });
}

// ======================================================================= 11. ILF (method)
{
  const s = newSlide("ILF: Teaching the Model to Use Memory at Inference");
  // write/read loop
  box(s, "Generated frame x_t", 0.5, 1.4, 2.6, 0.7, { fill: TINT, line: BLUE, bold: true });
  box(s, "Thin dark structures", 3.6, 1.4, 2.6, 0.7);
  box(s, "Persistent?\n(median of 7 frames)", 3.6, 2.55, 2.6, 0.85);
  box(s, "M_t = min(M_{t−1}, new)\nonly grows", 3.6, 3.85, 2.6, 0.85, { fill: TINT, line: BLUE, bold: true });
  box(s, "Hint channel\nfor frame t+1", 0.5, 3.85, 2.6, 0.85);
  arrow(s, 3.1, 1.75, 3.6, 1.75); arrow(s, 4.9, 2.1, 4.9, 2.55); arrow(s, 4.9, 3.4, 4.9, 3.85);
  arrow(s, 3.6, 4.275, 3.1, 4.275); arrow(s, 1.8, 3.85, 1.8, 2.1);
  // exposure bias
  box(s, "Training: memory written from real frames → always correct", 6.8, 1.4, 6.03, 0.85, { size: 16 });
  box(s, "Inference: memory written from its own frames → errors are confirmed every frame", 6.8, 2.5, 6.03, 0.85,
    { size: 16, fill: "FDF0EA", line: ORANGE, bold: true });
  label(s, "Goal: use self-written memory, without copying its errors or ignoring it.", 6.8, 3.6, 6.03, 0.9, { size: 18, bold: true });
  bullets(s, [
    ["Match training to inference:", " write memory from self-generated frames; down-weight targets that erase it."],
    ["Make memory necessary:", " erase cracks from S; vary memory strength by denoising stage."],
    ["Write when physics says so:", " a Kaiser gate at each new drift level; a learned or blurred memory."],
    ["Break the loop:", " open-loop memory, training-column memory as a prior, distribution-level loss."],
  ], X0, 4.95, CW, 1.95);
}

// ======================================================================= 12. references
{
  const s = newSlide("References");
  const refs = [
    "Diffusion Forcing, arXiv 2407.01392", "History-Guided Video Diffusion (DFoT), arXiv 2502.06764",
    "Resampling Forcing, arXiv 2512.15702", "Self Forcing, arXiv 2506.08009", "SVI (error recycling), arXiv 2510.09212",
    "Memory Forcing, arXiv 2510.03198", "Memory Forcing (write gate), arXiv 2610.11756", "In-Context Forcing, arXiv 2608.05237",
    "Ring Forcing, arXiv 2608.26794", "Clean Forcing, arXiv 2610.25634", "FreqForcing, arXiv 2607.27110",
    "Scheduled Sampling, arXiv 1506.03099", "Professor Forcing, arXiv 1610.09038", "GameNGen, arXiv 2408.14837",
    "Under-Development of Irreversible Processes, arXiv 2608.00617",
  ];
  const half = Math.ceil(refs.length / 2);
  [refs.slice(0, half), refs.slice(half)].forEach((col, i) => {
    s.addText(col.map((r, k) => ({ text: r, options: { bullet: { indent: 18 }, breakLine: k < col.length - 1, paraSpaceAfter: 8 } })),
      { x: X0 + i * 6.3, y: 1.4, w: 6.0, h: 5.3, fontFace: FONT, fontSize: 18, color: INK, valign: "top", margin: 0, isTextBox: true });
  });
}

pres.writeFile({ fileName: OUT }).then((f) => console.log("[deck] wrote", f));
