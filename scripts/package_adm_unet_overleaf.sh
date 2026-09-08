#!/usr/bin/env bash
# Zip PlotNeuralNet files for Overleaf (flat layout: all .sty in project root).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PNNDIR="${ROOT}/third_party/PlotNeuralNet"
STAGING="${ROOT}/results/architecture_diagrams/overleaf_staging"
OUTZIP="${ROOT}/results/architecture_diagrams/adm_unet_cfm_overleaf.zip"

rm -rf "${STAGING}"
mkdir -p "${STAGING}"

# Flat layout: init.tex + *.sty in same folder as main .tex (Overleaf-friendly).
cp "${PNNDIR}/layers/init.tex" "${STAGING}/"
cp "${PNNDIR}/layers/"*.sty "${STAGING}/"

OVERLEAF_FLAT=1 OVERLEAF_OUTDIR="${STAGING}" \
    python3 "${PNNDIR}/pyexamples/adm_unet_cfm.py"

rm -f "${OUTZIP}" "${STAGING}/output.pdf"
(cd "${STAGING}" && zip -r "${OUTZIP}" adm_unet_cfm.tex Ball.sty Box.sty RightBandedBox.sty)

cp -f "${STAGING}/adm_unet_cfm.tex" "${ROOT}/results/architecture_diagrams/adm_unet_cfm.tex"

echo "[package_adm_unet_overleaf] Created: ${OUTZIP}"
echo "Overleaf:"
echo "  1. Delete old project (or remove output.pdf if present)"
echo "  2. Upload zip as New Project"
echo "  3. Main document: adm_unet_cfm.tex"
echo "  4. Compiler: pdfLaTeX -> Recompile"
