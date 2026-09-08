#!/usr/bin/env bash
# Build ADMUNetCFM U-Net diagram with PlotNeuralNet.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PNNDIR="${ROOT}/third_party/PlotNeuralNet"
PYEX="${PNNDIR}/pyexamples"
OUTDIR="${ROOT}/results/architecture_diagrams"
NAME="adm_unet_cfm"

mkdir -p "${OUTDIR}"
cd "${PYEX}"

echo "[build_adm_unet_diagram] Generating ${NAME}.tex ..."
python3 "${NAME}.py"

if ! command -v pdflatex >/dev/null 2>&1; then
    echo "[build_adm_unet_diagram] pdflatex not found. Install TeX Live, then run:"
    echo "  cd ${PYEX} && pdflatex ${NAME}.tex"
    exit 0
fi

echo "[build_adm_unet_diagram] Compiling PDF ..."
if pdflatex -interaction=nonstopmode "${NAME}.tex" >/dev/null && \
   pdflatex -interaction=nonstopmode "${NAME}.tex" >/dev/null; then
    cp -f "${PYEX}/${NAME}.pdf" "${OUTDIR}/${NAME}.pdf"
    echo "[build_adm_unet_diagram] Done: ${OUTDIR}/${NAME}.pdf"
else
    echo "[build_adm_unet_diagram] pdflatex failed (need TeX Live: standalone, tikz, import)."
    echo "[build_adm_unet_diagram] .tex saved to ${OUTDIR}/${NAME}.tex — compile on Overleaf or local TeX."
fi

cp -f "${PYEX}/${NAME}.tex" "${OUTDIR}/${NAME}.tex"
