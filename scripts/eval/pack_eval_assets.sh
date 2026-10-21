#!/bin/bash
# Pack the evaluation files that git does not carry (Data/, results/ and *.pt are ignored) into one
# archive with checksums, to upload next to the code (GitHub Release, Hugging Face, lab NAS).
#
#   bash scripts/eval/pack_eval_assets.sh [out_dir]        → <out_dir>/eval_assets_v1.tar.gz + .sha256
#   (receiver, in the repo root)  tar xzf eval_assets_v1.tar.gz && sha256sum -c eval_assets_v1.files.sha256
#
# Contents (paths are kept, so extracting in the repo root puts everything in place;
# Data/DamagePattern/{images,mask,label.csv,id_list*.json} are already in git):
#   Data/DamagePattern_inpaint_selfcondnoDI/images   inpainted reference set (eval reference)
#   Data/DamagePattern/r_spall.csv   per-frame R_spall labels (the probe and kid_by_spall)
#   results/InterferenceDetector/v1/{best.pt,backbone_config.json,metrics.json}   interference detector
#       (best.pt holds its fine-tuned DINOv3 weights, backbone_config.json the architecture, so it
#       loads without Hugging Face access)
#   results/DamagePatternGeneration/CFMADM/selfcondnoDI/{args.json,.../vq9dirur/checkpoints/last.ckpt}
#       the CFM inpainter that removes interference (same one that made the reference set)
#   phase1_extractor/{args.json,best.ckpt}   GRU extractor; the inpainter's condition needs it
#       (its Hugging Face repo is private, so prepare_extractor_from_hf.sh only works for its owner)
# Not included: the metric backbone:
# DINOv3 (gated: accept the licence on its Hugging Face page, then `huggingface-cli login`) or
# DINOv2 (public, "dino_model": "facebook/dinov2-small" in the suite config).
# best.pt contains DINOv3 weights: check the DINOv3 licence terms before sharing the archive.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-dist}"
mkdir -p "${OUT}"
FILES=(
    Data/DamagePattern_inpaint_selfcondnoDI/images
    Data/DamagePattern/r_spall.csv
    results/InterferenceDetector/v1/best.pt
    results/InterferenceDetector/v1/backbone_config.json
    results/InterferenceDetector/v1/metrics.json
    results/DamagePatternGeneration/CFMADM/selfcondnoDI/args.json
    "results/DamagePatternGeneration/CFMADM/selfcondnoDI/Opensees Based damage Patterns Generation/vq9dirur/checkpoints/last.ckpt"
    phase1_extractor/args.json
    phase1_extractor/best.ckpt
)
for f in "${FILES[@]}"; do
    [[ -e "${f}" ]] || { echo "[pack] missing ${f}"; exit 1; }
done
find "${FILES[@]}" -type f -print0 | sort -z | xargs -0 sha256sum > "${OUT}/eval_assets_v1.files.sha256"
tar czf "${OUT}/eval_assets_v1.tar.gz" "${FILES[@]}"
sha256sum "${OUT}/eval_assets_v1.tar.gz" > "${OUT}/eval_assets_v1.tar.gz.sha256"
du -h "${OUT}/eval_assets_v1.tar.gz"
echo "[pack] wrote ${OUT}/eval_assets_v1.tar.gz, eval_assets_v1.files.sha256, eval_assets_v1.tar.gz.sha256"
