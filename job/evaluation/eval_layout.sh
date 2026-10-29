#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=4
#PBS -l place=shared
#PBS -o log/eval_layout.out
#PBS -e log/eval_layout.err
#PBS -N eval_layout

# Layout-conditioned evaluation: (render layouts →) eval suite → image-side spall metrics (area / IoU / components).
# Replaces eval_layout_oracle.sh, eval_layout_sensitivity.sh, phase1_coarse_ceiling.sh, phase1_c2_e2e.sh, phase1_ofp.sh.
#   EXP=oracle       (default) gn / control / layout-oracle; the layout model gets the true CTR1 / R307 layouts
#                    from its dataset (docs/redesign/phase0_layout_oracle.md)
#   EXP=coarse       real mask128 / iso32 / iso16 layouts → layout_scratch renderer (docs/redesign/phase1_layout_harness.md)
#   EXP=c2           C2 layout generator outputs → renderer; iso32 = real monotone 32x32 layout as ceiling (docs/redesign/phase1_c2.md)
#   EXP=ofp          OFP layouts → renderer, single frame + temporal (docs/redesign/phase1_ofp.md)
#   EXP=sensitivity  true / zero / max-drift / cross-pillar layout, same noise; needs RESULT_DIR, CKPT, TAG
#   STAGE=seg        only train the spall segmenter (measuring tool). It is also trained when missing.
#   qsub job/DamagePatternGeneration/eval_layout.sh
#   qsub -v EXP=ofp job/DamagePatternGeneration/eval_layout.sh
#   qsub -v EXP=sensitivity,RESULT_DIR=<run>,CKPT=<ckpt>,TAG=<name> job/DamagePatternGeneration/eval_layout.sh
# CONFIG, MODELS and TAG override the preset of each EXP. Rendered samples that already exist are reused.
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

EXP="${EXP:-oracle}"
STAGE="${STAGE:-all}"
CFG=job/DamagePatternGeneration/configs
OUT=results/phase0_layout_oracle
SEG="${OUT}/spallseg/best.pt"
mkdir -p log "${OUT}"

if [[ "${EXP}" == "sensitivity" ]]; then
    for split in test val; do
        python3 -u scripts/eval/phase0_layout_sensitivity.py --result_dir "${RESULT_DIR}" --checkpoint "${CKPT}" \
            --split "${split}" --k "${K:-4}" --out "${OUT}/sensitivity_${TAG}_${split}.json" || exit 1
    done
    exit 0
fi
if [[ "${STAGE}" == "seg" || ! -f "${SEG}" ]]; then
    python3 -u scripts/eval/phase0_layout_oracle_spallseg.py --out_dir "${OUT}/spallseg" || exit 1
fi
[[ "${STAGE}" == "seg" ]] && exit 0

# render <layouts npz> <out npz>, skipped when the output exists
render() {
    [[ -f "$2" ]] && return 0
    mkdir -p "$(dirname "$2")"
    python3 -u scripts/eval/render_layouts.py --layouts "$1" --out "$2" || exit 1
}

case "${EXP}" in
    oracle)
        P_CONFIG=${CFG}/eval_suite_layout_oracle.json; P_MODELS="gn layout_ctrl layout_oracle"; P_TAG=""
        ;;
    coarse)
        D=results/phase1_layout_harness
        for split in test val; do
            for pair in oracle128:mask128 iso32:iso32 iso16:iso16; do
                render "$D/layouts/real_${pair##*:}_${split}.npz" "$D/samples/${pair%%:*}_${split}.npz"
            done
        done
        P_CONFIG=${CFG}/phase1_coarse_ceiling.json; P_MODELS="gn oracle128 iso32 iso16"; P_TAG=coarse
        ;;
    c2)
        D=results/phase1_c2
        for pair in c2:prev_tau c2_noprev:noprev_tau iso32:REAL; do
            name=${pair%%:*}; run=${pair##*:}
            for split in test val; do
                src=$D/runs/${run}/gen_${split}.npz
                [[ "$run" == REAL ]] && src=results/phase1_layout_harness/layouts/real_iso32_${split}.npz
                render "${src}" "$D/samples/${name}_${split}.npz"
            done
        done
        P_CONFIG=${CFG}/eval_suite_phase1_c2.json; P_MODELS="gn gn_facfg c2 c2_noprev iso32"; P_TAG=c2
        ;;
    ofp)
        D=results/phase1_ofp
        for split in test val; do
            for pair in ofp_c:c_iou ofp_b:b; do
                render "$D/layouts_${split}_${pair##*:}.npz" "$D/samples/${pair%%:*}_${split}.npz"
            done
            # temporal: shared noise per sequence and k
            [[ -f $D/samples/ofp_c_tmp_${split}.npz ]] || python3 -u scripts/eval/render_ofp_temporal.py \
                --layouts $D/layouts_${split}_c_iou.npz --out $D/samples/ofp_c_tmp_${split}.npz || exit 1
        done
        P_CONFIG=${CFG}/eval_suite_ofp.json; P_MODELS="gn gn_facfg refiner v106F ofp_c ofp_b ofp_c_tmp"; P_TAG=ofp
        ;;
    *) echo "EXP must be oracle, coarse, c2, ofp or sensitivity"; exit 1 ;;
esac
CONFIG="${CONFIG:-${P_CONFIG}}"
MODELS="${MODELS:-${P_MODELS}}"
TAG="${TAG:-${P_TAG}}"

python3 -u scripts/eval/run_eval_suite.py --config "${CONFIG}" || exit 1

SUITE_OUT=$(python3 -c "import json;print(json.load(open('${CONFIG}'))['out_dir'])")
for split in test val; do
    args=()
    for name in ${MODELS}; do
        # A model given as a ready npz in the config is read from there; the others from the suite dir.
        npz=$(python3 -c "import json,sys;m={x['name']:x for x in json.load(open('${CONFIG}'))['models']};print(m['${name}'].get('npz',{}).get('${split}',''))")
        [[ -z "${npz}" ]] && npz="${SUITE_OUT}/${split}/samples/${name}.npz"
        args+=(--model "${name}=${npz}")
    done
    python3 -u scripts/eval/phase0_layout_oracle_metrics.py --split "${split}" "${args[@]}" \
        --segmenter "${SEG}" --out "${OUT}/spall_metrics${TAG:+_${TAG}}_${split}.json" || exit 1
done
