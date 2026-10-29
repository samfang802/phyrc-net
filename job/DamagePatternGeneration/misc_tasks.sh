#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=6
#PBS -l place=shared
#PBS -o log/misc_tasks.out
#PBS -e log/misc_tasks.err
#PBS -N misc_task

# One-shot data / probe jobs, one python call each. Output goes to log/${TASK}.{out,err}.
#   TASK=dataset_audit          per-frame dataset measurements (docs/redesign/dataset_audit.md)
#   TASK=export_inpaint_png     rebuild the inpainted reference set (V9 selfcondnoDI CFM, SAM masks, seed 42, 30 Heun
#                               steps) as lossless PNG instead of the PIL-default q75 JPEG; same args as
#                               Data/DamagePattern_inpaint_selfcondnoDI/meta.json, the JPEG set is left untouched
#   TASK=opa_free_cache         condition cache (GRU g, max drift, encoder progress P) of the image-free columns
#                               for OPA-Flow group C (docs/redesign/phase1_opaflow.md)
#   TASK=c2_dump_g              dump GRU g for the C2 layout generator (docs/redesign/phase1_c2.md)
#   TASK=phase0_critic          ordinal / magnitude critic check (docs/redesign/phase0_critic.md);
#                               EXTRA="--eval_only --samples ..." to only evaluate new samples
#   TASK=phase0_spd_onset       PEER SPD onset drift vs the 6 image columns; needs XML_DIR (per-specimen xml)
#   TASK=clock_condition_probe  does the KF damage clock tau add to the v11 condition? ridge + MLP probes
#   TASK=kid_jpeg               re-score finished suites with JPEG-q75-aligned DINOv3 metrics (docs/eval/protocol.md);
#                               SUITES="results/eval_suite/v105 ..." (default: every suite the docs quote)
#   qsub -v TASK=dataset_audit job/DamagePatternGeneration/misc_tasks.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

TASK="${TASK:?set TASK, see the header}"
mkdir -p log
exec > "log/${TASK}.out" 2> "log/${TASK}.err"
S=results/eval_suite
case "${TASK}" in
    dataset_audit)
        python3 -u scripts/data/audit_frames.py --bs 16 ;;
    export_inpaint_png)
        python3 -u scripts/export/export_inpaint_dataset.py \
            --out_dir Data/DamagePattern_inpaint_selfcondnoDI_png --image_ext png \
            --splits train,val,test --ode_steps 30 --seed 42 ;;
    opa_free_cache)
        python3 -u scripts/export/export_opa_free_cache.py --out Data/DamagePattern/opa_free_cache.npz ;;
    c2_dump_g)
        python3 -u scripts/data/dump_gru_g.py --tmp_dir "${TMPDIR:-/tmp}/c2_dump_g_${USER}" ;;
    phase0_critic)
        python3 -u scripts/eval/phase0_critic.py --out_dir "${OUT_DIR:-results/phase0/critic}" \
            --samples gn_test=$S/v105/test/samples/gn.npz gn_shared_test=$S/v105/test/samples/gn_shared.npz \
                      gn_val=$S/v106/val/samples/gn.npz ${EXTRA:-} ;;
    phase0_spd_onset)
        python3 -u scripts/eval/phase0_spd_onset.py --xml_dir "${XML_DIR:?XML_DIR is required}" ;;
    clock_condition_probe)
        python3 -u scripts/eval/clock_condition_probe.py ;;
    kid_jpeg)
        python3 -u scripts/eval/rerun_kid_jpeg.py ${SUITES:-$S/v105 $S/quick $S/v106 $S/v106C $S/v11 $S/v11_gru $S/v11_fog \
            $S/v11_refiner $S/v11app_refiner $S/tier0_churn $S/phase0_guidance $S/phase1_opaflow \
            results/phase0_layout_oracle/eval_suite results/phase0_layout_oracle/eval_suite_scratch \
            results/phase1_c2/eval_suite results/phase1_layout_harness/eval_suite_coarse results/phase1_ofp/eval_suite} ;;
    *) echo "unknown TASK=${TASK}"; exit 1 ;;
esac
