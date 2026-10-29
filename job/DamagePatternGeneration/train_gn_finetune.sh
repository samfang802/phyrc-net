#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=6
#PBS -l place=shared
#PBS -o log/train_gn_finetune.out
#PBS -e log/train_gn_finetune.err
#PBS -N gn_finetune

# Fine-tune gn (1024-dadaGNsplit, chjqbp9m) with the gn recipe plus one extra mechanism.
# Replaces train_layout_oracle.sh (EXP=layout) and train_opaflow.sh (EXP=opa).
#
# EXP=layout  Phase-0 layout-oracle (docs/redesign/phase0_layout_oracle.md), 5k steps.
#   LAYOUT_COND=True  : + one zero-init input channel with the AnyLabel spall SDF (train split only),
#                       dropped (set to 0) with probability LAYOUT_DROP_PROB per sample.
#   LAYOUT_COND=False : control. Same warm start, same 5k steps, no layout channel.
#   qsub -v EXP=layout,LAYOUT_COND=True,RUN_TAG=layout_oracle -o log/train_layout_oracle.out -e log/train_layout_oracle.err job/DamagePatternGeneration/train_gn_finetune.sh
#   qsub -v EXP=layout,LAYOUT_COND=False,RUN_TAG=layout_ctrl  -o log/train_layout_ctrl.out   -e log/train_layout_ctrl.err   job/DamagePatternGeneration/train_gn_finetune.sh
#   From scratch, 15k like gn (fair comparison; layout column default init):
#   qsub -v EXP=layout,SCRATCH=1,MAX_STEPS=15000,WARMUP_STEPS=1000,LAYOUT_DROP_PROB=0.1,LAYOUT_ZERO_INIT=False,RUN_TAG=layout_scratch -o log/train_layout_scratch.out -e log/train_layout_scratch.err job/DamagePatternGeneration/train_gn_finetune.sh
#   The layout cache is built on first use with scripts/export/export_anylabel_layout_sdf.py.
#
# EXP=opa  OPA-Flow minimal experiment, null anchor (docs/redesign/phase1_opaflow.md), 3k steps.
#   GROUP=A  continue FM only (control, same warm start / steps / schedule)
#   GROUP=B  + L_ord + L_mag of the frozen pixel critic on one-step predictions vs a real earlier frame
#   GROUP=C  B + image-free pair rollouts (needs Data/DamagePattern/opa_free_cache.npz,
#            qsub -v TASK=opa_free_cache job/DamagePatternGeneration/misc_tasks.sh)
#   qsub -v EXP=opa,GROUP=A -N opa_A -o log/opa_A.out -e log/opa_A.err job/DamagePatternGeneration/train_gn_finetune.sh
#   qsub -v EXP=opa,GROUP=B,LAM_ORD=0.05,LAM_MAG=0.05 ...
#
# Smoke (a few steps, no wandb): qsub -v SMOKE=1,EXP=... ...
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

EXP="${EXP:?set EXP=layout or EXP=opa}"
VELOCITY_INIT="${VELOCITY_INIT:-results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit/Opensees Based damage Patterns Generation/chjqbp9m/checkpoints/last.ckpt}"
WARMUP_STEPS="${WARMUP_STEPS:-250}"
EXP_ARGS=()
case "${EXP}" in
    layout)
        LAYOUT_COND="${LAYOUT_COND:-True}"
        LAYOUT_CACHE="${LAYOUT_CACHE:-Data/DamagePattern/anylabel_layout_sdf128.npz}"
        LAYOUT_DROP_PROB="${LAYOUT_DROP_PROB:-0.2}"
        [[ -n "${SCRATCH:-}" ]] && VELOCITY_INIT=""   # SCRATCH=1: no warm start
        MAX_STEPS="${MAX_STEPS:-5000}"
        if [[ -z "${RUN_TAG:-}" ]]; then
            [[ "${LAYOUT_COND}" == "True" ]] && RUN_TAG=layout_oracle || RUN_TAG=layout_ctrl
        fi
        if [[ "${LAYOUT_COND}" == "True" && ! -f "${LAYOUT_CACHE}" ]]; then
            echo "[gn_finetune] building ${LAYOUT_CACHE}"
            python3 -u scripts/export/export_anylabel_layout_sdf.py --out "${LAYOUT_CACHE}" || exit 1
        fi
        EXP_ARGS=(--layout_cond "${LAYOUT_COND}" --layout_cache "${LAYOUT_CACHE}"
                  --layout_drop_prob "${LAYOUT_DROP_PROB}" --layout_zero_init "${LAYOUT_ZERO_INIT:-True}")
        echo "[gn_finetune] LAYOUT_COND=${LAYOUT_COND} DROP=${LAYOUT_DROP_PROB}"
        ;;
    opa)
        GROUP="${GROUP:-A}"
        LAM_ORD="${LAM_ORD:-0.05}"
        LAM_MAG="${LAM_MAG:-0.05}"
        OPA_N="${OPA_N:-16}"
        OPA_FREE_N="${OPA_FREE_N:-8}"
        MAX_STEPS="${MAX_STEPS:-3000}"
        RUN_TAG="${RUN_TAG:-opa_${GROUP}}"
        case "${GROUP}" in
            A) ;;
            B) EXP_ARGS=(--opa_lambda_ord "${LAM_ORD}" --opa_lambda_mag "${LAM_MAG}" --opa_n "${OPA_N}") ;;
            C) EXP_ARGS=(--opa_lambda_ord "${LAM_ORD}" --opa_lambda_mag "${LAM_MAG}" --opa_n "${OPA_N}" --opa_free True --opa_free_n "${OPA_FREE_N}") ;;
            *) echo "GROUP must be A, B or C"; exit 1 ;;
        esac
        [[ -n "${SMOKE:-}" && "${GROUP}" != "A" ]] && EXP_ARGS+=(--opa_diag_every 2)
        echo "[gn_finetune] GROUP=${GROUP} lam_ord=${LAM_ORD} lam_mag=${LAM_MAG}"
        ;;
    *) echo "EXP must be layout or opa"; exit 1 ;;
esac

USE_WANDB_ARGS=(--use_wandb)
CKPT_EVERY=1000
if [[ -n "${SMOKE:-}" ]]; then
    MAX_STEPS="${SMOKE_STEPS:-6}"
    WARMUP_STEPS=2
    RUN_TAG="smoke_${RUN_TAG}"
    USE_WANDB_ARGS=()
    CKPT_EVERY=0
fi
if [[ -n "${VELOCITY_INIT}" && ! -f "${VELOCITY_INIT}" ]]; then
    echo "[gn_finetune] velocity_init not found: ${VELOCITY_INIT}"; exit 1
fi
INIT_ARGS=()
[[ -n "${VELOCITY_INIT}" ]] && INIT_ARGS=(--velocity_init "${VELOCITY_INIT}")
echo "[gn_finetune] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} EXP=${EXP} VELOCITY_INIT=${VELOCITY_INIT}"
echo "[gn_finetune] RUN_TAG=${RUN_TAG} MAX_STEPS=${MAX_STEPS} WARMUP_STEPS=${WARMUP_STEPS}"

# gn recipe: results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit/args.json (GRU g, split gated AdaGN,
# zero DI, no self-cond, Patch Diffusion 0.5/0.3/0.2, VSS L3/U3, min-SNR 5, hflip 0.5, CLAHE gray).
python3 -u train_mixed.py --lr 1e-4 --epoch 3000 --batch_size 128 --num_workers 2 --run_tag "${RUN_TAG}" \
    --train_virtual_repeat 100 --max_steps "${MAX_STEPS}" --warmup_steps "${WARMUP_STEPS}" --val_check_interval 5 \
    --architecture CFMADM --loss_type CFMADM --data_path id_list_val_R307.json \
    --extractor_path ./phase1_extractor "${USE_WANDB_ARGS[@]}" \
    "${INIT_ARGS[@]}" \
    "${EXP_ARGS[@]}" \
    --img_aug --use_grayscale_input True --use_clahe True --clahe_clip_limit 2.0 --clahe_tile_grid_size 8,8 \
    --aug_prob 0.5 --aug_blur_prob 0.15 --aug_brightness 0.2 --aug_contrast 0.2 --aug_saturation 0.0 --aug_hue 0.0 \
    --aug_hflip_prob 0.5 --aug_blur_kernel 3 --aug_blur_sigma_min 0.1 --aug_blur_sigma_max 0.5 \
    --aug_erase_prob 0.0 --aug_erase_scale_min 0.05 --aug_erase_scale_max 0.15 \
    --aug_erase_ratio_min 0.5 --aug_erase_ratio_max 2.5 \
    --condition_perturb --condition_perturb_dist uniform --condition_perturb_min 0.01 --condition_perturb_max 0.03 \
    --condition_mode split --zero_di_condition True --cfg_g_drop_prob 0.0 --gated_fusion_alpha_dim scalar \
    --use_physics_cross_attn False --cfm_physics_num_tokens 16 --cfm_physics_token_dim 128 \
    --self_condition False \
    --use_wavelet_unet False --use_vss_l3_u3 True --use_vss_l2 False --use_vss_u2 False --use_u1_extra_resblock False \
    --cfm_physics_adagn_dim 128 --physics_dim 1024 --physics_bottleneck_dim 0 --g_pca_dim 0 \
    --use_early_stopping False --early_stop_patience 15000 --early_stop_min_delta 0.0 \
    --checkpoint_every_n_steps "${CKPT_EVERY}" --checkpoint_start_step 1000 \
    --latent_conditioning True --direction_conditioning True --include_di_conditioning True --add_DI_loss False \
    --use_damage_mask_condition True --use_object_loss_mask False \
    --image_size 128 --cfm_model_channels 32 --cfm_num_res_blocks 1 --cfm_cond_dim 128 --cfm_dropout 0.1 \
    --cfm_mid_attn_heads 4 \
    --use_patch_diffusion True --use_patch_prob_schedule False --patch_size_probs 0.5,0.3,0.2 \
    --use_patch_t_schedule False \
    --cfm_solver heun --cfm_ode_steps 30 --lr_schedule warmup_cosine --min_lr_ratio 0.1 \
    --grad_clip_max_norm 1.0 --use_velocity_min_snr_weighting True --min_snr_gamma 5.0 \
    --inference_every_n_epochs 1000 --inference_num_images 4 \
    --inference_train_anchor_exp R615 --inference_val_anchor_exp R307 --inference_anchor_timestep -1 \
    --fixed_di_values 0.0,0.4,0.75,1.0
