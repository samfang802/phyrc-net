# PBS 工作腳本

所有腳本都在 repo 根目錄執行（`cd "${PBS_O_WORKDIR:-$HOME/intership-project}"`），環境是 conda `internship`。參數一律用 `qsub -v 名稱=值,...` 覆寫，不需要改檔案。GPU 編號和 `gpu_id` 的對應見 `docs/GPU.md`。

## DamagePatternGeneration（生成模型）

| 腳本 | 用途 | 常用覆寫 |
|---|---|---|
| `common.sh` | 共用環境設定（`adm_env`），其他腳本 `source` 它 | — |
| `train_ADMCFM.sh` | **主訓練腳本**：單幀 CFMADM（V9/V10 預設設定） | `RUN_TAG`、`MAX_STEPS`、`CFM_MODEL_CHANNELS`、`RESUME_CKPT`（`none` = 從頭訓練）、`EARLY_STOP_PATIENCE`、`CHECKPOINT_EVERY_N_STEPS`、`CHECKPOINT_START_STEP` |
| `train_ADMCFM_v12.sh` | v12 單幀：v11 appearance code + layout channel | `LAYOUT_CACHE`、`RUN_TAG`（`v12` / `v12gt` / `v12basis`） |
| `train_ADMCFM_v12refiner.sh` | v12 refiner 和它的變體 | `RECIPE`：`refiner`（預設）、`hspP`、`mem`、`memr`、`nbr`、`rf`；`LAYOUT_CACHE`、`BASE_CKPT`、`RUN_TAG`、`STOP_STEP` |
| `train_ADMCFM_refiner_residual.sh` | 殘差時序 refiner（V10 variant B），以 gn 為凍結基底 | `REFINER_BASE_DIR`、`REFINER_BASE_CKPT`、`MAX_STEPS` |
| `train_ADMCFM_oracle_dino.sh` | Oracle 上限：GRU g 換成 GT 的 DINOv3 特徵（見 `docs/eval/archive/oracle.md`） | `ORACLE_DINO_POS_BLUR`（`none` / `shuffle` / `shift`） |
| `test_ADMCFM.sh` | 單一 checkpoint 的固定 DI 推論與測試指標 | `CKPT_PATH`、`SPLIT`、`FIXED_DI_VALUES` |
| `prepare_extractor_from_hf.sh` | 從 Hugging Face 下載 GRU extractor（`./phase1_extractor`） | — |
| `viz_high_di_grid.sh` | 6×6 High-DI 總覽圖 | `MODE`：`plain`、`apg_g0`、`apg_facfg`、`hipass_g0`、`autoguide_fdg`、`refiner` |
| `train_gn_finetune.sh` | 從 gn 續訓加一個機制（gn 的設定原封不動） | `EXP`：`layout`（layout-oracle，`LAYOUT_COND`、`SCRATCH=1`）、`opa`（OPA-Flow，`GROUP`=A/B/C）；`SMOKE=1` |
| `train_layout_fm.sh` | C2 的 layout 生成器（`scripts/train_layout_fm.py`） | `RUNS="名稱\|參數;..."` |
| `misc_tasks.sh` | 單一 python 指令的小工作，輸出在 `log/${TASK}.{out,err}` | `TASK`：`dataset_audit`、`export_inpaint_png`、`opa_free_cache`、`c2_dump_g`、`phase0_critic`、`phase0_spd_onset`（`XML_DIR`）、`clock_condition_probe`、`fiber_clock`、`kid_jpeg` |

這兩個資料夾裡合併過的腳本只有一組 PBS 標頭（`gpu_id`、`-o`、`-N`），要換 GPU 或分開 log 時在 `qsub` 加 `-l select=...`、`-o`、`-e`、`-N`。

## evaluation（生成模型的評估）

共用 `DamagePatternGeneration/common.sh` 的環境。

| 腳本 | 用途 | 常用覆寫 |
|---|---|---|
| `eval_suite.sh` | **完整評估**：生成、去干擾物、三種評估、摘要（見 `docs/eval/README.md`） | `CONFIG`、`SPLITS`、`QUICK=1`（只跑 test 的 KID）、`FORCE=1` |
| `eval_final.sh`、`eval_cgan_clahe.sh` | 定案表（每版本一個 ckpt）和 cGAN 的 CLAHE 前處理 | — |
| `eval_fid_panels.sh` | 定案表的 K=8 Inception FID（test + val）和 V12 的 drift × DI 圖 | — |
| `configs/` | 還在用的清單：`eval_suite_models.json`（預設）、`eval_suite_final.json`、`eval_suite_cgan_clahe.json`、`eval_suite_v12.template.json`、`eval_suite_fiber_layout.json`。實驗腳本會把自己的清單寫進這個資料夾，那些不進 git（`.gitignore`） | — |
| `eval_v12.sh` | v12 的 layout 變體：生成、suite、剝落指標 | `RUN`、`STEPS`、`VARIANTS`、`QUICK=1`、`FORCE=1` |
| `eval_diagnostics.sh` | 模型診斷（不屬於評估流程） | `WHAT`：`shift`、`spectrum`、`all`（gn）；`train_area`、`velocity_fusion`（v11appr） |
| `phase1_fiber_layout.sh` | 纖維時鐘的 layout 快取、渲染、評估 | `STAGES` |
| `eval_v12rf.sh`、`eval_v12rfm.sh`、`eval_rfmv2.sh`、`eval_ilf.sh`、`eval_frame0.sh`、`eval_gravnoise.sh`、`frame0_hop.sh`、`ilf_oracle.sh`、`memory_guidance.sh`、`state_conf_sweep.sh` | 目前的 v12 比較：各寫一份清單再跑 suite。`QUICK=1` 只跑 test 的 KID，`FORCE=1` 重算 | 見各腳本開頭 |
| `irreversibility_curve.sh`、`refiner_memory_diag.sh`、`refiner_s_probe.sh`、`state_source_probe.sh`、`u_calib_probe.sh`、`physics_gate.sh`、`damage_rate.sh`、`sm_ablation.sh`、`late_copy.sh`、`rollout_probe.sh`、`memory_degraded.sh`、`memory_reliance.sh`、`uncertainty_map.sh`、`ilf_ae_probe.sh`、`ilf_memory_probe.sh`、`heun_hop_check.sh`、`speedup_check.sh` | 單一診斷，各呼叫一支 `scripts/eval/` | 見各腳本開頭 |
| `exp.sh` | 2026-11-11 文獻追蹤實驗（`scripts/eval/exp_probe.py`）：`MODE=probe` 比較取樣變體、`MODE=ft` 短微調 | 見腳本開頭 |
| `deck_copy_example.sh` | 簡報用範例圖（`scripts/viz/deck_copy_example.py`） | — |

### 常用指令

```bash
# 主模型
qsub job/DamagePatternGeneration/train_ADMCFM.sh
# autoguidance 用的「差模型」（原本的 train_ADMCFM_autoguide_bad.sh）
qsub -v CFM_MODEL_CHANNELS=16,MAX_STEPS=6000,EARLY_STOP_PATIENCE=3000,CHECKPOINT_EVERY_N_STEPS=1000,CHECKPOINT_START_STEP=1000,RESUME_CKPT=none,RUN_TAG=autoguide_bad \
    job/DamagePatternGeneration/train_ADMCFM.sh
# 評估（先編輯 job/evaluation/configs/eval_suite_models.json）
qsub job/evaluation/eval_suite.sh
# High-DI 圖
qsub -v MODE=apg_g0,SCALES="1.5 2.0" job/DamagePatternGeneration/viz_high_di_grid.sh
# 用另一份清單評估
qsub -v CONFIG=job/evaluation/configs/eval_suite_final.json job/evaluation/eval_suite.sh
# v12 refiner 的某個變體（log 和 GPU 在 qsub 上覆寫）
qsub -N CFM_v12rf -o log/train_v12rf.out -e log/train_v12rf.err -v RECIPE=rf job/DamagePatternGeneration/train_ADMCFM_v12refiner.sh
qsub -v EXP=opa,GROUP=B -N opa_B -o log/opa_B.out -e log/opa_B.err job/DamagePatternGeneration/train_gn_finetune.sh
```

## DamageQuantification（DI 回歸器）

python 程式在 `quantification/`，用 `python -m quantification.<script>` 從專案根目錄執行。

| 腳本 | 用途 |
|---|---|
| `train_resnet18_siamese.sh` / `test_resnet18_siamese.sh` | Siamese ResNet18 DI 評估器（目前的 DI evaluator） |
| `train_patch_weighted_siamese.sh` | patch 加權 Siamese |
| `train_resnet18_cbam.sh` | ResNet18 + CBAM |
| `train_backbone.sh` | torchvision backbone 回歸器：`BACKBONE` = `vit_b_16`、`vit_b_32`、`vit_l_16`、`vit_l_32`、`swin_b`、`resnet101`；`LR`、`EPOCH` 可覆寫 |

## HysteresisPrediction（遲滯迴圈預測）

| 腳本 | 用途 |
|---|---|
| `train_hysteresis.sh` | `ARCH=HystereticGRU`（預設，即 `phase1_extractor` 的設定）或 `HystereticLSTM` |

## 已刪除的腳本

2026-10-21 和 2026-10-29 兩次整理。以下腳本是舊實驗的整份複製，主腳本後來的設定都沒跟上，或者已經合併進上面的腳本。要重跑時用 `git show <commit>:<路徑>` 取回。

| 腳本 | 原因 | 取回 |
|---|---|---|
| `train_ADMCFM_cascade_sr.sh`、`train_ADMCFM_stage1_64.sh`、`train_ADMCFM_cascade_64to128.sh`、`export_inpaint_cascade.sh` | 多段生成（cascade），沒有比單段好的證據（`docs/archive/CASCADE_DIFFUSION.md`） | `d635bf6` |
| `train_ADMCFM_vae_mafp.sh`、`train_ADMCFM_patch_undamaged_ref.sh`、`train_laplacian_cfm.sh` | 選用的舊實驗（`docs/ARCHITECTURE_UPDATE_V9.md`），程式模組仍在 `src/model/exp/` | `d635bf6` |
| `train_ADMCFM_refiner.sh` | refiner variant A（paired bridge），已被 variant B 取代 | `80f71fa` |
| `train_ADMCFM_autoguide_bad.sh` | 只差 6 個參數，改成主腳本的覆寫（見上方指令） | `80f71fa` |
| `viz_high_{apg_g0,apg_facfg,hipass_g0,autoguide_fdg,refiner}.sh` | 合併成 `viz_high_di_grid.sh`（`MODE`） | `80f71fa` |
| `eval_base_vs_gt_shift.sh`、`eval_state_spectrum.sh` | 合併成 `eval_diagnostics.sh`（`WHAT`） | `80f71fa` |
| `eval_dinov3_kid.sh` | 被 `eval_suite.sh` 取代 | 未進 git |
| `DamageQuantification/train_{resnet101,swin_b,vit_*}.sh` | 合併成 `train_backbone.sh`；舊腳本指向舊 repo 路徑和已不存在的 `hysteresis` 環境 | `64b8499` |
| `HysteresisPrediction/train_{GRU,LSTM}.sh` | 合併成 `train_hysteresis.sh`，原因同上 | `64b8499` |
| `train_layout_oracle.sh`、`train_opaflow.sh` | 同一份 gn 續訓指令，合併成 `train_gn_finetune.sh`（`EXP=layout` / `opa`） | 未進 git |
| `eval_layout_oracle.sh`、`eval_layout_sensitivity.sh`、`phase1_coarse_ceiling.sh`、`phase1_c2_e2e.sh`、`phase1_ofp.sh` | 合併成 `eval_layout.sh`（`EXP`） | 未進 git |
| `phase0_guidance.sh`、`eval_opaflow.sh` | 合併成 `eval_facfg.sh`（`EXP=guidance` / `opaflow`） | 未進 git |
| `dataset_audit.sh`、`export_inpaint_png.sh`、`opa_free_cache.sh`、`c2_dump_g.sh`、`phase0_critic.sh`、`phase0_spd_onset.sh`、`clock_condition_probe.sh` | 合併成 `misc_tasks.sh`（`TASK`） | 未進 git |
| `v11appr_train_area.sh`、`velocity_fusion_stats.sh` | 併入 `eval_diagnostics.sh`（`WHAT=train_area` / `velocity_fusion`） | 未進 git |
| `tier0_churn.sh` | 和 `eval_suite.sh` 只差預設 `CONFIG` | 未進 git |
| `c2_layout_fm.sh` | 改名為 `train_layout_fm.sh` | — |
| `eval_suite.sh`、`eval_layout.sh`、`eval_facfg.sh`、`eval_diagnostics.sh`、`configs/` | 2026-10-29 搬到 `job/evaluation/` | — |
| `train_ADMCFM_v12mem.sh`、`v12memr.sh`、`v12nbr.sh`、`v12rf.sh`、`v12refiner_hspP.sh` | 只差預設旗標，併進 `train_ADMCFM_v12refiner.sh`（`RECIPE`） | 2026-11-08 |
| `eval_layout.sh`、`eval_facfg.sh`、`eval_imagenet_prior.sh`、`eval_v12_compare.sh`、`eval_v12fix.sh`、`eval_diversity.sh`、`panel_v12basis.sh`、`scratch_grav.sh`、`phase1_fiber_clock.sh` | 實驗已結束，結果在 `docs/redesign/` 或 `docs/eval/`。fiber clock 改成 `misc_tasks.sh` 的 `TASK=fiber_clock`。逐步產生的 suite 設定檔也刪了，腳本下次跑會重寫 | 2026-11-08 |
