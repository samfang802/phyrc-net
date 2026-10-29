# PBS 工作腳本

所有腳本都在 repo 根目錄執行（`cd "${PBS_O_WORKDIR:-$HOME/intership-project}"`），環境是 conda `internship`。參數一律用 `qsub -v 名稱=值,...` 覆寫，不需要改檔案。GPU 編號和 `gpu_id` 的對應見 `docs/GPU.md`。

## DamagePatternGeneration（生成模型）

| 腳本 | 用途 | 常用覆寫 |
|---|---|---|
| `common.sh` | 共用環境設定（`adm_env`），其他腳本 `source` 它 | — |
| `train_ADMCFM.sh` | **主訓練腳本**：單幀 CFMADM（V9/V10 預設設定） | `RUN_TAG`、`MAX_STEPS`、`CFM_MODEL_CHANNELS`、`RESUME_CKPT`（`none` = 從頭訓練）、`EARLY_STOP_PATIENCE`、`CHECKPOINT_EVERY_N_STEPS`、`CHECKPOINT_START_STEP` |
| `train_ADMCFM_refiner_residual.sh` | 殘差時序 refiner（V10 variant B），以 gn 為凍結基底 | `REFINER_BASE_DIR`、`REFINER_BASE_CKPT`、`MAX_STEPS` |
| `train_ADMCFM_oracle_dino.sh` | Oracle 上限：GRU g 換成 GT 的 DINOv3 特徵（見 `docs/eval/oracle.md`） | `ORACLE_DINO_POS_BLUR`（`none` / `shuffle` / `shift`） |
| `test_ADMCFM.sh` | 單一 checkpoint 的固定 DI 推論與測試指標 | `CKPT_PATH`、`SPLIT`、`FIXED_DI_VALUES` |
| `prepare_extractor_from_hf.sh` | 從 Hugging Face 下載 GRU extractor（`./phase1_extractor`） | — |
| `viz_high_di_grid.sh` | 6×6 High-DI 總覽圖 | `MODE`：`plain`、`apg_g0`、`apg_facfg`、`hipass_g0`、`autoguide_fdg`、`refiner` |
| `train_gn_finetune.sh` | 從 gn 續訓加一個機制（gn 的設定原封不動） | `EXP`：`layout`（layout-oracle，`LAYOUT_COND`、`SCRATCH=1`）、`opa`（OPA-Flow，`GROUP`=A/B/C）；`SMOKE=1` |
| `train_layout_fm.sh` | C2 的 layout 生成器（`scripts/train_layout_fm.py`） | `RUNS="名稱\|參數;..."` |
| `misc_tasks.sh` | 單一 python 指令的小工作，輸出在 `log/${TASK}.{out,err}` | `TASK`：`dataset_audit`、`export_inpaint_png`、`opa_free_cache`、`c2_dump_g`、`phase0_critic`、`phase0_spd_onset`（`XML_DIR`）、`clock_condition_probe` |

這兩個資料夾裡合併過的腳本只有一組 PBS 標頭（`gpu_id`、`-o`、`-N`），要換 GPU 或分開 log 時在 `qsub` 加 `-l select=...`、`-o`、`-e`、`-N`。

## evaluation（生成模型的評估）

共用 `DamagePatternGeneration/common.sh` 的環境。

| 腳本 | 用途 | 常用覆寫 |
|---|---|---|
| `eval_suite.sh` | **完整評估**：生成、去干擾物、三種評估、摘要（見 `docs/eval/README.md`） | `CONFIG`、`SPLITS`、`FORCE=1` |
| `configs/` | 評估腳本的模型清單（JSON）。預設是 `configs/eval_suite_models.json`；每次實驗的清單另存一份，例如 `eval_suite_v11.json`、`eval_suite_tier0_churn.json` | — |
| `eval_layout.sh` | （渲染 layout →）eval suite → 圖像側剝落指標 | `EXP`：`oracle`、`coarse`、`c2`、`ofp`、`sensitivity`；`STAGE=seg`；`CONFIG`、`MODELS`、`TAG` |
| `eval_facfg.sh` | FA-CFG 樣本 → eval suite | `EXP`：`guidance`（gn）、`opaflow`（`RUNS="A=<dir> ..."`） |
| `eval_diagnostics.sh` | 模型診斷（不屬於評估流程） | `WHAT`：`shift`、`spectrum`、`all`（gn）；`train_area`、`velocity_fusion`（v11appr） |

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
qsub -v CONFIG=job/evaluation/configs/eval_suite_v11.json job/evaluation/eval_suite.sh
# layout 實驗
qsub -v EXP=ofp job/evaluation/eval_layout.sh
qsub -v EXP=opa,GROUP=B -N opa_B -o log/opa_B.out -e log/opa_B.err job/DamagePatternGeneration/train_gn_finetune.sh
```

## DamageQuantification（DI 回歸器）

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
