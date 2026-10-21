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
| `eval_suite.sh` | **完整評估**：生成、去干擾物、三種評估、摘要（見 `docs/eval/README.md`） | `CONFIG`、`SPLITS`、`FORCE=1` |
| `eval_suite_models.json` | `eval_suite.sh` 的模型清單，要比較的模型加在這裡 | — |
| `viz_high_di_grid.sh` | 6×6 High-DI 總覽圖 | `MODE`：`plain`、`apg_g0`、`apg_facfg`、`hipass_g0`、`autoguide_fdg`、`refiner` |
| `eval_diagnostics.sh` | 單幀模型診斷：平移與殘差能量、功率譜 | `WHAT`：`shift`、`spectrum`、`all` |

### 常用指令

```bash
# 主模型
qsub job/DamagePatternGeneration/train_ADMCFM.sh
# autoguidance 用的「差模型」（原本的 train_ADMCFM_autoguide_bad.sh）
qsub -v CFM_MODEL_CHANNELS=16,MAX_STEPS=6000,EARLY_STOP_PATIENCE=3000,CHECKPOINT_EVERY_N_STEPS=1000,CHECKPOINT_START_STEP=1000,RESUME_CKPT=none,RUN_TAG=autoguide_bad \
    job/DamagePatternGeneration/train_ADMCFM.sh
# 評估（先編輯 eval_suite_models.json）
qsub job/DamagePatternGeneration/eval_suite.sh
# High-DI 圖
qsub -v MODE=apg_g0,SCALES="1.5 2.0" job/DamagePatternGeneration/viz_high_di_grid.sh
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

## 已刪除的腳本（2026-10-21 整理）

以下腳本是舊實驗的整份複製，主腳本後來的設定都沒跟上，或者已經合併進上面的腳本。要重跑時用 `git show <commit>:<路徑>` 取回。

| 腳本 | 原因 | 取回 |
|---|---|---|
| `train_ADMCFM_cascade_sr.sh`、`train_ADMCFM_stage1_64.sh`、`train_ADMCFM_cascade_64to128.sh`、`export_inpaint_cascade.sh` | 多段生成（cascade），沒有比單段好的證據（`docs/Cascade diffusion model.md`） | `d635bf6` |
| `train_ADMCFM_vae_mafp.sh`、`train_ADMCFM_patch_undamaged_ref.sh`、`train_laplacian_cfm.sh` | 選用的舊實驗（`docs/ARCHITECTURE_UPDATE_V9.md`），程式模組仍在 `src/model/exp/` | `d635bf6` |
| `train_ADMCFM_refiner.sh` | refiner variant A（paired bridge），已被 variant B 取代 | `80f71fa` |
| `train_ADMCFM_autoguide_bad.sh` | 只差 6 個參數，改成主腳本的覆寫（見上方指令） | `80f71fa` |
| `viz_high_{apg_g0,apg_facfg,hipass_g0,autoguide_fdg,refiner}.sh` | 合併成 `viz_high_di_grid.sh`（`MODE`） | `80f71fa` |
| `eval_base_vs_gt_shift.sh`、`eval_state_spectrum.sh` | 合併成 `eval_diagnostics.sh`（`WHAT`） | `80f71fa` |
| `eval_dinov3_kid.sh` | 被 `eval_suite.sh` 取代 | 未進 git |
| `DamageQuantification/train_{resnet101,swin_b,vit_*}.sh` | 合併成 `train_backbone.sh`；舊腳本指向舊 repo 路徑和已不存在的 `hysteresis` 環境 | `64b8499` |
| `HysteresisPrediction/train_{GRU,LSTM}.sh` | 合併成 `train_hysteresis.sh`，原因同上 | `64b8499` |
