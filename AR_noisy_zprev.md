# Auto-Regressive OT-CFM：Single-Stage Noisy GT `z_prev`

本文件整理本次實作的 **時序自回歸（temporal AR）訓練**：以噪聲增強的真實上一幀 latent 作為空間條件（`self_cond`），降低推論時暴露偏差（exposure bias）。

---

## 背景與目標

橋柱損傷圖隨準靜態週期載重逐步發展（固定漂移角遞增 + Push/Pull），影像序列在 `label.csv` 中依 `exp`、`direction`、`timestep` 排列。

目標：在標準 OT-CFM（`noise → z_target`）上，訓練時讓網路看見「上一幀」的結構先驗，推論時可餵入實際生成的上一幀以提高時序一致性。

---

## 方法：Single-Stage Noisy GT `z_prev`

### OT 路徑

- \(z_0 \sim \mathcal{N}(0, I)\)
- \(z_1 = z_{\text{target}}\)（當前幀經凍結編碼／像素空間目標）
- \(z_t = (1-t)\,z_0 + t\,z_{\text{target}}\)
- 速度目標：\(u = z_{\text{target}} - z_0\)

### `self_cond`（空間條件）

1. VAE／目標空間編碼：`z_prev = encode(image_prev)`（無 prev 幀時佔位，見下）
2. `self_cond = z_prev + ε`，樣本級 \(\sigma \sim U(\sigma_{\min}, \sigma_{\max})\)（預設 \(0.05\)–\(0.10\)）
3. Curriculum dropout：以 `drop_prob` 將部分樣本的 `self_cond` 置零
4. **僅使用 target 物理條件**（`gru_target` / AdaGN 的 DI·direction 等）做一次 forward
5. MSE\((v_\theta, u)\) 後反向更新

**不採用**「在 target OT 路徑上、用 prev 物理條件估 draft」的兩階段草稿法（與 OT 幾何不一致）。

### Dropout 課表

線性衰減：

| step | `drop_prob` |
|------|-------------|
| 0 | `ar_drop_prob_start`（預設 0.50） |
| `ar_drop_prob_end_step` | `ar_drop_prob_end`（預設 0.15） |
| 之後 | 維持 `ar_drop_prob_end` |

目前訓練腳本：`MAX_STEPS=20000`，`AR_DROP_PROB_END_STEP=20000`。

### 序列第一幀

同一 `(exp, direction)` 無前一張時：`has_prev=0`，Stage 中 **強制** `self_cond = 0`（`image_prev` 僅佔位）。

---

## Damage mask（物件遮罩）

`Data/DamagePattern/mask/` 的 rebar／物件遮罩（載入後混凝土區≈1、rebar≈0），經 `--use_damage_mask_condition` concat 進 U-Net 空間通道。

AR 開啟時使用聯集：

```text
mask_combined = max(mask_prev, mask_target)
```

與物理 DI／滯迴歷時無關。

---

## 資料管線改動

### `PatternLoopDataset`（`src/dataset.py`）

- Flag：`use_ar_temporal_pairing`（可由 `use_ar_self_conditioning` 開啟）
- 依 `(exp, direction)`、`timestep` 建立上一幀索引
- AR 模式 `__getitem__` 回傳 **dict**（含 `image_prev/target`、masks、`has_prev`、對應 hidden／DI／direction）
- Prev／target 共用水平翻轉；prev 不做 random erasing

### Transform（`src/utils.py`）

- `force_flip`、`apply_random_erasing` 供配對增強

### Batch unpack（`src/model/cfm/module.py`）

- 支援 dict / 舊版 tuple
- 編碼 `z_prev`、`z_target` 後呼叫 AR loss

### `set_dataset_args`（`argument/argument_mixed.py`）

- 修正 AR dict batch 下 `batch[:7]` 造成的 `TypeError`（訓練啟動崩潰）

### Patch Diffusion（`src/model/cfm/patch_mixin.py`）

- `_apply_patch_crop` 同步裁切 `z_prev`（`z_aux`），與 `z_target`／`z_0` 對齊

---

## 訓練 Loss 入口

`CFMSamplingMixin.ar_temporal_flow_matching_loss`（`src/model/cfm/sampling_mixin.py`）

- Curriculum：`_ar_self_cond_drop_prob_at_step`
- 日誌：`train/ar_drop_prob`、`train/self_cond_prob`（keep rate ≈ `1 - drop_prob`）

開啟 AR 時會自動開啟 `use_self_conditioning`，以便 `in_conv` 保留 `self_cond` 通道（約 3 ch），用來承載 noisy `z_prev`——**不是**舊的 batch-level Bernoulli「估當前 \(z_1\)」訓練路徑。

---

## CLI / 訓練腳本

| 參數 | 說明 | 現行 `train_ADMCFM.sh` |
|------|------|------------------------|
| `--use_ar_self_conditioning` | 啟用 AR + temporal pairing | `True` |
| `--use_self_conditioning` | 開啟 self_cond 通道 | `True` |
| `--ar_drop_prob_start` | drop 起點 | `0.5` |
| `--ar_drop_prob_end` | drop 終點 | `0.15` |
| `--ar_drop_prob_end_step` | 課表終點 step | `20000` |
| `--ar_noise_std_min` / `_max` | `z_prev` 高斯 σ 範圍 | `0.05` / `0.10` |
| `--max_steps` | 總訓練步數 | `20000` |
| `--use_damage_mask_condition` | 物件遮罩條件 | `True` |

腳本：`job/DamagePatternGeneration/train_ADMCFM.sh`

---

## 推論注意

訓練時 `self_cond` ≈ noisy GT 上一幀；推論長序列應餵入 **上一時間步生成結果**（或等價 latent），並與訓練相同的 mask／通道配置對齊，才與 teacher forcing 設定一致。

高斯噪聲是對生成誤差的粗近似，結構化誤差（裂縫位偏等）可能仍造成長序列漂移；後續可再考慮 chain／partial OT 等加強項。

---

## 建議對照實驗（非本 run 預設）

1. **No-AR**：其餘相同，關閉 AR／self_cond 通道——檢驗 `z_prev` 條件是否有增益
2. 其餘超參（noise、drop）細調可視 R²／時序連貫性再決定

評測建議同時看 test R² 與同 specimen 隨 timestep／DI 的視覺連貫性。
