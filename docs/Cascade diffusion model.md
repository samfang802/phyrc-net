# Cascade diffusion model（多段生成）

目前試過的「多段生成」：先在低解析度生成整體，再在高解析度補細節。V10 的時序 refiner 不在這份文件裡。

共有兩條路：

| 路線 | 是否訓練第二段 | 嘗試過的解析度 |
|---|---|---|
| A. FreCaS | 不需要（用已訓練好的單段模型） | 64 → 128 |
| B. Cascade SR（兩段都是訓練出來的 CFM） | 第二段需要訓練 | 64 → 128、128 → 256 |

**先講結論：兩條路都沒有量化比較**，只有圖和少量數字，不能判斷多段生成有沒有比單段更好地生成剝落細節。細節見第 4 節。

## 1. A 路線：FreCaS（免訓練，64 → 128）

### 方法

FreCaS（Frequency-aware Cascaded Sampling）是取樣時的技巧：不重新訓練，把一次 128 解析度的 ODE 取樣拆成兩段。

1. **第一段**在 64×64 解析度積分前半段 ODE（到某個停止步 L），先決定整體結構。
2. **解析度轉換**：把第一段的 latent 放大成 128，並用「SNR 對齊」決定第二段從哪一步開始。依據是解析度變大時，同一個噪聲量對應的訊噪比會下降：

   ```
   SNR(z_L) = SNR(z_F) · (s1 / s2)^γ
   ```

   `z_L` 是第一段結束的狀態，`z_F` 是第二段開始的狀態，`s1 = 64`、`s2 = 128`，`γ` 預設 2。流時間 t 和 SNR 的關係用 `SNR(t) = (t / (1 − t))²`。
3. **第二段**在 128 解析度繼續積分到 t = 1。
4. **頻率感知的輸出縮放**：第二段中，速度的高頻部分乘 `w_h`（預設 2.0）、低頻部分乘 `w_c`（預設 0.6），讓第二段專注補高頻細節，不要把第一段已定好的低頻結構推翻。
5. **Mid-block 自注意力**：第一段記錄 U-Net 中間層的注意力圖，第二段融合回去，維持結構一致。

這個專案裡有「單模型」版本（同一個 128 模型在 64 和 128 兩種解析度各跑一段，靠 Patch Diffusion 學過的多尺度）。取樣腳本只支援單模型版本。

### 參數

`use_frecas`、`frecas_stage1_size`（64）、`frecas_stage2_size`（128）、`frecas_stage1_stop_index`（L，−1 表示總步數的一半）、`frecas_gamma`、`frecas_w_h`、`frecas_w_c`、`frecas_pool_kernel`。

### 結果位置（`results/DamagePatternGeneration/CFMADM/`）

- `selfcondnoDI/inference_samples/fixed_di_test_step_14000_ode30_frecas.png`：單模型 FreCaS，固定 DI 面板。
- `2026-08-17_04-35-43/inference_samples/fixed_di_test_step_14000_ode30_frecas_64only.png`：只用 64 解析度的模型。
- 沒有找到 FreCaS 的 6×6 High-DI 圖。

## 2. B 路線：Cascade SR（兩段都訓練）

### 方法

**第一段（Stage 1）**：低解析度的 CFM，用相同的條件（DI、方向、g）生成整體結構。
**第二段（Stage 2）**：另一個 CFM，輸入多一個「低解析度條件圖」通道（`lr_cond`），輸出高解析度圖。

#### 第二段怎麼訓練

訓練時不需要第一段的樣本，而是用真值 HR 影像自己「降級」出條件：

1. 雙三次（bicubic）縮小到 LR 大小，再用高斯模糊（σ 在區間內隨機取）模糊，再雙三次放大回 HR 大小，得到 `lr_degraded`。
2. 在 `lr_degraded` 上加一點高斯噪聲（標準差在 `[0, 0.05]` 隨機取），當成條件 `lr_cond`。噪聲只加在條件上，目標不加噪聲。
3. 目標有兩種：
   - **殘差**（`cascade_predict_residual = True`）：目標 = `HR − lr_degraded`，第二段只需要學第一段沒有的高頻部分。
   - **直接**（`False`）：目標 = `HR`，`lr_cond` 只當條件。
4. 損失的 Min-SNR 權重在殘差模式下用殘差的變異數（程式裡有註解說明：「這個 crop 要補的高頻量」）。

模糊和噪聲的目的是讓第二段習慣「第一段的輸出是模糊、有誤差的」，而不是乾淨的降級真值，降低訓練和推論之間的落差。

#### 取樣

1. 第一段取樣得到 LR 圖 → 雙三次放大到 HR 大小，當作 `lr_cond`。
2. 第二段從噪聲出發，帶著 `lr_cond`，用 ODE 積分。
3. 殘差模式：輸出 = `clamp(第二段輸出 + lr_cond, −1, 1)`；直接模式：輸出就是第二段輸出。

訓練第二段時，驗證和推論面板如果沒有掛上第一段，就用 GT 降級（teacher forcing）；掛上第一段後才是真正的兩段取樣。

這條路線與 spall 生長遮罩、AR 鏈、paired bridge 不能同時使用，也不支援 VAE 潛空間。

### 嘗試過的組合

| Run | 角色 | 解析度 | 第二段目標 | 模糊 σ |
|---|---|---|---|---|
| `2026-10-08_01-01-46_cascade64` | 第一段（64 的 CFM） | 64 | — | 0.8–2.0（僅是設定，沒有第二段在用） |
| `2026-10-08_05-31-44_cascade128` | 第二段 | 64 → 128 | 殘差 | 0.5–1.5 |
| `2026-10-08_13-22-53_cascade128_basic` | 第二段 | 64 → 128 | 直接 | 0.5–1.5 |
| `Cascaded 256` | 第二段 | 128 → 256 | 殘差 | 0.8–2.0 |

共通：15000 步、ODE 30 步、Patch Diffusion。`cascade128` 和 `cascade128_basic` 的第一段都是 `cascade64` 的最後一個 checkpoint。`Cascaded 256` 的第一段是 `selfcondnoDI` 的 14000 步 checkpoint，且這個 run 開啟 self-conditioning，其餘兩個沒有。

### 訓練 job 腳本

`job/DamagePatternGeneration/` 底下：`train_ADMCFM_stage1_64.sh`（第一段）、`train_ADMCFM_cascade_64to128.sh`、`train_ADMCFM_cascade_sr.sh`。

### 結果位置（`results/DamagePatternGeneration/CFMADM/`）

**`2026-10-08_05-31-44_cascade128/inference_samples/`**（殘差版）
- `overview_di_3_high_6x6_selfcondOFF_step_14000_ode30_cascade.png`、`..._step_15000_ode30_cascade.png`
- 搭配 guidance 的版本：
  - `..._step_14000_ode30_coordg2p00_hp2p0_tmax0p55_cascade.png`（高通座標引導）
  - `..._step_15000_ode30_facfg_wl1p00_wh2p00_g0undamaged_cascade.png`
  - `..._step_15000_ode30_facfg_wl1p50_wh4p00_g0undamaged_cascade.png`
- GT：`overview_di_3_high_6x6_GT_test_CTR1.png`
- guidance 是加在第一段上（取樣時把 FA-CFG 的設定傳給第一段模型）。

**`2026-10-08_13-22-53_cascade128_basic/inference_samples/`**（直接版）
- `overview_di_3_high_6x6_selfcondOFF_step_15000_ode30_cascade.png`、GT 圖。

**`Cascaded 256/inference_samples/`**
- `overview_di_3_high_6x6_selfcondON_step_15000_ode30_cascade.png`、`fixed_di_test_step_15000_ode30_cascade_s1.png`（含第一段樣本的面板）、GT 圖。

**`2026-10-08_01-01-46_cascade64/`**：我沒有在 `inference_samples/` 找到 6×6 圖。

## 3. 現有的數字

| Run | val/loss（最後記錄） | 備註 |
|---|---|---|
| `cascade64` | 0.04007 | 64 解析度，val 的算法和 128 不同 |
| `cascade128`（殘差） | 0.01031 | 目標是殘差 |
| `cascade128_basic`（直接） | 0.00855 | 目標是完整影像 |
| `Cascaded 256` | 0.0069 | 256 解析度、殘差 |

**這些 val/loss 不能互相比較**：解析度、目標（殘差或完整影像）都不同，連單段 128 模型（例如 `selfcondnoDI` 0.01628）也不能直接比。

`Cascaded 256` 有做 DI 一致性評估（`eval_siamese_r2/test_metrics.json`，n = 124）：

| 模型 | generated R² | generated MAE |
|---|---|---|
| Cascaded 256 | 0.8612 | 0.0953 |
| 單段 `selfcondnoDI`（128） | 0.8685 | 0.0850 |

差距很小，而且兩者解析度不同，只能說兩段沒有在 DI 一致性上明顯勝出，不能說誰比較好。`selfcondnoDI` 另有 FID / SSIM / PSNR（`test_metrics/metrics_test_step_14000_ode30.json`：FID 210.1、SSIM 0.218、PSNR 14.4），但 cascade 沒有對應的數字。

## 4. 未確認事項

- **沒有任何 cascade 版本和單段模型在同一指標下的並排比較**，所以不知道兩段是否改善了剝落細節。
- `cascade128` 和 `cascade128_basic` 沒有 DI 評估或 FID，殘差版和直接版只能目視比較（第 2 節的圖），我還沒有目視比較過。
- 第二段看到的是 GT 降級出的條件，推論時看到的是第一段的樣本；兩者分佈的差距有多大，沒有量測。第一段如果沒有細節（例如剝落），第二段只能在模糊的基礎上補紋理。
- 在資料只有約 530 張（4 個試體）的限制下，第二段能學到的高頻細節本身就受限於資料，兩段結構不會憑空增加剝落的種類。
- `export_inpaint_cascade.sh` 與這裡的 cascade 無關（我沒有深入確認，只是從名稱判斷是 inpaint 資料匯出）。
