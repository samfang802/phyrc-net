# Guidance（取樣時的引導）

四種在 ODE 取樣時修改速度 `v` 的引導方法：APG、高通濾波引導（coord guidance + high-pass）、FA-CFG、autoguidance。
都是單幀 CFM 的取樣端設定，不需要重新訓練主模型（autoguidance 另外需要一個「壞模型」）。
**四種都沒有量化評估**，結果只有 6×6 High-DI 圖，沒有 FID / SSIM / LPIPS 或 DI 一致性的數字，也還沒有整理過目視比較的結論。

## 為什麼不能直接用一般 CFG

單幀模型訓練時沒有開 g-dropout（`CFG_G_DROP_PROB = 0`），所以沒有「無條件分支」。一般的 classifier-free guidance
`v_unc + s·(v_cond − v_unc)` 沒有現成的 `v_unc` 可以用。下面幾種方法各自用不同方式湊出一個「對照分支」。

## 1. APG（Adaptive Projected Guidance）

### 方法

APG（Sadat 等人）是在 CFG 的差 `diff = v_cond − v_uncond` 上做三件事，目的是避免高引導強度造成過飽和和偽影：

1. **反向動量**（reverse momentum，可選）：對 `diff` 做 `avg ← m·avg + diff`，`m` 為負值（預設 −0.5）。
2. **L2 縮放**：如果 `‖diff‖₂ > 閾值`，把 `diff` 縮到閾值長度（預設 15）。
3. **正交投影**：把 `diff` 拆成平行於 `v_cond` 和垂直於 `v_cond` 兩部分，`diff = diff⊥ + η·diff∥`。`η = 0` 表示完全丟掉平行部分（去飽和）；`η = 1` 表示保留，等於一般 CFG。

最後 `v = v_cond + (s − 1)·diff`。`s = 1` 時不改變輸出。

### 在這個專案裡的「對照分支」

沒有無條件分支，所以 APG 作用在**座標通道的有 / 無**這一對上：
- `v_cond`：正常輸入，帶 XY 座標通道。
- `v_uncond`：同一個模型、同樣的 `z_t, t, c, g`，但把 XY 座標通道換成全零。

所以它引導的是「座標位置資訊帶來的那部分差異」，而不是 g 或 DI。只有在 Patch Diffusion 開啟時有效，也不能和 `coord_guidance` 同時使用。

### 參數

| 參數 | 意思 |
|---|---|
| `apg_guidance_scale` | `s`（0 = 關閉） |
| `apg_eta` | 平行部分的權重，預設 0 |
| `apg_norm_threshold` | 縮放閾值，預設 15；≤ 0 表示不縮放 |
| `apg_momentum` | 動量係數，預設 −0.5 |

### 結果位置

`results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit/inference_samples/other_guidance/`（15k 步、ODE 30 步）：

- `..._apg1p50.png`、`..._apg2p00.png`、`..._apg4p00.png`、`..._apg7p50.png`：`s = 1.5 / 2 / 4 / 7.5`，其餘用預設（`η = 0`，閾值 15）。
- `..._apg2p00_mom.png`：檔名的 `_mom` 是腳本在 `η = 1` 且不縮放時加的標記，也就是接近一般 CFG（加反向動量）的設定，用來和投影版對照。

## 2. 高通濾波引導（coord guidance + high-pass）

### 方法

和 APG 用同一對分支（有座標 vs 全零座標），但是用最簡單的 CFG 形式：

```
Δ = v_xy − v_0
Δ = Δ − avgpool(Δ)          # 高通：去掉低頻
v = v_xy + γ·Δ
```

高通的原因：座標通道的差異如果帶有低頻成分，會把「大塊的剝落」一起放大。去掉低頻後，引導只加強細節（高頻）。
高通方式是用 `avg_pool2d` 取低頻再相減，核大小由 σ 決定：`k = 2·round(2σ) + 1`（σ = 2 → 9，σ = 6 → 25）。σ 越大，被當成「低頻」濾掉的範圍越大，保留的差異越細。

另可限制只在某個流時間範圍內作用：`t_min ≤ t < t_max`。t = 0 是噪聲、t = 1 是乾淨影像。例如 `t < 0.55` 表示只在前半段（影像結構成形期）引導。

### 參數

| 參數 | 意思 |
|---|---|
| `coord_guidance_scale` | `γ`（0 = 關閉） |
| `coord_guidance_hipass_sigma` | 高通的 σ（0 = 不濾波） |
| `coord_guidance_t_min` / `t_max` | 作用的 t 區間 |

### 結果位置

- `1024-dadaGNsplit/inference_samples/other_guidance/overview_di_3_high_6x6_selfcondOFF_step_15000_ode30_coordg2p00_hp2p0_tmax0p55.png`（`γ = 2`、σ = 2、t < 0.55）
- `2026-09-20_21-48-32/inference_samples/` 底下的另一批掃描（15k 步）：
  - `..._coordg2p00_hp2p0_tmax0p55.png`
  - `..._coordg2p00_hp6p0.png`
  - `..._coordg2p00_hp6p0_tmax0p55.png`
  - `..._coordg2p00_hp6p0_tmin0p55.png`
- `2026-10-08_05-31-44_cascade128/inference_samples/..._step_14000_ode30_coordg2p00_hp2p0_tmax0p55_cascade.png`：套在 cascade 模型上的版本。

## 3. FA-CFG（頻率感知 CFG）

### 方法

FA-CFG 來自 FreCaS：對 CFG 的低頻和高頻速度各給一個權重。

```
v = (1 − wl)·v_unc^低 + wl·v_c^低 + (1 − wh)·v_unc^高 + wh·v_c^高
```

- 低頻 = 把 `v` 雙線性縮小一半再放大回來；高頻 = `v` 減去低頻。
- `wl = wh = 1` 時等於 `v_c`，不改變。例如 `wl = 1, wh = 2`：低頻不動，高頻變成 `v_c^高 + (v_c^高 − v_unc^高)`，也就是只放大高頻的條件差。

### 在這個專案裡的「無條件」分支

沒有 g-dropout 的訓練，沒有真正的無條件 g。這裡用 `--facfg_g0 undamaged`：**同一個試體、同一個方向裡 DI 最低的那一幀的 g** 當作 `g_0`，其他輸入不變。所以 `v_unc` 是「這個柱子在沒損傷時的條件」下的預測，引導的是「g 從無損傷變到目前狀態」的差異。

### 參數

`facfg_wl`、`facfg_wh`、`facfg_g0`（目前只有 `undamaged` 一種）。

### 結果位置

- `1024-dadaGNsplit/inference_samples/other_guidance/..._facfg_wl1p00_wh2p00_g0undamaged.png`（`wl = 1, wh = 2`）
- `2026-10-08_05-31-44_cascade128/inference_samples/` 底下，套在 cascade 模型上：
  - `..._step_15000_ode30_facfg_wl1p00_wh2p00_g0undamaged_cascade.png`
  - `..._step_15000_ode30_facfg_wl1p50_wh4p00_g0undamaged_cascade.png`

## 4. Autoguidance

### 方法

Karras 等人（2024）：不用無條件分支，而是用「同架構、較弱的壞模型」`v_bad` 來引導主模型。壞模型要比主模型**更小或更少訓練**，這樣兩者的差就是「模型品質差」而不是「條件差」。
這裡的壞模型：16 通道（主模型 32）、`num_res_blocks = 1`、dropout 0.1、較短的訓練（約 3000 步，腳本現在設 6000 步）。兩個模型取樣時看到的 `z_t, t, c, g` 和座標完全相同。

### 更新式

```
d       = v − v_bad
d_low, d_high = split(d)                 # 同 FA-CFG 的頻率拆分
d       = wl·d_low + wh·d_high           # 頻率解耦權重（FDG）
d       = d⊥ + η·d∥                      # d∥ 是平行於 v 的部分
w(t)    = max(w_min, w·(1 − t)^γ)        # 時間排程
v       = v + w(t)·d
```

預設都是無作用：`wl = wh = η = 1`、`γ = 0`、`w_min = 0`，此時只剩 `v + w·(v − v_bad)`。
`γ` 越大，引導越集中在 t 小（噪聲）的階段，後期逐漸關掉；`w_min` 是下限。

### 參數

| 參數 | 意思 |
|---|---|
| `autoguide_checkpoint`、`autoguide_result_dir` | 壞模型的 checkpoint 和 run 資料夾 |
| `autoguide_weight` | `w`（0 = 關閉） |
| `autoguide_eta` | 平行部分權重（grid 腳本預設 0，其他入口預設 1） |
| `autoguide_gamma`、`autoguide_w_min` | 時間排程 |
| `autoguide_wl`、`autoguide_wh` | 低 / 高頻權重 |

只支援 ODE 取樣，不支援 SDE。

### 壞模型

訓練 run：`2026-10-12_03-59-39_autoguide_bad`，用過 3000 步和 6000 步兩個 checkpoint。

### 結果位置

`1024-dadaGNsplit/inference_samples/autoguide/`（主模型 15k 步、ODE 30 步）。檔名裡的標記：

| 標記 | 意思 |
|---|---|
| `ag0p50`、`ag2p00` | `w = 0.5 / 2.0` |
| `eta0p00` | `η = 0` |
| `g4p0` | `γ = 4`（時間排程） |
| `bad3000` | 壞模型用 3000 步的 checkpoint |
| `fdgL1p00H0p50` | `wl = 1.0`、`wh = 0.5`（只削弱高頻的引導） |

現有的檔案：

- `..._ag0p50.png`：最單純，只有 `w`
- `..._ag0p50_eta0p00.png`、`..._ag0p50_eta0p00_g4p0.png`、`..._ag2p00_eta0p00_g4p0.png`
- `..._ag0p50_g4p0.png`、`..._ag0p50_g4p0_bad3000.png`
- `..._ag0p50_g4p0_fdgL1p00H0p50.png`、`..._ag0p50_g4p0_bad3000_fdgL1p00H0p50.png`
- `..._ag0p50_eta0p00_g4p0_bad3000.png`

**注意：** 沒有 `bad…` 後綴的檔案，是在加上這個後綴之前產生的，檔名不會記錄當時用的是哪個壞模型 checkpoint。

## 5. 目前的限制

- **沒有定量評估。** 四種方法都只有 6×6 圖，沒有指標；哪個設定最好，我不能判斷。
- **APG、高通濾波、FA-CFG 的「無條件」分支都是替代品**（零座標、最低 DI 幀的 g），不是真正的無條件分支，效果是否等同一般 CFG 的直覺，沒有驗證。
- **對剝落細節的幫助未知。** 引導只能放大模型已經學到的差異；如果訓練資料裡沒有足夠的剝落細節（約 61 張 DI > 0.9 的訓練幀），引導放大的可能只是雜訊或過飽和。
- 兩種引導同時使用的組合（例如 autoguidance + FA-CFG）沒有系統性測過。

## 6. 結果目錄總覽

| 位置 | 內容 |
|---|---|
| `results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit/inference_samples/autoguide/` | autoguidance（含 FDG） |
| `results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit/inference_samples/other_guidance/` | APG、coord + 高通、FA-CFG，以及其他（`featg`、`lfguide`、`sag`、`repel`、`nocoord`） |
| `results/DamagePatternGeneration/CFMADM/2026-09-20_21-48-32/inference_samples/` | coord + 高通的 σ / t 區間掃描 |
| `results/DamagePatternGeneration/CFMADM/2026-10-08_05-31-44_cascade128/inference_samples/` | coord + 高通、FA-CFG 套在 cascade 模型上 |
