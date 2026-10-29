# 時序生成：FlowEdit

這次只改推論。G 仍是單幀模型 `1024-dadaGNsplit` 的 `chjqbp9m`（step 15000）。沒有重訓，沒有用 M。

## 為什麼要換

之前幾條時序都不夠好，原因是同一件事：每一幀的起點是標準高斯。

- 整張從噪聲重畫：單幀好看，閉環六十多幀以後整張變暗。高 DI 6×6 灰階大約 93，單幀大約 160，真實混凝土大約 170。
- 用 M 的硬範圍鎖住外面、裡面重畫：剝落像貼上去的一塊。`step 07555` 那張就是這樣，M 過 0.5 之後只剩底部一條。
- 用 M 的 sigmoid 當每格強度，從上一幀加噪再往前走（SDEdit）：淡影每幀都改一點，看起來像前幾幀疊在一起。把 0.5 以下清掉之後，疊影沒了，但 M 大多時刻過不了 0.5，序列停在第一幀。

M 的標籤 `m*` 在真實高 DI 上是一塊底部實心範圍。網路輸出的 `m` 是從下往上的淡漸層，test 高 DI 的 Dice 約 0.45。它現在不能當「哪裡可以改」的開關。

單幀 G 已經會畫格子、裂縫和剝落。時序要做的是：格子留在上一幀上，只有 g 變了的地方才動。

## 論文

Kulikov、Kleiner、Huberman-Spiegelglas、Michaeli，FlowEdit，ICCV 2025，arXiv:2412.08629。

常見編輯是把圖逆回噪聲，再換成新條件重畫。那條路一定經過高斯，結構容易掉。FlowEdit 不逆回噪聲。同一張噪聲、同一個時刻，用舊條件算一次速度，用新條件再算一次，乾淨圖只加上這兩個速度的差。噪聲在相減時消掉。條件沒變的地方差接近 0，圖留著；條件變了的地方才動。他們在 SD3 和 FLUX 上不需再訓練。

對到這裡：舊條件是上一幀的 g，新條件是這一幀的 g，乾淨圖是上一幀。

Hertz、Aberman、Cohen-Or 的 Delta Denoising Score（ICCV 2023）是擴散模型上的同一件事，用兩次分數的差。我們的模型是 flow，所以用 FlowEdit。

沒有採用的：

- SDEdit（Meng 等人，ICLR 2022）：上一幀加噪再積分。M 的淡影會讓每幀都改一點，結果像疊圖。0.5 門檻又把序列凍在第一幀。
- Differential Diffusion（Levin、Fried，arXiv:2306.00950）：每格不同的強度。強度圖若用現在的 M，和上面同一問題。
- Pyramidal Flow Matching（Jin 等人，ICLR 2025）：逐幀 flow，但是整個視頻模型重訓。這一步不上。

## 我們的時間

網路的 t = 0 是高斯，t = 1 是圖。

```
z_t = (1 - t) * noise + t * image
v   = image - noise
```

FlowEdit 論文把 t = 0 放在圖、t = 1 放在噪聲，和這裡相反。換成我們的 t 之後，從較噪的時刻走到 t = 1，每一步是：

```
noise  ~ N(0, I)          # 每步獨立抽一張
z_src  = (1 - t) * noise + t * x_prev
z_tar  = x_edit + z_src - x_prev
x_edit = x_edit + dt * (v(z_tar, g_t) - v(z_src, g_prev))
```

一開始 `x_edit` 就是上一幀。`z_tar` 和 `z_src` 共用這張噪聲，差只在圖被改過的那一點，以及 g。

`t_start` 是速度差從多噪開始算。0 是整段都算，改動最大。越靠近 1，越只動細紋。步數是這段裡的 Euler 步數。論文裡可以對多張噪聲取平均；這次每步只抽一張。

第一幀沒有上一幀，仍用原本的 Heun 30 步，從噪聲畫到 t = 0 的 g。

## 程式

- `src/model/cfm/ode_mixin.py`：`sample_flowedit_sequence`、`_flowedit_frame`
- `scripts/viz/export_di_tertile_overview_grids.py`：`--flowedit`、`--flowedit_t_start`、`--flowedit_steps`

G 的權重：

`results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit/Opensees Based damage Patterns Generation/chjqbp9m/checkpoints/last.ckpt`

高 DI 6×6 的格子順序鎖在 `2026-09-20_21-48-32` 那張 overview 的 items。test，CTR1，方向 (1, 0)。最後一列右邊是較早的時刻，不是破壞消失。

## 跑了兩次

兩次都是 61 幀，從 t = 0 走到 t = 1860。顯示的 36 格不是時間順序。

**t_start = 0.5，15 步。** 速度差只在比較靠近圖的那一半計算。格子留著，剝落幾乎沒有長出來。論文裡粗結構的差出現在較噪的時刻，0.5 以後主要是細紋，所以這組看不出破壞。

圖：`results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit/inference_samples/overview_di_3_high_6x6_selfcondOFF_step_15000_ode30_flowedit_mg.png`

**t_start = 0，20 步。** 從純噪聲時刻算到 t = 1。這張才看得出方法。

圖：`results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit/inference_samples/overview_di_3_high_6x6_selfcondOFF_step_15000_ode30_flowedit_full_mg.png`

格子還在，沒有底部貼片。隨著時刻，底部出現一塊越來越寬的暗影。最後一列右邊較早的時刻，暗影較淺，和格子順序一致。和單幀、和真實圖比，這塊暗影是糊的，沒有清晰裂縫，也沒有厚的剝落帶。整張仍偏灰。

## 這次說明了什麼

改動量可以只跟著 g 的差走，不必用 M 的遮罩。格子不會被切成兩塊圖。

速度差累積出來的是粗的變暗，不是單幀那種從噪聲直接畫出的紋理。每一步的噪聲是新抽的，細紋在相減和多幀累加之後留不住。論文用多張噪聲平均來穩這個差；這次 `n_avg = 1`。

第一幀仍從高斯畫。後面的幀再怎麼小步修改，都站在這張第一幀上。t = 0 的 g 幾乎沒有破壞，所以起點是一根灰柱。

## 還沒做

- 每步對多張噪聲取平均（論文的 `n_avg`）。
- 把 `t_start` 放在 0 和 0.5 之間，看暗影和裂縫哪個先出現。
- 重訓一個起點就在上一幀附近的 flow。那是 ResShift / 殘差 flow 的方向，這次沒有訓。
