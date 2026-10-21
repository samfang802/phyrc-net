# 研究筆記：DINOv3 KID 與其他指標的發展過程

> 這是評估指標從無到有的原始筆記，依時間順序記錄每一次實驗和修正。**較早章節的數字有些已被更正**（SE 低估、演進幅度的偏差等），整理後的說明與目前的結果請看 [README.md](README.md) 和 [results.md](results.md)。

用 DINOv3 抽特徵，再用不偏的 MMD² 比較生成圖和真實圖。目的是在 test 只有 124 張（CTR1 一根柱）的情況下，仍能客觀比較不同模型，並附上誤差與顯著性。

## 為什麼不用現在的 Inception FID

- **小樣本偏差**：124 張圖估 2048 維共變異，秩最多 123。FID 在有限樣本下有偏差，而且偏差大小因模型而異（Chong & Forsyth 2020）。
- **特徵不對**：Inception 特徵和 ImageNet 類別機率只差一個仿射變換（Kynkäänniemi 2023）。混凝土裂縫不在任何類別裡。DINOv2/v3 這類自監督特徵和人的判斷較一致（Stein 2023）。
- **領域特徵有風險**：Woodland 2024 發現醫學領域的特徵反而不如 ImageNet 自監督特徵，也警告用生成器自己的資料訓練特徵抽取器會有循環偏差。所以這裡用**凍結的通用 DINOv3**，不微調。

## 檔案

| 檔案 | 作用 |
|---|---|
| `src/metrics/dino_mmd.py` | 特徵抽取、核函數、分組 U-statistic、jackknife、相對檢定、PRDC、最近鄰 |
| `scripts/eval/generate_eval_samples.py` | 每個 test 條件生成 K 張，存成 npz |
| `scripts/eval/eval_dinov3_kid.py` | 主評估，輸出 `dinov3_kid.json` |
| `scripts/eval/dinov3_kid_sanity.py` | 驗證指標本身：逐級失真時分數是否單調上升 |
| `src/metrics/damage_texture.py`、`scripts/eval/eval_damage_texture.py` | 損傷型態的像素統計（白塊、灰霧、裂縫與訓練柱的重疊），輸出 `damage_texture.json` |
| `job/DamagePatternGeneration/eval_dinov3_kid.sh` | PBS：生成、評估、驗證一次跑完 |

## 流程

```
checkpoint ──generate_eval_samples──▶ npz
                                      fake (N,K,128,128) uint8
                                      real (N,128,128)   uint8  ← dataset GT，同一套量化
                                      DI, filename, cond
npz ──eval_dinov3_kid──▶ DINOv3 特徵 ──▶ 分組 U-statistic ──▶ dinov3_kid.json
```

### 前處理（Clean-FID 的教訓）

真實圖和生成圖走同一條路：128 灰階 CLAHE → uint8 → antialias bicubic 放大到 224 → 複製成 3 通道 → ImageNet 正規化。

- 生成圖：`x*0.5+0.5`，四捨五入成 uint8。
- 真實圖：取 dataset 的 GT 張量，做同一套量化。
- 評估腳本會把 npz 裡的真實圖和「直接讀檔再做同樣前處理」的結果比對，印出 `npz-vs-file mean |Δ|`。目前是 0.000。

### 特徵

- 模型：`facebook/dinov3-vits16-pretrain-lvd1689m`，凍結、不微調。
- **全域**：`[CLS ; patch 平均]`，768 維。
- **局部**：每張圖隨機取 16 個最後一層 patch token，384 維。

### 估計量

每個樣本都帶一個 **group**，也就是它來自第幾個 test 條件。真實圖 i 和從條件 i 生成的 K 張共用 group i。

- 同一 group 的樣本彼此相依，所有估計量都**排除同 group 的配對**（block U-statistic）。
- 在同一集合內，這等於去掉對角線，所以就是 KID 的不偏估計。
- 跨集合時，它也去掉了「生成圖和自己的 GT」這種相依配對；不排除的話，cross 項會偏高，讓模型看起來比實際好。
- patch token 繼承所屬圖的 group，所以同一張圖的 patch 互相配對也會被排除。

所有核函數總和先壓成 G×G 的 group-sum 矩陣（G=124）。任何條件子集的 MMD²，例如 jackknife、DI 三分位、隨機對半，都直接由這個矩陣算出。

| 指標 | 核 | 說明 |
|---|---|---|
| `kid_global` | `(xᵀy/d + 1)³` | 主要指標（Bińkowski 2018） |
| `mmd_rbf_global` | RBF，σ = 真實圖特徵兩兩距離的中位數 | CMMD 式（Jayasumana 2024）。σ 只由參考集決定，每次執行都相同 |
| `kid_patch` | 多項式核，作用在 patch token | 局部紋理 |
| `kid_by_di` | 同 `kid_global` | 依參考集 DI 三分位分層，每層約 41 個條件 |

### 不確定度與比較

- **SE**：以條件為單位的 leave-one-out jackknife。不用 bootstrap，因為有放回抽樣產生的重複樣本會讓 U-statistic 產生偏差。
- **兩模型比較**：兩者共用同一組真實圖，直接對差值 D = MMD²(A,R) − MMD²(B,R) 做配對 jackknife，回報 SE 和 z。概念參考 Bounliphone 2016 的相對相似度檢定。配對後差值的 SE 大約是單一模型 SE 的一半，比各自畫誤差棒再看重疊靈敏。

### 參考刻度

- `real_halves_kid`：test 真實圖隨機切兩半互比 50 次。期望值約 0，不偏估計可以是負的。
- `other_real_val_kid`：val 柱（R307）的真實圖對 test 柱。代表「換一根真實柱子」的距離。

### 條件一致性：線性 probe

- DINOv3 凍結，在 **train** 真實圖上擬合 ridge 回歸（標準化後 RidgeCV），預測 DI 和 R_spall。
- 同一個 probe 套到 test 真實圖得到上限，套到生成圖得到分數（K 張分別算再平均）。
- probe 從沒看過 test 圖，和 Siamese 回歸器是不同的特徵與訓練方式，可以互相對照。

### 背圖檢查

- 每張生成圖到 train 真實圖的最近鄰距離。
- 基準是 train 內部的**跨柱最近鄰**（leave-one-column-out）：一根「新柱子」的真實圖離其他柱子有多近。
- `frac_fake_below_loco_q05` 明顯大於 0.05，就表示有背圖嫌疑。`closest_pairs` 列出最近的 16 組，供人工檢查。

## 試過但拿掉的設計

### 聯合核 MMD（FJD 的不偏版本）

用 `k_img · k_cond` 的乘積核，再把生成圖的條件打亂，預期分數會變差。實測有兩個問題：

1. **用 GRU g（1024 維）當條件時**，真實圖的影像距離和 g 距離幾乎無關（Spearman 0.06）。用 DI 當條件則有 0.55。
2. **改用 DI 當條件後**，把真實圖自己的條件打亂，聯合 MMD 只上升 0.014，遠小於模型的邊際差距（0.3–0.8）。生成圖打亂條件後分數反而下降，因為生成圖之間依條件形成的結構很強，但生成圖和真實圖的交叉項幾乎是平的。

結論：邊際分佈差距大的時候，這個測試沒有意義。改用上面的線性 probe。

### Precision / Recall / Density / Coverage

`prdc` 仍然會輸出，但在這個參考集上會退化：test 是同一根柱的連續幀，5-NN 半徑中位數只有 2.9，而**任何**其他圖到它的距離都在 8 左右。

- 生成圖的 PRDC 是 0。
- 另一根真實柱（val）的 PRDC 也是 0，見 `other_real_val_prdc`。

除非參考集涵蓋多根柱子，否則不要用 PRDC 下結論。

### 用 test 真實圖當背圖基準

一開始拿「test 真實圖到 train 的距離」當基準，結果 78% 的生成圖被判為背圖。原因是 CTR1 本身就離 train 柱很遠：

| 到 train 最近鄰的距離（中位數） | 值 |
|---|---|
| test 真實圖（CTR1） | 6.77 |
| val 真實圖（R307） | 4.37 |
| train 跨柱 | 4.55 |
| 30 步生成圖 | 5.75 |

這是柱子之間的差異，不是背圖。所以改用跨柱基準。

## 指標本身的驗證（`dinov3_kid_sanity.py`）

對 test 真實圖做 5 種失真，每種 6 級，和乾淨原圖比較。所有指標的 Spearman ρ(等級, 分數) 都是 +1.00，只有局部 KID 和 DINOv3 FD 在擦裂縫這項是 +0.94（最後一級飽和）。

| 失真 | KID 第 1 級 / 最後一級 | Inception FID 第 1 級 / 最後一級 |
|---|---|---|
| 模糊 σ 0.5 → 3 | 0.0058 / 0.33 | 14.7 / 363 |
| 雜訊 std 4 → 32 | 0.020 / 0.34 | 45 / 314 |
| 擦裂縫（closing 3 → 11） | 0.23 / 0.52 | 224 / 344 |
| 對比降低 10% → 80% | 0.00006 / 0.066 | 0.8 / 88 |
| JPEG 品質 90 → 10 | 0.018 / 0.17 | 13 / 149 |

注意事項：

- 這些失真太粗，**所有指標都通過**，所以這個測試只證明 DINOv3 KID「沒有壞」，沒有證明它比 Inception FID 好。要分出高下，下一步需要更細的失真，例如只移動剝落區、只改裂縫密度、換成鄰近幀，或請專家排名。
- grayscale closing 也會擦掉柱面上畫的格線，所以「擦裂縫」其實是「擦所有細暗線」。

## 第一次實測（K=2，初步）

單幀 base（`2026-09-20_21-48-32`，step 15000），30 步和 5 步 ODE：

| | KID ± SE | RBF MMD | patch KID | probe R² DI / R_spall |
|---|---|---|---|---|
| 30 步 | 0.306 ± 0.013 | 0.769 | 0.145 | 0.58 / 0.33 |
| 5 步 | 0.411 ± 0.017 | 0.883 | 0.218 | 0.54 / 0.38 |
| 參考：test 對半 | ≈ 0（±0.002） | | | |
| 參考：val 柱 vs test 柱 | 0.216 | | | |
| 參考：probe 用在 test 真實圖 | | | | 0.73 / 0.90 |

- 30 步 vs 5 步：KID Δ = −0.106，配對 SE 0.006，z = −17.4；patch KID z = −16.7。差異非常顯著，方向也符合預期。
- 30 步的分數仍比「換一根真實柱子」差（0.31 > 0.22）。
- DI 三分位：中段 DI 最差（0.41），高 DI 最好（0.25）。
- R_spall 的 probe R²（0.33）遠低於真實圖上限（0.90）。剝落面積是目前最明顯的落差。
- 兩個模型都沒有背圖跡象：`frac_fake_below_loco_q05` = 0。

正式數字請用 K=8 重跑（job 預設）。

## 不同 split、cGAN 的比較（K=2，初步）

單幀 base 模型分別在 test、val、train 上生成，各自和該 split 的真實圖比。cGAN 用 `cGANckpt/CTR1` 的 124 張存檔圖（K=1）。

| 參考集 | n | 模型 | KID ± SE | patch KID |
|---|---|---|---|---|
| test（CTR1，留出） | 124 | base | 0.306 ± 0.013 | 0.145 |
| test（CTR1，留出） | 124 | cGAN | 0.247 ± 0.010 | 0.093 |
| val（R307，留出） | 128 | base | 0.164 ± 0.006 | 0.054 |
| train（4 根柱） | 533 | base | 0.069 ± 0.001 | 0.037 |

**不同 split 的 KID 不能直接比大小**，因為參考集的柱子不同。要先看「真實圖自己離訓練集多遠」：

| 真實圖對 train 真實圖的 KID | 值 |
|---|---|
| CTR1（test） | 0.154 |
| R307（val） | 0.029 |
| train 內：一根柱 vs 其他三根 | 0.034–0.057 |

- **val vs test**：模型分數扣掉這個底線後，val 約 0.135，test 約 0.152，差不多。也就是 test 的 KID 比 val 高很多，主要是因為 CTR1 本身就離訓練柱很遠，模型自己的誤差並沒有差這麼多。這是粗略的減法（KID 並非嚴格可加），只用來判斷量級。
- **train**：0.069，略高於「換一根柱」的底線（約 0.046）。train 的分數遠低於 val 和 test，所以有泛化落差，但沒有背圖跡象（見下）。
- **背圖檢查**：val 的 `frac_fake_below_loco_q05` = 0。train 沒有檢查（train 本來就是訓練圖，最近鄰沒有意義）。

### base vs cGAN（test，CTR1）

| | KID | patch KID | probe R² DI | probe R² R_spall |
|---|---|---|---|---|
| base | 0.306 | 0.145 | 0.58 | 0.33 |
| cGAN | 0.247 | 0.093 | 0.80 | 0.28 |
| 真實圖（上限） | | | 0.73 | 0.90 |

- 配對比較 KID 差 +0.059（z = +9.3），patch KID 差 +0.051（z = +13.5）：cGAN 在分佈上較接近真實圖，這個差異顯著。
- cGAN 的 DI R² 高於真實圖上限，表示它畫出的損傷程度比真實圖更容易被 probe 讀出，不代表它比真實圖更準。
- R_spall 兩者都遠低於上限 0.90，剝落面積是共同的弱點。

比較時要注意：

- **訓練切分不同**：cGAN 用 `id_list_old.json`，訓練集含 R307 和各柱的部分幀，val 也是各柱混合；CTR1 在兩邊都是留出的，所以 test 上可以比，但它們看過的訓練圖不完全一樣。
- **cGAN 圖不能再套 CLAHE**：它輸出的亮度（156.7）已經和 CLAHE 後的真實圖（156.5）一致，再套一次 CLAHE 會失真，所以直接當 uint8 灰階使用。
- cGAN 只有 K=1。KID 是不偏估計，K 小只會讓 SE 變大，不會讓數值偏掉。

### cGAN 的色彩與背圖（追加檢查）

**色彩不影響排名。** cGAN 存的是 RGB jpg，但實際上接近灰階：R、G、B 通道平均差只有約 4/255（JPEG 色度雜訊）。我們把它轉成灰階（luma）再抽特徵，改用單一通道的 KID 在 0.250–0.262，luma 是 0.247，都低於 base 的 0.306，所以轉換方式不改變結論。

**cGAN 有背圖，而且 KID 沒有偵測到。**

- 把 cGAN 的圖和 cGAN 自己的訓練集（`id_list_old`）逐像素比最近鄰：cGAN 圖（左）幾乎就是訓練集 R315 的畫面（中），連**彎鉤形鋼筋**的形狀都相同；真實的 CTR1（右）沒有這個鉤子，它是一個黑色方塊。
- 124 張 cGAN 圖的最近鄰全部來自 R330（69 張）和 R315（51 張），只有 4 張來自其他柱。
- 這個複製沒被 DINOv3 特徵的最近鄰檢查抓到（`frac_fake_below_loco_q05` = 0）。全域特徵對「整體畫面像不像」很敏感，對「某個局部物件是不是照抄」不敏感。

**base 沒有同類問題。** base 的最近鄰訓練畫面和它自己的圖差異很大，沒有複製特定物件（沒有鉤子）。像素最近鄰距離偏低是因為 base 的畫面平滑、偏暗，容易靠近任何一張圖，不是複製。因此**像素距離不能單獨當背圖判準**，一定要搭配目視最近鄰配對。

**為什麼 KID 偏好視覺上較差的 cGAN。**

| | 影像標準差 | Laplacian 能量 |
|---|---|---|
| 真實 test | 32.1 | 26.7 |
| cGAN | 25.3 | 15.5 |
| base | 19.5 | 22.9 |

- 原先的假設是「base 的亮度和對比偏離真實，所以 KID 較差」。**這個假設後來被實測否定**：把每張圖的亮度、對比正規化後，排名完全不變，見下一節。base 的整體平均亮度（154.6）其實和真實（156.5）差不多，只是影像標準差較小（19.5 對 32.1，但真實圖的標準差包含了那根黑色立柱）。
- 另外，cGAN 的內容是錯的：它畫出別根柱子的鉤子，卻幾乎沒有裂縫；base 至少有畫出細裂縫。
- 兩者「生成圖和自己對應的 GT 有多像」都接近隨機：自己的 GT 在 124 張中的排名中位數，cGAN 38.5、base 43.5（隨機約 62，越小越好）。

結論：**全域 KID 衡量的是整體畫面的相似度，抓不到「是不是這根柱子、裂縫對不對」。** 它對 cGAN 照抄的鉤子沒有反應。

可行的改進，尚未實作：

1. **裂縫專用特徵**：先做高通（影像減去低通），或只取暗細線的區域，再抽 DINOv3 特徵算 KID，讓牆面背景和對比不再主導。
2. **排除固定區域**：下緣的方塊、鉤子這種固定物件不是損傷，可以遮掉或裁掉。
3. **配對指標**：生成圖對自己 GT 的距離，與隨機配對比較（上面的排名就是粗略版本）。
4. **背圖檢查加上目視最近鄰面板**，並用像素和特徵兩個空間各查一次。

### 亮度正規化，以及用補圖（inpainted）當參考集

**補圖資料集**是 `Data/DamagePattern_inpaint_selfcondnoDI`：用 CFM 把 SAM 遮罩出的物件補掉。CTR1 柱面下緣那根黑色立柱（和下方的方塊）被補成混凝土，其他區域只有 2–5 個灰階的微小變動。補圖是 128 且已經 CLAHE 過，所以評估時要加 `--no_clahe --image_dir Data/DamagePattern_inpaint_selfcondnoDI/images`，不能再套一次 CLAHE。

**亮度正規化**（`--normalize`）在特徵抽取前，對**所有**圖組（參考集、生成圖、probe 的訓練集）做同一種處理：

| 模式 | 作法 |
|---|---|
| `mean` | 每張圖平均亮度調到 128 |
| `meanstd` | 平均 128、標準差 32（也去掉全域對比；裂縫密度部分就是對比，要小心） |
| `flatfield` | 圖 − 高斯模糊(σ=12) + 128，去掉照明和整體色調，保留細線 |

KID（越小越好），base vs cGAN，test（CTR1）：

| 參考集 | 正規化 | base | cGAN | 差 base−cGAN（z） | patch KID base / cGAN |
|---|---|---|---|---|---|
| 原圖 | none | 0.306 | 0.247 | +0.059（9.3） | 0.145 / 0.093 |
| 原圖 | mean | 0.314 | 0.248 | +0.066（10.4） | 0.147 / 0.091 |
| 原圖 | flatfield | 0.299 | 0.218 | +0.080（10.5） | 0.138 / 0.077 |
| 原圖 | meanstd | 0.326 | 0.242 | +0.083（11.5） | 0.156 / 0.091 |
| 補圖 | none | 0.213 | 0.211 | +0.002（0.4） | 0.111 / 0.074 |
| 補圖 | mean | 0.214 | 0.202 | +0.013（2.5） | 0.111 / 0.070 |
| 補圖 | flatfield | 0.198 | 0.188 | +0.010（1.6） | 0.103 / 0.066 |
| 補圖 | meanstd | 0.202 | 0.191 | +0.011（2.0） | 0.104 / 0.066 |

**結論：**

1. **亮度和色調不是原因。** 不管哪種正規化，base 的 KID 只在 0.298–0.326 之間變動，cGAN 在 0.218–0.248，排名和差距都沒變。base 的 KID 較差不是因為色調。
2. **補圖參考集讓全域 KID 的差距消失。** 去掉那根立柱後，base 的 KID 從 0.306 降到 0.213，cGAN 從 0.247 降到 0.211，差距只剩 +0.002（z = 0.4），統計上持平；其他正規化下 z 也只有 1.6–2.5。base 和 cGAN 的圖裡都沒有這根立柱，所以真實圖裡的立柱原本就是兩者共同的誤差來源，但 base 吃虧更多。
3. **patch KID 仍然偏向 cGAN**（0.111 對 0.074，z ≈ 11），不受正規化影響。局部紋理上 cGAN 的確比較接近真實，但這個指標一樣看不出它照抄的鉤子。
4. **跨柱真實圖距離大半是立柱造成的。** val 柱對 test 柱的真實圖 KID，原圖是 0.216，補圖只有 0.072。也就是 CTR1 離其他柱子很遠，主要因為那根固定物件，不是損傷內容。**用補圖當參考集，才是比較能反映損傷本身的刻度。**
5. **條件 probe 的結果因參考集而翻轉。** 補圖上 base 的 DI R² 是 0.79（上限 0.82），cGAN 是 0.61；R_spall 則 cGAN 較高（0.37 對 0.29）。原圖上方向相反。probe 受固定物件影響，不能單看。

**對訓練的意見：** 實測顯示 base 的問題不在亮度，所以「訓練時統一 trainset 亮度」的優先度不高。真的要做的話，是對訓練目標做每張圖的 `mean` 或 `flatfield` 正規化再 CLAHE，這需要重訓，應當當成獨立實驗，並用上面的補圖加 `--normalize mean` 的設定評估。

### 高剝落幀：KID 為什麼仍然偏向 cGAN

test 124 張裡，R_spall ≤ 0.05 有 107 張（86%），R_spall > 0.3 只有 15 張。依 R_spall 分層（原圖參考集；兩邊都只用該層的真實圖和生成圖）：

| R_spall | n | base KID | cGAN KID |
|---|---|---|---|
| ≤ 0.05 | 107 | 0.345 ± 0.012 | 0.276 ± 0.009 |
| > 0.3 | 15 | 0.238 ± 0.009 | 0.187 ± 0.008 |

（0.05–0.3 只有 2 張，沒有 SE，略。）cGAN 在兩層都比較低，包括高剝落那層。

**兩個模型在高剝落幀上都沒有畫出剝落。** 用凍結 DINOv3 加 ridge（只用 train 真實圖擬合）預測 R_spall，對這 15 張：

| | 預測的 R_spall 平均 |
|---|---|
| 真實標註 | 0.57 |
| probe 用在真實圖 | 0.46 |
| base | 0.18 |
| cGAN | 0.14 |

最高剝落的 12 張並排看：真實圖是大片碎石和鋼筋。base 試著畫碎石，但常常變成爆白、爆黑的高對比團塊；cGAN 多半畫成平整乾淨的牆面加一個鉤子，幾乎沒有剝落，偶爾有暗色污漬。

**偽影的量化：** 飽和像素（≤5 或 ≥250）超過 1% 的幀，真實 0 張、cGAN 0 張、base 4 張；在這 12 張的下半部，飽和像素比例真實 0.02%、base 2.7%、cGAN 0.0005%。

**為什麼 KID 這樣打分：**

1. **KID 依樣本數加權。** 86% 的幀是乾淨牆面，cGAN 在這些幀上畫得乾淨清楚，分數自然好；剝落幀只占約 12%，畫不出來的代價被稀釋。
2. **「畫不出來」和「畫壞」的代價不同。** cGAN 失敗的方式是一張看起來合理的乾淨牆面，在 DINOv3 特徵空間裡仍落在真實圖分佈附近；base 失敗的方式是爆白爆黑的團塊，離任何真實圖都遠。MMD 重罰後者，對前者只罰「少了一個模式」。這是分佈型指標對稀有模式（mode dropping）的盲點。
3. **KID 沒有「剝落存在」這一項。** 它只比整體特徵分佈，不比「這個條件該有多少剝落」。

**因此，KID 回答的是「整體看起來像不像真實圖」，你關心的是「高 DI 時剝落有沒有畫出來、有沒有偽影」，這兩件事要用不同的指標回答。** 目前能直接回答後者的有：R_spall probe（base 略好於 cGAN，兩者都遠低於真實）、飽和像素偽影率（cGAN 比 base 好）。尚缺一個可靠的剝落量測：probe 對真實圖也只有 0.46（真值 0.57），太粗；可以改用 `anylabel` 的剝落標註訓練逐 patch 的剝落分割。

### 加入 `1024-dadaGNsplit`（step 15000，ODE 30 步，K=2）

前面的「base」指 `2026-09-20_21-48-32`（base0920）。`1024-dadaGNsplit`（gn）是另一個單幀 CFMADM run，`inference_samples/high_di_gn.png` 就是它的高 DI 掃描圖。

| | KID（原圖參考） | KID（補圖參考） | 補圖 patch KID | R_spall>0.3 的 probe 讀數 | 飽和幀比例 |
|---|---|---|---|---|---|
| gn | 0.292 | 0.198 | 0.103 | 0.21 | 2.8% |
| base0920 | 0.306 | 0.213 | 0.111 | 0.18 | 3.2% |
| cGAN | 0.247 | 0.211 | 0.074 | 0.14 | 0% |
| 真實 | | | | 0.46（真值 0.57） | 0% |

- gn 的 KID 比 base0920 略好（原圖差 −0.014，z = −3.8），仍輸 cGAN（原圖 +0.045，z = +7.2）。補圖參考集上 gn 的全域 KID 反而比 cGAN 好（−0.013，z = −2.6），但 patch KID 仍是 cGAN 好（z = +9.4）。
- **在高剝落條件下，gn 畫出的剝落最多**（probe 讀數 0.21，cGAN 0.14），仍只有真實圖的一半不到。
- 最高 DI 的 12 個條件中，gn 有 3 個（25%）出現飽和偽影，cGAN 0 個。
- 同一個條件、不同種子，gn 的剝落差異很大：有的種子畫出大片碎石，有的幾乎沒有。條件到剝落的連結還不穩。

評估腳本現在會輸出 `kid_by_spall`、`saturated_frames`、`probe_spall_top`。

### GT 鄰居對比（`nbr_global`、`nbr_patch`）：唯一通過打亂檢查的條件指標

**定義**（`dm.gt_neighbors`、`dm.nbr_contrast`）

1. 對每個條件 i，取 GT 的鄰居 N(i)：同一根柱、同一個面、DI 最接近的 4 張真實幀，不含 GT 本身。
2. 對比值 = 生成圖到「N(i) 和 GT 以外所有真實圖」的平均距離 − 生成圖到 N(i) 的平均距離。
3. 越大越好。減掉「到其他圖」那一項，扣除了整體真實感的落差，只留下和條件有關的部分；GT 本身不在任何一邊，所以不會獎勵「剛好像某一張」。
4. K 張生成圖各算一次再平均。
5. `nbr_global` 用全域特徵；`nbr_patch` 用 16 個固定位置的 patch token，同位置互相比再平均，看得到「損傷長在柱子哪裡」。

**檢查與刻度**

- **打亂檢查**：把生成圖隨機換到別的條件，對比值必須掉到約 0。所有模型、所有設定都通過。`kid_soft_di` 打亂後分數反而變好，沒有通過。
- **上限**：每根其他柱各取一張 DI 最接近的真實照片，算同樣的對比值再平均（`nbr_*_ceiling_other_specimens`）。不用 CTR1 自己的相鄰幀當上限：它們是同一根柱、同一張臉，裂縫位置一幀一幀延續，距離只有 3.1；生成器按設計不知道是哪根柱，不可能畫出 CTR1 自己的裂縫佈局。
- `frac_of_ceiling` = 模型對比值 ÷ 上限。

**結果（K=8；cGAN 只有存檔的 K=1）**

| 參考集 | | gn | base0920 | cGAN | 上限 |
|---|---|---|---|---|---|
| test 原圖 | global | 0.346（61%） | 0.331（58%） | 0.321（57%） | 0.566 |
| test 原圖 | patch | 0.227（62%） | 0.221（60%） | 0.204（55%） | 0.369 |
| test 補圖 | global | 0.397（57%） | 0.358（52%） | 0.320（46%） | 0.692 |
| test 補圖 | patch | 0.258（69%） | 0.240（64%） | 0.212（57%） | 0.374 |
| val 原圖 | global | 0.319（55%） | 0.345（59%） | — | 0.585 |
| val 補圖 | global | 0.423（53%） | 0.413（52%） | — | 0.798 |

- test 上四種設定的排序都是 gn > base > cGAN，和目視判斷一致；同一份資料上，KID 的排序是 cGAN 最好。
- **但差異都不顯著**：test 上 gn vs base 最高 z = 1.6，gn vs cGAN z ≤ 0.6。val 上 gn vs base 的方向在兩種參考集之間相反，也不顯著。
- 結論：這個指標**有效**（真的看條件、方向正確、有可解釋的上限），但在目前的資料量下，**解析度不足以分出這三個模型**。三者都抓到約 50–70% 的「換一根真實柱」能達到的條件訊號。
- cGAN 只能用 K=1，雜訊比另外兩個大；要公平比較，應該用 `cGANckpt/epoch=02999.ckpt` 多抽幾個 z。cGAN 也不能用 val 比，因為 R307 在它的訓練集裡。

### 修正：所有 SE 改成 block jackknife

相鄰幀幾乎是同一張圖，leave-one-condition-out 會把 124 個高度相關的條件當成 124 個獨立樣本，SE 嚴重低估。現在所有 SE（KID、patch KID、分層 KID、`kid_soft_di`、鄰居對比）都改成 leave-one-block-out，一個區塊 = 同一面、同一 drift 級（`dm.frame_block`）。test 只有 **22 個區塊**。

- SE 約為原本的 2–2.5 倍，z 值約減半。
- **本文件前面各節的 SE 與 z 值都是舊算法，高估了顯著性。** 例如 test 原圖上 base vs cGAN 的 KID，z 從 9.3 降為 4.1；gn vs base 從 −5.0 降為 −3.1。原圖上 KID 的模型差異大致仍顯著；補圖上 gn vs cGAN（z = −1.7）、base vs cGAN（z = 0.0）不顯著。

### 干擾物：偵測、移除、另外報

**為什麼要處理**：GT 有干擾物（CTR1 下緣右側的立柱），我們的模型不畫干擾物，cGAN 會畫別根柱的鉤子。干擾物又大又明顯，DINO 一定會看它，所以它會混進每一個指標。

**偵測器**（`src/metrics/interference.py`、`scripts/eval/train_interference_detector.py`）

- DINOv3-S/16，輸入 448，微調最後 2 個 block 和 final norm；卷積解碼器加一條像素層級的 skip，輸出 128×128 遮罩。
- 目標是 `Data/DamagePattern/mask` 的 SAM 遮罩，也就是產生補圖參考集用的那批遮罩。
- 正樣本：train 原圖。困難負樣本：同一批幀的補圖版本，遮罩全 0，讓它學會「補過的地方不是干擾物」。
- 40 epoch，以 val IoU 選模型。權重在 `results/InterferenceDetector/v1/best.pt`。

| | 像素 IoU | 影像層級 P / R | 補圖被誤判比例 |
|---|---|---|---|
| val（R307） | 0.93 | 1.00 / 1.00 | 5% |
| test（CTR1） | 0.63 | 1.00 / 1.00 | 19% |

test IoU 偏低是標註寬度不一致：CTR1 的 SAM 遮罩比立柱本體寬一圈，訓練柱的遮罩貼著物件邊緣；預測位置正確，只是比較窄（漏標的像素 35%，誤標的 3%）。補圖時取樣器本來就會擴張 5 px。

**清理規則**（`clean_mask`）：

- 面積 ≥ 150 px。val、test 的 SAM 元件全部 ≥ 339 px；我們模型被標記的元件 90% < 150 px（下緣暗色碎石上的小點）；cGAN 的元件 98% > 150 px。
- 貼著下緣或左右邊緣 3 px 以內：874 個 SAM 元件全部符合。

**生成圖的移除**（`scripts/eval/inpaint_generated_samples.py`）：被標記的生成圖，用和參考集**同一個** CFM（`selfcondnoDI`，`sample_with_ode_inpaint`）、**同一幀的遲滯條件**補掉。輸出同格式的 npz，多了 `interference_mask`、`interference_flag`；`--png_dir` 會存「生成圖 | 偵測遮罩 | 補圖結果」對照圖。`eval_dinov3_kid.py` 讀到這兩個欄位時，會另外報 `interference`。

| | 被標記的生成圖 | 標記面積（平均） |
|---|---|---|
| gn | 0.3%（3/992） | 177 px |
| base0920 | 0.3%（3/992） | 161 px |
| cGAN | 96%（119/124） | 750 px |

gn、base 被標記的 3 張都是下緣中央的半圓形暗塊，位置正好是訓練柱鉤子的位置，比較像模型畫出的淡淡鉤子殘影。cGAN 的補圖結果在 `results/InterferenceDetector/v1/cgan_CTR1_inpainted/png/`。

### 兩邊都去掉干擾物之後的結果（test，K=8；cGAN K=1）

參考集用補圖，生成圖也去掉干擾物。另外做了兩個對照：

- **安慰劑對照**（`--add_gt_mask`）：每張生成圖都在 GT 的立柱位置再補一次，讓三個模型都帶著和參考集一樣的補圖痕跡。
- **亮度正規化**：`mean`、`flatfield`、`meanstd`。

| 設定 | gn KID | base KID | cGAN KID | gn − cGAN（z） | gn − base（z） |
|---|---|---|---|---|---|
| 去干擾物 | 0.198 | 0.212 | 0.110 | +0.088（7.9） | −0.013（−3.5） |
| ＋安慰劑對照 | 0.194 | 0.208 | 0.116 | +0.078（8.2） | −0.014（−4.1） |
| ＋安慰劑＋mean | 0.195 | 0.209 | 0.117 | +0.077（8.3） | −0.014（−4.1） |
| ＋安慰劑＋flatfield | 0.179 | 0.192 | 0.106 | +0.074（7.9） | −0.013（−3.7） |

- **安慰劑對照幾乎不改變數字**（gn 0.198 → 0.194），所以 cGAN 的優勢不是來自補圖痕跡。
- **亮度正規化也不改變結論**。
- **不是背圖**：在 DINO 空間，cGAN 到最近訓練幀的距離中位數 4.50，比訓練柱之間的跨柱最近鄰（3.97）還遠，0% 的 cGAN 圖比同柱相鄰幀（2.55）更接近任何訓練幀。之前像素空間的「照抄」證據主要是鉤子本身。
- **差距在低剝落幀**：R_spall ≤ 0.05 時 gn 0.231、cGAN 0.126；R_spall > 0.3 時 gn 0.180、cGAN 0.128。
- **條件指標仍然分不開**：鄰居對比 gn − cGAN z ≈ 0；R_spall probe 校準斜率三者都約 0.30，高剝落條件的 probe 讀數三者都約 0.17（真實 0.44）。
- **偽影**：飽和幀 gn 2.3%、base 2.2%、cGAN 0%。

**解讀**：在把干擾物、補圖痕跡、亮度都控制住之後，全域 DINOv3 KID 仍然認為 cGAN 比我們的模型更像真實照片，差距顯著；但「有沒有跟著條件畫出損傷」的指標在三者之間沒有顯著差異。這和目視判斷（cGAN 較差，尤其高 DI 剝落）不一致。目前的證據不足以判斷是指標漏掉了人看重的東西，還是目視判斷有偏差，需要人評實驗來定。

### 損傷演進（`progression_global`、`progression_patch`）

**要回答的問題**：drift 越大，生成圖有沒有越來越「不像沒損傷的樣子」？變化的量和形狀有沒有跟 GT 一樣？KID 把所有圖混在一起比，看不到這件事。

**定義**（`dm.drift_levels`、`dm.progression_curve`、`dm.progression`）

1. 依檔名把 test 分成兩個面（CTR1/NE、CTR1/SW），每個面再依 drift 代碼（00375、00500、…、08000）分級。
2. 每一級取所有圖的 DINOv3 特徵平均，得到這一級的「中心」。
3. 量每一級的中心離最低一級中心多遠，得到一條「演進曲線」。GT、每個模型各一條。
4. 三個分數，最低一級一律不計（它離自己的距離永遠是 0，算進去會讓任何曲線都假裝和 GT 相關）：
   - `ratio`：模型曲線的總和 ÷ GT 曲線的總和。1 = 和 GT 變化一樣多。
   - `corr`：兩條曲線的 Pearson 相關。量的是形狀像不像。
   - `monotone`：曲線對 drift 的 Spearman 相關。1 = drift 越大離得越遠，沒有倒退。

**兩個關鍵設計**

- **不偏的中心距離**（`_centroid_sqdist`）：直接拿樣本平均算距離會被高估約 tr(Σ)/n。每級的圖越少（cGAN K=1，每級約 6 張；gn K=8，約 48 張），高估越多。改用線性核的不偏 MMD² 估計，並排除同一個條件的配對，再取平方根。合成資料測試：真值 1.0 時，不偏估計 0.96（K=1）、1.04（K=8），天真估計 21.8、18.2。
- **以循環為單位的 jackknife**：檔名最後一碼是同一 drift 級的第幾個循環。拿掉一個循環編號，等於拿掉一整條 drift 掃描的一次重複，兩條曲線一起重算。test 只有 7 個單位，z 值要當 t 分佈（自由度 6）看：z = 2 約對應 p ≈ 0.09，z = 3 約 p ≈ 0.02。

**檢查與參考**

- **打亂檢查**：生成圖隨機換到別的條件，等於把 drift 級混在一起。所有模型打亂後 ratio ≈ 0.05、corr ≈ 0、monotone ≈ 0，通過。
- **其他真實柱**：每個條件從其他 5 根柱各取一張 DI 最接近的真實照片，算同樣的分數，代表「換一根真實柱子」能做到的程度。

**結果**（兩邊去干擾物、補圖參考集、K=8；cGAN K=1）

| test（CTR1） | ratio | corr | monotone |
|---|---|---|---|
| 其他真實柱 | 0.53 ± 0.04 | 0.83 ± 0.02 | — |
| gn | 0.38 ± 0.06 | **0.81 ± 0.04** | 1.00 |
| base0920 | 0.37 ± 0.07 | 0.78 ± 0.04 | 1.00 |
| cGAN | 0.40 ± 0.05 | 0.62 ± 0.10 | 0.89 |

（以上是全域特徵；patch 版本結論相同。）

| val（R307） | ratio | corr |
|---|---|---|
| 其他真實柱 | 0.96 | 0.90 |
| gn | 0.66 ± 0.02 | **0.86** |
| base0920 | 0.66 ± 0.03 | 0.82 |

配對比較（corr 的差）：

- test：gn − base +0.029（z = 2.7）、gn − cGAN +0.196（z = 2.1）、base − cGAN +0.167（z = 1.6）。
- val：gn − base +0.039（z = 3.9）。
- ratio 的差都不顯著（|z| < 1）。

**怎麼讀**

- **ratio 分不開三個模型**，都只有 GT 的約 0.4；連其他真實柱也只有 0.53。曲線圖（`progression.png`）顯示 CTR1 在最小 drift 就跳到很高，之後幾乎是平的。對照 panel，那是 2% 起大量出現的**標記筆裂縫描線**；其他真實柱是緩慢上升。所以 CTR1 的變化量有一大部分是這根柱、這場實驗的描線習慣，不是生成器能從遲滯迴圈學到的。在 val（R307）上，其他真實柱的 ratio 是 0.96，模型 0.66，比較能代表「模型少畫了多少變化」。
- **corr 分得開**：gn 的曲線形狀最像 GT，和其他真實柱同級（0.81 對 0.83）。cGAN 在 NE 面 4%→8% 幾乎是平的（高 drift 時剝落沒長出來），所以 corr 和 monotone 都比較低，和目視判斷一致。
- **gn > base 在 test 和 val 上方向一致**，這是目前唯一在兩根留出柱上都重現的模型差異。

**更正**：上一輪用「每張圖到 0% 中心的平均距離」手算的演進幅度（gn 72%、cGAN 40–63%）有偏差：同一級內的雜訊也被算成「變化」，而且每級圖數不同的來源偏差也不同。請以本節的不偏版本為準。

### 不要讀的數字

- **val 的 R_spall probe（R² = −2.25）**：R307 幾乎沒有剝落，真實圖自己的上限也是 −0.67，這個 R² 沒有意義。
- **train 的 probe（DI 0.07、上限 0.29）**：train 上的 probe 是用 val 擬合的，val 只有一根柱，probe 太弱，上限都不高。要評 train 的條件一致性，需要改用「留一根柱」的交叉驗證，目前沒有做。

### 修正

`generate_eval_samples.py` 在 train 模式下原本會跑 1666 批次（約 5 萬張），因為 train 資料集預設把 split 重複很多次。現在強制 `train_virtual_repeat = 1`，每張圖只生成一次。

### 損傷型態的像素統計（`eval_damage_texture.py`）

DINOv3 只能說差多遠，不能說差在哪裡。這一節把目視看到的三種問題直接數值化：

- gn 剝落區裡有白塊；
- cGAN 高 DI 時是灰霧，不是剝落；
- cGAN 低 DI 時畫的是訓練柱的裂縫。

所有門檻都從**訓練柱 drift 代碼 ≤ 250 的完好幀**（54 張）的百分位數得到，沒有在 test 上調。

**像素分類**（7×7 視窗）。`d` = 局部平均亮度減去這張圖的表面亮度，表面亮度取上方 30% 列的中位數，所以整張偏暗或偏亮不會被當成損傷。`E` = 細紋理能量，即 |x − 高斯模糊(x)| 的局部平均。

| 類別 | 條件 | 意義 |
|---|---|---|
| rough | d < 完好 p0.5，且 E ≥ 完好中位數 | 又暗又有細節：露出的混凝土、鋼筋 |
| fog | d < 完好 p5，且 E < 完好中位數 | 比表面暗，卻比完好混凝土還平滑：灰霧 |
| white | d > 完好 p99.5 | 比任何完好表面都亮：白塊 |

報告的量：

- 損傷面積；
- 損傷裡各類別的占比（`white_share`、`fog_share`、`black_share`）；
- 白塊的個數和平均大小；
- 損傷區的平均 E。

**裂縫**。black-hat 取出細暗結構，用列／行投影去掉畫的格線，再去掉小點。兩張裂縫圖用 3 px 容忍度的 F1 比較。

- `f1_gt`：和自己 GT 的 F1。
- `f1_train_max`：和 533 張訓練圖中最像的那張的 F1。
- 基準：GT 自己的 `f1_train_max`，也就是一根沒看過的真實柱碰巧和訓練柱重疊多少。
- `copy_excess` = 模型的 `f1_train_max` − GT 的。
- `before_loading`：每個面最低 drift 那一級的裂縫面積。真實柱在這一級幾乎沒有裂縫。

**白塊會不會讓 DINOv3 低估距離（`--dino`）**。同一個條件的 K = 8 張圖，白塊多寡不同。把每張圖到「自己最低 drift 中心」的距離，也就是演進指標加總的那個量，對白塊面積做組內迴歸，同時控制損傷面積和灰霧。係數為負，表示白塊讓圖在 DINOv3 眼中看起來比較沒損傷。

**結果（test，兩邊都去干擾物；cGAN K = 1）**，高 DI 三分位：

| | 損傷面積 | white_share | 白塊大小 (px) | fog_share | 損傷區 E |
|---|---|---|---|---|---|
| GT（CTR1） | 0.164 | 0.005 | 23 | 0.086 | 10.8 |
| 訓練柱（同 DI 區間） | 0.131 | 0.151 | 38 | 0.076 | 12.0 |
| gn | 0.106 ± 0.039 | 0.067 ± 0.009 | 28 | 0.121 ± 0.015 | 10.7 ± 0.2 |
| base0920 | 0.096 ± 0.039 | 0.051 ± 0.020 | 30 | 0.115 ± 0.025 | 11.4 ± 0.3 |
| cGAN | 0.065 ± 0.023 | 0.009 ± 0.009 | 35 | **0.441 ± 0.090** | **7.7 ± 0.7** |

| 裂縫（低＋中 DI，82 個條件） | f1_gt | f1_train_max | copy_excess | 最低 drift 的裂縫面積 |
|---|---|---|---|---|
| GT | — | 0.395 | — | 0.0027 |
| 另一根真實柱 R307 | — | 0.420 | — | — |
| gn | 0.112 | 0.434 | +0.039 ± 0.025 | 0.0105 |
| base0920 | 0.118 | 0.433 | +0.038 ± 0.023 | 0.0082 |
| cGAN | 0.109 | 0.455 | **+0.060 ± 0.019** | **0.0231** |

**怎麼讀**

- **cGAN 的灰霧可以量化，而且差距最大。** 損傷區有 44% 是灰霧，GT 是 9%，訓練柱是 8%。損傷區的細紋理能量 7.7，其他三者都在 10.7 以上。這是第一個把「灰霧不是剝落」變成數字的量，而 KID 完全看不到它。
- **cGAN 在加載前就有裂縫，並且偏向訓練柱的裂縫配置。** 最低 drift 的裂縫面積是 GT 的 8.5 倍（gn 3.9 倍）。和訓練圖的最佳 F1 比 GT 自己高 0.06（z ≈ 3.2），gn 和 base 約 +0.04（z ≈ 1.6）。不過這個超出量不大：另一根真實柱 R307 碰巧也有 0.42。所以比較準確的說法是「cGAN 不管條件都畫一套和訓練柱相近的裂縫」，而不是逐張複製某一張訓練圖。最像的配對分數最高只有 0.65。
- **gn 的白塊：比 CTR1 多，但沒有比訓練柱多。** gn 的 white_share 是 CTR1 的 13 倍，但只有訓練柱的一半左右。訓練柱的剝落裡本來就有很多白色混凝土碎塊，CTR1 剛好很少。gn 的白塊比訓練柱的小（28 對 38 px），符合「比較散」；個數沒有比較多。至於目視覺得比較多，可能來自形狀，例如圓、邊緣銳利、旁邊是黑色塗抹，這些簡單統計沒有捕捉。
  - 用 −110 為門檻時，gn 的極暗像素比 GT 多；改用校準過的門檻（−137）後差異就消失了，所以不當結論。
- **白塊不會讓 DINOv3 低估演進距離。** 組內迴歸的白塊係數，gn 是 −3.4 ± 5.5（z = −0.6），base 是 +2.2 ± 7.3。高 DI 時 GT 到最低級中心的距離比 gn 多 3.17，其中能歸給白塊的只有 0.016。即使取 95% 區間的極端值，也不到 2%。損傷面積不足大約解釋 7%。剩下的差距主要是 CTR1 特有的東西，例如標記筆描線、鋼筋細節；這和其他真實柱的演進 ratio 只有 0.53 一致。
- **限制**：
  - 門檻是像素層級的手工定義，無法分辨「白色碎塊」和「白色塗漆」。
  - R307 低 DI 的大量 white，是補圖在干擾物位置留下的亮塊，不是剝落，所以 R307 那一列的 share 不可用。
  - cGAN 只有 K = 1，沒有做組內迴歸。

## 用法

```bash
# 一次跑完（PBS）。要比較的模型加進 RUNS
qsub job/DamagePatternGeneration/eval_dinov3_kid.sh

# 手動
python scripts/eval/generate_eval_samples.py --cfm_checkpoint <ckpt> --cfm_result_dir <dir> \
    --out out/a.npz --k 8 [--set apg_guidance_scale=1.5]
python scripts/eval/eval_dinov3_kid.py --samples A=out/a.npz B=out/b.npz --out_dir out/kid
python scripts/eval/dinov3_kid_sanity.py --out_dir out/sanity --save_examples

# val / train 當參考集：加 --split，並關掉會洩漏或沒意義的項目
python scripts/eval/generate_eval_samples.py ... --split val --out out/val.npz
python scripts/eval/eval_dinov3_kid.py --samples base=out/val.npz --other_split "" --out_dir out/kid_val
python scripts/eval/eval_dinov3_kid.py --samples base=out/train.npz --other_split "" --memo_split "" \
    --probe_split val --out_dir out/kid_train
```

外部生成圖（例如 cGAN）：用 `save_samples_npz` 轉成 npz，`real`、`filename` 必須和 test 的 npz 完全一致，才能放在同一次評估裡比較。

- 時序或軌跡取樣器（refiner、bridge、FlowEdit）不能走 `generate_eval_samples.py` 的單幀路徑，改在各自的腳本裡呼叫 `save_samples_npz` 輸出同樣格式。
- 真實圖特徵快取在 `<out_dir>/feature_cache/`。

## 注意事項

- **`oracle_dino*` 這幾個 run 不能用這個指標評**：它們用 GT 幀的 DINOv3 embedding 當條件，等於拿評分用的特徵空間來當輸入，是循環的。
- KID 用原始數值回報，沒有乘 1000。
- 分數只在**同一個參考集、同一個 DINOv3 模型、同一份前處理**下才能互相比較。σ、patch 數、解析度在看到結果前就定好，之後不要改。
- K 越大，生成端造成的變異越小；但真實端固定 124 張，SE 主要來自真實端。

## 參考文獻

| 論文 | 這裡用到什麼 |
|---|---|
| Heusel et al. 2017, *GANs Trained by a Two Time-Scale Update Rule…* (arXiv 1706.08500) | FID 定義，也就是要取代的基準 |
| Bińkowski et al. 2018, *Demystifying MMD GANs* (1801.01401) | KID：多項式核、不偏 U-statistic、子集估計誤差 |
| Chong & Forsyth 2020, *Effectively Unbiased FID and Inception Score…* (1911.07023) | FID 的有限樣本偏差因模型而異，是改用不偏估計的理由 |
| Parmar et al. 2022, *On Aliased Resizing and Surprising Subtleties in GAN Evaluation* (2104.11222) | 兩邊用同一套 resize 和量化，並做 npz 對檔案的檢查 |
| Kynkäänniemi et al. 2023, *The Role of ImageNet Classes in FID* (2203.06026) | 不用 Inception 特徵的理由 |
| Stein et al. 2023, *Exposing Flaws of Generative Model Evaluation Metrics…* (2306.04675) | 選 DINO 系列特徵；小資料要檢查背圖 |
| Jayasumana et al. 2024, *Rethinking FID* (2401.09603) | RBF-MMD 版本；用逐級失真驗證指標 |
| Woodland et al. 2024, *Feature Extraction for Generative Medical Imaging Evaluation* (2311.13717) | 不微調領域特徵；循環偏差的警告（也是排除 oracle_dino 的理由） |
| Bounliphone et al. 2016, *A Test of Relative Similarity for Model Selection* | 共用參考集時，用配對差值比較兩個模型 |
| Kynkäänniemi et al. 2019, *Improved Precision and Recall* (1904.06991) | `prdc` 的 precision / recall |
| Naeem et al. 2020, *Reliable Fidelity and Diversity Metrics* (2002.09797) | `prdc` 的 density / coverage |
| Meehan et al. 2020, *A Non-Parametric Test to Detect Data-Copying* (2004.05675) | 背圖檢查：拿最近鄰距離和留出的真實圖比較 |
| Alaa et al. 2022, *How Faithful is your Synthetic Data?* (2102.08921) | 「真實性」：生成圖不該比新的真實樣本更接近訓練集 |
| DeVries et al. 2019, *On the Evaluation of Conditional GANs* (FJD, 1907.08175) | 聯合核 MMD 的出發點（試過後拿掉，原因見上） |
| Ravuri & Vinyals 2019, *Classification Accuracy Score* (1905.10887)；Shmelkov et al. 2018, *How Good is My GAN?* (1807.09499) | 用預測器評估條件一致性，也就是線性 probe 的想法 |
| Borji 2022, *Pros and Cons of GAN Evaluation Measures: New Developments* (2103.09396) | 綜述，用來決定整體指標組合 |
| Siméoni et al. 2025, *DINOv3* (2508.10104) | 特徵抽取器；dense 特徵品質是用 patch 特徵和干擾物分割的理由 |
| Gretton et al. 2012, *A Kernel Two-Sample Test*, JMLR 13 | MMD 與其不偏估計；演進曲線的中心距離就是線性核的不偏 MMD² |
| Soloveitchik et al. 2021, *Conditional Fréchet Inception Distance* (2103.11521) | 「輸出逼真但和輸入無關」要用條件指標才抓得到；GT 鄰居對比與演進指標的動機 |
| Kriegeskorte et al. 2008, *Representational Similarity Analysis*, Front. Syst. Neurosci. 2:4 | 不直接比特徵，而比「距離結構」；演進曲線的 corr 是同一個想法 |
| Kornblith et al. 2019, *Similarity of Neural Network Representations Revisited*（CKA，1905.00414） | 比較兩組表徵的幾何結構，與 RSA 互補 |
| Karras et al. 2019, *StyleGAN*（Perceptual Path Length，1812.04948） | 沿著一條路徑量影像在特徵空間變化多少；演進指標是「沿 drift 路徑」的版本 |
| Unterthiner et al. 2018, *FVD* (1812.01717) | 時序生成要看整段序列，不只單張 |
| Künsch 1989, *The Jackknife and the Bootstrap for General Stationary Observations*, Ann. Statist. 17(3) | 相依資料要整塊刪除（block jackknife）；區塊與循環單位的依據 |
| Phipson & Smyth 2010, *Permutation P-values Should Never Be Zero* | 打亂（置換）檢查的統計基礎 |

標註 arXiv 編號的條目中，1911.07023、2203.06026、2306.04675、2401.09603、2311.13717、2103.09396 已在網路上查證過，其餘依記憶整理，引用前請再核對。
