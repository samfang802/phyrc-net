# 分佈指標：DINOv3 KID（可換 DINOv2）

**它回答**：生成圖整體看起來像不像真實照片。

**它不回答**：有沒有跟著條件畫對。把生成圖隨機換到別的條件，KID 完全不變。

## 為什麼不用 Inception FID

- **小樣本偏差**：124 張圖要估 2048 維的共變異矩陣，秩最多 123。FID 在有限樣本下有偏差，而且偏差大小因模型而異（Chong & Forsyth 2020），不同模型之間的比較因此不公平。
- **特徵不對**：Inception 特徵本質上是 ImageNet 類別機率，兩者只差一個仿射變換（Kynkäänniemi 2023）。混凝土裂縫不在任何一個類別裡。自監督特徵（DINOv2/v3）和人的判斷比較一致（Stein 2023）。
- **不微調領域特徵**：Woodland 2024 發現，醫學影像上的領域特徵反而不如 ImageNet 自監督特徵；也警告用生成器自己的資料訓練特徵抽取器，會產生循環偏差。所以這裡用**凍結**的通用 DINOv3。

## 特徵

- 模型：`facebook/dinov3-vits16-pretrain-lvd1689m`，凍結。
- **全域特徵**：`[CLS ; patch 平均]`，768 維。
- **局部特徵**：最後一層 patch token，取固定 4×4 網格上的 16 個位置，384 維。每張圖都取相同位置，所以同位置可以互相比較。

## Backbone 的選擇

DINOv3 在 Hugging Face 需要申請權限。`eval_dinov3_kid.py --dino_model facebook/dinov2-small`（suite 設定檔的 `"dino_model"`）可以換成不用申請的 DINOv2 ViT-S/14；224 px 輸入一樣是 16×16 個 patch，其餘計算完全相同。為了確認換掉 backbone 不會改變結論，我們用同一批樣本（eval-v1 的 `_eq` 樣本）兩種都算了一次（2026-10-21）：

| | test（7 個模型） | val（5 個模型） |
|---|---|---|
| 排名一致性（Spearman ρ）：KID / 鄰居對比 / 演進 ratio / 演進 corr | 0.96 / 0.89 / 0.86 / 0.96 | 0.90 / 1.00 / 1.00 / 1.00 |
| refiner − gn，KID 的 z（DINOv3 / DINOv2） | −5.4 / −5.3 | −5.7 / −5.3 |
| V9 − gn，KID 的 z | −4.5 / −3.4 | −7.7 / −5.0 |
| V9 − gn，鄰居對比的 z | +2.0 / +1.2 | +3.7 / +3.9 |
| gn_shared − gn，KID 的 z | −3.5 / −3.5 | −3.6 / −3.2 |
| cGAN − gn，KID 的 z | −8.2 / −7.0 | — |
| oracle − V9，KID 的 z | −1.8 / **−4.8** | −1.5 / −1.8 |
| oracle − V9，鄰居對比的 z | −3.2 / −3.1 ※ | −6.3 / −5.3 ※ |

※ 比較方向是 V9 − oracle（oracle 較好時為負）。

- **結論不依賴 backbone**：兩個 split 的排名幾乎一樣，refiner、gn_shared 優於 gn、cGAN 的 KID 最低這幾點在兩種特徵下都顯著。（這張表是 JPEG 對齊前算的；「cGAN 的 KID 最低」在對齊後已經不成立，見 [results.md](results.md#對齊前後差多少)。DINOv2 下的對齊版本還沒有算。）
- **變弱的一項**：test 上 V9 比 gn 好的鄰居對比，在 DINOv2 下不顯著（z = 1.2）；val 上兩種都顯著。
- **KID 原始值不能跨 backbone 比**：DINOv2 的 KID 尺度大約是 DINOv3 的 500 倍。要並列時除以「另一根真實柱」的 KID（test：DINOv3 0.072，DINOv2 40.0）。
- **oracle**：見 [oracle.md](oracle.md#換-backbone-的檢查)。

`run_eval_suite.py` 的範例設定檔用 DINOv3 當主 backbone，並設 `"extra_dino_models": ["facebook/dinov2-small"]`，每次都附上這個檢查。

## 估計量

| 輸出鍵 | 核 | 說明 |
|---|---|---|
| `kid_global` | 多項式核 `(xᵀy/d + 1)³` | 主要指標（Bińkowski 2018） |
| `mmd_rbf_global` | RBF，σ = 參考集特徵兩兩距離的中位數 | CMMD 式（Jayasumana 2024）。σ 只由參考集決定，每次執行都一樣 |
| `kid_patch` | 多項式核，作用在 patch token | 局部紋理 |
| `kid_by_di` | 同 `kid_global` | 依參考集 DI 三分位分層，每層約 41 個條件 |
| `kid_by_spall` | 同 `kid_global` | 依 R_spall 分層：≤ 0.05、0.05–0.3、> 0.3 |

**不偏的關鍵：排除同 group 的配對。** 真實圖 i 和從條件 i 生成的 K 張共用 group i，所有核函數的總和都排除同 group 的配對：

- 同一集合內：等於去掉對角線，也就是 KID 原本的不偏 U-statistic。
- 跨集合時：同時去掉了「生成圖和自己的 GT」這種相依配對。不排除的話，cross 項會偏高，讓模型看起來比實際好。
- patch token 繼承所屬圖的 group，所以同一張圖的 patch 互相配對也會被排除。

**計算方式**：所有核函數總和先壓成 G×G 的 group-sum 矩陣（G = 條件數）。任何條件子集的 MMD²，例如 jackknife、分層、隨機對半，都直接從這個矩陣算，不必重算核函數。

KID 用原始數值回報，沒有乘 1000。不偏估計可以是負數，代表和雜訊底線分不開。

## 參考刻度

- `real_halves_kid`：test 真實圖隨機切兩半互比 50 次，期望值約 0。
- `other_real_val_kid`：R307 的真實圖對 test。補圖參考集上是 0.072。模型分數如果接近這個值，就代表「和換一根真實柱子差不多像」。

## 附帶的偽影統計

- `saturated_frames`：飽和像素（≤ 5 或 ≥ 250）超過 1% 的幀所占的比例，生成圖和真實圖各報一個。
- `probe_spall_top`：R_spall > 0.3 的條件上，probe 讀到的剝落量，真實圖和生成圖各報一個（probe 見 [condition.md](condition.md#線性-probe)）。

## 背圖檢查

- 量每張生成圖到 train 真實圖的最近鄰距離（DINOv3 全域特徵）。
- 基準是 train 內部的**跨柱最近鄰**（leave-one-column-out）：一根「新柱子」的真實圖離其他柱子有多近。
- `frac_fake_below_loco_q05` 明顯大於 0.05，就表示有背圖嫌疑。`closest_pairs` 列出最近的 16 組，供人工檢查。

注意，全域特徵對「整體畫面像不像」敏感，對「某個局部物件是不是照抄」不敏感。cGAN 照抄訓練柱的彎鉤，這個檢查沒有抓到。像素空間的最近鄰也要搭配目視才能下判斷。裂縫配置的沿用改用 [damage_texture.md](damage_texture.md#裂縫) 的方法量測。

**不要用 test 真實圖當基準**：CTR1 本身就離 train 柱很遠，用它當基準時，78% 的生成圖都會被判為背圖。這是柱子之間的差異，不是背圖。

## 指標本身的驗證（`dinov3_kid_sanity.py`）

對 test 真實圖做 5 種失真，每種 6 級，看分數是否隨失真等級單調上升：

- 模糊 σ 0.5 → 3
- 雜訊 std 4 → 32
- 擦細暗線（closing 3 → 11）
- 對比降低 10% → 80%
- JPEG 品質 90 → 10

結果所有指標的 Spearman ρ 都 ≥ 0.94，Inception FID 也一樣。所以這個測試只證明 DINOv3 KID「沒有壞」，沒有證明它比 FID 好。要分出高下，需要更細的失真，或請專家排名。

## 已知的盲點

1. **不看條件**：見上。
2. **依樣本數加權**：test 有 86% 的幀幾乎沒有剝落。模型在這些幀上畫得乾淨，分數就好；剝落幀只占約 12%，畫不出來的代價被稀釋。
3. **對「沒畫」和「畫壞」的懲罰不同**：「乾淨牆面加一點灰霧」在特徵空間裡仍落在真實分佈附近，只被罰「少了一個模式」；爆白爆黑的團塊離任何真實圖都遠，會被重罰。這是分佈指標對稀有模式（mode dropping）的盲點。以前我們用它來解釋「cGAN 的 KID 比較好、目視卻比較差」，但 2026-10-28 發現那個差距主要來自 JPEG：參考集和 cGAN 是 JPEG，我們的生成圖不是。對齊後 v11appr、V106C 的 KID 都贏 cGAN（[results.md](results.md#真實感與條件一致性)）。這個盲點仍然存在，只是不再是主因。
4. **對壓縮痕跡敏感**：DINOv3 分得出 JPEG q75 的痕跡，而且影響大到能翻轉模型排名（test KID 差 0.04–0.06）。和參考集比之前，兩邊的壓縮狀態要一致，做法見 [protocol.md](protocol.md#參考集是-jpeg生成圖也要過一次-jpeg)。

所以 KID 一定要和條件指標、損傷型態指標一起報告。

## 試過但不採用

| 設計 | 問題 |
|---|---|
| 聯合核 MMD（`k_img · k_cond`，FJD 的不偏版本） | 用 GRU 特徵當條件時，真實圖的影像距離和條件距離幾乎無關（Spearman 0.06）。改用 DI 當條件時，打亂檢查方向相反：邊際分佈差距大的時候，聯合項沒有解析度 |
| `kid_soft_di`（依 DI 做高斯加權的 KID） | 打亂檢查後分數反而變好。仍會輸出，但不要用 |
| Precision / Recall / Density / Coverage（`prdc`） | test 是同一根柱的連續幀，5-NN 半徑中位數只有 2.9，而任何其他圖到它的距離都在 8 左右，連另一根真實柱的 PRDC 都是 0。除非參考集涵蓋多根柱，否則不要用 |
