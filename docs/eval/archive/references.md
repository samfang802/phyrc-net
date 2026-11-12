# 參考文獻

依「在這套評估裡用在哪裡」分類。標 ✓ 的條目已在網路上核對過出處；其餘依記憶整理，正式引用前請再核對。

## 為什麼不用 Inception FID、改用 DINOv3

| 論文 | 用在哪裡 |
|---|---|
| Heusel et al. 2017, *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*（arXiv 1706.08500） | FID 的定義，也就是被取代的基準 |
| ✓ Chong & Forsyth 2020, *Effectively Unbiased FID and Inception Score and Where to Find Them*（1911.07023） | FID 的有限樣本偏差因模型而異 |
| ✓ Kynkäänniemi et al. 2023, *The Role of ImageNet Classes in Fréchet Inception Distance*（2203.06026） | Inception 特徵就是 ImageNet 類別資訊 |
| ✓ Stein et al. 2023, *Exposing Flaws of Generative Model Evaluation Metrics and Their Unfair Treatment of Diffusion Models*（2306.04675） | 選 DINO 系列特徵；小資料要檢查背圖 |
| ✓ Woodland et al. 2024, *Feature Extraction for Generative Medical Imaging Evaluation*（2311.13717） | 不微調領域特徵；循環偏差（排除 `oracle_dino*`） |
| ✓ Siméoni et al. 2025, *DINOv3*（[2508.10104](https://arxiv.org/abs/2508.10104)） | 特徵抽取器；dense 特徵品質是用 patch 特徵和做干擾物分割的理由 |
| Parmar et al. 2022, *On Aliased Resizing and Surprising Subtleties in GAN Evaluation*（2104.11222） | 真實圖和生成圖走同一套 resize 和量化 |

## 分佈指標（[distribution.md](distribution.md)）

| 論文 | 用在哪裡 |
|---|---|
| Bińkowski et al. 2018, *Demystifying MMD GANs*（1801.01401） | KID：多項式核、不偏 U-statistic |
| ✓ Jayasumana et al. 2024, *Rethinking FID: Towards a Better Evaluation Metric for Image Generation*（2401.09603） | RBF-MMD（CMMD）；用逐級失真驗證指標 |
| ✓ Gretton et al. 2012, *A Kernel Two-Sample Test*, JMLR 13（[連結](https://www.jmlr.org/papers/v13/gretton12a.html)） | MMD 和它的不偏估計 |
| Kynkäänniemi et al. 2019, *Improved Precision and Recall Metric for Assessing Generative Models*（1904.06991） | `prdc` 的 precision / recall |
| Naeem et al. 2020, *Reliable Fidelity and Diversity Metrics for Generative Models*（2002.09797） | `prdc` 的 density / coverage |
| Meehan et al. 2020, *A Non-Parametric Test to Detect Data-Copying in Generative Models*（2004.05675） | 背圖檢查：和留出的真實圖比最近鄰距離 |
| Alaa et al. 2022, *How Faithful is your Synthetic Data?*（2102.08921） | 生成圖不該比新的真實樣本更接近訓練集 |
| ✓ Borji 2022, *Pros and Cons of GAN Evaluation Measures: New Developments*（2103.09396） | 綜述，用來決定整體指標組合 |

## 條件指標（[condition.md](condition.md)）

| 論文 | 用在哪裡 |
|---|---|
| ✓ Soloveitchik et al. 2021, *Conditional Fréchet Inception Distance*（[2103.11521](https://arxiv.org/abs/2103.11521)） | 「輸出逼真但和輸入無關」要用條件指標才抓得到；鄰居對比與演進指標的動機 |
| DeVries et al. 2019, *On the Evaluation of Conditional GANs*（FJD，1907.08175） | 聯合核 MMD 的出發點（試過後移除） |
| Ravuri & Vinyals 2019, *Classification Accuracy Score for Conditional Generative Models*（1905.10887） | 用預測器評估條件一致性：線性 probe 的想法 |
| Shmelkov et al. 2018, *How Good is My GAN?*（1807.09499） | 同上 |
| ✓ Kriegeskorte et al. 2008, *Representational Similarity Analysis*, Front. Syst. Neurosci. 2:4（[PubMed](https://pubmed.ncbi.nlm.nih.gov/19104670/)） | 不直接比特徵，改比距離結構；演進曲線 `corr` 的想法 |
| Kornblith et al. 2019, *Similarity of Neural Network Representations Revisited*（CKA，1905.00414） | 比較兩組表徵幾何的另一種方法 |
| ✓ Karras et al. 2019, *A Style-Based Generator Architecture for GANs*（Perceptual Path Length，[1812.04948](https://arxiv.org/abs/1812.04948)） | 沿著一條路徑量特徵空間的變化；演進指標是「沿 drift 路徑」的版本 |
| Unterthiner et al. 2018, *Towards Accurate Generative Models of Video*（FVD，1812.01717） | 序列生成要看整段，不只看單張 |

## 統計方法（[protocol.md](protocol.md)）

| 論文 | 用在哪裡 |
|---|---|
| Bounliphone et al. 2016, *A Test of Relative Similarity for Model Selection in Generative Models*（ICLR） | 共用參考集時，對兩模型的差值做配對檢定 |
| ✓ Künsch 1989, *The Jackknife and the Bootstrap for General Stationary Observations*, Ann. Statist. 17(3)（[連結](https://projecteuclid.org/journals/annals-of-statistics/volume-17/issue-3/The-Jackknife-and-the-Bootstrap-for-General-Stationary-Observations/10.1214/aos/1176347265.full)） | 相依資料要整塊刪除（區塊 jackknife） |
| Phipson & Smyth 2010, *Permutation P-values Should Never Be Zero* | 打亂（置換）檢查的統計基礎 |
