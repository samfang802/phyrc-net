# 調研 D / E：小資料真實感 與 條件跟隨 / 外插（可加在現行像素 CFM + ADM U-Net 上）

範圍：原 prompt 第 5 節 D、E；並依協調者更新加上「原創性」「通用性」兩欄，以及一個原創方案（O1）。
查證標記：✓ = 已用 WebFetch 打開 arXiv abs（或會議頁）確認標題 / 作者 / 編號；「未查證」= 依記憶或無法打開。某些細節（例如某篇用哪個 backbone）若只靠記憶，會在該句另外標「細節依記憶」。

專案事實（讀自 repo 文件，非推論）：
- 評估指標（KID global/patch、演進 corr/ratio、鄰居對比）全部用 **DINOv3 ViT-S/16**（`facebook/dinov3-vits16-pretrain-lvd1689m`）特徵；替代 backbone 是 DINOv2-small。損傷型態評估的偵測器 `best.pt` **也是 DINOv3 backbone**（`docs/eval/README.md`）。
- 生成器眼中 CTR1 的條件幾乎疊在 R315 / R330 同 drift 等級上（`docs/eval/condition_g.md` §7），換成損傷狀態編碼器也「沒有改善 CTR1」（`ARCHITECTURE_UPDATE_V11.md` §1）。外插失敗的主因在條件端，生成器端的方法只能放大條件裡已有的差異。
- 已試過：gmix（等同 CcGAN/CCDM 的 vicinal 想法）、FA-CFG 以「最低 DI 幀的 g」當負分支（等同「退化條件引導」）、autoguidance、APG、ControlNet 式時序 refiner（v10.5，相鄰幀 S_{t-1}）、paired bridge。**guidance 類都只有 6×6 目視，沒有量化**（`docs/guidance.md` §5）。

---

## 0. 循環偏差（DINOv3）——所有 D/E 方案共用的規則

**論文實證**
- Kynkäänniemi et al.（ICLR 2023）證明：只要讓生成集在 Inception 特徵空間的 Top-N 類別直方圖對齊真實集，FID 就能大幅下降「而影像品質沒有變好」。也就是「在評估特徵空間裡優化」可以直接刷分。✓
- Projected GAN（Sauer et al. 2021）刻意不用 Inception 當投影網路，並用 KID、SwAV-FID、CLIP-FID 等多個特徵空間確認排名一致。✓
- Vision-aided GAN（Kumari et al. 2022）的判別器庫刻意排除 Inception，另外報 SwAV-FID，理由就是避免「攻擊評估指標」。✓

**建議（推論）**
1. 訓練端（REPA 目標、判別器 backbone、任何 critic）**禁用 DINOv3 的所有尺寸**；DINOv2 同一系列、特徵相近，而且已經是本專案的替代評估 backbone，所以也避開。可選的有 SAM2-S（iREPA 顯示它空間結構好、語意弱，REPA 效果仍佳）、EfficientNet-Lite（Projected GAN 的最佳選擇）、MAE，或從頭訓練的小型 in-domain CNN。
2. 每個實驗至少在兩個特徵空間報 KID：DINOv3（既有）加一個「訓練沒用到」的空間（例如 DINOv2-small，前提是訓練端沒用 DINOv2；或 SigLIP / Inception）。只在 DINOv3 變好、另一個空間不動，就視為刷分。
3. 補上與特徵網路無關的檢查：損傷區的灰階直方圖、徑向功率譜 / 梯度幅值分布、`damage_texture` 的 fog_share / E_damage（但它的偵測器也是 DINOv3，只能算半獨立），以及盲測目視 A/B。
4. O1 的 critic 一律在像素上從頭訓練，不接 DINO。這樣 DINOv3 指標對它來說是獨立的驗證。

---

## 1. 候選方案

### 總表

| # | 方案 | 解決第 4 節哪幾個問題 | 實作量 | 24 GB | 直接加到現行模型？ | 原創性 | 通用性 |
|---|---|---|---|---|---|---|---|
| D1 | 條件式對抗後訓練（非 DINO 判別器＋鄰居錯配負例） | 1 真實感、5 色調；附帶 2 鄰居對比 | 3–4 天 | 可 | 可（從 gn / v11 ckpt 微調） | 低：工程套用，只有「鄰居錯配負例」是小變形 | 高 |
| D2 | 表徵正則化（iREPA / U-REPA 對齊非 DINO 編碼器；或無外部編碼器的 SRA / Dispersive） | 3 剝落破碎（推測）、1 | 1–2 天 | 可 | 可 | 無：工程套用 | 高 |
| E1 | 序列感知的 Contrastive Flow Matching（ΔFM，負例依條件距離挑） | 2 演進幅度、鄰居對比 | 1 天 | 可 | 可（多一項 loss） | 低到中：負例選法是小變形 | 高 |
| E2 | 物理正規化的「損傷時鐘」條件（drift ÷ 力學模型預測的剝落起始 drift） | 4 外插 | 2–3 天 | 可（CPU） | 可（多一個條件標量） | 低到中：土木經驗式 + 條件設計，組合少見但屬領域工程 | 低到中 |
| **O1** | **OPA-Flow：序對錨定 flow + 序數 / 幅度 critic + 無影像條件 rollout（原創）** | 2、4，附帶 5（時序色調）；單幀與時序合一 | 最小驗證 2 天；完整 6–8 天 | 可 | 可（從 gn / v11 微調；錨定版要加輸入通道） | 中到高（推論，見 §1.5） | 高 |

---

### D1. 條件式對抗後訓練（Conditional Adversarial Post-Training, CAPT）

**核心想法。** 在現有 CFM 上保留 FM 損失，另外加一個小權重的對抗損失。生成端取隨機 t ∈ [0.3, 1] 的一步乾淨預測 x̂₁ = x_t + (1−t)·v_θ（不用回傳整條 ODE）送進判別器。判別器 = 凍結的非 DINO 特徵網路（EfficientNet-Lite 或 SAM2-S）＋多尺度輕量 head（Projected GAN / ADD 的作法），加上 projection 條件項（Miyato & Koyama）吃 [z | 損傷標量 | 面 | 正負號]。判別器的輸入做 DiffAugment，正則用 R1 + R2（R3GAN）。
**加一個變形：鄰居錯配負例。** 真實影像 x_j 配上同柱同面、相隔 ≥ 1 個 drift 等級的條件 c_k，判成「假」（GAN-CLS 的 matching-aware 判別器，推廣到連續、有序的條件）。這直接懲罰「對相鄰條件畫一樣的圖」，對準鄰居對比 0.41 → 0.69 的缺口。
**最便宜的前置版本：** Discriminator Guidance（Kim et al. 2023）。主模型完全不動，只訓練一個時間條件判別器，取樣時加上 ∇log(D/(1−D)) 的修正項。一天內能看出判別器信號是否指向正確方向。

- **解決**：1（紋理、裂縫邊緣、剝落斷面是判別器最擅長的局部統計）、5（偏離分布的色調會被懲罰）；錯配負例對 2 有幫助。
- **依據**：Projected GAN（≤1k 張的 Pokemon 833、Art Painting 1000、AnimalFace-Dog 389 都有 SOTA）✓；Vision-aided GAN（100-shot 到 1k，FID 大幅下降）✓；ADD（凍結 DINOv2 ViT-S 判別器、條件 projection）✓；ADT（保留 diffusion loss 防判別器 hacking、對多步取樣做對抗監督）✓；CAFM（以判別器取代 MSE，主要做 flow-matching 模型的後訓練，ECCV 2026）✓；Seaweed-APT ✓；R3GAN ✓；DiffAugment ✓；ADA ✓；Discriminator Guidance ✓；Projection discriminator ✓。
- **實作量**：3–4 天（判別器 + 訓練迴圈 1.5 天，調權重 / 正則 1.5 天）。
- **24 GB**：可。128×128 灰階、U-Net 寬 32；EfficientNet-Lite 約 5M 參數，SAM2-S 也放得下。
- **最大風險**：(a) 530 張判別器很快背下訓練集，對抗梯度變成雜訊或把生成器推向背圖（舊 cGAN 的 mode collapse 是前例：同條件換噪聲只差約 3 灰階）；(b) 在 DINO 空間刷分（見 §0）。緩解：FM 損失權重保持 1、對抗權重 0.01–0.1、DiffAugment、監控「同條件兩張圖的平均差」不可掉到 15 灰階以下。
- **能否直接加**：可，從 gn 或 v11 的 ckpt 微調 3–5k 步。
- **原創性**：**無原創性，屬工程套用**。最接近的是 ADD / ADT / CAFM（凍結特徵判別器 + 保留擴散損失）和 GAN-CLS（錯配負例）。把錯配負例改成「同序列、相隔 k 等級的有序條件」只是小變形，不足以單獨構成貢獻。
- **通用性**：高。只假設「有真實影像可以當判別目標」。需要針對領域調整的只有特徵網路（灰階、非自然影像）；醫學、材料影像同樣可用。錯配負例要求條件是有序的序列，這在縱向資料普遍成立。

### D2. 表徵正則化（iREPA / U-REPA，或不用外部編碼器的 SRA / Dispersive）

**核心想法。** 把 U-Net 中段特徵（U-REPA：middle stage 最好，先過 MLP 再上採樣）對齊凍結視覺編碼器的 patch 特徵。用 iREPA 的作法：卷積投影加空間正規化，強調空間結構。編碼器用 SAM2-S（不用 DINO 系列）。另一條路完全不用外部編碼器：SRA（模型自己的高噪聲淺層對齊低噪聲深層）或 Dispersive Loss（batch 內特徵互推，沒有正例）。

- **解決**：iREPA 的結論是「空間結構比語意重要」，因此可能幫助 3（大片剝落連成一塊、形狀合理）。對 1 是否有幫助不確定。
- **依據**：REPA ✓、iREPA（27 個編碼器；SAM2-S 只有約 24% ImageNet linear-probe 準確率，效果仍勝過 PE-Core-G；做過像素空間 JiT）✓、U-REPA（U-Net 版，NeurIPS 2025）✓、HASTE（後期關掉對齊）✓、SRA ✓、Dispersive ✓。
- **實作量**：1–2 天。
- **24 GB**：可。
- **最大風險**：所有證據都是 ImageNet 規模的**收斂速度**，沒有任何一篇做過幾百張的小資料。對單一領域灰階影像，SAM2 / DINO 的 patch 特徵是否有用也不確定。Dispersive 把 batch 內特徵推開，可能**加大**同條件樣本變異（問題 5 已經偏大，這點是推論）。HASTE 指出對齊到後期會礙事。
- **能否直接加**：可。
- **原創性**：**無原創性，屬工程套用**。
- **通用性**：高，但效益未知。

### E1. 序列感知的 Contrastive Flow Matching

**核心想法。** ΔFM 的損失是 ‖v − u⁺‖² − λ‖v − u⁻‖²，負例 u⁻ 是 batch 內另一個樣本的 flow（論文用 λ = 0.05；λ = 0.15 就明顯變差）。原論文的負例是隨機抽的，也沒分析「負例跟正例很像」的情況。這裡有一個本專案特有的陷阱：相鄰幀影像平均絕對差只有 0.02，若把它們當負例，等於叫模型把幾乎相同的圖推開。所以改成依條件距離挑負例：
- 同柱同面、相隔 ≥ 2 個 drift 等級的幀當「硬負例」；
- 相隔 0–1 等級的排除；
- λ 乘上一個隨條件距離（損傷標量差）遞增的權重。

可選：在 U-Net 瓶頸特徵上加 Rank-N-Contrast，標籤用損傷標量，讓內部表徵依損傷程度排序。

- **解決**：2（演進 ratio 0.38 → 真實 0.53、鄰居對比 0.41 → 0.69）。ΔFM 的設計目的正是「不同條件的 flow 要分開」。
- **依據**：Contrastive Flow Matching（ICCV 2025）✓、Rank-N-Contrast（NeurIPS 2023）✓、hard negative 背景（Robinson et al. ICLR 2021，未查證）。
- **實作量**：1 天（loss 十幾行；負例索引要用 dataset 的 specimen / face / rank）。
- **24 GB**：可，不需要額外 forward。
- **最大風險**：ΔFM 的增益隨 batch 變大而增加，本專案 batch 小；負例推開的方向可能是亮度 / 色調而不是損傷（問題 5 變更糟）。λ 要小。這項只能放大條件裡已有的差異，**對 CTR1 外插（問題 4）幫助有限**（推論，依 condition_g.md 的診斷）。
- **能否直接加**：可。
- **原創性**：低到中。最接近的是 ΔFM（隨機負例）；RobustSpeechFlow（arXiv 2605.22083，只在搜尋結果看到，**未打開 abs，未查證**）用結構化負例做 TTS。我沒搜到「連續 / 有序條件下依條件距離挑 ΔFM 負例」的工作，但這個差異偏小，單獨不足以構成主要貢獻，比較適合當 O1 的一個元件。
- **通用性**：高。只假設條件有距離或順序。

### E2. 物理正規化的損傷時鐘（physics-normalized damage clock）

**核心想法。** CTR1 的軸力、箍筋比、斷面都高於訓練柱。生成器沒看到柱參數，只看到 drift / 能量等狀態，所以把 CTR1 畫成 R315 / R330。改法是給生成器一個「無因次的進度時鐘」：τ = 已達最大 drift ÷ 力學經驗式預測的剝落起始 drift（以及鋼筋挫屈起始 drift）。Berry & Eberhard（PEER 2003-18）的模型以軸壓比、長寬比等柱參數預測保護層剝落與鋼筋挫屈的起始變形，用 450 多根柱的 UW-PEER 資料庫校準。若 CTR1 的軸力讓預測的剝落起始 drift 變小，同樣 drift 下 τ 就比訓練柱大，「外插」就有機會變成「在訓練過的 τ 範圍內內插」。
另一條路（**未查證**）：專案的 `info_w_mander_big.csv` 欄位（configuration、transverse_type、failureType…）看起來來自 PEER Structural Performance Database。該資料庫的原始版本可能記錄了部分試體的剝落 / 挫屈觀測位移。如果有，就能在 124 根無影像柱上直接監督「剝落起始」，而不必只靠 4 根影像柱。這要先查原始資料。

- **解決**：4（外插）。這是 D/E 裡**唯一直接處理「CTR1 條件看起來像 R315」這個根因**的方案。
- **依據**：Berry & Eberhard, PEER Report 2003-18 ✓（報告頁已打開；公式本身的係數**未查證**，PDF 無法抽文字）；PEER SPD User's Manual（Berry, Parrish, Eberhard 2004）——搜尋有看到，**未打開，未查證**。DDOM（ICML 2023）的條件重加權＋CFG 聲稱能產生超過資料最大值的目標 ✓，可作為「條件外推」的一般參考。
- **實作量**：2–3 天（經驗式 0.5 天、驗證 τ 對 6 根影像柱的剝落面積是否單調、且 CTR1 落在訓練範圍內 1 天、接進 v11 的標量分支 0.5 天）。
- **24 GB**：不相關（CPU）。
- **最大風險**：經驗式的離散度約 25–30%（CoV，挫屈模型的數字，見 PEER 頁），而且它預測的是「起始」不是「面積」。若 CTR1 的 τ 仍落在訓練範圍外，就沒有用。
- **能否直接加**：可，當 v11 `e_d` 分支的第 5 個標量。
- **原創性**：低到中。用力學經驗式把條件正規化，在土木是標準作法；用它當生成器條件來把外插轉成內插，我沒查到前例，但這屬於領域工程。
- **通用性**：低到中。通用的原則是「用機制模型把外部驅動條件正規化成無因次的進度時鐘」，例如醫學上用預測的發病時間對齊病程。但具體公式完全是 RC 柱特有的。

### O1（原創，推論）. OPA-Flow：Ordinal Pair-Anchored Flow

> 以下整節是**我的推論與設計**，不是任何論文的結果。

**一句話。** 單幀和時序用同一個網路：條件是「目標狀態 c_j」加上一個可有可無的「較早的錨定幀 (x_i, c_i)」。監督有三層：一般 FM 損失，加上兩種不需要更多影像的訊號——序列內序對的**序數 / 幅度一致性**，以及 124 根**無影像遲滯迴圈上的生成 rollout**。

**架構（統一單幀與時序）**
- 網路輸入：x_t ‖ 座標 ‖ 錨定影像 a（null 時為 0）‖ 錨定旗標；條件 h = e_c + α·e_z(c_j) + α_d·e_d(c_j) + β·e_Δ(c_j − c_i)（null 時 e_Δ = 0，β 初值 0）。
- 訓練時，對目標幀 j，以機率 p_null = 0.3 不給錨定（= 單幀任務）；否則從同柱同面「所有較早的幀」（含加載前那一幀）均勻抽 i < j。
- 推論：單幀 = null 錨定；時序 = 錨定到前一幀，或錨定到稀疏關鍵幀，以減少誤差累積。不再有「單幀模型 + 另接 refiner」。
- 小資料理由 1：序對數量是 O(n²)。8 條序列（4 柱 × 2 面）、每條約 66 幀，就有約 1.7 萬個 (i, j) 序對，而且是任意間隔。現行 refiner（v10.5/10.6）只用相鄰幀，大約 530 對。

**序數 / 幅度 critic R_ψ（先訓練、凍結）**
- R_ψ(x_a, x_b) 輸出兩個量：「b 比 a 晚」的 logit，以及估計的進度差 m̂。用同柱同面的真實序對訓練，數量是 O(n²)。網路是像素小 CNN（不用 DINO）；兩張圖各自做強亮度 / 對比增強，讓它**不能靠色調判斷先後**。
- m̂ 的目標是**與柱無關的量**：損傷狀態編碼器預測的強度比下降、累積能量差。這些量來自遲滯迴圈，所有 160 根柱都有。
- 小資料理由 2：相對判斷（「哪張比較晚、差多少」）在各柱之間比絕對外觀更容易遷移。單調剝落是所有柱共有的規律；絕對紋理則是各柱特有。

**生成器的額外損失**（取 t ∈ [0.5, 1] 的一步預測 x̂ = x_t + (1−t)v_θ，梯度不穿過整條 ODE）
- L_ord = softplus(−logit R(a, x̂_j))：生成結果必須「看起來比錨定晚」。
- L_mag = Huber(m̂(a, x̂_j) − Δenc_ij)：變化幅度要符合遲滯迴圈所說的損傷進度差。這一項直接對準演進 ratio 0.38 vs 0.53。
- **無影像條件 rollout**：從 124 + 30 根無影像柱抽兩個狀態 s < s′，用**同一個起始噪聲**各產生一次一步預測（共享噪聲讓裂縫佈局相同，差別只來自條件）。對這一對施加 L_ord 和 L_mag。生成器因此在訓練時就**看過 CTR1 類條件**（160 根柱裡有高軸力柱），而且有梯度。小資料理由 3：影像只有 4 根柱，但條件空間的覆蓋可以擴到 154 根。
- 總損失：L = L_FM + λ_o·L_ord + λ_m·L_mag + λ_f·(L_ord + L_mag)_imagefree，λ 都取小值（0.01–0.1）。

**依賴的前提**
1. 進度大致單調（剝落只增不減）。
2. critic 在留柱外的柱上仍然有效（最小實驗第一天就檢查）。
3. 編碼器的 Δenc 對 CTR1 真的比 R315 大。如果編碼器把 CTR1 看成 R315（目前的診斷），L_mag 也救不了外插，這時要和 E2 的物理時鐘合用，把 τ 的差也放進 Δenc。

**和最接近的 prior work 比較**（都實際搜尋過；「沒搜到」不代表不存在）

| 工作 | 相同處 | 差異 |
|---|---|---|
| TADM / TADM-3D（Litrico et al., MICCAI 2024 / arXiv 2025）✓ | 以 baseline 掃描為錨定、預測增量；用預訓練的 brain-age estimator 讓生成結果符合預期的年齡差；3D 版有雙向時間正則 | 他們的 critic 是**絕對**年齡回歸器，用的是有影像的配對資料；O1 用**相對**序數 / 幅度 critic，從序列內序對學，而且在**無影像的外部驅動序列**上施加；另外有 null 錨定，把單幀合進來 |
| ArrowGAN（Hong et al., Neurocomputing）✓ | 用時間方向分類當輔助任務 | 那是判別器的輔助任務、無條件影片；O1 是凍結的 critic，加上外部物理量校準的「幅度」項，用於條件 flow |
| RankCGAN（Saquil et al., BMVC 2018）✓ | 以成對比較的 ranker 控制連續屬性 | 不是時序、沒有外部驅動條件、沒有無影像的條件 rollout |
| DRaFT / DDPO ✓ | 在「沒有對應影像的條件」上用可微 reward 微調 | reward 是逐樣本的絕對分數；O1 的訊號是同一 rollout 內兩個條件之間的**相對**一致性，critic 也是從序列序對學來的 |
| MCVD ✓、SADM ✓、Dao et al. 2024 ✓、本專案 v10.5 refiner | 一個模型同時處理有 / 無前幀，或以 baseline + 時間間隔為條件 | 只有「統一錨定」這一層和 O1 重疊；O1 的貢獻在訓練目標（序對 critic + 無影像 rollout），不是錨定架構本身 |
| Δ-LFM（Chen et al., ICLR 2026）✓ | 讓潛空間軌跡隨嚴重度單調 | 需要每張影像的嚴重度標籤、在潛空間；不使用無影像條件 |
| CcGAN / CCDM vicinal ✓ | 讓條件附近的資料互借 | 等同已經試過的 gmix；不提供序數或幅度訊號 |

搜尋「序數 critic reward + 無影像條件 + 時序生成」「arrow-of-time 當生成損失」「單調進程擴散 + 無配對條件」「all-pairs 錨定 + null anchor 統一」都沒有找到同樣的組合（見 §2 的搜尋紀錄）。我判斷**「由序列內序對學來、以外部驅動量校準幅度的相對 critic，施加在無影像的驅動序列上」**是主要的原創點；錨定架構本身不算。

**通用性**
- 適用條件：有少量帶順序的影像序列；有大量**沒有影像但有驅動條件**的序列；進度大致單調；有一個能把驅動條件轉成與個體無關的進度量的編碼器。
- 可推廣的例子：縱向 MRI 的大型臨床隊列（很多人只有臨床量表、沒有影像）；材料演化（大量製程紀錄 / 模擬，只有少量顯微照片）；植物生長（氣象 / 灌溉紀錄多，照片少）。
- RC 柱特有的部分：遲滯迴圈編碼器、損傷標量的選擇、剝落單調。
- 不適用：非單調進程（癒合、季節性）。這時可以拿掉 L_ord，只留帶號的 L_mag。

---

## 2. 論文清單

**D：表徵對齊**
- Yu et al., 2024/ICLR 2025, *Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think*, arXiv:2410.06940 — https://arxiv.org/abs/2410.06940 ✓ — REPA 原始方法；證據是 ImageNet 上的收斂加速，不是小資料。
- Singh et al., 2025/ICLR 2026, *What matters for Representation Alignment: Global Information or Spatial Structure?*, arXiv:2512.10794 — https://arxiv.org/abs/2512.10794 ✓ — 空間結構比語意重要，SAM2-S 也有效，有像素空間 JiT 結果；是選「非 DINO 目標」的依據。
- Tian et al., 2025/NeurIPS 2025, *U-REPA: Aligning Diffusion U-Nets to ViTs*, arXiv:2503.18414 — https://arxiv.org/abs/2503.18414 ✓ — U-Net 版 REPA（中段對齊、manifold loss），最接近本專案骨幹。
- Wang et al., 2025, *REPA Works Until It Doesn't: Early-Stopped, Holistic Alignment…* (HASTE), arXiv:2505.16792 — https://arxiv.org/abs/2505.16792 ✓ — 對齊在後期會礙事，要提早關掉。
- Jiang et al., 2025/ICLR 2026, *No Other Representation Component Is Needed…* (SRA), arXiv:2505.02831 — https://arxiv.org/abs/2505.02831 ✓ — 不用外部編碼器的自我對齊，天然避開循環偏差。
- Wang & He, 2025, *Diffuse and Disperse: Image Generation with Representation Regularization*, arXiv:2506.09027 — https://arxiv.org/abs/2506.09027 ✓ — 無正例的特徵分散；可能加大同條件變異。

**D：判別器 / 對抗後訓練**
- Sauer et al., NeurIPS 2021, *Projected GANs Converge Faster*, arXiv:2111.01007 — https://arxiv.org/abs/2111.01007 ✓ — 凍結特徵判別器在 389–1000 張的資料集上達 SOTA；刻意避開 Inception。
- Kumari et al., CVPR 2022, *Ensembling Off-the-shelf Models for GAN Training*, arXiv:2112.09130 — https://arxiv.org/abs/2112.09130 ✓ — 100-shot 到 1k 張大幅改善；示範如何避免攻擊評估指標。
- Sauer et al., 2023, *Adversarial Diffusion Distillation*, arXiv:2311.17042 — https://arxiv.org/abs/2311.17042 ✓ — 凍結 DINOv2 ViT-S 判別器加條件 projection（HTML 全文確認）；本專案要把 DINO 換掉。
- Sauer et al., 2024, *Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation* (LADD), arXiv:2403.12015 — https://arxiv.org/abs/2403.12015 ✓ — 用生成模型自己的特徵當判別器；目標是蒸餾加速，不是本題的重點。
- Lin et al., ICML 2025, *Diffusion Adversarial Post-Training for One-Step Video Generation* (Seaweed-APT), arXiv:2501.08316 — https://arxiv.org/abs/2501.08316 ✓ — 擴散預訓練後接對抗後訓練的範式。
- Lin et al., 2026/ECCV 2026, *Continuous Adversarial Flow Models*, arXiv:2604.11521 — https://arxiv.org/abs/2604.11521 ✓ — 以判別器取代 FM 的 MSE，主要做 flow-matching 模型的後訓練；與 D1 最接近。
- Shen et al., 2025, *ADT: Tuning Diffusion Models with Adversarial Supervision*, arXiv:2504.11423 — https://arxiv.org/abs/2504.11423 ✓ — 對多步推論路徑做對抗監督，保留擴散損失防 hacking。
- Kim et al., ICML 2023, *Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models*, arXiv:2211.17091 — https://arxiv.org/abs/2211.17091 ✓ — 不重訓主模型、只加判別器修正；D1 的一天前置檢查。
- Wang et al., ICLR 2023, *Diffusion-GAN: Training GANs with Diffusion*, arXiv:2206.02262 — https://arxiv.org/abs/2206.02262 ✓ — 對判別器輸入加噪的穩定化；本題只借「t 條件判別器」的想法。
- Huang et al., NeurIPS 2024, *The GAN is dead; long live the GAN! A Modern GAN Baseline* (R3GAN), arXiv:2501.05441 — https://arxiv.org/abs/2501.05441 ✓ — R1 + R2 相對論損失，用來穩定小資料判別器。
- Miyato & Koyama, ICLR 2018, *cGANs with Projection Discriminator*, arXiv:1802.05637 — https://arxiv.org/abs/1802.05637 ✓ — 條件判別器的 projection 形式。
- Reed et al., ICML 2016, *Generative Adversarial Text to Image Synthesis*, arXiv:1605.05396 — https://arxiv.org/abs/1605.05396 ✓（論文）；matching-aware 判別器（真實圖 + 錯配條件當負例）這個細節**依記憶**，abs 沒寫。
- Kynkäänniemi et al., ICLR 2023, *The Role of ImageNet Classes in Fréchet Inception Distance*, arXiv:2203.06026 — https://arxiv.org/abs/2203.06026 ✓ — 在評估特徵空間優化可以刷分不提升品質，是 §0 的依據。
- Stein et al., NeurIPS 2023, *Exposing flaws of generative model evaluation metrics…*, arXiv:2306.04675 — https://arxiv.org/abs/2306.04675 ✓ — 建議 DINOv2 特徵評估；本專案用 DINOv3，因此訓練端更要避開 DINO 系列。

**D：資料效率增強 / few-shot / 檢索**
- Zhao et al., NeurIPS 2020, *Differentiable Augmentation for Data-Efficient GAN Training*, arXiv:2006.10738 — https://arxiv.org/abs/2006.10738 ✓ — 100 張即可訓練；用在 D1 判別器的輸入。
- Karras et al., 2020, *Training Generative Adversarial Networks with Limited Data* (ADA), arXiv:2006.06676 — https://arxiv.org/abs/2006.06676 ✓（NeurIPS 2020 這個 venue 依記憶）— 自適應增強機率，可替代 DiffAugment。
- Karras et al., NeurIPS 2022, *Elucidating the Design Space of Diffusion-Based Generative Models* (EDM), arXiv:2206.00364 — https://arxiv.org/abs/2206.00364 ✓（論文）；「以增強參數為條件的非洩漏增強」是**依記憶**的細節，abs 沒提。對應本專案：翻轉 / 亮度增強可以當條件輸入，避免亮度漂移被學進去。
- Hou et al., 2025, *Score Augmentation for Diffusion Models*, arXiv:2508.07926 — https://arxiv.org/abs/2508.07926 ✓ — 在加噪後的資料上做變換以減少過擬合；是增強條件化的新版作法。
- Zhu et al., 2023, *DomainStudio: Fine-Tuning Diffusion Models for Domain-Driven Image Generation using Limited Data*, arXiv:2306.14153 — https://arxiv.org/abs/2306.14153 ✓ — few-shot 擴散的多樣性保持；需要預訓練擴散模型（屬 A 方向）。
- Few-shot Image Generation with Diffusion Models, arXiv:2211.03264 — 只在搜尋結果看到，**未打開 abs，未查證** — DDPM-PA，成對相似度保持。
- Blattmann et al., NeurIPS 2022, *Semi-Parametric Neural Image Synthesis* (RDM), arXiv:2204.11824 — https://arxiv.org/abs/2204.11824 ✓ — 檢索增強擴散；見 §5 不建議的理由。
- Sheynin et al., 2022, *KNN-Diffusion: Image Generation via Large-Scale Retrieval*, arXiv:2204.02849 — https://arxiv.org/abs/2204.02849 ✓ — 同上，需要大型資料庫。
- Somepalli et al., 2023, *Understanding and Mitigating Copying in Diffusion Models*, arXiv:2305.20086 — https://arxiv.org/abs/2305.20086 ✓ — 條件化模型比無條件模型更容易複製訓練圖；檢索條件會加劇。
- Gu et al., TMLR 2025, *On Memorization in Diffusion Models*, arXiv:2310.02664 — https://arxiv.org/abs/2310.02664 ✓ — 小資料集與「沒有資訊量的條件」會觸發記憶。

**E：條件跟隨 / 連續條件**
- Stoica et al., ICCV 2025, *Contrastive Flow Matching*, arXiv:2506.05350 — https://arxiv.org/abs/2506.05350 ✓（ICCV 由 CVF openaccess 搜尋結果確認）— E1 的基礎；隨機負例、λ = 0.05。
- Zha et al., NeurIPS 2023, *Rank-N-Contrast: Learning Continuous Representations for Regression*, arXiv:2210.01189 — https://arxiv.org/abs/2210.01189 ✓ — 依目標值排序做對比；可用在瓶頸特徵。
- Ding et al., *CcGAN: … Novel Empirical Losses and Label Input Mechanisms*, arXiv:2011.07466（TPAMI）— https://arxiv.org/abs/2011.07466 ✓ — 連續條件的 vicinal 損失；等同已試過的 gmix。
- Ding et al., 2024, *CCDM: Continuous Conditional Diffusion Models for Image Generation*, arXiv:2405.03546 — https://arxiv.org/abs/2405.03546 ✓ — 擴散版 vicinal 損失 + 依標籤的噪聲共變異；核心機制與 gmix 重疊。
- Ding et al., 2026, *Enhancing Diffusion-Based Quantitatively Controllable Image Generation via Matrix-Form EDM and Adaptive Vicinal Training* (iCCDM), arXiv:2602.02114 — https://arxiv.org/abs/2602.02114 ✓ — CCDM 的後續（adaptive vicinal 的細節未讀）。
- Ramgopal et al., 2026, *Data-efficient continuous conditional denoising diffusion model for microstructure generation*, arXiv:2607.10429 — https://arxiv.org/abs/2607.10429 ✓ — 小資料材料影像使用 vicinal 損失；是通用性的旁證。
- Cai et al., CVPR 2024, *Condition-Aware Neural Network for Controlled Image Generation*, arXiv:2404.01143 — https://arxiv.org/abs/2404.01143 ✓ — 由條件生成權重，是 AdaGN 以外的注入方式。
- Gandikota et al., 2023, *Concept Sliders*, arXiv:2311.12092 — https://arxiv.org/abs/2311.12092 ✓ — 用 LoRA 做連續屬性滑桿；需要文字模型，不適用。
- Bansal et al., 2023, *Universal Guidance for Diffusion Models*, arXiv:2302.07121 — https://arxiv.org/abs/2302.07121 ✓ — 用任意可微估計器引導；若估計器是 DINOv3 偵測器，就會循環。
- Han et al., CVPR 2026, *Guiding Diffusion Models with Semantically Degraded Conditions*, arXiv:2603.10780 — https://arxiv.org/abs/2603.10780 ✓ — 以「差一點的條件」取代 null；機制與本專案已試的 FA-CFG g0 = undamaged 相同。
- Chang et al., ICML 2026, *ContrastiveCFG*, arXiv:2411.17077 — https://arxiv.org/abs/2411.17077 ✓ — 以對比損失做正 / 負概念引導；仍屬 CFG 家族。
- Krishnamoorthy et al., ICML 2023, *Diffusion Models for Black-Box Optimization* (DDOM), arXiv:2306.07180 — https://arxiv.org/abs/2306.07180 ✓ — 重加權 + CFG 聲稱可以超過資料最大值；是條件外推的一般參考。
- Okawa et al., NeurIPS 2023, *Compositional Abilities Emerge Multiplicatively*, arXiv:2310.09336 — https://arxiv.org/abs/2310.09336 ✓ — 組合泛化需要大量資料；是不走 compositional 的依據。
- Pfrommer et al., NeurIPS 2025, *Is Your Conditional Diffusion Model Actually Denoising?*, arXiv:2512.18736 — https://arxiv.org/abs/2512.18736 ✓ — 條件模型在條件空間有平滑歸納偏差，會「抹平」不同條件的 flow，可以解釋演進 ratio 偏低（推論）。
- Isobe et al., 2024, *Extended Flow Matching*, arXiv:2402.18839 — https://arxiv.org/abs/2402.18839 ✓ — 以廣義連續方程控制輸出分布對條件的敏感度；理論參考。
- Tsakonas et al., 2026, *Native Extrapolation Awareness in Flow-Based Conditional Generation*, arXiv:2602.13061 — https://arxiv.org/abs/2602.13061 ✓ — 只「偵測」外插，不改善。
- Berry & Eberhard, 2003, *Performance Models for Flexural Damage in Reinforced Concrete Columns*, PEER Report 2003-18 — https://peer.berkeley.edu/node/1519 ✓（報告頁）；公式係數**未查證** — E2 的物理時鐘。

**O1 的原創性比對**
- Litrico et al., MICCAI 2024, *TADM*, arXiv:2406.12411 — https://arxiv.org/abs/2406.12411 ✓（venue 由搜尋結果的 MICCAI 頁確認）— 最接近：用 brain-age estimator 讓年齡差一致。
- Litrico et al., 2025, *TADM-3D … Bidirectional Temporal Regularisation*, arXiv:2509.03141 — https://arxiv.org/abs/2509.03141 ✓ — 雙向時間正則。
- Hong et al., *ArrowGAN*, arXiv:2101.03710（Neurocomputing）— https://arxiv.org/abs/2101.03710 ✓ — 以時間方向當判別器的輔助任務。
- Saquil et al., BMVC 2018, *Ranking CGANs*, arXiv:1804.04082 — 只在搜尋結果看到，**未打開 abs，未查證** — 以成對比較的 ranker 控制屬性。
- Clark et al., ICLR 2024, *DRaFT*, arXiv:2309.17400 — https://arxiv.org/abs/2309.17400 ✓；Black et al., 2023, *DDPO*, arXiv:2305.13301 — https://arxiv.org/abs/2305.13301 ✓ — 在沒有影像的條件上用 reward 微調。
- Voleti et al., NeurIPS 2022, *MCVD*, arXiv:2205.09853 — https://arxiv.org/abs/2205.09853 ✓ — 隨機遮罩前 / 後幀，一個模型做預測、生成、內插。
- Yoon et al., IPMI 2023, *SADM*, arXiv:2212.08228 — https://arxiv.org/abs/2212.08228 ✓；Dao et al., 2024, arXiv:2411.05860 — https://arxiv.org/abs/2411.05860 ✓；Puglisi et al., 2024, *BrLP*, arXiv:2405.03328 — https://arxiv.org/abs/2405.03328 ✓ — 縱向醫學影像生成的錨定 / 序列條件。
- Chen et al., ICLR 2026, *Δ-LFM*, arXiv:2512.09185 — https://arxiv.org/abs/2512.09185 ✓ — 潛空間中隨嚴重度單調的軌跡。

搜尋紀錄（原創性檢查用）：「temporal order classifier reward diffusion unpaired conditions」、「arrow of time classifier as loss for video generation」、「diffusion fine-tuning pairwise ranking … progression reward」、「longitudinal … all pairs … null anchor unify」、「contrastive flow matching hard negatives condition-aware」。都沒有找到 O1 的組合。

---

## 3. 推薦排序（同時考慮可行性、原創性、通用性）

| 排名 | 方案 | 可行性 | 原創性 | 通用性 | 主要對準 |
|---|---|---|---|---|---|
| 1 | **O1 OPA-Flow**（E1 當它的一個元件） | 中（最小驗證 2 天；第一天就有 go / no-go） | 中到高 | 高 | 2、4，單幀與時序合一 |
| 2 | **D1 條件式對抗後訓練** | 中到高 | 低（工程套用） | 高 | 1、5（唯一直接攻 KID 的方案） |
| 3 | **E1 序列感知 ΔFM** | 最高（1 天） | 低到中 | 高 | 2 |
| （輔） | E2 物理時鐘 | 高 | 低到中 | 低到中 | 4；若 O1 的前置檢查顯示編碼器分不出 CTR1，就和 O1 合用 |
| 後 | D2 表徵對齊 | 高 | 無 | 高 | 證據最弱 |

**論文實證**
- 凍結特徵判別器在 100–1000 張的資料上能大幅改善 FID（Projected GAN、Vision-aided GAN）；保留擴散 / FM 損失可以抑制判別器 hacking（ADT、CAFM）。
- ΔFM 能讓不同條件的 flow 分開，並降低 FID（ImageNet 規模、隨機負例）。
- 在評估特徵空間裡優化會刷分（Kynkäänniemi 2023）。
- 用外部估計器讓生成結果符合時間差，在縱向醫學影像有效（TADM，絕對年齡估計器）。
- 組合 / 外插需要的資料量遠大於本專案（Okawa 2023）；條件模型對條件有平滑偏差（Pfrommer 2025）。

**我的推論**
- 本專案的問題 2（ratio、鄰居對比）是「條件差異沒有被放大」，問題 4 則是「條件本身沒有差異」。只靠生成器端的方法（D1、E1）可能改善 2，但改善不了 4。O1 是唯一同時給生成器「更多相對監督」和「條件空間覆蓋」的方案；但它對 4 的效果仍受限於編碼器（或 E2 時鐘）能不能把 CTR1 和 R315 分開。
- D1 是真實感（KID）唯一有力的工具。它和 O1 是正交的，可以疊加；建議先做 O1 的第一天前置檢查和 D1 的 discriminator guidance 檢查，兩者都不必重訓主模型。
- E1 便宜，但單獨的原創性和效果上限都有限，所以放進 O1 的消融裡跑，不另開路線。
- 循環偏差：D1 / D2 必須照 §0 換特徵網路並報兩個空間的 KID。O1 的 critic 是像素 CNN，DINOv3 指標對它來說是獨立驗證。

---

## 4. 最小驗證實驗（每個 1–2 天）

通則：以 gn（或 v11 完成後的 ckpt）為起點，所有組別微調**相同步數**（含一個「只續訓 FM」的對照組，排除「多訓練幾步就變好」）。固定 seed、K = 8、ODE 30 步 Heun。val（R307）與 test（CTR1）都跑，用 eval suite 的配對 z。test 只有 7 個循環單位，所以要求 **val 和 test 方向一致**，不只看單邊顯著。

### O1（2 天）
- **第 1 天，前置檢查（不動生成器）。** 訓練 critic R_ψ（1–2 小時 GPU），4 根訓練柱做留一柱。指標：
  (a) 在 R307 / CTR1 的真實序對上，相隔 ≥ 1 個 drift 等級的先後判斷準確率；
  (b) m̂ 與編碼器 Δenc 的 Spearman；
  (c) 把 critic 用在既有的 gn 樣本上：同一序對的 m̂(gn) / m̂(real) 是否約 0.7，與 DINOv3 ratio 0.38 / 0.53 的結論一致。這是兩個獨立特徵空間的交叉驗證。
  **過關條件**：CTR1 上 (a) ≥ 0.85、(b) ≥ 0.6，(c) 和 DINOv3 方向一致。另外檢查 Δenc 對 CTR1 是否大於 R315 同 drift；若不是，第 2 天的 C 組改用 Δenc + E2 的 τ。
- **第 2 天，生成器（先只做 null 錨定，隔離「訓練目標」這個因素）。** 從 gn 微調 3k 步，三組：
  A = 只續訓 FM；B = + L_ord + L_mag（只用有影像的條件）；C = B + 無影像條件 rollout。
  看演進 ratio、鄰居對比、corr、CTR1 剝落面積偏差（probe / damage_texture 的損傷面積）、KID（DINOv3 + 第二空間）、同條件樣本差。
  **值得繼續**：C − A 的 ratio ≥ +0.05、鄰居對比 ≥ +0.05，CTR1 剝落低估縮小 ≥ 0.05，val / test 方向一致，KID 變差不超過 1 個 SE。若 B ≈ C，代表無影像 rollout 沒有貢獻；若 B、C 都 ≈ A，代表 critic 訊號太弱（再看 λ）。
- 通過後的第二階段（另外 2–3 天）：加錨定通道與 all-pairs 抽樣，和 v10.5 / 10.6 refiner 比時序指標（tone_jump、演進曲線）。

### D1（2 天）
- **第 1 天**：discriminator guidance。用 gn 生成約 2k 張（訓練條件），訓練 t 條件判別器（EfficientNet-Lite 特徵 + head，DiffAugment），取樣時加修正項，掃權重 0 / 0.5 / 1。主模型不動。看 KID（DINOv3 + DINOv2）、fog_share、灰階直方圖。
- **第 2 天**：對抗微調 3k 步，四組：A = 只續訓 FM；B = + 無條件對抗；C = + 條件 projection；D = C + 鄰居錯配負例。
- **值得繼續**：C 或 D 的 KID 在**兩個特徵空間**都下降 ≥ 0.03（目前 SE 約 0.026，配對 z 會更小）；ratio、鄰居對比不下降（D 組希望上升）；同條件樣本差維持在 15 灰階以上（不 collapse）；損傷面積不減少。只有 DINOv3 變好就判為刷分。

### E1（1 天，可併入 O1 的 B 組當消融）
- 從 gn 微調 3k 步，三組：λ = 0；λ = 0.05 隨機負例（原版 ΔFM）；λ = 0.05 序列感知負例。
- **值得繼續**：序列感知版的 ratio 與鄰居對比比 λ = 0 高 ≥ 0.05、比隨機版高；KID 不變差；同條件亮度標準差不增加（檢查負例是否只推開亮度）。

---

## 5. 不建議的方向

- **REPA / 判別器 / critic 用 DINOv3（或 DINOv2）特徵**：與評估指標同源，會刷分（§0）。
- **Retrieval-augmented（用物理狀態檢索最近訓練幀當額外條件）**：
  (1) 只有 4 根柱，相鄰幀又幾乎相同，檢索到的幾乎就是答案，模型會學成「複製檢索圖」（Somepalli 2023：條件化模型更容易複製；Gu 2023：小資料容易記憶）；
  (2) 生成器眼中 CTR1 的條件就在 R315 / R330 旁邊（condition_g.md §7），檢索只會更確定地拿 R315 的圖，**把外插低估固化**；
  (3) RDM / KNN-Diffusion 的優勢來自大型資料庫，本題沒有。
  唯一可能的用途是在 O1 中隨機抽較早的錨定幀，那其實是錨定條件而不是檢索。
- **CCDM / CcGAN 的 vicinal 損失**：核心機制等同已試過的 gmix。CCDM 其他部分（依標籤的噪聲共變異、DDIM 250 步）對本題沒有明顯對應。
- **ADD / LADD / Seaweed-APT 這類以「少步數蒸餾」為目標的方法**：目標是速度，不是小資料真實感。純對抗目標在 530 張上有重演 cGAN mode collapse 的風險。只借用它們的判別器設計（D1）。
- **從零訓練 Diffusion-GAN / 純 GAN**：等於回到 cGAN 路線；cGAN 的 KID 好，是靠「乾淨、不畫損傷」和 collapse（results.md）。
- **Compositional generation、Concept Sliders**：前者需要大量組合資料（Okawa 2023），後者依賴文字條件的預訓練模型。
- **新的 CFG 變體（Condition-Degradation Guidance、ContrastiveCFG）**：機制和已試過的 FA-CFG（g0 = undamaged）相同，都是拿「差一點的條件」當負分支。但要注意，既有 guidance **從沒量化過**。拿現有 FA-CFG 跑一次 eval suite 的 ratio / 鄰居對比，是零訓練、半天就能做的檢查，值得補上，但不值得開新方法。
- **Universal guidance / 以偵測器引導剝落面積**：專案的偵測器是 DINOv3 backbone，會循環。若要做，需要另訓一個像素 CNN 的剝落估計器，而這基本上就是 O1 的 critic。
- **Dispersive Loss**：沒有條件感知的特徵分散，可能加大同條件變異（問題 5），目前沒有證據顯示它對條件跟隨有幫助。
- **Native Extrapolation Awareness（Diverging Flows）**：只偵測外插，不改善。
