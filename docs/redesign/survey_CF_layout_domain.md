# 文獻調查：C（先結構後紋理）與 F（領域相關工作）

範圍：原 prompt 第 5 節的 C 與 F，加上協調者的更新（原創性、通用性、自己的原創方案）。
查證：標 ✓ 的條目都用 WebFetch 打開過 arXiv abs 頁、期刊頁、Crossref 或會議頁，確認了標題、作者、年份與編號。標「未查證」的只憑記憶或專案文件。
日期：2026-10-25。

---

## 0. 先講一個會影響 C 方向的資料發現

原 prompt 說「沒有像素級標註，只有用亮度規則算的粗 m*」。但專案裡其實有人工的剝落多邊形：

- `Data/DamagePattern/anylabel/*.json`：785 張圖（train/val/test 全部）都有 AnyLabeling 檔。標籤只有兩類：`spall`（662 個多邊形）與 `interference`（878 個，干擾物）。
- `anylabel_r_spall.csv`：每張圖的剝落面積比 `R_spall`。訓練柱中有剝落多邊形的圖：R615 44 張、R315 51 張、R330 37 張、R1015 26 張。最大 `R_spall` 約 0.62–0.68。CTR1 有 39 張，最大 0.568。
- 目前只有 `scripts/export/export_anylabel_r_spall.py` 用到它，算 `R_spall`。生成器和 M 都沒拿它當 layout 標籤。

所以剝落 layout 可以直接用人工多邊形，不必只靠 m*。m* 仍可以當弱標籤，補「比混凝土暗、但沒被框成 spall」的區域。**裂縫與鋼筋外露沒有標註**，這兩類只能靠弱監督或留給紋理階段隱式處理（見 §1 各方案）。

序列結構：train 是 8 條「柱 × 柱面」序列（每條 61–70 幀），val 2 條（R307 NW/SE），test 2 條（CTR1 SW/NE）。下面所有「小資料」的論證都以「8 條訓練序列」為單位。整條序列層級的分佈只有 8 個樣本。

---

## 1. 候選方案

共 5 個。C1–C4 來自文獻，C5 是我自己的原創方案。

### 總表

| | 方案 | 解決第 4 節哪些問題 | 實作量 | 24 GB | 原創性 | 通用性 |
|---|---|---|---|---|---|---|
| C1 | 影像＋剝落 SDF 聯合生成（joint image–layout） | 3（形狀）、部分 2、4 | 1–2 天 | 可 | 無，屬工程套用 | 高 |
| C2 | 兩階段：layout 生成器 → layout 條件渲染器 | 3、4、部分 1、2 | 4–6 天 | 可 | 低 | 高 |
| C3 | 數量先行：單調回歸剝落面積軌跡，再拿來校準 | 4、部分 2 | 0.5–1 天 | 可（CPU 即可） | 無，屬工程套用（BrLP 模式） | 高 |
| C4 | Layout-oracle 診斷（給真實 layout 的上限實驗） | 不直接解決，決定 C2/C5 值不值得做 | 1 天 | 可 | 無（診斷用） | — |
| **C5（原創）** | **單調起始場 OFP：整段序列是一張「起始時刻場」，單幀是它的切片** | 2、3、4、5，以及時序一致性 | layout 部分 1–2 天，完整 5–8 天 | 可（layout 模型只有幾千參數） | 中：組件都有前例，組合與「單幀＝場的切片」的設定沒有找到前例 | 高，前提是 §C5 列的假設成立 |

---

### C1 影像＋layout 聯合生成（SatSynth 式）

**核心想法**：在現行像素 CFM 的 z 上多加 1–2 個通道，同時生成影像與剝落的截斷 SDF（由 anylabel 多邊形算，沒有多邊形的圖用 m* 補）。速度場同時預測兩者，推論時一起解 ODE。SDF 通道給網路一個「連續、平滑的大區塊」目標。直接預測暗像素的話，大塊剝落會被平均成碎塊；SDF 比較不會。

**依據**：
- SatSynth 把類別標籤編成 ⌈log₂K⌉ 個 bit 通道，和 RGB 串接，在像素空間用 DDPM 於 128×128 聯合生成影像與遮罩。one-hot 編碼在他們的消融裡比較差。
- DiffAtlas 也是影像–遮罩聯合 diffusion，在 limited-data 設定下增益最大。
- Bogensperger 2023 與 FlowSDF 2024 說明，對遮罩做 diffusion / flow matching 時，SDF 表示比二值遮罩平滑。FlowSDF 本身就是 flow matching。

**解決的問題**：#3 大片剝落破碎（SDF 通道強迫連續）。#4 面積可以直接從 SDF 通道讀，也能拿來監控。#2 有一部分：layout 通道的損失給條件一個「容易學、幅度大」的訊號，可能把條件反應拉高，但這是推論。

**M 與 m***：m* 降成補充標籤。M 不變（時序仍可用它），但也可以改用生成出來的 SDF 通道當「這一幀的剝落」。

**24 GB**：只多 1–2 個輸入／輸出通道，記憶體幾乎不變。

**最大風險**：兩個通道的不一致，也就是影像裡沒畫出 SDF 標的剝落。SatSynth 的資料量是幾千張以上，530 張下聯合分佈學不學得好沒有證據。

**原創性**：無，屬工程套用。最接近的前例是 SatSynth（CVPR 2024）、DiffAtlas（2025）、JoDiffusion（AAAI 2026）。

**通用性**：高。任何「影像＋可得的粗標註」都能用：醫學病灶、材料相、植物葉面積。前提是至少有弱標註。

---

### C2 兩階段：layout 生成器 → layout 條件渲染器

**核心想法**：
- 第一階段：一個小的條件 flow matching 模型，在 32×32 生成剝落的截斷 SDF。條件是物理狀態 g（或損傷狀態編碼器的 256 維 state 加 4 個標量）、柱面、座標，以及上一幀的 layout（時序時）。輸出後做單調投影：新 layout = max(上一幀 layout, 新生成)。
- 第二階段：渲染器就是現在的 G。把 layout 以 SPADE / SDM 方式注入 decoder 各層（SDM 的做法：噪聲圖進 encoder，layout 經 spatially-adaptive normalization 進 decoder），或沿用現有的 zero-init 輸入通道。訓練時用真實 layout（anylabel），推論時用第一階段的輸出。
- 用 Konz 2024 的 mask-ablated training：訓練時隨機丟掉 layout，讓渲染器容忍不完整或錯誤的 layout。這很重要，因為 layout 標籤是粗的。

**依據**：
- SegGen（Text2Mask → Mask2Img）、DiffInfinite（先生成遮罩，再以遮罩條件生成病理影像）、MedDiff-FT（輕量 DDPM 遮罩生成器加微調的 SD，約 30 對影像遮罩、24 GB 上約 30 分鐘）。
- Feng et al. MICCAI 2024：先用 diffusion 生成病灶 mask，再只在 mask 內 inpaint 健康眼底，論文明說這比「只靠病灶 mask 生成整張圖」容易學。
- CrackSegFlow 2026：一個 flow matching 模型生成裂縫 mask（可控覆蓋率），另一個從 mask 生成裂縫影像。
- 渲染器：SPADE、OASIS、SDM、Konz 2024。Konz 2024 報告在 mask 忠實度上贏過 SPADE 與 ControlNet。

**解決的問題**：
- #3：形狀由 SDF 生成器決定，SDF 的零等值線天然平滑。
- #4：面積由第一階段決定，可單獨校準（接 C3）。
- #1 有一部分：渲染器只要學「給定 layout 下的紋理」，條件比較強。Oracle 實驗顯示，給對條件後灰霧與白塊會消失。
- #2 有一部分：layout 隨條件變化的幅度可以直接量、直接調。

**M 與 m***：第一階段取代 M。M 是逐格 BCE、讀上一幀「生成影像」的池化；新的第一階段讀上一幀的 layout，不讀影像，所以生成影像的錯誤不會回灌到 layout。m* 降成補充標籤。

**24 GB**：第一階段 32×32 的小 U-Net，訓練幾小時。第二階段同現行 G。

**最大風險**：
- 第一階段在 8 條序列上學「整張 layout 的分佈」，樣本太少，會背訓練柱的剝落位置。這正是 C5 想解決的。
- 推論時渲染器拿到的是生成 layout，和訓練時的真實 layout 有分佈差（exposure bias）。mask-ablated training 和 layout 擾動只能部分緩解。

**原創性**：低。「先 mask 後影像」的兩階段已經是標準作法（SegGen、DiffInfinite、MedDiff-FT、Feng 2024、CrackSegFlow）。差異只在第一階段的條件是物理序列，並加單調投影。這個差異偏工程，不足以單獨構成方法貢獻。

**通用性**：高。醫學（病灶）、材料（相分佈）、植物（葉片遮罩）都有同型做法。依賴的假設：有可分割的結構類別，而且紋理主要由結構決定。

---

### C3 數量先行：先預測剝落面積軌跡，再校準生成

**核心想法**：用 8 條訓練序列的 anylabel `R_spall(k)`，擬合一個低參數、對物理標量單調的回歸 A(k) = f(累積能量, 勁度比, 強度比, 最大 drift)。可以用 UMNN，或正權重的小網路，或 logistic 形式。拿它做兩件事：

1. 當條件標量，和 BrLP 把預測的區域體積當條件一樣。
2. 在 C2 / C5 裡校準 layout 的閾值，讓生成的剝落面積等於 A(k)。

**依據**：
- BrLP：輔助模型（線性回歸或 Disease Course Mapping）先預測目標年齡的區域腦容量，再和 metadata 一起經 cross-attention 條件 LDM。
- Drees et al. 2023：作物生長影像生成時，把模擬的生物量當條件，衍生的表型性狀明顯變準。
- 兩者都是「先預測一個低維、可校準的量，再生成影像」。

**解決的問題**：#4 是主要目標。CTR1 低估 0.2–0.35 的面積比例，問題可能出在「面積這個量」沒有被單獨約束、而是混在 1024 維 g 的 AdaGN 裡。單調回歸在條件超出訓練範圍時會繼續往同一方向走，不會飽和。這一點是推論，要靠 §4 的實驗驗證。

**M 與 m***：不影響 M。A(k) 也可以拿 m* 的面積當第二個標籤源。

**最大風險**：8 條序列擬合的單調回歸，外插到 CTR1 是否正確，取決於選的標量是否真的決定剝落量。如果 CTR1 的高軸力讓同樣的能量對應更多剝落，任何只看這些標量的回歸都會低估。

**原創性**：無，屬工程套用（BrLP、Drees 2023 的模式）。

**通用性**：高。醫學（體積）、植物（生物量）、材料（相分率）都有對應的「可量測總量」。

---

### C4 Layout-oracle 診斷

不是新架構，是做 C2 / C5 之前必須先跑的上限實驗：把 CTR1 的真實剝落 layout（anylabel 多邊形）餵給 layout 條件渲染器，看 #1、#3、#4 改善多少。邏輯和現有的 `oracle_dino` 一樣，但 oracle 的是「結構」，不是 DINOv3 全域特徵。所以它沒有 DINOv3 指標的循環偏差，`oracle.md` 指出的那個問題在這裡不會出現。

如果給了真實 layout，剝落面積偏差與破碎仍然不改善，就代表瓶頸在渲染器本身，C2 / C5 都不值得做。

---

### C5（原創方案）單調起始場 OFP（Onset-Field Progression）

> 以下全部是我的推論與設計。組件各有前例，見「原創性」。

#### 核心想法

把一整條序列（一根柱的一個柱面）的損傷結構，表示成一張**起始時刻場**：

- T(x) = 位置 x 在「損傷時鐘」τ 上第一次變成剝落的時刻。沒剝落的位置記為 +∞。
- 第 k 幀的剝落 layout 是場的一個切片：L_k(x) = 1[T(x) ≤ τ_k]。
- 損傷時鐘 τ_k 是物理標量的單調函數（累積能量、勁度比等，就是 C3 的 f）。它不是幀號。
- 「損傷年齡」A_k(x) = clip(τ_k − T(x), 0, A_max)：這塊剝落已經存在多久。它決定紋理，例如新鮮斷面和已經掉到鋼筋外露的差別。

渲染器只有一個，就是現行 G。它看 (L_k, A_k, g_k, 柱面, 座標) 畫出第 k 幀。

- **單幀模式**：抽一張 T，取 τ_k 切片，渲染一張。
- **時序模式**：同一張 T，對所有 k 切片。前後幀的增量區 {τ_{k−1} < T ≤ τ_k} 就是現有機制裡的「可畫區」，其餘位置鎖住，沿用上一幀（現有的 MaskFlow 式路徑與 `spall_soft_paint` 直接可用）。

所以單幀和時序是**同一個模型、同一個場**，差別只在取樣程序。沒有另外接的 refiner。

#### T 怎麼生成：隨機局部 hazard 細胞自動機（不是整場 diffusion）

在 16×16（或 32×32）格上，每格是吸收態：0 = 完好，1 = 剝落，一旦變 1 就永遠是 1。每一步的轉移機率是

  p(S_{k+1}(c)=1 | S_k(c)=0) = σ( h_θ( S_k 的 3×3 / 5×5 鄰域, 座標, 柱面, τ_k, Δτ_k ) )

- h_θ 是幾千參數的小卷積。對 τ 與 Δτ 單調（正權重或 UMNN），讓 hazard 隨損傷時鐘只增不減。
- 拆成兩項：「成核」（周圍沒有剝落時自發出現）和「蔓延」（貼著已剝落格往外長）。蔓延項讓剝落從邊緣長出來，所以會是**連續的大區塊**，不是散點（問題 #3）。
- 訓練是 discrete-time survival likelihood。對每個「這一步仍完好」的格子，它在下一步有沒有變剝落，做 Bernoulli 似然，也就是只在 at-risk 格子上算的 BCE。
- 狀態 S_k 就是 layout 本身，teacher forcing 用真實 layout，推論時讀自己的 layout，**兩者分佈一致**。現行 M 讀「生成影像」的池化，推論時看到的是訓練沒見過的生成錯誤。
- 一次 rollout 就得到一整張 T。多次取樣得到不同但合理的剝落形狀。
- T 在格上是整數步，放大到 128 時用雙線性插值，再取等值線 {T ≤ τ}。粗格的 T 場切出來會是平滑的次格邊界，形狀細節不必學。

#### 標籤

- T*（真實起始場）由 anylabel 多邊形逐幀算出 16×16 覆蓋率，再對每格的時間序列做**單調投影**，也就是 isotonic regression。這就是 Gorinevsky et al. 2009 的「逐像素單調時間序列估計」，用來壓掉相機錯位與標註抖動。
- 第一次越過 0.5 的時刻記為 T*。沒有多邊形的區域可用 m* 補。
- 裂縫沒有標註：第一版不做裂縫的 T，裂縫留給渲染器，以 g_k 與 A_k 隱式畫。第二版可以用公開的裂縫分割模型自動標註，做第二張 T_crack。OmniCrack30k 的基準顯示 nnU-Net 跨材料遷移最好，但權重是否公開我沒查到，要再確認。

#### 為什麼在小資料下合理

1. **樣本單位變小**：要學的是局部轉移規則，不是整張 layout 的分佈。8 條序列 × 256 格 × 約 65 步 ≈ 13 萬個 cell-step，事件數（新剝落格）有數百到上千個。C2 的第一階段只有 8 個「整張 T」樣本，C5 不必學那個分佈。
2. **單調性是結構保證**，不必從資料學。剝落不會「長回去」，前後幀一致性免費得到。
3. **外插路徑明確**：CTR1 在損傷時鐘上若走得比訓練柱遠，hazard 對 τ 單調，剝落會繼續長，不會飽和。這直接對準 #4，但前提是 τ 的定義抓到了 CTR1 的差異。
4. **紋理與結構分開**：渲染器只學「給定 layout 與年齡的外觀」。530 張圖的每一個剝落像素都是這個條件分佈的樣本，比學「g → 整張圖」容易。

#### 解決的問題

- #2：條件改變 → τ 改變 → layout 切片改變，幅度由 hazard 決定，不會被 AdaGN 稀釋。
- #3：蔓延項加上 SDF 式的等值線切片。
- #4：單調 τ 加上可校準的面積（C3）。
- #5：同一條件下的變異集中在「T 的取樣」，也就是結構的合理變化。亮度與色調可以用共享的紋理噪聲固定。
- 時序：同一張 T，結構上自動一致。

#### M 與 m* 怎麼處理

**取代 M**：hazard CA 就是新版的 M，從「逐幀 BCE、讀影像」改成「吸收態 survival、讀 layout」。現行 M 的 `logit = conv(pool(x_{t-1})) + s(g)·conv(x,y)` 結構很接近，可以從它改起：把 pool(x_{t-1}) 換成 S_k，加上鄰域項，把損失換成 at-risk BCE。

**m***：降成補充標籤，主標籤換成 anylabel 多邊形。

**可畫區／鎖住區機制保留**，只是可畫區由 T 的增量決定。

#### 24 GB

hazard CA 幾千參數，CPU 都能訓。渲染器同現行 G，只多 2 個 zero-init 輸入通道（L_k、A_k）。

#### 最大風險

1. **τ 的定義**：如果 CTR1 低估的原因是「同樣的累積能量下剝落更多」（軸力效應），任何只吃這些標量的 τ 都會繼承低估。這時要讓 τ 吃損傷狀態編碼器的 state，但外插性會變差。
2. **局部規則可能學不到「剝落會先出現在柱底塑鉸區」這類全域位置偏好**。要靠座標輸入和成核項；8 條序列夠不夠學位置先驗，不確定。
3. **鄰居對比（#2）不一定改善**：相鄰條件的 τ 差很小時，切片差也很小。這是對的行為，但若評估要求相鄰條件分得開，要看真實資料的增量是否真的比模型大。
4. 只處理剝落，裂縫與鋼筋外露仍是隱式的。

#### 原創性：最接近的工作與差異

| 最接近的工作 | 它做了什麼 | OFP 的差異 |
|---|---|---|
| DeepGrowth（Chen et al. 2024，arXiv 2404.02614） | 腫瘤用 SDF＋latent 表示，ConvLSTM 預測未來 latent，確定性 | 不生成影像紋理；每個時刻各一個 SDF，沒有「整段序列是一張起始場」的設定；沒有單調保證 |
| Gorinevsky et al. 2009（IEEE TSP） | 從雜訊影像序列估計逐像素單調的損傷累積，QP 求解 | 是估計，不是生成；我們拿它來做 T* 標籤 |
| Yu et al. 2026（GMD，wildfire diffusion surrogate） | diffusion 模擬機率細胞自動機的下一步燃燒狀態 | 狀態到狀態的自回歸，沒有單調保證，也沒有影像渲染；它的資料生成機制（機率 CA）正是 OFP 的 T 生成器形式，但我們直接學 CA，不是用 diffusion 模仿它 |
| Hoskere et al. 2022（PBGM） | 非線性 FE 的塑性應變／壓碎損傷 → 程序化渲染裂縫、剝落、鋼筋外露 | 結構與渲染都是手寫規則，不從影像學；需要每根柱的 FE 模型 |
| BrLP（Puglisi et al. 2025） | 輔助模型先預測區域體積 → 條件 LDM＋ControlNet | 中介量是全域標量，不是空間場；沒有單調的空間表示 |
| Growing NCA（Mordvintsev et al. 2020） | 學局部更新規則長出圖形 | 確定性、無外部驅動、無 survival 似然、不渲染真實影像 |
| 現行專案的 M＋可畫區（MaskFlow 式） | 逐幀 BCE 粗遮罩，讀上一幀影像 | OFP 把它重寫成吸收態、讀 layout、整段一張場；單幀是切片 |

搜尋紀錄：我查了 "arrival time map generative"、"onset time map image sequence generation"、"time-dependent SDF / level set generation"、"4D dynamic SDF diffusion"、"per-pixel survival hazard map damage"、"monotone growth diffusion"。只找到上表這些和 4D 變形式的 SDF 生成（Han et al. ECCV 2024、ShapeGen4D 2025）。後兩者處理的是物體運動與變形，不是單調增長。**沒有找到「以單一起始時刻場表示整段影像序列、外部驅動的隨機 hazard 生成這張場、再以切片和損傷年齡條件渲染影像」的工作。** 但搜尋有限，這不等於沒人做過。

**差異是否構成貢獻**：單看每個組件，都不新（survival analysis、細胞自動機、level set、mask-conditioned diffusion 各有前例）。新的是：

1. 「單幀＝序列場的切片」，把單幀與時序生成統一。
2. 用 at-risk survival 似然在極少序列上學局部 hazard。
3. 用損傷年齡圖條件紋理。

我判斷這足以撐起一篇應用方法論文（例如 CACAIE、MICCAI 的縱向生成 workshop 這一級），前提是實驗上能贏過「單幀 CFM＋refiner」與 C2。單靠概念不夠，需要至少兩個領域的結果才能主張通用（見下）。

#### 通用性

依賴四個假設：

| 假設 | RC 柱 | 醫學縱向影像 | 材料演化 | 植物生長 |
|---|---|---|---|---|
| (a) 結構變化在固定座標上是單調、不可逆的（吸收態） | 成立（剝落不會長回去） | GA（地圖狀萎縮）病灶擴大、多發性硬化的萎縮、未治療腫瘤：大致成立；治療後腫瘤縮小：不成立 | 腐蝕、損傷、裂縫：成立；晶粒成長、spinodal 分解：不成立（像素不是吸收態） | 冠層覆蓋率大致單調；葉片移動、新葉遮擋：不成立 |
| (b) 影像可對齊到固定座標（或只差幾個像素） | 大致成立，粗格吸收錯位 | 需要配準，腦 MRI 常規可做 | 模擬資料天然成立 | 田間相機難 |
| (c) 某處的外觀主要由「結構狀態＋損傷年齡＋全域狀態」決定 | 推論：剝落斷面→骨料→鋼筋外露，隨年齡變化 | GA 病灶邊緣與內部外觀不同，合理 | 合理 | 葉片顏色隨年齡變化，合理 |
| (d) 有外部驅動序列，可以定義單調時鐘 τ | 遲滯迴圈 → 累積能量 | 年齡、病程分數 | 溫度／時間歷程 | 積溫、生物量（Drees 2023 就用模擬生物量） |

**RC 柱特有的**：

- 剝落的亮度定義（m* 規則）。
- 柱面、加載正負號、鋼筋白線遮罩。
- 「損傷時鐘」用累積能量。

**可直接檢驗的跨領域資料**：

- ImageFlowNet 用的 GA 眼底序列，以及 Domalpally / Salvi 等 GA 成長研究：GA 是最乾淨的單調病灶擴大案例。
- Najafi Koopas 2024 的混凝土相場斷裂模擬（470 組）：損傷場在相場模型裡本來就不可逆。

這兩者都符合 (a)–(d)，可以當 OFP 的第二、第三個領域。

---

## 2. 論文清單

### 2.1 C 方向：layout → texture、mask 生成、單調表示、弱標註

**語意影像合成（渲染器）**

- Park, Liu, Wang, Zhu. *Semantic Image Synthesis with Spatially-Adaptive Normalization* (SPADE). CVPR 2019. [arXiv:1903.07291](https://arxiv.org/abs/1903.07291) ✓。layout 經 spatially-adaptive norm 注入。可以拿來取代或補充現行的 AdaGN，讓結構以空間方式進入每層。
- Sushko, Schönfeld, Zhang, Gall, Schiele, Khoreva. *You Only Need Adversarial Supervision for Semantic Image Synthesis* (OASIS). ICLR 2021. [arXiv:2012.04781](https://arxiv.org/abs/2012.04781) ✓。判別器是以 label map 為真值的分割網路，加 3D 噪聲做局部重抽。如果渲染器要加判別器（D 方向），這是 layout 感知的版本。
- Wang, Bao, Zhou, Chen, Chen, Yuan, Li. *Semantic Image Synthesis via Diffusion Models* (SDM). arXiv 2022. [arXiv:2207.00050](https://arxiv.org/abs/2207.00050) ✓。噪聲圖進 encoder，layout 經多層 SPADE 進 decoder，再加丟 mask 的 CFG。C2 渲染器最直接的範本。
- Konz, Chen, Dong, Mazurowski. *Anatomically-Controllable Medical Image Generation with Segmentation-Guided Diffusion Models*. MICCAI 2024. [arXiv:2402.05210](https://arxiv.org/abs/2402.05210) ✓。mask-ablated training 讓模型容忍缺漏或錯誤的 mask，mask 忠實度贏過 SPADE 與 ControlNet。我們的 layout 標籤是粗的，這點很關鍵。
- Zhang, Rao, Agrawala. *Adding Conditional Control to Text-to-Image Diffusion Models* (ControlNet). [arXiv:2302.05543](https://arxiv.org/abs/2302.05543) ✓。專案已在時序上試過 ControlNet 式分支，效果有限。這裡只當對照，不推薦。
- Cano-Ortiz, Sainz-Ortiz, Lloret Iglesias, Martínez Ruiz del Árbol, Castro-Fresno. *Enhancing pavement crack segmentation via semantic diffusion synthesis model for strategic road assessment*. Results in Engineering 23 (2024) 102745. DOI 10.1016/j.rineng.2024.102745 ✓（Crossref）。在裂縫上把 SDM 的複雜度、噪聲排程、condition scaling 調過。證明 mask → 裂縫影像這段在土木影像上可行。

**先 mask 後影像（兩階段）與聯合生成**

- Toker, Eisenberger, Cremers, Leal-Taixé. *SatSynth: Augmenting Image-Mask Pairs through Diffusion Models for Aerial Semantic Segmentation*. CVPR 2024. [arXiv:2403.16605](https://arxiv.org/abs/2403.16605) ✓。像素空間 DDPM 在 128×128 聯合生成影像與 bit 編碼的遮罩通道。C1 的直接範本。
- Ye, Kuen, Liu, Lin, Price, Xu. *SegGen: Supercharging Segmentation Models with Text2Mask and Mask2Img Synthesis*. ECCV 2024. [arXiv:2311.03355](https://arxiv.org/abs/2311.03355) ✓。標準的 mask 生成 → 影像生成兩段式（基於 SDXL）。C2 的原型。
- Aversa et al. *DiffInfinite: Large Mask-Image Synthesis via Parallel Random Patch Diffusion in Histopathology*. arXiv 2023. [arXiv:2306.13384](https://arxiv.org/abs/2306.13384) ✓。先生成分割 mask，再生成影像，用 CFG 結合少量標註與大量未標註資料。和我們「少量剝落標註＋多張未標註圖」的情況同型。
- Xie, Zhang, Weng, Zhu, Luo. *MedDiff-FT: Data-Efficient Diffusion Model Fine-tuning with Structural Guidance for Controllable Medical Image Synthesis*. MICCAI 2025. DOI 10.1007/978-3-032-04965-0_29 ✓（MICCAI 頁）。輕量 DDPM 遮罩生成器加微調的 SD 1.5，約 30 對資料、24 GB 約 30 分鐘（數字來自審稿與 rebuttal）。小資料兩段式的直接證據。
- Feng, Zhang, He, Gao, Wei, Yuan. *Diversified and Structure-realistic Fundus Image Synthesis for Diabetic Retinopathy Lesion Segmentation*. MICCAI 2024, LNCS 15012, pp. 77–86. DOI 10.1007/978-3-031-72390-2_8 ✓。diffusion 生成病灶 mask，再只在 mask 內 inpaint 健康眼底，並說明這比從 mask 生成整張圖容易學。這等於「鎖住區／可畫區」的單幀版本。
- Mao et al. *MedSegFactory: Text-Guided Generation of Medical Image-Mask Pairs*. [arXiv:2504.06897](https://arxiv.org/abs/2504.06897) ✓。雙分支 diffusion 加 joint cross-attention，同時生成影像與 mask。C1 的另一種結構。
- Wang et al. *JoDiffusion: Jointly Diffusing Image with Pixel-Level Annotations for Semantic Segmentation Promotion*. AAAI 2026. [arXiv:2512.13014](https://arxiv.org/abs/2512.13014) ✓。另訓一個 mask VAE 進同一個 latent 做聯合 diffusion。需要 latent 與 VAE，在我們的規模下不推薦，列為對照。
- Zhang, Liu, Yang, Guo, Wang, Fua. *DiffAtlas: GenAI-fying Atlas Segmentation via Image-Mask Diffusion*. [arXiv:2503.06748](https://arxiv.org/abs/2503.06748) ✓。影像–遮罩聯合 diffusion，limited-data 下增益最大。
- Asadi, Wu, Golparvar-Fard, Hajj. *CrackSegFlow: Controllable Flow-Matching Synthesis for Generalizable Crack Segmentation with the CSF-50K Benchmark*. [arXiv:2601.03637](https://arxiv.org/abs/2601.03637) ✓。兩個 flow matching 模型（mask 生成，mask → 影像），用 topology-preserving mask injection 保持細裂縫連續。和我們同是 flow matching，最接近的土木兩段式。

**mask / 形狀的生成表示**

- Bogensperger, Narnhofer, Ilic, Pock. *Score-Based Generative Models for Medical Image Segmentation using Signed Distance Functions*. [arXiv:2303.05966](https://arxiv.org/abs/2303.05966) ✓。在 SDF 上做 diffusion，分佈比二值 mask 平滑。支持用 SDF 表示剝落。
- Bogensperger, Narnhofer, Falk, Schindler, Pock. *FlowSDF: Flow Matching for Medical Image Segmentation Using Distance Transforms*. [arXiv:2405.18087](https://arxiv.org/abs/2405.18087) ✓。SDF 上的 conditional flow matching，可以直接接到現行的 CFM 程式。
- Chen, Wang, Shan. *BerDiff: Conditional Bernoulli Diffusion Model for Medical Image Segmentation*. [arXiv:2304.04429](https://arxiv.org/abs/2304.04429) ✓。二值 mask 用 Bernoulli 噪聲。這是影像條件下的分割，我們推論時沒有影像，參考價值低。
- Chen, Zhang, Hinton. *Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning*. ICLR 2023. [arXiv:2208.04202](https://arxiv.org/abs/2208.04202) ✓。離散標籤的 bit 編碼，SatSynth 的編碼來源。多類別 layout（剝落／鋼筋外露）可用。
- Gupta, Samaras, Chen. *TopoDiffusionNet: A Topology-aware Diffusion Model*. ICLR 2025. [arXiv:2410.16646](https://arxiv.org/abs/2410.16646) ✓。用 persistent homology 損失控制生成結構的 Betti 數（連通塊數）。直接對準「大片剝落不要碎」，但成本高，見 §5。
- Mazé, Ahmed. *TopoDiff: A Performance and Constraint-Guided Diffusion Model for Topology Optimization*（AAAI 2023 題名 *Diffusion Models Beat GANs on Topology Optimization*）. [arXiv:2208.09591](https://arxiv.org/abs/2208.09591) ✓。由物理條件（荷載、邊界條件）生成結構 layout，用代理模型 guidance，對 OOD 邊界條件泛化較好，贏過 cGAN。這是「物理條件 → 結構 layout」最接近的工程前例。

**單調增長、累積場**

- Osher, Sethian. *Fronts propagating with curvature-dependent speed: Algorithms based on Hamilton-Jacobi formulations*. J. Comput. Phys. 79(1):12–49, 1988 ✓（Crossref，DOI 10.1016/0021-9991(88)90002-2）。level set／到達時間表示的源頭。OFP 的 T 場就是到達時間函數。
- Gorinevsky, Kim, Beard, Boyd, Gordon. *Optimal Estimation of Deterioration from Diagnostic Image Sequence*. IEEE TSP 57(3):1030–1043, 2009 ✓（[Stanford 頁](https://stanford.edu/~boyd/papers/opt_det_est.html)）。從雜訊影像序列估計逐像素單調的損傷累積，應用在 SHM。OFP 用它做 T* 標籤清理。
- Chen, Wolterink, et al. *Vestibular schwannoma growth prediction from longitudinal MRI by time conditioned neural fields* (DeepGrowth). [arXiv:2404.02614](https://arxiv.org/abs/2404.02614) ✓。腫瘤以 SDF＋latent 表示，ConvLSTM 預測未來形狀。最接近的「結構化生長表示」，但沒有單調保證，也不渲染影像。
- Wehenkel, Louppe. *Unconstrained Monotonic Neural Networks*. NeurIPS 2019. [arXiv:1908.05164](https://arxiv.org/abs/1908.05164) ✓。對輸入單調的網路。用在 C3 的 A(k) 與 OFP 的 hazard 對 τ 的單調性。
- Mordvintsev, Randazzo, Niklasson, Levin. *Growing Neural Cellular Automata*. Distill 2020. DOI 10.23915/distill.00023 ✓。學局部規則長出整體形狀。OFP 的 hazard CA 是它的隨機、外部驅動、survival 版本。
- Yang, Xie, Zisserman. *Made to Order: Discovering monotonic temporal changes via self-supervised video ordering*. ECCV 2024. [arXiv:2404.16828](https://arxiv.org/abs/2404.16828) ✓。用排序代理任務找出影像序列中的單調變化區域。可以當「單調變化區」的自動標註器，取代亮度規則的 m*（未試過）。
- Jeong, Jung, Sohn, Suk. *Deep Geometric Learning with Monotonicity Constraints for Alzheimer's Disease Progression*. [arXiv:2310.03353](https://arxiv.org/abs/2310.03353) ✓。在病程軌跡上加不可逆的單調約束。醫學端的單調先驗前例（非影像生成）。

**弱監督、自動標註**

- Zhang et al. *DatasetGAN: Efficient Labeled Data Factory with Minimal Human Effort*. CVPR 2021. [arXiv:2104.06490](https://arxiv.org/abs/2104.06490) ✓。從生成器特徵用少量標註解碼出分割。可以從我們 G 的特徵解碼剝落／裂縫 layout。
- Baranchuk, Rubachev, Voynov, Khrulkov, Babenko. *Label-Efficient Semantic Segmentation with Diffusion Models*. ICLR 2022. [arXiv:2112.03126](https://arxiv.org/abs/2112.03126) ✓。用 diffusion U-Net 的中間特徵做少樣本分割。可以用 anylabel 的少量多邊形訓練出裂縫／鋼筋外露分割，給 layout 加類別。
- Flotzinger, Rösch, Braml. *dacl10k: Benchmark for Semantic Bridge Damage Segmentation*. WACV 2024. [arXiv:2309.00460](https://arxiv.org/abs/2309.00460) ✓。9,920 張橋梁檢測影像，類別包含 Crack、Spalling、Exposed Rebars 等。可以用來預訓練自動標註器，補我們缺的裂縫與鋼筋外露 layout。
- Benz, Rodehorst. *OmniCrack30k: A Benchmark for Crack Segmentation and the Reasonable Effectiveness of Transfer Learning*. CVPRW 2024, pp. 3876–3886. DOI 10.1109/CVPRW63382.2024.00392 ✓（mlanthology）。跨 20 多個資料集的裂縫分割基準，nnU-Net 最好。可以當裂縫自動標註器的來源，但權重是否公開未確認。

**專案內已引用（未由我查證）**

- MaskFlow（Xu, Yong, Yang, Gong, Lv, arXiv:2608.06929）：專案 `DAMAGE_PATTERN_CURRENT.md` 引用的區域編輯 flow，現行可畫區機制的依據。**未查證**。

### 2.2 F 方向：領域相關工作

**土木、SHM：損傷影像生成**

- Wu, Wu, Wang, Lin, Chang. *Development of a high-fidelity failure prediction system for reinforced concrete bridge columns using generative adversarial networks*. Engineering Structures 286 (2023). DOI 10.1016/j.engstruct.2023.116130 ✓（[NTU 頁](https://scholars.lib.ntu.edu.tw/handle/123456789/631328)）。用 110 張反覆載重試驗損傷圖訓練 cGAN，條件是設計參數＋目標 DI，判別器加分類器與回歸器，FID 102.6。**和本題最接近**：同樣是 NCREE 反覆載重 RC 柱、同樣小資料。差別是它把 DI 與設計參數當條件，本題刻意不用。專案裡的 cGAN 基線很可能是這一系的模型（我推論，未確認）。
- Lin, Chang, Wang, Wu, Saddek, Chang, Dzeng. *Damage Scenario Prediction for Concrete Bridge Columns Using Deep Generative Networks*. Structural Control and Health Monitoring 2024, 5526537. DOI 10.1155/2024/5526537 ✓（[NYCU 頁](https://scholar.nycu.edu.tw/en/publications/damage-scenario-prediction-for-concrete-bridge-columns-using-deep/)）。比較灰階、黑白、去障礙物三種影像表示，以 DI 為額外約束。「先把影像簡化成黑白（類 layout）」在這裡已經出現雛形。
- Wu, Wu, Wang, Lin, Chang. *Prognostic Modeling of Seismic Failure for RC Bridge Columns Leveraging Deep Generative Learning*. WCEE 2024 Proceedings ✓（[NTU 頁](https://scholars.lib.ntu.edu.tw/handle/123456789/736245)）。同一組人的 CGAN，從遲滯迴圈抽特徵改善表現。是「遲滯迴圈 → 損傷圖」的直接前例。
- Hoskere, Narazaki, Spencer. *Physics-Based Graphics Models in 3D Synthetic Environments as Autonomous Vision-Based Inspection Testbeds*. Sensors 22(2):532, 2022. DOI 10.3390/s22020532 ✓（[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8779227)）。非線性 FE 的塑性應變與壓碎損傷驅動裂縫、剝落、鋼筋外露的程序化渲染，損傷狀態由 drift 易損性抽樣。這是「物理 → 結構 layout → 紋理」的手工版，概念上就是 C2 / C5，只是兩段都不學。相關資料集 QuakeCity（[sail.cive.uh.edu/quakecity](https://sail.cive.uh.edu/quakecity/) ✓）有裂縫、剝落、鋼筋外露三類 mask。
- Li, Le, Zhao. *Style-Controlled Image Synthesis of Concrete Damages Based on Fusion of Convolutional Encoder and Attention-Enhanced Conditional Generative Adversarial Network* (DamageGAN-AE). J. Comput. Civ. Eng. 38(6), 2024. DOI 10.1061/JCCEE5.CPENG-6007 ✓（Crossref）。用 encoder 控制風格的 cGAN 合成混凝土損傷。資料增強用途，不處理演進。
- Rakowski, Bourlai. *On Enhancing Crack Semantic Segmentation Using StyleGAN and Brownian Bridge Diffusion*. IEEE Access 12:34769–34784, 2024. DOI 10.1109/ACCESS.2024.3368376 ✓（Crossref）。StyleGAN3＋BBDM 合成裂縫影像。只是資料增強。
- Cano-Ortiz 2024、CrackSegFlow 2026：見 2.1。土木界的趨勢是「mask → 影像」的 semantic synthesis，做演進的很少。

**材料：顯微組織、損傷演化**

- Najafi Koopas, Rezaei, Rauter, Ostwald, Lammering. *A spatiotemporal deep learning framework for prediction of crack dynamics in heterogeneous solids*. [arXiv:2407.15665](https://arxiv.org/abs/2407.15665) ✓。UNet 逐步預測相場損傷場（目前損傷 → 下一步），用 470 組模擬。結構化的損傷場本身可以逐步學。可以當 OFP 通用性的第二領域。
- Tiwari, Satpute, Ghosh. *Time series forecasting of multiphase microstructure evolution using deep learning*. [arXiv:2408.13111](https://arxiv.org/abs/2408.13111) ✓。卷積 AE 加 ConvRNN 在 latent 做時序預測。典型的「latent 自回歸」路線，沒有單調結構。
- Zhou, Zahran, Chen, Zhang, Wang. *A joint voxel flow – phase field framework for ultra-long microstructure evolution prediction with physical regularization*. [arXiv:2601.04898](https://arxiv.org/abs/2601.04898) ✓。voxel flow 網路預測，週期性插入相場模擬修正累積誤差。啟示：長序列自回歸要靠「物理錨點」止血。OFP 的單調場是另一種錨點。
- TopoDiff（見 2.1）：物理條件生成結構，OOD 泛化較 cGAN 好。

**醫學：疾病進程、老化、縱向生成**

- Puglisi, Alexander, Ravì. *Brain Latent Progression: Individual-based Spatiotemporal Disease Progression on 3D Brain MRIs via Latent Diffusion* (BrLP). [arXiv:2502.08560](https://arxiv.org/abs/2502.08560) ✓。輔助模型（線性回歸或 DCM）先預測區域腦容量，當 LDM 的 cross-attention 條件，ControlNet 吃基線影像，LAS 對多次取樣平均以穩定時序。是 C3 的範本。LAS 也提示：時序穩定可以用多樣本平均換。
- Litrico, Guarnera, Giuffrida, Ravì, Battiato. *TADM: Temporally-Aware Diffusion Model for Neurodegenerative Progression on Brain MRI*. MICCAI 2024. [arXiv:2406.12411](https://arxiv.org/abs/2406.12411) ✓。預測「兩次掃描間的強度差」，再加回基線，用腦齡估計器正則。和現行 refiner 的「增量」思路相同。
- Yoon, Zhang, Suk, Guo, Li. *SADM: Sequence-Aware Diffusion Model for Longitudinal Medical Image Generation*. [arXiv:2212.08228](https://arxiv.org/abs/2212.08228) ✓。transformer 讀整段序列當條件，可處理缺幀，自回歸生成。是 B 方向「序列 token」的醫學前例。
- Liu et al. *ImageFlowNet: Forecasting Multiscale Image-Level Trajectories of Disease Progression with Irregularly-Sampled Longitudinal Medical Images*. ICASSP 2025. [arXiv:2406.14794](https://arxiv.org/abs/2406.14794) ✓。在多尺度表示上用 neural ODE/SDE 預測，跨病人共享以對付資料稀少，資料含 GA 眼底。GA 是單調擴大病灶，是 OFP 最適合的第二領域。
- Xia, Chartsias, Wang, Tsaftaris. *Learning to synthesise the ageing brain without longitudinal data*. Medical Image Analysis 73:102169, 2021. [arXiv:1912.02620](https://arxiv.org/abs/1912.02620) ✓。年齡用「前 a 個為 1」的 ordinal 二元向量編碼，消融比 one-hot 好，只用橫斷面資料。連續嚴重度的條件編碼法，可以給 E 方向。
- Schmidt, Berens, Müller. *Ordinal Diffusion Models for Color Fundus Images*. MICCAI 2026. [arXiv:2602.24013](https://arxiv.org/abs/2602.24013) ✓。DR 分級改成標量嚴重度條件，相鄰分級過渡平滑，FID 與 κ 都改善。對應「演進 ratio／鄰居對比」問題。
- Zhou, Wang, He, Cui, Shao. *DR-GAN: Conditional Generative Adversarial Network for Fine-Grained Lesion Synthesis on Diabetic Retinopathy Images*. IEEE JBHI 2020. [arXiv:1912.04670](https://arxiv.org/abs/1912.04670) ✓。條件是結構 mask＋病灶 mask＋分級向量。「layout＋嚴重度」雙條件的早期版本。
- Feng et al. MICCAI 2024（見 2.1）：病灶 mask 生成 → 只在 mask 內 inpaint。

**植物生長、其他**

- Drees, Junker-Frohn, Kierdorf, Roscher. *Temporal Prediction and Evaluation of Brassica Growth in the Field using Conditional Generative Adversarial Networks*. Comput. Electron. Agric. 190:106415, 2021. [arXiv:2105.07789](https://arxiv.org/abs/2105.07789) ✓。cGAN 預測個別植株未來外觀。
- Drees et al. *Data-driven Crop Growth Simulation on Time-varying Generated Images using Multi-conditional Generative Adversarial Networks*. [arXiv:2312.03443](https://arxiv.org/abs/2312.03443) ✓。條件是初始影像、時間、處理，以及**模擬生物量**，加了生物量後性狀明顯變準。又一個「數量先行」的證據。
- Yu, Ghosh, Finn, Arcucci, Bocquet, Cheng. *A probabilistic approach to wildfire spread prediction using a denoising diffusion surrogate model*. Geosci. Model Dev. 19:1027–1054, 2026. DOI 10.5194/gmd-19-1027-2026 ✓。diffusion 學機率細胞自動機的下一步燃燒狀態，ensemble 給燃燒機率。「單調擴張的二值場」在火災領域用機率 CA 生成，支持 OFP 的 hazard CA 形式。
- Yuan et al. *MagicTime: Time-lapse Video Generation Models as Metamorphic Simulators*. [arXiv:2404.05014](https://arxiv.org/abs/2404.05014) ✓。微調大型 T2V 生成縮時「變形」影片（生長、融化）。需要大型預訓練與 2k 多段影片，不適合本題，列在 §5。

**F 方向的整體觀察**（用來檢驗通用性）：

1. **「先預測一個結構化或低維中介量，再生成影像」在三個領域獨立出現**：BrLP 的區域體積、Drees 的生物量、Hoskere 的 FE 損傷場、Feng 與 DiffInfinite 的病灶 mask。這支持 C2 / C3 / C5 的方向不是 RC 柱特有的。
2. **縱向生成幾乎都是在影像或 latent 上做自回歸**（SADM、TADM、ImageFlowNet、Tiwari、Yu）。沒有一篇用「單一單調起始場」表示整段序列。這是 OFP 想填的空缺。
3. **土木領域的損傷影像生成主要是 cGAN＋DI 或設計參數條件**（Wu 2023、Lin 2024），或 mask → 影像的資料增強。沒有處理「隨載重歷程演進」的結構化表示。

---

## 3. 推薦排序（可行性 × 原創性 × 通用性）

| 排名 | 方案 | 可行性 | 原創性 | 通用性 | 理由 |
|---|---|---|---|---|---|
| **0（前置）** | C4 layout-oracle | 高（1 天） | — | — | 決定 C2 / C5 值不值得做的閘門，必須先跑 |
| **1** | **C5 OFP（含 C3 的單調時鐘）** | 中：layout 部分 1–2 天可驗證，完整 5–8 天 | 中（組合新，組件舊） | 高（單調、可對齊、有驅動的序列都適用） | 唯一同時處理 #3、#4、時序一致性，並把單幀與時序統一的方案；小資料下有結構性理由（局部規則、吸收態） |
| **2** | C2 兩階段（SDF-CFM layout＋SDM / mask-ablated 渲染器） | 中高（4–6 天，全是成熟組件） | 低 | 高 | 是 C5 失敗時的退路，渲染器與 C5 共用，所以投入不浪費 |
| 3 | C3 數量先行（單調面積回歸） | 很高（半天） | 無 | 高 | 便宜。直接量出「面積這個量能不能外插到 CTR1」，結果同時決定 C5 的 τ 怎麼定 |
| 4 | C1 聯合生成 | 很高（1–2 天） | 無 | 高 | 最快試出「SDF 通道能否讓剝落不碎」。但不處理時序，也不處理外插的根因 |

### 論文實證

- 小資料兩段式（mask 生成器＋mask 條件生成）可行：MedDiff-FT（約 30 對、24 GB）、Feng 2024、DiffInfinite（少量標註＋未標註）。
- SDF 表示讓 mask 的生成分佈較平滑：Bogensperger 2023、FlowSDF 2024。
- mask 條件生成時，容忍不完整的 mask 有助於忠實度與真實感：Konz 2024。
- 先預測低維量再生成能提升可量測性狀的準確度：BrLP（體積誤差改善）、Drees 2023（生物量）。
- 物理條件生成結構 layout，OOD 泛化比 cGAN 好：TopoDiff（拓撲最佳化領域，不是影像紋理）。
- 單調的逐像素損傷估計在 SHM 影像上有效：Gorinevsky 2009。

### 我的推論（沒有直接證據）

- 剝落破碎（#3）主要是「速度場在大區塊上被平均」造成的，給明確 layout 會修好。要靠 C4 驗證。
- CTR1 低估（#4）有一部分是「面積沒有被單獨約束」。如果是 CTR1 的物理差異（軸力）沒有被條件表示，C3 / C5 都救不了。要靠 C3 驗證。
- 局部 hazard CA 在 8 條序列上學得到合理的位置先驗與蔓延規則。沒有任何論文在這麼少的序列上做過。
- 損傷年齡圖 A_k 能改善紋理（新鮮斷面 vs. 鋼筋外露）。目前沒有證據。
- C5 的單幀＝切片設計會讓時序一致性（`tone_jump`）大幅改善。結構部分是保證的，紋理部分仍取決於渲染器的鎖住機制。

---

## 4. 最小驗證實驗（每個 1–2 天）

共同設定：

- 對照組是 V9（`selfcondnoDI`），同切分（`id_list_val_R307.json`）、同 15000 step、Heun 30 步。現在的 g 正在換成損傷狀態編碼器，兩邊用同一個條件版本。
- 指標用現有 `eval_suite`：DINOv3 KID global、演進 ratio / corr、鄰居對比、CTR1 高 DI 剝落面積偏差（用 anylabel 算 GT，生成圖用 m* 規則或 anylabel 訓練的剝落 probe 量）。
- 另加兩個結構指標：高 DI 三分位的「剝落連通塊數」與「最大連通塊佔剝落面積比例」。
- 報 val（R307）與 test（CTR1）兩根，SE 用區塊 jackknife。

### 實驗 0（C4 layout-oracle，1 天，最先做）

- **做法**：在 V9 上加兩個 zero-init 輸入通道（剝落 layout 的截斷 SDF、損傷年齡 A；第一版可只放 SDF），用 anylabel 算的真實 layout 微調 5k step，或從頭 15k step。推論時餵 CTR1 / R307 的真實 layout。
- **比較**：V9 vs. layout-oracle。
- **固定**：條件 g、取樣器、步數、種子。
- **值得繼續的條件**：
  - CTR1 高 DI 剝落面積偏差從 −0.2～−0.35 縮到 |偏差| < 0.1。
  - 最大連通塊比例接近 GT（差 < 0.1）。
  - KID 不比 V9 差超過 1 個 SE。
- **代表不值得做 C2 / C5 的結果**：面積仍低估超過 0.15，或剝落仍碎。這表示渲染器無法照 layout 畫，要先修渲染器（D 方向）。

### 實驗 1（C5 OFP 的 layout 部分，1–2 天，不需要渲染）

- **做法**：
  1. 用 anylabel 多邊形算每幀 16×16 覆蓋率，逐格做 isotonic regression（單調不減），以 0.5 為界得 T*。
  2. 訓練 hazard CA，三個變體：(a) 只有成核項，(b) 成核＋蔓延（3×3 鄰域），(c) (b)＋單調時鐘 τ（UMNN，輸入累積能量、勁度比、強度比、最大 drift）。
  3. 對照組是現行 M 逐幀 rollout：讀真實上一幀，以及讀自己的上一幀二值化結果。
- **評估**（R307、CTR1，每個條件抽 16 個樣本）：
  - 剝落面積軌跡的 signed bias（全段與高 DI）。
  - 最終 layout 與 GT 的 IoU。
  - 連通塊數、最大連通塊比例。
  - 單調違反次數（M 會有，CA 為 0）。
  - 樣本間 IoU（多樣性）。
- **值得繼續的條件**：
  - (c) 在 CTR1 的高 DI 面積偏差 |偏差| < 0.1，而且明顯好於 M rollout。
  - 連通塊數 ≤ GT + 1。
  - R307 上不比 M 差。
- **若 (c) 不比 (b) 好**：τ 沒有帶來外插能力，要改用損傷狀態編碼器的 state 當 τ 的輸入，或接受外插只能靠 C3。

### 實驗 2（C5 完整串接，接在實驗 0、1 之後，1 天）

- **做法**：用實驗 0 的 layout 條件渲染器，餵實驗 1 抽樣得到的 T 切片（L_k、A_k）。
  - 單幀模式：每個條件獨立抽 T。
  - 時序模式：同一張 T 切所有幀，用現有可畫區／鎖住區機制，增量區由 T 決定。
- **比較**：V9、V9＋refiner（現行時序）、OFP 單幀、OFP 時序。
- **值得繼續的條件**：
  - OFP 拿到實驗 0 oracle 改善量的一半以上（面積偏差、連通塊）。
  - 演進 ratio 提高至少 0.05。
  - 時序模式的 `tone_jump` 低於 refiner。
  - KID 不比 V9 差。

### 實驗 3（C3，半天，可和實驗 0 並行）

- **做法**：用 8 條訓練序列的 `R_spall(k)` 擬合單調回歸 A(k)，比較三種：線性、logistic、UMNN。輸入是 4 個損傷標量。用 leave-one-column-out 選模型，再預測 R307 與 CTR1。
- **值得繼續的條件**：CTR1 高 DI 的面積預測偏差 |偏差| < 0.1。
  - 若成立：低估的根因在「生成器沒照面積畫」，C3 / C5 的校準有效。
  - 若不成立：連這 4 個標量都不足以外插 CTR1，要把資訊來源換成損傷狀態編碼器或 160 根柱的自監督表示（E 方向），C5 的 τ 也要跟著換。

### 實驗 4（C1，1–2 天，可選）

- **做法**：z 多一個剝落 SDF 通道（anylabel，截斷到 ±8 px 再正規化），從頭訓 15k step。
- **評估**：KID、ratio、鄰居對比、面積偏差（直接讀 SDF 通道，也用 m* 量影像，兩者差值即一致性）、連通塊。
- **值得繼續的條件**：面積偏差改善 ≥ 0.1，連通塊比例改善，KID 不變差，通道與影像的面積差 < 0.05。

---

## 5. 不建議的方向

| 方向 | 理由 |
|---|---|
| 在整張 layout（或整張 T 場）上訓練無條件或全域條件的 diffusion（D3PM、Bernoulli diffusion、Analog bits 直接套） | 序列層級只有 8 個樣本，整張 layout 分佈學不到，會背訓練柱的剝落位置。BerDiff 這類是「有影像時的分割」，我們推論時沒有影像。要用局部規則（C5）或至少以上一幀 layout 為條件的局部生成（C2） |
| layout 階段在 layout 上做逐幀自回歸 diffusion（像 Yu 2026 的狀態到狀態） | 沒有單調保證，誤差會累積。8 條序列也不夠學穩定的轉移分佈。吸收態 CA 用一個 Bernoulli 就能表達同樣的事 |
| TopoDiffusionNet 式的 persistent homology 損失 | 每步要算 persistence diagram，成本高。「連續大塊」用表示法（SDF、蔓延項）解決比較便宜。如果 SDF 仍破碎，再考慮加連通塊數懲罰 |
| JoDiffusion 式「另訓 mask VAE、在 latent 聯合 diffusion」 | 需要另外兩個 VAE。530 張圖、灰階 128 下，VAE 本身的重建誤差就可能吃掉紋理（A/B 方向會評估 latent 路線） |
| 以 ControlNet＋SD 當 layout 渲染器 | 專案已在時序上試過 ControlNet 式分支，效果有限。MICCAI 2024 的一份補充材料也提到 ControlNet 在醫學資料上表現差（只見於搜尋摘要，未讀全文）。灰階柱面與 SD 的 domain gap 大。要用預訓練先驗交給 A 方向評估，不要和 layout 綁在一起 |
| PBGM / FE 驅動的 sim-to-real（Hoskere 2022） | 每根柱都要非線性 FE 模型，而我們只有遲滯迴圈。概念可借（OFP 的 T 可以來自 FE），但工作量是週到月 |
| 大型時序影片模型（MagicTime 一類） | 需要大型 T2V 預訓練與上千段影片。灰階柱面與自然縮時影片差太多，24 GB 也很勉強 |
| 用亮度規則自動標裂縫 layout | m* 的亮度規則對剝落都只是粗標籤，裂縫只有 1–2 px，規則會被陰影與標記筆線干擾（`interference` 標註有 878 個）。裂縫要麼留給渲染器隱式處理，要麼用 dacl10k / OmniCrack 預訓練的分割器 |
| 只靠換條件編碼器解決 #4 | 已經試過，改善有限。C3 的實驗能直接判斷「條件裡是否有足夠的外插資訊」，比再換一個編碼器有診斷力 |
