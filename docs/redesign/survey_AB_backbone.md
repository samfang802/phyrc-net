# 方向 A／B 文獻調查：預訓練先驗、latent + Transformer、序列條件，以及一個原創方案

範圍：原 prompt 第 5 節的 A（借用大型預訓練先驗）和 B（latent + Transformer 骨幹、半循環 token 序列條件）。C/F、D/E 由另外兩位研究員負責，這裡不重複。
依 2026-10-25 的更新需求，每個方案多兩欄：「原創性」和「通用性」。另外提出一個以骨幹與條件注入為主軸的原創方案（O1），並以「可行性、原創性、通用性」三者一起排序。

標記方式：✓ 表示我已打開 arXiv abs/HTML 頁、會議頁或官方 repo，確認標題、作者、年份和編號。「venue 未查證」表示 arXiv 編號已確認，但會議名稱是憑記憶寫的。「未查證」表示整筆都是憑記憶。
專案內的事實來自 `docs/DAMAGE_PATTERN_CURRENT.md`、`docs/ARCHITECTURE_UPDATE_V11.md`、`docs/DAMAGE_STATE_ENCODER.md`、`docs/eval/results.md`、`docs/archive/ARCHITECTURE_UPDATE_V8.md`、`src/model/adm.py`。

**和本題直接相關的專案事實：**

- `results.md` 量到：給完美條件（oracle）時，KID 只改善 −0.009。作者的結論是「真實感主要是生成器自己的極限」。這是方向 A（換一個有先驗的骨幹）最主要的理由。
- `adm.py` 已有 `CrossAttention2d`，接在 mid／U3／U2。V8 用過 `--use_physics_cross_attn`，但當時是把**單一個** 1024 維 g 線性切成 16 個偽 token，不是真的歷史序列，V9 以後關掉。所以 B 的「每個半循環一個 token」在這個專案**沒有被測過**。
- 損傷狀態編碼器 A_info 本來就是「每個半循環一個 token → 單向 GRU」，每個 k 都有 z_k。B／O1 需要的序列已經有了，不必另外造。
- 現行 G 已經能吃「上一幀粗亮度 + 可畫區 α」兩個通道，沒有上一幀時填 0。所以「單幀與時序同一個網路」在專案裡已部分存在。O1 要擴充的是上一幀的形式（全解析度、連續噪聲等級）和條件路徑，不是從零開始。

---

## 1. 候選方案

### 總表

| # | 方案 | 解決第 4 節哪些問題 | 實作量 | 24 GB 可行性 | 原創性（摘要） | 通用性（摘要） |
|---|---|---|---|---|---|---|
| A1 | **PT-ADM128**：OpenAI ImageNet-128 class-conditional ADM（像素空間）整網微調，物理條件取代 class embedding | 1、5（主要），3（次要） | 3–4 天 | 可（全微調約 4 億參數，batch 16–32 加 checkpointing） | 無原創性，屬工程套用 | 高 |
| A2 | **PT-SiT-latent**：ImageNet-256 預訓練 SiT-XL/2（SD-VAE latent，linear interpolant + velocity，和現行目標相同），物理走 adaLN，歷史 token 走新加的 gated cross-attention，上一幀 latent 走 channel concat | 1、3、5，2（透過 cross-attn） | 5–7 天（含 VAE 檢查） | 可（675M 全微調約 17–20 GB；DiffFit/LoRA 小於 8 GB） | 低：DiffFit + GLIGEN + GameNGen 的組合 | 中（要求 VAE 能還原該領域細節） |
| B2 | **SeqXAttn**：保留現行像素 ADM-32，加上對「真的半循環序列 z_1..z_k」的因果 cross-attention（Bastek 式），池化結果也加到 AdaGN | 2、4（假設） | 1.5–2 天 | 很輕 | 低：Bastek & Kochmann 已用逐步 token 做 cross-attn | 高 |
| O1 | **PC-DF（Physics-Clocked Diffusion Forcing，我的原創提案）**：單幀與時序同一個網路，上一幀是帶連續噪聲等級的槽位；歷史 token 的位置編碼用單調的「物理時鐘」（累積能量、已到達最大 drift），不用 index；可選擇讓編碼器和 124 根無影像柱的反應預測共同訓練 | 2、4（主要），單幀／時序統一，3（間接） | 第一階段 3 天，第二階段 +2 天 | 很輕（建在 ADM-32 上），之後可換成 A1 骨幹 | 中：沒有找到相同做法（§1.5） | 高（任何「少量影像、單調演進、外部驅動」的資料） |

沒有列成候選、放到 §5 的有：T2I 骨幹（SD/PixArt/SD3/FLUX）把文字換成物理 token、自訓 VAE + 從零訓 DiT、VA-VAE/RAE 這類 DINO 對齊的 latent。

---

### 1.1 A1：PT-ADM128（預訓練像素 ADM，class embedding 換成物理）

**核心想法**

guided-diffusion 有現成的 ImageNet 128×128 class-conditional 權重（MIT license）。flags 是 channels 256、2 res blocks、attention 在 32/16/8、scale-shift norm、learn_sigma ✓（官方 repo）。

這個架構和現行 ADM-32 幾乎是同一家族：

- class embedding 加到時間嵌入後驅動 scale-shift norm，就是現行 AdaGN 的路徑。
- 像素空間、128 解析度，和現在完全一樣。不需要 VAE，細裂縫不會因壓縮而流失。

做法：

1. 把 `label_emb` 換成 v11 的條件 MLP，輸出 `e_c + α·e_g + α_d·e_d`。
2. 灰階複製成 3 通道，輸出取 3 通道平均。
3. XY 座標、上一幀粗亮度、α 都當成新增的輸入通道，權重從 0 開始。
4. 整網微調。

RCG ✓ 已示範「用連續表徵向量直接取代 ADM/DiT 的 class embedding」可行（RCG 是重新訓練生成器，不是微調時抽換，所以這只是部分證據）。

**解決的問題**

- 1（紋理真實感）：主要針對這點。oracle 實驗顯示瓶頸在生成器本身。現行寬度 32 的網路容量和先驗都很弱，ImageNet 128 的低層卷積濾波器帶有自然紋理與邊緣的先驗。
- 5（同條件下亮度不穩）：可能有幫助，屬推論。
- 3（剝落碎）：較深的網路加上 32/16/8 的 attention 有較大感受野，屬推論。

**依據**

- 預訓練後微調在少量資料下勝過從零訓練：DDPM-PA ✓（Zhu et al. 2022，10-shot，指出直接微調品質較好但會丟失多樣性）、DomainStudio ✓。
- Diff-Tuning ✓：可遷移性沿逆向鏈遞減。
- T-LoRA ✓：高噪聲步最容易過擬合，所以高噪聲端的調整要節制。
- 跨大 domain gap 的灰階醫學影像微調可行：Chambon et al. 2022 ✓、RoentGen ✓（但資料量遠大於 530）。

**實作量**：3–4 天。

- 載入權重和改名：0.5 天。
- 條件替換與「錨點初始化」：0.5 天。做法是讓物理 MLP 一開始輸出某個固定 ImageNet 類別（例如 wall/stone 類）的 embedding，再加上一個從 0 起始的殘差。這是我的推論。
- 處理目標不一致：ADM 原本是 ε 預測 + learned σ + linear β。要改成現行 CFM velocity，輸出頭需要重新適應，學習率要壓低（例如 2e-5）並加 warm-up。另一個選擇是先用原生 ε 目標（約 +1 天）。
- 訓練與評估：1.5 天。

**24 GB 估算**（推估，未實測）

- 參數量：依官方 flags 手算約 3.5–5 億。channel mult (1,1,2,3,4)：各層 resblock 約 18c²，attention 約 4c²。
- AdamW 混合精度全微調約 16 B/參數，也就是 6–8 GB。
- 128² 解析度、最寬 1024 通道的 activation，batch 16 不加 checkpointing 估計約 10–14 GB；加 gradient checkpointing 後 batch 32 可以放進 24 GB。8-bit Adam 還可以再省約 3 GB。
- 推論：36 格 × K=8 × Heun 30 步（約 59 NFE），在 A5000 上大約幾分鐘。
- 每步算力約為 ADM-32 的數十倍（寬度 8 倍，conv FLOPs 約 ×64），需要實測。530 張只需要約 100–300 epoch（batch 32 時約 2–5k step），一天內應可完成。

**最大風險**

ImageNet 自然影像的先驗未必適合灰階、單一場景的混凝土；而且 4 億參數記住 530 張圖非常容易，條件跟隨可能反而變差（模型背整根柱）。驗證時要同時看 DINOv3 最近鄰距離。

**原創性**：無原創性，屬工程套用。最接近的是 DDPM-PA／DiffFit／RCG：預訓練 ADM/DiT 加上抽換條件嵌入。

**通用性**：高。只要求「影像 ≤ 256、能接受 3 通道複製、條件可以表成一個向量」。依賴的假設是 ImageNet 低層特徵對目標領域有用；這點在醫學灰階影像上有部分證據（Chambon/RoentGen）。RC 柱特有的只有條件 MLP 的內容。

---

### 1.2 A2：PT-SiT-latent（預訓練 latent SiT + 歷史 token cross-attn + 上一幀 latent concat）

**核心想法**

SiT-XL/2 ✓ 的設定：

- 用 SD-VAE latent（256² → 32×32×4）、patch 2，共 256 個 token。
- 最佳設定是 Linear interpolant + velocity 預測，條件走 adaLN-Zero（沿用 DiT）。時間慣例是 t=0 為資料、t=1 為噪聲，和專案相反，需要翻轉。
- 也就是說，它的訓練目標和現行 OT-CFM 本質相同，損失和採樣器幾乎不用改。

做法：

1. 128 → 256 雙三次上採樣，灰階複製 3 通道，latent 先離線算好快取。
2. y_embedder 換成物理 MLP（和 A1 相同的錨點初始化）。
3. 每個 block 加一層 GLIGEN 式 gated cross-attention，attend 到 z_1..z_k（tanh(γ) 閘門從 0 開始；細節依記憶）。
4. 上一幀 latent 以 channel concat 接到 patch embed，新增權重從 0 開始。這是 GameNGen ✓ 的做法：它試過把過去幀改用 cross-attn，沒有明顯改善，所以選 concat。上一幀噪聲增強沿用 GameNGen（最大 0.7、分桶後輸入網路），和現行 0–0.15 的 OT 推回一致。
5. 時序的可畫區混合改在 latent 32×32 上做。專案的 16×16 粗格剛好對應 2×2 latent。依據是 Blended Latent Diffusion ✓。

**解決的問題**

- 1、5：主要針對這兩點。
- 3：全域 self-attention 有利於畫出連續大塊，屬推論。
- 2：歷史 cross-attn 可能有幫助。

**依據**

- DiT ✓ 比較了條件注入方式：adaLN-Zero FID 19.47，優於 cross-attn 26.14、in-context 35.24（DiT-XL/2，400K step）。所以「向量條件走 adaLN、序列條件才用 cross-attn」。
- DiffFit ✓：DiT-XL/2 從 ImageNet 遷移到 8 個下游資料集，只調 bias/norm/class emb/γ（0.12% 參數），平均 FID 15.39，全微調 16.59。**注意：這 8 個資料集最小也有數千張，沒有接近 530 張的證據。**
- GameNGen ✓：SD 1.4 全微調，文字 cross-attn 換成動作 token 序列，過去幀用 latent concat，並微調 VAE decoder。
- OminiControl ✓、FLUX.1 Kontext ✓：上一幀改成 in-context token 串進序列的替代做法，token 數會變成兩倍。

**實作量**：5–7 天（SiT 程式碼整合 2 天、VAE 檢查 0.5 天、cross-attn 和 concat 1 天、時序接回 1–2 天）。

**24 GB 估算**（推估）

- 675M 參數 × 16 B 約 10.8 GB。
- activation 約為 256 token × 1152 × 28 層。batch 32 加 checkpointing 約 3–5 GB，合計約 15–18 GB。不加 checkpointing 時 batch 約 16。
- DiffFit/LoRA 時 frozen bf16 權重約 1.4 GB，batch 64 沒問題。
- 算力：DiT-XL/2 forward 是 118.6 GFLOPs ✓，訓練約 3 倍。假設 A5000 有效 15–30 TFLOPS（未實測），每秒約 15–40 張，一天大約 1–3M 樣本，也就是 2000 epoch 以上。**算力不是瓶頸，過擬合才是。**

**最大風險**

1. SD-VAE f8 在灰階細裂縫（128 px 時 1–2 px 寬）上的還原度未知。必須先過 §4 的 VAE 檢查。
2. 675M 的 transformer 在 530 張上很容易背圖。Li et al. CVPR 2024 ✓ 明確寫到：transformer 骨幹「因為缺少 inductive bias，從零訓練比 U-Net 困難」，而且 PixArt-α 變體要花更多 iteration 才追上 U-Net。這個風險在微調時比從零訓練小，但沒有消失。

**原創性**：低。屬於 DiffFit（參數高效遷移）、GLIGEN（gated 新增注入層）、GameNGen（latent concat 上一幀與噪聲增強）三者的組合。差異只在條件是物理序列，**不足以構成方法貢獻**。

**通用性**：中。依賴兩個假設：(i) 現成 VAE 對目標影像的還原夠好（顯微、超音波、裂縫這類高頻細節常常不成立）；(ii) 影像可以上採樣到 256。RC 柱特有的只有條件內容。

---

### 1.3 B2：SeqXAttn（現行像素骨幹 + 真實半循環序列的因果 cross-attention）

**核心想法**

不換骨幹，只改條件形式：

- 每個 k 的 A_info 狀態 z_j（j ≤ k，最多取最近 N=16–32 個半循環），加上 9 個 token 原始特徵。
- 經 Linear 變成 key/value，在 `adm.py` 現有的 mid/U3/U2 CrossAttention2d 位置注入，輸出乘上從 0 開始的閘門。
- 再做一次 attention pooling，結果加到 AdaGN 的條件向量。

這是 Bastek & Kochmann ✓ 的做法：應力—應變曲線每個點經 Linear 變成一個 token，影像在對應步驟用 cross-attn 讀取，同時 token 平均後經 MLP 加到時間嵌入。和 V8 的差別：V8 是把一個向量切成偽 token；這裡是**真的時間序列**，影像可以 attend 到「哪一次大位移」。

**解決的問題**：2（條件反應弱）、4（外插），都是假設。現行向量條件的有效秩只有約 16，歷史被壓成一個點。序列讓空間位置可以讀不同的歷史事件。

**依據**

- Bastek & Kochmann 2023 ✓：曲線 token 的 cross-attn 加上池化到時間嵌入。
- SADM ✓：IPMI 2023，用 sequence-aware transformer 當縱向醫學影像生成的條件模組。
- DreamDiffusion ✓：EEG 編碼器的 token 經投影進 SD 的 cross-attn，只訓練編碼器和 cross-attn 部分。
- MindEye2 ✓：先在多受試者上預訓練共享對映，新受試者只用 1 小時資料。對應到這裡就是「編碼器先在 124 根柱上訓練」。

**實作量**：1.5–2 天。dataset 要回傳 z 序列加遮罩，模型端已有 CrossAttention2d。

**24 GB 可行性**：很輕。cross-attn 只在 ≤ 16×16 的解析度做，序列長 ≤ 32。

**最大風險**：4 根柱的歷史序列形狀很容易被當成柱子 ID。這個方案正好挑戰「柱編號不進生成器」的原則。需要做 token dropout 和截斷，並用打亂歷史的對照確認模型真的有用歷史。

**原創性**：低。Bastek & Kochmann 在逐步 token 上做過幾乎一樣的事，差別只在應用領域。這個方案在 §4 被併入 O1 當作消融組 R1。

**通用性**：高。只要條件本身是序列就適用。

---

### 1.4 O1（原創提案）：PC-DF，Physics-Clocked Diffusion Forcing

> 以下整節都是**我的推論**，不是文獻結論。

**動機**

問題 2 和 4 都指向同一件事：生成器把「現在處於損傷演進的哪個階段」讀得太弱，外插時讀錯。問題 4 中 CTR1 的軸力與箍筋比在訓練範圍之外。同一個 drift 歷程下，它的能量消散和破壞時程都不同；但 index 式的時間（第幾個半循環）對所有柱都一樣。

**三個組件**

**(a) 統一的「上一幀槽位」**

網路輸入是

```
[x_τ（這一幀加噪）, x̃_prev = (1−σ_p)·x_prev + σ_p·ε, XY, α]
```

σ_p 經 embedding 加到時間嵌入。

- 訓練時，40% 機率令 σ_p = 1（看不到上一幀，也就是單幀模式），其餘從 U(0, 0.3) 抽（時序模式）。
- 推論時，單幀用 σ_p = 1；時序 rollout 用 σ_p 約 0.05–0.1，並保留專案現有的 α 鎖定混合。
- 可以加上 DFoT ✓ 的 vanilla history guidance：v = v(σ_p=1) + w·[v(σ_p 小) − v(σ_p=1)]，用 w 控制要多信任上一幀。

這樣同一組權重同時是第 0 幀來源和 refiner，530 張圖同時訓練兩種模式。**這部分本身不是原創**，見下方「原創性」。

**(b) 物理時鐘的因果 cross-attention（核心原創點）**

歷史 token h_j（j ≤ k，內容是 z_j 加上 token 特徵）的位置編碼不是 index j，而是單調的物理量：

```
c_j = [log1p(累積消散能量_j), 已到達最大 drift_j]
```

兩者都可以從模擬迴圈算出，推論時已知。這兩個量對每一根柱都是單調不減的「損傷時鐘」。

- key 使用 Fourier(c_j)，query 端使用 Fourier(c_k)，attention 加上相對時鐘差 c_k − c_j 的偏置（類似 continuous-time attention / RoPE 的連續版本）。
- 加上因果遮罩（只看 j ≤ k）。
- 注入位置同 B2（mid/U3/U2，16×16 與 8×8），閘門從 0 開始。

**(c) 可選：反應共訓練（第二階段）**

編碼器不再凍結。總損失是

```
L = L_flow（4 根有影像的柱） + λ·L_resp（124 根無影像柱 + 4 根有影像柱的實驗強度比、勁度比、累積能量）
```

無影像的柱等同於「影像槽位永遠缺席」，概念上是 UWM ✓ 的模態專屬噪聲等級，但方向相反：UWM 是動作缺席、影像充足；這裡是影像缺席、驅動與反應序列充足。λ 取大值、編碼器學習率取小值，讓 h 被 124 根柱的物理錨定，同時允許影像梯度微調它。

**小資料下的理由**

1. **時間扭曲的 inductive bias。** Tallec & Ollivier ✓ 證明 gated RNN 對時間重新參數化有近似不變性。把位置換成物理時鐘，等於明確告訴網路「循環次數本身不重要，消散了多少能量、到過多大位移才重要」。不同加載協議的柱在時鐘座標上會對齊，4 根柱就能覆蓋較寬的「階段」範圍。CTR1 的外插從「沒見過的循環序列」變成「見過的時鐘區段」。這是假設，需要實驗驗證。
2. **參數共享。** 單幀和時序共用同一個網路，不再有一個只用部分資料訓練的 refiner。cross-attn 只在低解析度做，新增參數約 10⁵ 等級。
3. **用上無影像資料。** (c) 讓 124 根柱直接約束條件路徑。現行做法只在編碼器預訓練時用到它們，生成器訓練時編碼器是凍結的。

**解決的問題**

- 2、4：主要針對這兩點。
- 單幀與時序統一。
- 3（間接）：歷史 attention 可以讓剝落區「記得」它在哪一次大位移開始。

**實作量**

- 第一階段 (a)+(b) 約 3 天：dataset 輸出時鐘與序列 0.5 天、σ_p 槽位 0.5 天、時鐘 attention 1 天、評估 1 天。
- 第二階段 (c) 約 +2 天。

**24 GB 可行性**：建在 ADM-32 上時很輕。若 A1 成功，換成 PT-ADM128 骨幹，記憶體同 A1。

**最大風險**

- 物理時鐘來自**模擬**迴圈。如果 CTR1 的模擬能量本身偏差很大，時鐘就是錯的。
- (c) 的影像梯度可能把 h 拉向「背 4 根柱」。

**原創性**（已搜尋確認，見 §1.5）

| 組件 | 最接近的 prior work | 差異 | 是否足以構成貢獻 |
|---|---|---|---|
| (a) 上一幀的連續噪聲槽位、單幀與時序統一 | Diffusion Forcing ✓、DFoT/History Guidance ✓、GameNGen ✓、MCVD（未查證） | 幾乎相同 | 否，屬工程套用 |
| (b) 物理時鐘位置編碼 | mTAN ✓（不規則取樣時序的連續時間 attention）；Bastek ✓（以應變步為 index 的 token） | 用**驅動訊號導出的單調損傷量**取代時間或 index，當作條件 cross-attn 的座標，以處理不同加載協議的對齊。兩次搜尋（擴充模式）都沒有找到在生成模型或結構反應模型中這樣做的工作 | 是，但貢獻偏小而且簡單。必須用外插改善（CTR1）和另一個領域的轉移來支撐 |
| (c) 反向模態缺席的共訓練 | UWM ✓（無動作標註的影片）、Multimodal Diffusion Forcing ✓（力訊號 + 影像）、MindEye2 ✓ | 資料比例反過來（影像稀少、驅動與反應充足），而且反應不是生成目標，只是錨定條件路徑 | 中：組合式創新 |

整體評估：(b) + (c) 合起來可以寫成一個「小資料、外部驅動的影像演進生成」的方法貢獻，但目前只有我的推論支持，**沒有文獻實證**。

**通用性**

依賴的假設：

1. 存在一個能從驅動訊號算出的單調進程量。
2. 影像是固定視角、已配準的。
3. 影像變化大致是累積的。
4. 有大量沒有影像的驅動或反應序列。

在其他領域上的對應：

- **醫學縱向影像**：時鐘是累積劑量（放療、藥物），token 是給藥事件，無影像資料是只有檢驗值的病歷。
- **材料演化**：時鐘是累積塑性應變或輻照通量，token 是加載區塊，無影像資料是沒有顯微照的機械試驗曲線。
- **植物生長**：時鐘是農學上的生長度日（growing degree days），token 是每日氣象，無影像資料是只有產量的氣象紀錄。

RC 柱特有的部分：半循環的切法、柱面與正負號的對稱、剝落鎖定規則、CLAHE 灰階。

### 1.5 原創性搜尋紀錄（O1）

- 「diffusion 以累積能量／劑量／進程變數取代 time index 當位置或條件」：沒有找到。最接近的是 PDE 場重建的 cross-attn，以及討論 timestep embedding 的文章，都不是物理進程量。
- 「transformer 位置編碼用累積損傷取代時間步（結構反覆載重）」：沒有找到。只有用一般序列 transformer 預測 316H 鋼反覆反應的工作。
- 「影像稀少、動作或感測序列充足的共享骨幹 diffusion」：找到 UWM（方向相反）、Multimodal Diffusion Forcing、Dual-Stream Diffusion、Vid2World、Causal World Modeling。都是影像充足或預訓練影片模型，沒有找到「驅動與反應充足、影像稀疏」的設定。
- 「以遲滯迴圈或加載歷史條件化的 RC 損傷影像 diffusion」：沒有找到。結果都是傳統遲滯模型。
- 結論：沒有找到完全相同的工作。不過搜尋範圍有限（WebSearch 加上少數全文），不能排除 2026 年未索引或非英文的工作。

---

## 2. 論文清單

### A：預訓練先驗與微調

| 論文 | 出處 | 與本問題的關聯 | 狀態 |
|---|---|---|---|
| Dhariwal & Nichol, *Diffusion Models Beat GANs on Image Synthesis*, 2021 | [arXiv:2105.05233](https://arxiv.org/abs/2105.05233)；權重見 [openai/guided-diffusion](https://github.com/openai/guided-diffusion) | A1 的 ImageNet-128 class-conditional ADM 權重來源，架構和現行 ADM 同家族 | ✓（arXiv、repo flags 與 MIT license 已確認；NeurIPS 2021 未查證） |
| Xie et al., *DiffFit: Unlocking Transferability of Large Diffusion Models via Simple Parameter-Efficient Fine-Tuning*, 2023 | [arXiv:2304.06648](https://arxiv.org/abs/2304.06648) | 只調 bias/norm/class emb/γ 就能把 DiT-XL/2 遷移到 8 個資料集，平均 FID 優於全微調；但資料量都在數千張以上 | ✓ |
| Zhong et al., *Diffusion Tuning: Transferring Diffusion Models via Chain of Forgetting*, 2024 | [arXiv:2406.00773](https://arxiv.org/abs/2406.00773) | 可遷移性沿逆向鏈遞減，可以保留靠近資料端的預訓練知識，作為 A1/A2 依 timestep 區分微調強度的依據 | ✓（NeurIPS 2024 未查證） |
| Soboleva et al., *T-LoRA: Single Image Diffusion Model Customization Without Overfitting*, 2025 | [arXiv:2507.05964](https://arxiv.org/abs/2507.05964)，AAAI 2026 | 高噪聲步最容易過擬合，所以要依 timestep 調整 rank；適用於 530 張的背圖風險 | ✓ |
| Zhu et al., *Few-shot Image Generation with Diffusion Models*（DDPM-PA）, 2022 | [arXiv:2211.03264](https://arxiv.org/abs/2211.03264) | 少量資料時，預訓練微調優於從零訓練，但會丟失多樣性；pairwise 相似度損失 | ✓ |
| Zhu et al., *DomainStudio: Fine-Tuning Diffusion Models for Domain-Driven Image Generation using Limited Data*, 2023 | [arXiv:2306.14153](https://arxiv.org/abs/2306.14153) | 有限資料微調（包含條件模型）的過擬合與高頻細節問題 | ✓ |
| Xiang et al., *A Closer Look at Parameter-Efficient Tuning in Diffusion Models*, 2023 | [arXiv:2303.18181](https://arxiv.org/abs/2303.18181) | adapter 輸入接在 cross-attn 之後效果最好，約 0.75% 參數就能追上全微調 | ✓ |
| Chambon et al., *Adapting Pretrained Vision-Language Foundational Models to Medical Imaging Domains*, 2022 | [arXiv:2210.04133](https://arxiv.org/abs/2210.04133)，NeurIPS 2022 FMDM workshop | SD 的 VAE/U-Net/文字編碼器分別微調到灰階胸部 X 光，是跨大 domain gap 的證據 | ✓ |
| Chambon et al., *RoentGen: Vision-Language Foundation Model for Chest X-ray Generation*, 2022 | [arXiv:2211.12737](https://arxiv.org/abs/2211.12737) | 同上，資料量大很多，只能當上限參考 | ✓ |
| Ruiz et al., *DreamBooth*, 2022 | [arXiv:2208.12242](https://arxiv.org/abs/2208.12242)，CVPR 2023 | 3–5 張的主體客製化；目標不同（見 §5） | ✓ |
| Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021 | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) | LoRA 原始論文 | ✓（ICLR 2022 未查證） |
| Valevski et al., *Diffusion Models Are Real-Time Game Engines*（GameNGen）, 2024 | [arXiv:2408.14837](https://arxiv.org/abs/2408.14837)，ICLR 2025 | SD 1.4 全微調；動作 token 序列取代文字 cross-attn；過去幀 latent channel concat（試過 cross-attn 沒有改善）；context 噪聲增強最大 0.7；微調 decoder | ✓ |
| Scotti et al., *MindEye2: Shared-Subject Models Enable fMRI-To-Image With 1 Hour of Data*, 2024 | [arXiv:2403.11207](https://arxiv.org/abs/2403.11207)，ICML 2024 | 非影像訊號經共享預訓練後條件化 SDXL，新受試者只需 1 小時資料；對應「124 根柱預訓練條件路徑」 | ✓ |
| Bai et al., *DreamDiffusion: Generating High-Quality Images from Brain EEG Signals*, 2023 | [arXiv:2306.16934](https://arxiv.org/abs/2306.16934) | EEG 時序編碼器的 token 經投影進 SD cross-attn，只訓練編碼器與 cross-attn | ✓ |
| Li, Katabi, He, *Return of Unconditional Generation: A Self-supervised Representation Generation Method*（RCG）, 2023 | [arXiv:2312.03701](https://arxiv.org/abs/2312.03701)，NeurIPS 2024 | 在 ADM/LDM/DiT 中直接用連續表徵向量取代 class embedding | ✓ |
| Li et al., *GLIGEN: Open-Set Grounded Text-to-Image Generation*, 2023 | [arXiv:2301.07093](https://arxiv.org/abs/2301.07093) | 凍結預訓練模型，用新增的 gated 層注入新的條件 token | ✓（CVPR 2023 和 tanh(γ) 閘門細節未查證） |
| Ye et al., *IP-Adapter*, 2023 | [arXiv:2308.06721](https://arxiv.org/abs/2308.06721) | 解耦的 cross-attn，22M 參數就能加入非文字條件 | ✓ |
| Tan et al., *OminiControl: Minimal and Universal Control for Diffusion Transformer*, 2024 | [arXiv:2411.15098](https://arxiv.org/abs/2411.15098)，ICCV 2025 | 條件影像 token 和影像 token 串成同一序列，只多 0.1% 參數；可作為上一幀 in-context 的替代做法 | ✓ |
| Black Forest Labs, *FLUX.1 Kontext*, 2025 | [arXiv:2506.15742](https://arxiv.org/abs/2506.15742) | 序列串接式的 in-context 條件與多輪一致性；FLUX 類模型的代表 | ✓ |
| HF Blog, *(LoRA) Fine-Tuning FLUX.1-dev on Consumer Hardware*, 2025-06-19 | [huggingface.co/blog/flux-qlora](https://huggingface.co/blog/flux-qlora) | QLoRA 在 4090 上峰值約 9 GB（512×768、batch 1、rank 4），700 step 約 41 分鐘 | ✓ |
| PixArt-α 256 權重 | [HF: PixArt-XL-2-256x256](https://huggingface.co/PixArt-alpha/PixArt-XL-2-256x256) | 確認有 0.6B 的 256 版本（openrail++） | ✓ |
| Chen et al., *PixArt-α*, 2023 | [arXiv:2310.00426](https://arxiv.org/abs/2310.00426) | DiT 加文字 cross-attn 的 T2I；訓練成本約為 SD1.5 的 10.8% | ✓（ICLR 2024 未查證） |
| Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*（SD3, MMDiT）, 2024 | [arXiv:2403.03206](https://arxiv.org/abs/2403.03206) | MMDiT 的文字和影像各自一套權重；rectified flow | ✓（ICML 2024 未查證） |
| Xie et al., *SANA*, 2024 | [arXiv:2410.10629](https://arxiv.org/abs/2410.10629) | 0.6B 線性 DiT 加 32× DC-AE，16 GB 筆電 GPU 可部署 | ✓ |
| Ding et al., *CCDM: Continuous Conditional Diffusion Models for Image Generation*, 2024 | [arXiv:2405.03546](https://arxiv.org/abs/2405.03546) | 連續標量條件的專用 label embedding，64–192 px；屬方向 E，這裡只列注入方式 | ✓ |

### B：latent、Transformer 骨幹、序列條件

| 論文 | 出處 | 與本問題的關聯 | 狀態 |
|---|---|---|---|
| Peebles & Xie, *Scalable Diffusion Models with Transformers*（DiT）, 2022 | [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)，[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf) | 條件注入的消融：adaLN-Zero 19.47 < adaLN 25.21 < cross-attn 26.14 < in-context 35.24；XL/2 為 675M、118.6 GFLOPs | ✓ |
| Ma et al., *SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers*, 2024 | [arXiv:2401.08740](https://arxiv.org/abs/2401.08740)，ECCV 2024 | Linear interpolant + velocity，和現行 CFM 目標相同；SD-VAE 32×32×4；adaLN-Zero | ✓ |
| Bao et al., *All are Worth Words: A ViT Backbone for Diffusion Models*（U-ViT）, 2022 | [arXiv:2209.12152](https://arxiv.org/abs/2209.12152)，CVPR 2023 | 時間、條件、影像全部當 token；long skip 是關鍵；和同尺寸 U-Net 相當 | ✓ |
| Li et al., *On the Scalability of Diffusion-based Text-to-Image Generation*, 2024 | [arXiv:2404.02883](https://arxiv.org/abs/2404.02883)，CVPR 2024 | 明確寫出 transformer 骨幹「缺少 inductive bias，從零訓練比 U-Net 難」，PixArt-α 變體要更多 iteration 才追上 U-Net。這是「小資料下 transformer 較差」最直接的文字證據（不過是大資料設定） | ✓ |
| Zheng et al., *Fast Training of Diffusion Models with Masked Transformers*（MaskDiT）, 2023 | [arXiv:2306.09305](https://arxiv.org/abs/2306.09305) | 遮 50% patch，訓練時間約為 DiT 的 30%；abs 頁沒有少量資料的主張 | ✓ |
| Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, 2021 | [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)，CVPR 2022 | SD-VAE 的來源 | ✓ |
| Chen et al., *Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models*（DC-AE）, 2024 | [arXiv:2410.10733](https://arxiv.org/abs/2410.10733)，ICLR 2025 | 32–128× 壓縮；對 128 px 太粗（見 §5） | ✓ |
| Karras et al., *Analyzing and Improving the Training Dynamics of Diffusion Models*（EDM2）, 2023 | [arXiv:2312.02696](https://arxiv.org/abs/2312.02696) | 小而強的 ImageNet-512 latent class-conditional U-Net，可作為 A 的替代；post-hoc EMA | ✓（CVPR 2024 和各尺寸參數量未查證） |
| Yao et al., *Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models*（VA-VAE/LightningDiT）, 2025 | [arXiv:2501.01423](https://arxiv.org/abs/2501.01423) | latent 對齊 VFM，收斂快 21 倍以上；見 §5 的指標耦合風險 | ✓（CVPR 2025 未查證） |
| Zheng et al., *Diffusion Transformers with Representation Autoencoders*（RAE）, 2025 | [arXiv:2510.11690](https://arxiv.org/abs/2510.11690) | 以 DINO 等表徵編碼器當 latent；同樣有指標耦合風險 | ✓ |
| Avrahami et al., *Blended Latent Diffusion*, 2022 | [arXiv:2206.02779](https://arxiv.org/abs/2206.02779)，SIGGRAPH 2023 | 在 latent 空間做遮罩混合，支持 A2 把 α 鎖定移到 latent | ✓ |
| Alonso et al., *Diffusion for World Modeling: Visual Details Matter in Atari*（DIAMOND）, 2024 | [arXiv:2405.12399](https://arxiv.org/abs/2405.12399)，NeurIPS 2024 | 過去 4 幀 channel concat、動作走 AdaGN（和現行專案相同）；EDM 式目標在少步時比 DDPM 穩 | ✓ |
| Bastek & Kochmann, *Inverse-design of nonlinear mechanical metamaterials via video denoising diffusion models*, 2023 | [arXiv:2305.19836](https://arxiv.org/abs/2305.19836) | 應力—應變曲線 11 個點各自經 Linear 變成 token，cross-attn 到對應應變步，平均後再加到時間嵌入；最接近 B2 的做法（資料 53k 組） | ✓（期刊名未查證） |
| Yoon et al., *SADM: Sequence-Aware Diffusion Model for Longitudinal Medical Image Generation*, 2022 | [arXiv:2212.08228](https://arxiv.org/abs/2212.08228)，IPMI 2023 | 序列 transformer 當條件模組，可處理缺失幀，自回歸生成 | ✓ |
| Liu et al., *ImageFlowNet*, 2024 | [arXiv:2406.14794](https://arxiv.org/abs/2406.14794)，ICASSP 2025 | 不規則取樣的縱向影像，在多尺度表徵中用 ODE/SDE 連接；跨病人共享以緩解資料少 | ✓ |
| Puglisi et al., *Enhancing Spatiotemporal Disease Progression Models via Latent Diffusion and Prior Knowledge*（BrLP）, 2024 | [arXiv:2405.03328](https://arxiv.org/abs/2405.03328) | 疾病進程的 latent diffusion，加上輔助模型預測體積變化 | ✓（MICCAI 2024 未查證；頁面只列 Springer DOI） |

### O1 的最接近工作

| 論文 | 出處 | 關聯 | 狀態 |
|---|---|---|---|
| Chen et al., *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion*, 2024 | [arXiv:2407.01392](https://arxiv.org/abs/2407.01392) | 每個 token 各自的噪聲等級，同時涵蓋單步預測與整段生成；O1 (a) 的來源 | ✓（NeurIPS 2024 未查證） |
| Song et al., *History-Guided Video Diffusion*（DFoT）, 2025 | [arXiv:2502.06764](https://arxiv.org/abs/2502.06764)，ICML 2025 | 可變長度歷史；history guidance | ✓ |
| Zhu et al., *Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets*, 2025 | [arXiv:2504.02792](https://arxiv.org/abs/2504.02792)，RSS 2025 | 模態專屬的 diffusion timestep，可用無動作的影片；O1 (c) 的反方向對照 | ✓ |
| Huang et al., *Multimodal Diffusion Forcing for Forceful Manipulation*, 2025 | [arXiv:2511.04812](https://arxiv.org/abs/2511.04812) | 力訊號加影像的多模態軌跡、隨機部分遮罩 | ✓ |
| Li et al., *Unified Video Action Model*（UVA）, 2025 | [arXiv:2503.00200](https://arxiv.org/abs/2503.00200) | 聯合影片與動作 latent，用遮罩切換任務 | ✓ |
| Shukla & Marlin, *Multi-Time Attention Networks for Irregularly Sampled Time Series*（mTAN）, 2021 | [arXiv:2101.10318](https://arxiv.org/abs/2101.10318)，ICLR 2021 | 連續時間 embedding 加 attention；O1 (b) 最接近的機制（但用的是時間，不是物理進程量） | ✓ |
| Tallec & Ollivier, *Can recurrent neural networks warp time?*, 2018 | [arXiv:1804.11188](https://arxiv.org/abs/1804.11188) | gated RNN 對時間扭曲近似不變；O1 (b) 的理論動機 | ✓（ICLR 2018 未查證） |
| Voleti et al., *MCVD: Masked Conditional Video Diffusion*, 2022 | 未查證 | 用遮罩把預測、生成、內插統一成同一個模型 | 未查證 |

---

## 3. 推薦排序（同時考慮可行性、原創性、通用性）

| 名次 | 方案 | 可行性 | 原創性 | 通用性 | 一句話理由 |
|---|---|---|---|---|---|
| 1 | **O1 PC-DF**（第一階段，建在 ADM-32 上；B2 是它的消融組） | 高（現有 CrossAttention2d、A_info 序列、上一幀通道都能沿用） | 中（物理時鐘與反向模態共訓練沒有找到前例） | 高 | 唯一同時針對問題 2、4 和單幀／時序統一的方案，又能寫成通用方法 |
| 2 | **A1 PT-ADM128** | 最高（同家族架構、像素空間、同解析度） | 無 | 高 | 最直接攻擊 oracle 實驗指出的瓶頸（生成器本身的真實感），而且可以和 O1 疊加：O1 的骨幹換成 PT-ADM128 |
| 3 | **A2 PT-SiT-latent** | 中（要先過 VAE 檢查、程式整合量較大） | 低 | 中 | 目標函數和現行相同，有全域 attention；只在 A1 失敗或 VAE 檢查通過時再做 |

**有論文實證的部分**

- 少量資料時，預訓練微調優於從零訓練（DDPM-PA、DomainStudio，10-shot 等級；DiffFit 是數千張等級）。
- 灰階醫學影像可以跨 domain 微調 SD（Chambon、RoentGen）。
- 向量條件走 adaLN 優於 cross-attn 和 in-context（DiT 消融）。
- 曲線或序列可以逐點 token 化後用 cross-attn 條件化生成（Bastek、SADM、DreamDiffusion）。
- 非影像訊號經過共享預訓練後，少量配對資料就能條件化大型生成模型（MindEye2）。
- transformer 從零訓練比 U-Net 難（Li et al. CVPR 2024）。
- 上一幀用 channel concat 加噪聲增強，可以穩定自回歸（GameNGen、DIAMOND）。

**我的推論（沒有直接實證）**

- 530 張、單一場景灰階的情況下，ImageNet 先驗能讓 KID 明顯下降。沒有任何論文在這個規模和 domain 上做過。
- 物理時鐘位置編碼能改善 CTR1 外插。
- 序列條件能提高演進 ratio 和鄰居對比，而不是讓模型背柱子。
- 單幀和時序共用權重不會互相拖累。

**排序的取捨**

如果只看「最可能讓第 4 節的指標變好」，A1 應該排第一，因為它直接針對最大的落差（KID 0.194 對 0.116／0.072）。依新需求把原創性和通用性納入後，O1 排第一，但它的實證最弱。建議**兩者並行**：O1 和 A1 的最小實驗互不依賴，兩張卡各跑 2 天，結果可以直接疊加。

---

## 4. 最小驗證實驗（每個 1–2 天）

所有實驗共同固定的條件：

- `id_list_val_R307.json` 分割；v11 條件（z、4 個損傷標量、柱面、正負號、flip 保留柱面）。
- Patch Diffusion 設定、Heun 30 步、K=8、test 6×6 格子順序。
- 評估一律用專案的 block-jackknife z。
- 每組同時報：KID global/patch（CTR1、R307）、演進 ratio/corr、鄰居對比、CTR1 高 DI 剝落面積偏差、同條件樣本標準差（問題 5）、DINOv3 最近鄰到訓練圖的距離（背圖檢查）。

### 4.1 A1：PT-ADM128（2 天）

**組別**

- (a) 現行 v11（已有）。
- (b) ADM-128 架構**隨機初始化**，條件注入、步數和 (c) 相同。
- (c) ADM-128 **ImageNet 預訓練**，物理取代 label_emb（錨點初始化），全微調，lr 2e-5，約 3–5k step、batch 32。
- (d)（有時間再做）DiffFit 式部分微調：bias、norm、條件 MLP、in/out conv。

**固定**：同樣的 epoch 數、同樣的評估。

**怎麼判斷**

- (c) 對 (a)：看 KID 是否下降。
- (c) 對 (b)：區分「先驗」和「容量」，這是最關鍵的對照。
- (c) 對 (d)：看背圖程度。

**值得繼續的條件**

- (c) 的 CTR1 KID ≤ 0.16（現行 0.194），z ≥ 2，而且顯著低於 (b)。
- 演進 ratio 和鄰居對比沒有比 (a) 低超過 1 個 SE。
- 最近鄰距離沒有塌到訓練圖。

如果 (c) 的 KID 好但 ratio 掉，就改用 (d) 或 T-LoRA 式的「高噪聲步低 rank」，再試一次。

### 4.2 A2：PT-SiT-latent（2 天）

**第 1 天上午：VAE 檢查**

- 把 test/val 的影像從 128 上採樣到 256，用 SD-VAE-ft-EMA 編碼再解碼。可另外比較一個 16 通道 VAE，但那會讓 SiT 的 patch embed 失去預訓練。
- 用相同的 DINOv3 流程算 KID(real, recon)，並算剝落面積誤差和裂縫 F1。
- 通過條件：KID_recon ≤ 0.02（遠低於真實柱對真實柱的 0.072）、剝落面積誤差 ≤ 0.01、裂縫 F1 下降 ≤ 10%。沒有通過就停止 A2。

**第 1–2 天：訓練與比較**

- (a) v11。
- (b) SiT-B/2 在同一個 latent 上**從零訓練**。這組直接回答 B 的「小資料下 transformer 是否更差」。
- (c) SiT-XL/2 預訓練，physics adaLN，全微調。
- 這一輪先不加歷史 cross-attn，以免混淆。

**值得繼續的條件**：(c) 的 KID 至少和 A1 (c) 一樣好，而且 (b) 明顯差於 (a)（證實增益來自先驗，不是 latent 本身）。之後才加歷史 cross-attn 和上一幀 concat。

### 4.3 O1：PC-DF 第一階段（2 天，兩張卡或縮短步數）

**組別**（每組都和現行 v11 同步數；必要時全部統一縮短到 8k step）

- R0：v11 向量條件（已有）。
- R1：加上因果歷史 cross-attn，位置是 **index**（就是 B2）。
- R2：和 R1 只差一點，位置改成**物理時鐘**（log 累積能量、最大 drift）。
- R3：R2 加上 σ_p 上一幀槽位（統一模型）。單幀格子用 σ_p = 1 評估；時序閉環用 σ_p = 0.05，和現行 refiner 比較。

**對照檢查**

- 歷史截斷：推論時只保留最後一個 token。如果輸出幾乎不變，代表 cross-attn 沒有被用到。
- CTR1 時鐘替換：拿 CTR1 的 token 分別配 index 位置和時鐘位置，看剝落面積偏差的差異。
- attention 熵：檢查有沒有退化成只看最後一個 token。

**值得繼續的條件**

- R1 對 R0：演進 ratio +0.05，或鄰居對比 +0.05，z ≥ 2；KID 沒有變差超過 1 個 SE。
- R2 對 R1：CTR1 高 DI 剝落面積偏差的絕對值減少 ≥ 0.05（例如 −0.25 → −0.20），R307 沒有變差。**這是原創點能不能成立的關鍵判準。**
- R3：單幀模式的 KID、ratio、鄰居對比都在 R2 的 1 個 SE 內；時序模式的 KID 不高於現行 refiner，flicker_ratio 比 0.41 更接近 1。
- 只有 R2 有增益時，才進入第二階段 (c) 反應共訓練。第二階段額外做 leave-one-specimen-out 的探針，確認 h 沒有變成柱子 ID。

---

## 5. 不建議的方向

| 方向 | 理由 |
|---|---|
| FLUX.1 類 12B 模型（QLoRA） | 24 GB 雖然放得下（HF blog：峰值約 9 GB，但 batch 1，700 step 約 41 分鐘，A5000 更慢），一天只能看約數萬個樣本，很難做多組對照。T5 文字流要整個換掉，大部分先驗也綁在文字對齊和 1024 px 上。對 128 px 灰階來說，代價和收益不成比例。FLUX.1-dev 授權是非商用（依記憶）。 |
| SD3/3.5（MMDiT）全微調 | 2–2.5B 參數全微調需要約 32–40 GB 的優化器狀態，超出 24 GB。MMDiT 為條件流另外配一整套權重，對約 260 維的物理向量和不到 40 個 token 的序列來說過重。只做 LoRA 時，又失去「骨幹適應新 domain」的主要好處。 |
| T2I 骨幹（SD1.5/2.1、PixArt-α-256）把文字換成物理 token（GameNGen 配方） | 技術上可行（SD U-Net 860M 全微調約 14 GB 狀態，加 8-bit Adam 和 checkpointing 在 256 px 可以放進 24 GB；PixArt-α 有 0.6B、256 px 的權重 ✓）。但 GameNGen 用的是 7000 萬筆資料。文字先驗無法表達連續物理量；cross-attn 本來是為語意對齊設計的。和 A1/A2 比，這條路多了 VAE 風險和 512 px 原生解析度的落差，好處卻沒有更多，所以放到第二輪。 |
| SD1.5 直接在 128 px 上用（latent 16×16）；DC-AE f32/f64 在 128 px（latent 4×4） | 離預訓練解析度太遠，裂縫細節會被壓掉。若要用就得上採樣到 512 以上，算力變 16 倍，資訊卻沒有增加。 |
| 自訓 VAE + 從零訓 DiT/U-ViT/MMDiT | 530 張連 VAE 本身都會過擬合。transformer 缺少 locality 的 inductive bias，Li et al. CVPR 2024 明確指出從零訓練比 U-Net 難，而且找不到任何支持「小資料下 latent transformer 從零訓練較好」的論文。A2 的 (b) 組會順便量化這一點。 |
| DiT 的 in-context 條件 token（向量條件） | DiT 自己的消融顯示這是最差的注入方式（FID 35.24）。 |
| VA-VAE、RAE 等 DINO 對齊的 latent | 評估指標本身就是 DINOv3 特徵，會出現「優化指標而不是真實感」的耦合。這類 latent 也需要 ImageNet 規模的資料來訓練。如果真要用，評估要換成非 DINO 的 backbone（專案已有 DINOv2 檢查，但仍屬同一家族）。 |
| DreamBooth 式主體客製化與 prior preservation | 它的目標是「一個主體的身分」，不是「隨條件變化的分布」。用通用混凝土的 class image 做 prior preservation，可能會把損傷沖淡。 |
