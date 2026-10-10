# Method: ADM U-Net for Conditional Flow Matching

本節說明損害圖樣生成所採用之 **ADMUNetCFM**（Adaptive Diffusion Model U-Net for Conditional Flow Matching）架構。該網路以 U-Net 為骨幹，預測 conditional flow matching（CFM）之速度場；物理與結構條件經 **AdaGN** 注入各殘差區塊，瓶頸處辅以 **自注意力** 與 **Vision State Space（VSS）** 模組以強化長程空間依賴。

---

## 1. 問題設定與網路角色

CFM 將生成建模為從噪聲分布流向資料分布的常微分方程（ODE）。訓練時，給定 flow time \(t \in [0,1]\)、條件向量 \(\mathbf{c}\) 與當前狀態 \(\mathbf{z}_t\)，U-Net 預測速度 \(\mathbf{v}_\theta(\mathbf{z}_t, t, \mathbf{c})\)，使

\[
\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{t,\mathbf{z}_0,\mathbf{z}_1}\left[\left\|\mathbf{v}_\theta(\mathbf{z}_t, t, \mathbf{c}) - (\mathbf{z}_1 - \mathbf{z}_0)\right\|^2\right],
\]

其中 \(\mathbf{z}_t = (1-t)\mathbf{z}_0 + t\mathbf{z}_1\)。推論時以 ODE 積分由 \(\mathbf{z}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})\) 得到生成樣本。

ADMUNetCFM 採 **encoder–bottleneck–decoder** 對稱結構，並以 skip connection 保留多尺度特徵。

| 符號 | 語意 |
|------|------|
| \(\mathbf{z}_t\) | Flow 狀態（像素空間影像通道，可含座標與 mask 等附加通道） |
| \(t\) | Flow 時間標量 |
| \(\mathbf{c}\) | 純量條件（如 hidden state、載向、DI 之拼接） |
| \(\mathbf{v}_\theta\) | 網路輸出之速度場 |

---

## 2. 條件注入：時間嵌入與聯合調制向量

條件分兩路進入網路：

1. **時間路徑**：\(t\) 經 sinusoidal positional embedding 與 MLP 映射為 \(\mathbf{e}_t\)。
2. **語意路徑**：\(\mathbf{c}\) 經 MLP 映射為 \(\mathbf{e}_c\)。

二者相加後經聯合 MLP，得到所有 AdaGN 區塊共用的調制向量：

\[
\mathbf{h}_{\mathrm{cond}} = \mathrm{MLP}\!\left(\mathbf{e}_t + \mathbf{e}_c\right).
\]

| 條件分量 | 典型來源 | 注入方式 |
|----------|----------|----------|
| Flow time \(t\) | 均勻抽樣 \(t \sim \mathcal{U}(0,1)\) | 時間嵌入 → \(\mathbf{e}_t\) |
| Hidden state | 遲滯迴圈 extractor | 拼接於 \(\mathbf{c}\) |
| Direction | Pull / Push 載向 | 拼接於 \(\mathbf{c}\) |
| DI | 標籤或指定值 | 拼接於 \(\mathbf{c}\) |
| Object mask（可選） | 影像 mask 檔 | 額外 spatial channel concat 於輸入 |

**設計原則**：純量條件僅經 AdaGN 調制特徵；空間 mask 則在輸入層與 \(\mathbf{z}_t\) 通道拼接，使網路能同時利用全局語意與局部幾何先驗。

---

## 3. AdaGN 殘差區塊（ResBlock）

骨幹由 **AdaGN ResBlock** 堆疊而成，結構類似擴散模型中常見之兩層卷積殘差單元，但以 **Adaptive Group Normalization（AdaGN）** 取代固定仿射參數，使 \(\mathbf{h}_{\mathrm{cond}}\) 能逐通道調制特徵。

單一 ResBlock 可寫為：

\[
\mathbf{h}' = \mathcal{F}(\mathbf{x}; \mathbf{h}_{\mathrm{cond}}) + \mathcal{S}(\mathbf{x}),
\]

其中 \(\mathcal{S}\) 為 shortcut（通道數改變時以 \(1\times1\) 卷積對齊），\(\mathcal{F}\) 為：

\[
\begin{aligned}
\hat{\mathbf{h}}_1 &= \mathrm{GN}(\mathbf{x}), \\
\mathbf{u}_1 &= \mathrm{SiLU}\!\left(\hat{\mathbf{h}}_1 \odot (\boldsymbol{\gamma}_1 + \mathbf{1}) + \boldsymbol{\beta}_1\right), \\
\mathbf{u}_2 &= \mathrm{Conv}(\mathbf{u}_1), \\
\hat{\mathbf{h}}_2 &= \mathrm{GN}(\mathbf{u}_2), \\
\mathbf{u}_3 &= \mathrm{SiLU}\!\left(\hat{\mathbf{h}}_2 \odot (\boldsymbol{\gamma}_2 + \mathbf{1}) + \boldsymbol{\beta}_2\right), \\
\mathcal{F}(\mathbf{x}) &= \mathrm{Conv}(\mathrm{Dropout}(\mathbf{u}_3)).
\end{aligned}
\]

此處 \(\mathrm{GN}(\cdot)\) 為 Group Normalization；\(\boldsymbol{\gamma}_i, \boldsymbol{\beta}_i\) 由線性層自 \(\mathbf{h}_{\mathrm{cond}}\) 預測，並 broadcast 至空間維度。

### AdaGN 之意義

| 項目 | 說明 |
|------|------|
| GroupNorm | 消除通道絕對尺度，保留空間 pattern |
| \(\boldsymbol{\gamma}, \boldsymbol{\beta}\) | 由條件決定之逐通道 scale / shift |
| \((\boldsymbol{\gamma}+\mathbf{1})\) | 使初始化時接近 identity，利於 CFM 穩定訓練 |
| 零初始化 | 調制層權重與偏置初始為 0，訓練初期近似無條件 ResBlock |

AdaGN 可視為 **FiLM（Feature-wise Linear Modulation）** 在 GroupNorm 特徵上之實例：條件為全局向量，調制為 channel-wise、spatially shared，適合編碼「在給定 DI 與時間 \(t\) 下應生成何種 damage 型態」這類全局語意。

---

## 4. U-Net 拓撲與多尺度設計

編碼端逐級下採樣並展寬通道；解碼端上採樣並與同尺度 skip 拼接。每個尺度至少一個 AdaGN ResBlock。

| 階段 | 功能 |
|------|------|
| Encoder | 局部紋理與邊緣提取；逐級降低解析度 |
| Bottleneck | 最低解析度之語意整合 |
| Decoder | 融合 skip 之高頻細節並恢復空間解析度 |

Downsample 以 stride-2 卷積實現；Upsample 以 nearest-neighbor 插值後接卷積平滑。Skip connection 採 **concatenation**，解碼端 ResBlock 輸入為當前特徵與 skip 之通道拼接。

---

## 5. 瓶頸自注意力（Mid Self-Attention）

於最低解析度特徵圖上，兩個 AdaGN ResBlock 之間插入 **multi-head self-attention**，以 residual 形式更新：

\[
\mathbf{x}' = \mathbf{x} + \mathrm{Proj}\!\left(\mathrm{Softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}}\right)\mathbf{V}\right),
\]

其中 \(\mathbf{Q}, \mathbf{k}, \mathbf{V}\) 由 \(1\times1\) 卷積自 normalized 特徵產生，並將空間維展平為 token 序列。

| 特性 | 說明 |
|------|------|
| 作用範圍 | 單一特徵圖內任意兩位置直接交互 |
| 互補性 | 彌補卷積感受野局部性 |
| 放置位置 | 僅 bottleneck，控制計算成本 |
| 正規化 | Pre-GroupNorm + residual |

對 damage pattern 生成，attention 有助整合相距較遠之 damage 區與背景（如混凝土區域）的全局一致性。

---

## 6. Vision State Space 區塊（VSS）

受 Vision Mamba / VMamba 啟發，於 **encoder 最深層（L3）** 與 **decoder 對應淺層（U3）** 各插入一 **VSS block**，在與 bottleneck 相近之解析度進行長程空間混合。實作為 pure PyTorch 之 SS2D 風格模組（非官方 CUDA selective scan），以兼顧可移植性與可變輸入尺寸。

### 6.1 四向 Cross-Scan

將二維特徵 \(\mathbf{X} \in \mathbb{R}^{H \times W \times C}\) 展開為四條一維序列：

| 方向 | 掃描順序 |
|------|----------|
| Row | 列優先（左→右、上→下） |
| Column | 行優先（上→下、左→右） |
| Row reverse | 列優先反向 |
| Column reverse | 行優先反向 |

四序列經 **selective scan**（簡化為 gated cumulative aggregation）後，再 merge 回二維特徵（四向平均）。

### 6.2 VSS Block 結構

\[
\begin{aligned}
\mathbf{X}_1 &= \mathbf{X} + \mathrm{SS2D}(\mathrm{LN}(\mathbf{X})), \\
\mathbf{X}_2 &= \mathbf{X}_1 + \mathrm{MLP}(\mathrm{LN}(\mathbf{X}_1)).
\end{aligned}
\]

U-Net 特徵為 \((B,C,H,W)\) 格式，VSS 內部以 \((B,H,W,C)\) 運算，前後以維度置换適配。

| 模組 | 複雜度（序列長 \(N=HW\)） | 全局交互方式 |
|------|---------------------------|--------------|
| Self-Attention | \(O(N^2)\) | 任意 token  pairwise |
| VSS / SS2D | \(O(N)\) per direction | 沿四向序列之狀態傳遞 |

**互補角色**：Mid Attention 提供 bottleneck 內 dense 全局重配；VSS 在 encoder/decoder 關鍵尺度以線性成本強化 directional long-range dependency，且 **不** 直接依賴 scalar condition。

---

## 7. 輸入、輸出與 Patch Diffusion（概要）

| 通道類型 | 訓練 / 推論角色 |
|----------|-----------------|
| 影像 RGB | Flow 狀態主通道 |
| 絕對座標 \((x,y)\) | Patch diffusion 下標定 crop 於全圖之位置 |
| Object mask | 空間條件（train 用真實 mask；val/inference 可為全 1） |

Patch diffusion 訓練時對已增強之全圖隨機 crop，影像、mask 與座標 grid 共用同一 crop window；損失僅計算影像速度通道，座標通道不參與 CFM 目標。

Mask 新增通道採 **zero initialization**：擴展輸入卷積時，新 channel 權重初始為 0，使載入舊 checkpoint 時行為不變。

---

## 8. 模組分工總表

| 元件 | 主要功能 | 條件依賴 |
|------|----------|----------|
| AdaGN ResBlock | 多尺度卷積特徵提取 | \(\mathbf{h}_{\mathrm{cond}}\)（\(t\) + \(\mathbf{c}\)） |
| Mid Self-Attention | Bottleneck 全局 spatial 重配 | 無 |
| VSS（L3 / U3） | 四向長程空間混合 | 無 |
| 輸入 mask concat | 局部幾何 / 桿件區域先驗 | 空間 mask（非 AdaGN） |
| Skip connections | 高頻細節還原 | 無 |

---

## 9. 方法小結

ADMUNetCFM 將 **conditional flow matching** 與 **ADM 風格 U-Net** 結合：以 AdaGN 將 flow time 與結構條件（hidden state、載向、DI）注入殘差路徑；以 bottleneck attention 與 L3/U3 VSS 在不同計算成本下增強全局與長程空間建模；並可選地以 mask 通道提供空間先驗。整體設計在生成品質、條件可控性與訓練穩定性之間取得平衡，適用於結構 damage pattern 之條件生成任務。
