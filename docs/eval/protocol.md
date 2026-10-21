# 評估設定（所有指標共用）

## 資料切分

`Data/DamagePattern/id_list_val_R307.json`：

| split | 柱 | 張數 | 用途 |
|---|---|---|---|
| train | R615、R1015、R315、R330 | 533 | 訓練；也拿來擬合 probe、校準像素門檻、做背圖基準 |
| val | R307 | 128 | 留出柱，用來確認結論能不能重現 |
| test | CTR1 | 124 | 主要的留出柱 |

檔名格式 `CTR1_NE_06000_3.jpg`：柱、面（NE = Push、SW = Pull）、drift 代碼（06000 = 6%）、循環編號。

注意：cGAN（`cGANckpt/`）用的是舊切分 `id_list_old.json`，它的訓練集包含 R307。所以 cGAN 只能在 test 上比，不能在 val 上比。

## 參考集：用補圖版

建議的參考集是 `Data/DamagePattern_inpaint_selfcondnoDI/images`，也就是用 CFM 把 SAM 遮罩出的實驗器材補成混凝土之後的 GT。

- **為什麼**：CTR1 下緣有一根黑色立柱，生成器不會畫它。它又大又顯眼，DINOv3 一定會看。用原圖當參考集時，「換一根真實柱」的 KID 是 0.216；用補圖時只有 0.072。也就是說，柱與柱之間的差距大半是這根立柱造成的，而不是損傷內容。
- 補圖已經是 128×128 並做過 CLAHE，所以評估時要加 `--no_clahe`，不能再套一次。

## 生成圖也要處理干擾物

如果參考集去掉了干擾物，生成圖也要去掉，兩邊才公平。步驟見 [interference.md](interference.md)。

建議加 `--add_gt_mask`（安慰劑對照）：每張生成圖都在 GT 的干擾物位置再補一次，讓每個模型都帶著和參考集相同的補圖痕跡。實測這個對照幾乎不改變數字，但加上之後可以直接排除「補圖痕跡造成差異」這個解釋。

## 前處理

真實圖和生成圖走完全相同的一條路（Clean-FID 的原則，見 Parmar 2022）：

1. 128 灰階 CLAHE，轉成 uint8。生成圖是 `x*0.5+0.5` 再四捨五入。
2. 用 antialias bicubic 放大到 224。
3. 複製成 3 通道，做 ImageNet 正規化。

`eval_dinov3_kid.py` 會把 npz 裡的真實圖和「直接讀檔再做同樣前處理」的結果比對，印出 `npz-vs-file mean |Δ|`。這個值應該是 0。

外部模型（例如 cGAN 的存檔圖）：

- 已經是 CLAHE 後的亮度範圍，就不要再套 CLAHE。
- 彩色 jpg 轉灰階。

## 樣本檔（npz）格式

`generate_eval_samples.py` 和 `save_samples_npz` 都輸出同一種格式：

| 欄位 | 形狀 | 內容 |
|---|---|---|
| `fake` | (N, K, 128, 128) uint8 | 每個條件 K 張生成圖 |
| `real` | (N, 128, 128) uint8 | 該條件的 GT |
| `cond`、`DI`、`filename` | (N, …) | 條件、DI、GT 檔名 |
| `meta` | JSON 字串 | checkpoint、split、是否去干擾物等 |
| `interference_mask`、`interference_flag` | 選用 | 由 `inpaint_generated_samples.py` 寫入 |

- 列的順序是 `Data/DamagePattern/label.csv` 裡該 split 的順序，不是 `id_list` 的順序。要對齊 GT 時，**一律用 npz 自己的 `filename` 欄位**。
- 同一次評估的所有 npz，`filename` 必須完全一致。
- K 建議 8。K 小時 SE 會變大，但不偏估計量不會因此偏掉。

## 不確定度

### 樣本的分組

每張圖都帶一個 group，也就是它來自第幾個條件。真實圖 i 和從條件 i 生成的 K 張，共用 group i。同一 group 的樣本彼此相依，所以所有 MMD 類估計量都排除同 group 的配對（block U-statistic）。細節見 [distribution.md](distribution.md#估計量)。

### 區塊 jackknife

相鄰幀幾乎是同一張圖。如果把每個條件當成一個獨立樣本，SE 會被嚴重低估：實測低估約 2–2.5 倍，z 值高估約 2 倍。

- **KID、鄰居對比、像素統計**：leave-one-block-out。一個區塊 = 同一面、同一 drift 級（`dm.frame_block`，例如 `CTR1/NE/06000`）。test 有 22 個區塊，val 有 26 個。
- **演進指標**：以循環編號為單位（leave-one-cycle-out）。test 只有 7 個單位，所以 z 要用 t 分佈（自由度 6）來讀：z = 2 約對應 p ≈ 0.09，z = 3 約對應 p ≈ 0.02。
- **不用 bootstrap**：有放回抽樣會產生重複樣本，讓 U-statistic 偏掉。

### 兩個模型怎麼比

兩個模型共用同一組真實圖，所以直接對差值 D = score(A) − score(B) 做配對 jackknife，報 SE 和 z（Bounliphone 2016 的相對相似度檢定）。配對後差值的 SE 大約只有單一模型 SE 的一半，比「各自畫誤差棒、看有沒有重疊」靈敏得多。輸出在 `dinov3_kid.json` 的 `comparisons`。

## 打亂檢查：條件指標的必要條件

一個號稱「看條件」的指標，必須通過這個檢查：**把生成圖隨機換到別的條件，分數要掉到約 0。** 掉不下來，就表示它其實沒有在看條件。

| 指標 | 打亂後 | 結論 |
|---|---|---|
| `kid_global` | 不變 | 不看條件（本來就是分佈指標） |
| `kid_soft_di` | 反而變好 | **沒通過**，不使用 |
| 聯合核 MMD | 方向相反 | **沒通過**，已移除 |
| `nbr_global`、`nbr_patch` | ≈ 0 | 通過 |
| `progression_*` | ratio ≈ 0.05、corr ≈ 0 | 通過 |

`nbr_*` 和 `progression_*` 每次執行都會自動做這個檢查，輸出 `shuffled_*`。

## 參考刻度

每個指標都要有一個「好到什麼程度算合理」的刻度：

| 參考列 | 意義 |
|---|---|
| `real_halves_kid` | test 真實圖隨機切兩半互比，是雜訊底線（≈ 0） |
| `other_real_val_kid` | 另一根真實柱（R307）對 test 柱，代表「換一根真實柱」的距離 |
| `*_ceiling_other_specimens`、`progression_*_other_specimens` | 其他柱在相同 DI 的真實照片能達到的分數 |

**為什麼條件指標的上限要用「其他柱」，而不是 CTR1 自己的相鄰幀**：生成器不知道是哪一根柱，不可能畫出 CTR1 自己的裂縫配置。換一根真實柱能做到的，才是生成器可以追的目標。

## 可比性規則

- 只有在**同一個參考集、同一個 backbone、同一套前處理**下的分數，才能互相比較。DINOv3 和 DINOv2 的 KID 尺度差約 500 倍，不能並列（[distribution.md](distribution.md#backbone-的選擇)）。
- 不同 split 的 KID 不能直接比大小，因為參考柱本身離訓練集的遠近不同（CTR1 遠、R307 近）。
- σ、patch 數、解析度、像素門檻在看到結果前就要固定，之後不能為了結果去改。
- **循環偏差**：用 GT 幀 DINOv3 embedding 當條件的模型（`oracle_dino*`）不能用 DINOv3 的條件指標評，因為它把評分用的特徵空間當成輸入了（Woodland 2024）。改用 DINOv2 評可以降低偏差（[oracle.md](oracle.md#換-backbone-的檢查)）。
