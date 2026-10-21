# 干擾物：偵測、移除

## 問題

GT 上有實驗器材，例如 CTR1 下緣右側的立柱。各模型的情況不同：

- 我們的模型不畫這些器材；
- cGAN 會畫出別根柱的器材（例如訓練柱的彎鉤）。

這些器材又大又明顯，DINOv3 一定會看，所以會混進每一個指標。處理方式分三步：

1. **偵測**：找出生成圖上的器材。
2. **移除**：用和參考集相同的補圖模型補掉。
3. **另外報**：「畫了器材」本身就是一種缺陷，用 `interference` 單獨報告，而不是混在其他分數裡。

## 偵測器

程式：`src/metrics/interference.py`、`scripts/eval/train_interference_detector.py`。

- **架構**：DINOv3 ViT-S/16，輸入 448。微調最後 2 個 block 和 final norm，其餘凍結。解碼器是卷積層，加一條像素層級的 skip，輸出 128×128 機率圖。
- **載入不需要 DINOv3 權限**：`best.pt` 存了完整的 backbone 權重（包含微調過的部分），旁邊的 `backbone_config.json` 是架構設定，`load_segmenter` 直接用它建模型，不連 Hugging Face。新訓練的偵測器會把架構設定直接存進 checkpoint。所以指標 backbone 換成 DINOv2 時，偵測器照舊是同一個。
- **標註**：`Data/DamagePattern/mask` 的 SAM 遮罩，也就是產生補圖參考集用的同一批。
- **正樣本**：train 原圖和它的遮罩。
- **困難負樣本**：同一批幀的補圖版本，遮罩全 0。這讓模型學會「補過的地方不是器材」，避免在生成圖上誤報。
- **訓練**：資料增強用翻轉、小幅仿射、亮度對比和雜訊；損失是 BCE（pos_weight 4）加 Dice。訓練 40 epoch，以 val IoU 選模型。
- **權重**：`results/InterferenceDetector/v1/best.pt`。


|            | 像素 IoU | 影像層級 P / R  | 補圖被誤判比例 |
| ---------- | ------ | ----------- | ------- |
| val（R307）  | 0.93   | 1.00 / 1.00 | 5%      |
| test（CTR1） | 0.63   | 1.00 / 1.00 | 19%     |


test 的 IoU 偏低，是標註寬度不一致：CTR1 的 SAM 遮罩比立柱本體寬一圈。預測的位置是對的，只是比遮罩窄。補圖時取樣器本來就會把洞擴張 5 px，所以影響不大。

### 後處理規則（`clean_mask`）

兩條規則都從資料統計得到，不是看結果調的：

- **面積 ≥ 150 px**：val、test 的 SAM 元件全部 ≥ 339 px。我們模型被標記的元件有 90% < 150 px，都是下緣暗色碎石上的小點。
- **貼著下緣或左右邊緣（3 px 以內）**：874 個 SAM 元件全部符合。



## 生成圖的補圖

程式：`scripts/eval/inpaint_generated_samples.py`。

1. 對每張生成圖跑偵測器和 `clean_mask`。
2. 被標記的圖，用和參考集**同一個** CFM 補圖模型（`selfcondnoDI`，`sample_with_ode_inpaint`），以**同一幀的遲滯條件**補掉。沒被標記的圖原樣保留。
3. 輸出同格式的 npz，多了 `interference_mask` 和 `interference_flag` 兩個欄位。`eval_dinov3_kid.py` 讀到這兩個欄位時，會輸出 `interference`。


| 參數              | 預設  | 說明                           |
| --------------- | --- | ---------------------------- |
| `--dilate_px`   | 2   | 送進取樣器前再擴張幾 px（取樣器自己會再擴 5 px） |
| `--add_gt_mask` | 關   | 安慰劑對照：每張圖都在 GT 的器材位置補一次      |
| `--png_dir`     | —   | 存「生成圖 | 偵測遮罩 | 補圖結果」的對照圖和總覽  |




## 目前的偵測結果（test）


|          | 被標記的生成圖      | 平均標記面積 |
| -------- | ------------ | ------ |
| gn       | 0.3%（3/992）  | 177 px |
| base0920 | 0.3%（3/992）  | 161 px |
| cGAN     | 96%（119/124） | 750 px |


gn 和 base 被標記的 3 張，都是下緣中央的半圓形暗塊，位置正好是訓練柱彎鉤的位置。cGAN 的補圖結果在 `results/InterferenceDetector/v1/cgan_CTR1_inpainted/`。