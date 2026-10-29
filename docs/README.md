# 專案說明

倉庫首頁仍是上一層的 `README.md`。這裡是架構和實驗筆記。PBS 腳本的用法見 [`../job/README.md`](../job/README.md)，機器的 GPU 編號見 [`GPU.md`](GPU.md)。

## 生成器架構（依版本）

每份只寫和前一版的差異。

| 檔案 | 內容 |
|---|---|
| [`ARCHITECTURE_UPDATE_V9.md`](ARCHITECTURE_UPDATE_V9.md) | 單幀生成器 `selfcondnoDI`，主訓練腳本 `train_ADMCFM.sh` 的基準 |
| [`ARCHITECTURE_UPDATE_V10.md`](ARCHITECTURE_UPDATE_V10.md) | autoguidance 取樣、資料快取與前兩幀、殘差時序 refiner（variant B） |
| [`ARCHITECTURE_UPDATE_V10_5.md`](ARCHITECTURE_UPDATE_V10_5.md) | refiner 改成 ControlNet 分支的隱式殘差，加曝光修正 |
| [`ARCHITECTURE_UPDATE_V10_6.md`](ARCHITECTURE_UPDATE_V10_6.md) | 回到串接 `S_{t-1}` 的 refiner，目標改成 x1，保留 v10.5 的曝光修正 |
| [`ARCHITECTURE_UPDATE_V11.md`](ARCHITECTURE_UPDATE_V11.md) | gn 的條件換成損傷狀態編碼器（z + 損傷標量 + 加載正負號） |

V2–V8 和更早的方法筆記在 [`archive/`](archive/)，預設搜尋不會讀它們。

## 單幀模型的元件

| 檔案 | 內容 |
|---|---|
| [`DAMAGE_STATE_ENCODER.md`](DAMAGE_STATE_ENCODER.md) | 取代 GRU g 的損傷狀態編碼器（版本 A token／B 逐點），V11 的條件 |
| [`archive/SINGLE_FRAME_OPTIMIZATION.md`](archive/SINGLE_FRAME_OPTIMIZATION.md) | 條件瓶頸、patch 取樣、g 增強三項設定 |
| [`guidance.md`](guidance.md) | 取樣時的引導：APG、高通濾波、FA-CFG、autoguidance |
| [`archive/DAMAGE_PATTERN_CURRENT.md`](archive/DAMAGE_PATTERN_CURRENT.md) | 2026-10-16 時的 G、M、m*，以及 soft-paint 採用了哪些論文 |

## 時序生成（2026-10-11 那一輪，已歸檔）

V10 的 refiner 之前的時序設計，依時間順序：

| 檔案 | 內容 |
|---|---|
| [`archive/FLOWEDIT_TEMPORAL.md`](archive/FLOWEDIT_TEMPORAL.md) | 不重訓的 FlowEdit 時序推論，以及為什麼高 DI 會糊 |
| [`archive/PAIRED_BRIDGE.md`](archive/PAIRED_BRIDGE.md) | 疊影的原因，BBDM、I²SB、MasaCtrl、TokenFlow，以及要改成的配對橋 |
| [`archive/TEMPORAL_FINAL.md`](archive/TEMPORAL_FINAL.md) | 同一個速度損失、`s` 的尾巴、凍結單幀只當起點、飽和時才重畫 |
| [`archive/SPALL_GROWTH_MASK.md`](archive/SPALL_GROWTH_MASK.md) | 剝落增量遮罩的設計說明 |

## 評估與圖

| 資料夾 | 內容 |
|---|---|
| [`eval/`](eval/README.md) | 生成品質評估：DINOv3 KID、條件指標、損傷型態、干擾物處理、目前結果與參考文獻 |
| [`figures/`](figures/) | 文件用的流程圖 |
