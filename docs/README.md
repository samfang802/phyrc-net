# 專案說明

倉庫首頁仍是上一層的 `README.md`。這裡是架構和實驗筆記。

## 現在這條線

| 檔案 | 內容 |
|---|---|
| `TEMPORAL_FINAL.md` | 時序生成的最終版：同一個速度損失、`s` 的尾巴、凍結單幀只當起點、飽和時才重畫 |
| `DAMAGE_PATTERN_CURRENT.md` | 目前的 G、M、m*，以及 soft-paint 採用了哪些論文 |
| `FLOWEDIT_TEMPORAL.md` | 不重訓的 FlowEdit 時序推論，以及為什麼高 DI 會糊 |
| `PAIRED_BRIDGE.md` | 疊影的原因，BBDM、I²SB、MasaCtrl、TokenFlow，以及要改成的配對橋 |
| `SPALL_GROWTH_MASK.md` | 增量遮罩那一版的設計說明 |
| `GPU.md` | 這台機器的 PBS `gpu_id` 和 `nvidia-smi` 編號 |

## 架構沿革

目前的生成器是 [`ARCHITECTURE_UPDATE_V9.md`](ARCHITECTURE_UPDATE_V9.md)。更早的筆記在 [`archive/`](archive/)，預設搜尋不會讀它們。
