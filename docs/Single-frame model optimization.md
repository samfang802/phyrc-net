# Single-frame model optimization

單幀 CFM（`CFMADM`，ADM U-Net，像素空間，Patch Diffusion）上三項和「條件怎麼進網路、patch 怎麼取樣、g 怎麼增強」有關的設定，加上高 DI 6×6 樣本的位置。
沒有做過對照實驗的地方會直接寫明。

## 0. 先交代條件路徑（三項設定都接在這裡）

網路的條件分兩條，各自轉成同維度向量後相加：

```
e_c = LN( MLP_c( t, direction [, DI] ) )     # 標量條件：流時間、方向（zero-DI 模式下 DI 不輸入）
e_g = LN( Linear_g( g ) )                    # g：GRU 物理特徵，1024 維
h_cond = e_c + alpha * e_g                   # alpha 是可學參數，初始為 0
```

`h_cond` 再去驅動 U-Net 各層 AdaGN 的 scale / shift。因為 `alpha` 初始為 0，訓練一開始網路完全不看 g，等 `alpha` 被學大之後 g 才逐步影響生成。以下的 bottleneck 和 gmix 都是在改變「g 這條路」。

## 1. Condition bottleneck（g 的瓶頸層）

### 機制

`k = 0`（預設）時，g 只經過一個 `Linear(1024 → cond_dim)`，1024 個軸全部都能影響條件。

`0 < k < 1024` 時，改成先壓到 k 維再升回來：

```
g (1024) →  LayerNorm → Linear(1024→k) → RMSNorm(k) → Linear(k→cond_dim) → e_g
```

- **沒有 skip**。如果加 skip，細軸會從旁邊繞過去，瓶頸就失去意義；這裡的目的就是強迫條件只能通過 k 個方向。
- 效果上等於把 `Linear_g` 限制成秩 ≤ k 的映射。g 裡面和損傷程度無關、或只是試體個別差異的細軸會被丟掉，網路只能用 k 個組合來區分「現在是哪個損傷階段」。
- 參數量從 `1024 × cond_dim` 變成 `1024 × k + k × cond_dim`，k 小時明顯變少。
- 瓶頸中間用 RMSNorm 而不是 LayerNorm：LayerNorm 會減均值，把各軸的偏移抹掉；RMSNorm 只縮放長度，保留軸的偏移和相對能量大小。
- 和 `g_pca_dim` 是兩種做法（PCA 是在資料端先降維、瓶頸是在網路內學），不能同時使用。



### 為什麼想用它

訓練集只有 4 個試體，g 是 1024 維，維度遠大於試體的獨立狀態數。條件軸過多時，網路可能用 g 的細節去「認出是哪一幀」，而不是學損傷階段與影像的關係。瓶頸是降低這種記憶的一個手段。（這是動機，不是已驗證的結論。）

### 目前狀態

**現行腳本和 15k refiner run 都沒有使用**（`physics_bottleneck_dim = 0`）。

**舊 run 的痕跡：** 結果資料夾的 `args.json` 顯示有兩個 run 設了 `physics_bottleneck_dim = 16`：

| Run | bottleneck | 其他相關設定 | val/loss（最後記錄） |
|---|---|---|---|
| `2026-09-18_02-07-49` | 16 | gmix λ=0.3 | 0.01839 |
| `2026-09-18_02-15-49` | 0 | gmix λ=0.3 | 0.01957 |
| `2026-09-19_03-49-14` | 16 | 無 gmix、`g_pca_dim=0` | 0.01547 |
| `2026-09-18_20-29-32` | 0（改用 `g_pca_dim=16`） | 無 gmix | 0.01969 |

前兩列的 `args.json` 只有 `physics_bottleneck_dim` 不同（16 vs 0），是最乾淨的一對，但只有單次運行、差距約 0.001，不能下結論。

**注意：** 這些只是 `args.json` 的紀錄。我沒有找到對應的訓練 job 腳本或 PBS 日誌（現行 `train_ADMCFM.sh` 把它設為 0，`log/` 裡也沒有這些 run 的訓練日誌），所以無法確認當時的完整指令。

## 2. Patch 機率排程



### 機制：Patch Diffusion 的隨機裁切

訓練時不是每步都用整張 128×128，而是每個 batch 抽一個共用的裁切尺寸：


| 尺寸      | 邊長  | 佔整張面積 |
| ------- | --- | ----- |
| full    | 128 | 1     |
| half    | 64  | 1/4   |
| quarter | 32  | 1/16  |


- 同一個 batch 共用一個尺寸，但每個樣本的裁切位置各自隨機。
- 網路另外收到裁切區的 XY 座標通道（對應在整張圖中的絕對位置），所以它知道自己看的是哪個位置。損失只算影像通道。
- 好處：小 patch 讓每步更便宜、並讓同一張圖產生很多不同的局部訓練樣本，對資料少的情況有幫助；full 則保留全域結構（格線、柱體輪廓）。
- 取樣（推論）時一律使用整張圖。

預設的靜態機率是 `(full, half, quarter) = (0.5, 0.3, 0.2)`，期望的像素質量是 `M = p_full + 0.25·p_half + 0.0625·p_quarter = 0.5875`。

### 機制：排程（想讓機率隨訓練步數變）

兩種模式：

- **two_segment**：機率分三段。`[0, mid]` 從 start 線性走到中間值，`(mid, end_start]` 保持不變，`(end_start, total]` 再線性走到 end。
- **pixel_mass**：不直接排機率，而是排「期望像素質量 M」。在 `[0, total_steps]` 上讓 M 從 `mass_start` 線性走到 `mass_end`，quarter 機率另外從 `quarter_start` 走到 `quarter_end`，再由 `M = p_full + 0.25·p_half + 0.0625·p_quarter` 反推 full / half。
  - 例：`M: 0.55 → 0.64`、`p_quarter: 0.15 → 0.10`。起點約 `(0.44, 0.41, 0.15)`，終點約 `(0.55, 0.36, 0.10)`。也就是前期較多小 patch（學局部紋理），後期逐步增加整張圖的比例（把全域一致性補回來）。

另有一個**依流時間 t 的加成**（`use_patch_t_schedule`）：t 越接近 1（越乾淨）時，把一部分機率從 full 移到 half。理由是 t 小的時候影像接近純噪聲，需要看全域；t 大時局部細節比較重要。因為 batch 平均的 t 約在 0.5，會讓斜坡失效，所以是對每個樣本分別加成後再平均。

### 目前狀態

**現行腳本和 15k refiner run 是關閉的**（`use_patch_prob_schedule = False`、`use_patch_t_schedule = False`），用靜態 `0.5, 0.3, 0.2`。

**舊 run 的痕跡：** `args.json` 顯示 9/16 有三個 run 開過排程（都是 15k 步）：

| Run | 排程設定 | val/loss（最後記錄） |
|---|---|---|
| `2026-09-16_00-44-38` | two_segment：start `0.8,0.2,0` → mid `0.5,0.3,0.2` → end `0.7,0.25,0.05` | 0.01638 |
| `2026-09-16_15-53-35` | pixel_mass：M 0.85→0.62，quarter 0→0.05 | 0.01963 |
| `2026-09-16_22-22-50` | pixel_mass：M 0.55→0.64，quarter 0.15→0.10，並開 t 加成（half +0.20） | 0.01607 |

後續 9/18 起的 run 都改回靜態機率。三個 run 的其他設定我沒有逐項比對，且沒有靜態機率的同期對照，所以不能由 val/loss 判斷排程有沒有幫助。這些同樣只是 `args.json` 的紀錄，我沒有找到對應的訓練 job 腳本或 PBS 日誌。

## 3. g_traj_mix（gmix）



### 機制

這是**只在訓練時**對 g 做的資料增強：

1. 把同一個試體、同一個方向的標註幀，依時間順序排成一條軌跡。
2. 對每個訓練樣本，以機率 λ 決定要不要換 g。
3. 要換的話，從該幀前後最多 `delta` 個標註幀（預設 ±2，也就是最多 4 個鄰居）裡隨機挑一個，把它的 g 拿來用。
4. **影像、遮罩、DI 都保持原幀不變**，也就是「影像是第 t 幀，條件 g 卻來自 t±1 或 t±2」。

鄰居是用「標註幀的順序」算的，不是用時間戳差。

### 為什麼合理

- 相鄰幀的影像幾乎一樣（我量過，128 px 下平均絕對差約 0.02），所以把 g 換成相鄰幀的 g，條件和影像仍然大致相容，只是 g 帶有一點時間上的位移。
- 效果是讓網路學到「g 附近的一小段範圍都對應這種影像」，而不是把某個精確的 g 向量對到某張精確的圖。這是一種標籤平滑，對 g 的小變化更穩健；在 4 個試體、約 530 張高度相似的訓練幀下，這種防止背單幀的正則化尤其有可能有用。
- λ 是機率，不是混合係數：沒被選中的樣本完全不變。λ = 0.5 表示一半的樣本帶鄰居 g。
- 推論時不做任何混合，用的是該幀自己精確的 g。



### 風險

- 把 g 換成鄰居，等於故意讓條件和影像略微不一致。如果 λ 太大或 delta 太大，會讓網路忽略 g 的細節（包括真正有用的損傷進程資訊）。
- 損傷快速變化的區段（例如剝落剛出現），±2 幀的差異可能不小，換 g 會引入真正的矛盾。



### 現有結果（`results/DamagePatternGeneration/CFMADM/<run>/`）


| Run                             | λ   | 6×6 High DI                                                                 | val/loss（最後記錄） |
| ------------------------------- | --- | --------------------------------------------------------------------------- | -------------- |
| `2026-09-18_02-07-49`           | 0.3 | `inference_samples/overview_di_3_high_6x6_selfcondOFF_step_14000_ode30.png` | 0.01839        |
| `2026-09-18_02-15-49`           | 0.3 | 同上檔名                                                                        | 0.01957        |
| `2026-10-08_00-33-21_g_caption` | 0.5 | `inference_samples/overview_di_3_high_6x6_selfcondOFF_step_15000_ode30.png` | 0.01769        |


**這些數字不能用來判斷 gmix 有沒有效。** 前兩個 run 之間只有 `physics_bottleneck_dim` 不同（見第 1 節），沒有 λ = 0 的同期對照；`g_caption` 則和另外兩個有許多設定不同（spall 相關參數、`use_paired_bridge`、`velocity_init`、`cfg_g_drop_prob` 等）。val/loss 的差距約 0.002，沒有重複種子，6×6 也只有目視、沒有量化指標。

**待做：** 以同一份設定、同種子跑 λ = 0 / 0.3 / 0.5，再用固定的 test 評估指標比較。

## 4. 高 DI 6×6 inference 樣本位置



### 單幀 base（`1024-dadaGNsplit/inference_samples/`）

- `baseline_6x6/overview_di_3_high_6x6_selfcondOFF_step_15000_ode30.png`
- GT：`baseline_6x6/overview_di_3_high_6x6_GT_test_all.png`、`..._GT_test_CTR1.png`
- Autoguide 版本：`autoguide/`（檔名帶 `ag…`、`g…`、`bad…`、`fdgL…H…`）
- 其他：`baseline_6x6/` 還有 low / mid 的 6×6 和 10×10 uniform-DI 圖；`other_guidance/`、`flowedit_sdedit/` 為其他 guidance 和 FlowEdit / SDEdit 的結果。



### 15k refiner run（`2026-10-13_03-02-40_refiner_res_15k/inference_samples/`）

- 15k，灰度投影：`overview_di_3_high_6x6_selfcondOFF_step_15000_ode30_refiner_gray.png`
- 15k，無投影：`..._step_15000_ode30_refiner.png`
- 10k，灰度投影：`..._step_10000_ode30_refiner_gray.png`
- GT：`overview_di_3_high_6x6_GT_test_all.png`
- 5 個 test 樣本：`test5_refiner_step15000_gray.png`。**已知問題：** 圖中 `CTR1 t=1830` 重複出現，因為同一幀在兩個 direction 欄位中重複，選取時沒有去重；尚未修正。



### 其他 run 的 6×6

`2026-09-18_02-07-49`、`2026-09-18_02-15-49`、`2026-10-08_00-33-21_g_caption`（gmix）、`2026-10-08_05-31-44_cascade128`、`2026-10-08_13-22-53_cascade128_basic`、`2026-10-07_18-22-12_paired_bridge` 等，各自的 `inference_samples/` 內都有 `overview_di_3_high_6x6_*.png`。

## 5. 未確認事項

- 三項設定都沒有可靠的對照實驗（只有 `args.json` 的零散痕跡，缺訓練 job 腳本和日誌），不能判定對 spalling 細節是否有幫助。
- 6×6 圖可在這些舊 run 的 `inference_samples/` 找到（`2026-09-16_00-44-38`、`2026-09-16_22-22-50`、`2026-09-18_02-07-49`、`2026-09-19_03-49-14`、`2026-09-18_20-29-32`），尚未目視比較。
- 在約 61 張 DI > 0.9 的訓練幀、4 個試體的限制下，這類條件和增強的效果可能被資料量蓋過。
- 「為什麼想用」「為什麼合理」兩段是設計動機的說明，不是實驗結論。

