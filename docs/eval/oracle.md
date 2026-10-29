# Oracle 版本：生成器在「完美條件」下的上限

## 它是什麼

正常的模型用 GRU 從遲滯迴圈算出隱狀態 g，當成生成的條件。Oracle 版本把 g 換成 **GT 那一幀的 DINOv3 影像特徵**，其他架構完全不動。

- 程式：`src/model/cfm/oracle_dino.py`（`DinoOracleEncoder`）。接線在 `src/model/cfm/condition_mixin.py` 的 `_maybe_oracle_hidden_state`：每個 batch 用 GT 影像算 embedding，取代 `hidden_state`。
- 特徵：`facebook/dinov3-vits16-pretrain-lvd1689m`（凍結），`[CLS ; patch 平均]` 共 768 維，再做 layer norm，讓尺度接近 GRU g。
- 進入模型的路徑和 GRU g 一樣：physics Linear → gated AdaGN。
- 訓練設定：V9（`selfcondnoDI`）的設定，同切分（`id_list_val_R307.json`），同樣 15000 步。訓練 job 是 `job/DamagePatternGeneration/train_ADMCFM_oracle_dino.sh`。

| run | `oracle_dino_pos_blur` | 意義 |
|---|---|---|
| `2026-10-19_12-04-44_oracle_dino_none` | `none` | GT 幀原樣嵌入：條件裡有損傷的量、質感，也有位置 |
| `2026-10-19_12-04-44_oracle_dino_shuffle` | `shuffle` | GT 幀切成 4×4 塊打亂後再嵌入（推論時用固定的排列）：保留質感，拿掉位置 |

另外還有 `shift`（訓練時隨機平移、翻轉），目前沒有完整的 run。

## 要回答的問題

> 如果條件完美地描述了「這一幀長什麼樣」，生成器能畫到多好？

- Oracle 和 V9（GRU 條件）之間的差距，就是「上游條件再改進」最多能帶來的空間。
- Oracle 本身仍不夠好的部分，是生成器（架構、訓練、資料）自己的極限，換再好的條件也救不了。

**對照組一定要用 V9**（`selfcondnoDI`，同架構、同設定、只差條件），不能用 gn。gn 沒有 self-conditioning，和 oracle 不只差在條件。

## 它不是什麼

- **它不是可用的模型**：推論時要有 GT 影像才能算條件，真實應用時根本沒有 GT。
- **它是寬鬆的上限**：DINOv3 特徵從 GT 影像直接算出，不只帶了損傷，也帶了 CTR1 這根柱自己的風格（例如標記筆描線、表面紋理）。遲滯迴圈不可能提供這些資訊。所以它是「任何由影像得到的條件」的上限，比「從遲滯迴圈能得到的條件」寬鬆得多。

## 評估時的循環偏差

Oracle 的條件和 DINOv3 指標用的是**同一個特徵模型**。生成器被訓練成「讓輸出的 DINOv3 特徵符合條件」，再用 DINOv3 特徵評分，等於拿考卷當參考答案（Woodland 2024 警告的循環偏差）。`eval_dinov3_kid.py` 用 DINOv3 評到 `oracle_dino` 時會印出警告，摘要裡這些數字加括號；換成別的 backbone 時不加。

| 指標 | 對 oracle 是否可信 |
|---|---|
| 鄰居對比、演進（ratio、corr）、probe | **不可信**。實測鄰居對比達上限的 103–115%，演進 ratio 和 corr 也超過「換一根真實柱」，這不可能是真的 |
| KID、patch KID | 部分受影響：條件只約束每張圖的特徵，不約束整體分佈，所以偏差較小，但仍可能偏好 oracle |
| 像素統計（損傷面積、灰霧、白塊、細紋理、裂縫與 GT 的 F1）、時序指標 | **可信**，和 DINOv3 無關 |

所以 oracle 的上限要看像素統計，DINO 指標則要再用另一個 backbone 確認。

### 換 backbone 的檢查

用 DINOv2 ViT-S/14 重算同一批樣本（2026-10-21）。DINOv2 不是 oracle 的條件來源，但同樣是 DINO 系列的自監督模型，所以只算「循環偏差較小」，不算完全獨立的評分者。

| | test DINOv3 | test DINOv2 | val DINOv3 | val DINOv2 |
|---|---|---|---|---|
| oracle 鄰居對比（達上限） | 105% | 98% | 115% | 113% |
| V9 鄰居對比 | 66% | 58% | 66% | 59% |
| oracle 演進 ratio / corr | 0.69 / 0.92 | 0.66 / 0.90 | 0.88 / 0.93 | 0.92 / 0.92 |
| oracle − V9，KID 的 z | −1.8 | **−4.8** | −1.5 | −1.8 |

> 2026-10-29：這張表和下面「目前的結果」的所有 DINOv3 / DINOv2 數字（KID、鄰居對比、演進）都是 JPEG 對齊前的。oracle 和 V9 的樣本已刪除，無法用 JPEG q75 重算，所以 oracle − V9 的比較也是對齊前的；JPEG 讓我們模型的 KID 一律降 0.04–0.06（見 [results.md](results.md)），兩者降幅接近，差值應該變化不大，但沒有驗證。oracle 在 DINOv3 下本來就有循環偏差，這些數字只當參考。

- **條件一致性的進步不是循環造成的**：鄰居對比和演進在 DINOv2 下幾乎一樣高，和 V9 的差距仍然顯著（鄰居對比 z：test 3.1、val 5.3）。GT 的特徵帶有的資訊（試體、位置、損傷程度）確實讓生成圖更貼近對應的那一幀。鄰居對比仍接近或超過 100%，是因為「上限」是另一根真實柱，它不知道這根柱的細節，oracle 知道。
- **真實感的進步只在 test 上顯著**：DINOv2 下 test 的 KID 差距變顯著（z = 4.8），val 仍不顯著（z = 1.8）。結論維持「真實感幾乎沒進步」，test 上的差距可能是 CTR1 特有的，和損傷面積的情況一樣。

## 目前的結果

數字見 [results.md](results.md#oracle-上限)。摘要（DINOv3 數字是 JPEG 對齊前的，見上面 2026-10-29 的註記；像素統計不受影響）：

- **真實感幾乎沒進步**：KID 只比 V9 好 0.009（test z = 1.8、val z = 1.5，不顯著）。DINOv2 下 test 變顯著（z = 4.8），val 仍不顯著（z = 1.8），見上面的檢查。所以真實感的瓶頸主要在生成器本身，不在條件。
- **條件一致性大幅進步，而且不是循環造成的**：DINOv2 下鄰居對比仍有 98–113%（V9 58–59%）。
- **test（CTR1）上剝落量大幅改善**：高 DI 的損傷面積從 0.104（V9）升到 0.146–0.150（GT 0.164）。白塊、灰霧都降到 GT 以下，加載前的假裂縫也消失了，裂縫和 GT 的重疊從 0.12 升到 0.16–0.17。
- **val（R307）上沒有重現**：損傷面積 0.053，反而比 V9（0.098）和 GT（0.079）都少。R307 的剝落少、型態和 CTR1 不同，oracle 在這根柱上沒有畫出比較多的損傷。所以「條件是剝落量的瓶頸」目前只在 CTR1 上成立，證據還不夠。
- **位置資訊不需要**：打亂版和完整版的結果差不多。
- **時序上沒有幫助**：oracle 也是單幀模型，逐幀獨立抽樣時一樣會閃爍（`tone_jump` 17，GT 4）。
