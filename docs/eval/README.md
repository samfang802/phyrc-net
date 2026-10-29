# 損傷影像生成的評估指標

這個資料夾說明如何客觀評估 RC 柱損傷影像生成器，涵蓋三件事：

- 生成圖像不像真的；
- 有沒有跟著條件（遲滯、DI、drift）畫出對的損傷；
- 失敗時是哪一種失敗。

所有指標都針對這個資料的限制設計：

- test 只有一根柱（CTR1，124 張）。
- 相鄰幀幾乎相同。
- GT 上有實驗器材（干擾物），生成器不會畫。

## 先看這張表

| 指標 | 回答的問題 | 會看條件嗎 | 說明 |
|---|---|---|---|
| `kid_global`、`kid_patch`、`mmd_rbf_global` | 整體看起來像不像真實照片 | **不會** | [distribution.md](distribution.md) |
| `probe`（DI、R_spall） | 從圖讀得出條件該有的損傷程度嗎 | 會 | [condition.md](condition.md#線性-probe) |
| `nbr_global`、`nbr_patch` | 生成圖比較像 GT 附近的真實幀，還是別的幀 | 會 | [condition.md](condition.md#gt-鄰居對比) |
| `progression_global`、`progression_patch` | drift 變大時，損傷有沒有跟 GT 一樣地長出來 | 會 | [condition.md](condition.md#損傷演進) |
| `damage_texture`（灰霧、白塊、裂縫） | 畫的是哪一種損傷；有沒有沿用訓練柱的裂縫 | 部分 | [damage_texture.md](damage_texture.md) |
| `flicker`、`tone_jump`、`persistence`、`crack_keep` | 照時間排起來像不像真實的加載過程：會不會閃、損傷會不會消失 | 不會 | [temporal.md](temporal.md) |
| `interference` | 生成圖畫了多少干擾物 | — | [interference.md](interference.md) |
| `saturated_frames` | 爆白、爆黑的偽影比例 | — | [distribution.md](distribution.md#附帶的偽影統計) |
| `memorization` | 有沒有背訓練圖 | — | [distribution.md](distribution.md#背圖檢查) |

**最重要的一點：KID 不看條件。** 把生成圖隨機換到別的條件，KID 不變。所以 KID 只能回答「像不像真的」，不能回答「這個 DI 該有的剝落有沒有畫出來」。後者要看條件指標。

## 建議的報告組合

比較兩個模型時，至少報告下面這些。所有數字都附 SE 和配對比較的 z。

1. **真實感**：`kid_global` 和 `kid_patch`。生成圖要先做 JPEG q75 對齊，因為參考集是 JPEG（[protocol.md](protocol.md#參考集是-jpeg生成圖也要過一次-jpeg)）。
2. **條件一致性**：
   - `progression_global` 的 `corr`：目前唯一在兩根留出柱上都分得開模型的指標。
   - `nbr_global`，附上 `frac_of_ceiling`。
3. **失敗型態**：
   - `damage_texture` 高 DI 區間的 `fog_share`、`white_share`、`E_damage`；
   - 低 DI 的 `before_loading` 和 `copy_excess`。
4. **缺陷**：`interference`、`saturated_frames`。
5. **刻度**：每個指標旁邊列出參考列，例如「換一根真實柱」能達到多少。沒有刻度，數字無法解讀。

不要拿來下結論的數字：

- `prdc`：單一柱的參考集讓它退化成 0。
- `kid_soft_di`：沒通過打亂檢查。
- 用 val（R307）算的 R_spall probe：R307 剝落很少（8% 時約 0.17）。

原因見各指標文件。

## 安裝

1. **程式與資料**：clone 這個 repo。`Data/DamagePattern/`（GT 影像、`label.csv`、切分檔、SAM 遮罩）已經在 git 裡。
2. **環境**：conda 環境 `internship`。目前驗證過的版本：

   | 套件 | 版本 |
   |---|---|
   | torch | 2.13 |
   | transformers | 5.14（DINOv3 需要 ≥ 4.56） |
   | numpy / scipy / scikit-learn | 2.2 / 1.15 / 1.7 |
   | pandas / pillow / matplotlib / opencv | 2.3 / 12.3 / 3.10 / 5.0 |

3. **不在 git 裡的檔案**：向維護者拿 `eval_assets_v1.tar.gz`，在 repo 根目錄解開並檢查：

   ```bash
   tar xzf eval_assets_v1.tar.gz && sha256sum -c eval_assets_v1.files.sha256
   ```

   壓縮檔內含（約 380 MB）：
   - 補圖版參考集 `Data/DamagePattern_inpaint_selfcondnoDI/images`；
   - 每幀的剝落比例標籤 `Data/DamagePattern/anylabel_r_spall.csv`（人工剝落多邊形，2026-10-25 起的預設，`src/metrics/spall_labels.py`）；舊的 `r_spall.csv` 也一起打包，只用來重現舊數字（`--r_spall_csv old`）；
   - 干擾物偵測器 `results/InterferenceDetector/v1/best.pt` 與 `backbone_config.json`；
   - 去干擾物用的 CFM 補圖模型（V9 `selfcondnoDI` 的 `args.json` 和 `last.ckpt`，和做參考集的是同一個）；
   - 補圖模型條件需要的 GRU extractor `phase1_extractor/`。

   維護者用 `bash scripts/eval/pack_eval_assets.sh` 產生它。偵測器的 DINOv3 權重已經在 `best.pt` 裡，`backbone_config.json` 是架構設定，所以載入偵測器不需要 Hugging Face 權限。

4. **指標用的 backbone**，二選一：
   - **DINOv3**（預設，和既有數字可比）：`facebook/dinov3-vits16-pretrain-lvd1689m` 在 Hugging Face 需要申請，權限綁帳號。先到模型頁面同意授權，再執行 `huggingface-cli login`。
   - **DINOv2**（不用申請）：在設定檔寫 `"dino_model": "facebook/dinov2-small"`。模型排名和主要結論跟 DINOv3 一致（[distribution.md](distribution.md#backbone-的選擇)），但數字不能和 DINOv3 的直接比較。

   第一次執行時會下載到快取，之後 `eval_suite.sh` 預設 `HF_HUB_OFFLINE=1` 離線執行；第一次要用 `qsub -v HF_HUB_OFFLINE=0`。登入的 token 過期時，連公開模型也會下載失敗，`huggingface-cli logout` 或重新登入即可。
5. **GRU extractor**（`./phase1_extractor`）已經在壓縮檔裡：去干擾物和用本 repo 的模型生成都要用它。`prepare_extractor_from_hf.sh` 下載的 Hugging Face repo 是私人的，只有擁有者能用。

## 一條指令跑完

1. 編輯模型清單 `job/evaluation/configs/eval_suite_models.json`。格式見 `scripts/eval/run_eval_suite.py` 開頭的說明，每個模型可以是下面三種之一：
   - 本 repo 的 CFM run：填 `result_dir` 和 `checkpoint`；
   - 一個存好的圖檔資料夾：填 `images`；
   - 一個做好的 npz：填 `npz`。

   沒有經過 JPEG 的模型（本 repo 的模型都是）要加 `"jpeg_quality": 75`；cGAN 這類本來就存成 JPEG 的不要加。原因見 [protocol.md](protocol.md#參考集是-jpeg生成圖也要過一次-jpeg)。
2. 執行：

   ```bash
   qsub job/evaluation/eval_suite.sh                  # PBS
   # 或直接在有 GPU 的機器上
   CUDA_VISIBLE_DEVICES=0 python scripts/eval/run_eval_suite.py --config job/evaluation/configs/eval_suite_models.json
   ```

3. 看結果：`<out_dir>/summary.md`，每個 split 一份報告，裡面是所有指標的比較表和配對 z。各步驟的 log 在 `<out_dir>/logs/`。

說明：

- **開始前會先檢查**：參考集、偵測器、backbone 權限、checkpoint 都在不在。缺什麼會直接說，不會跑到一半才失敗。
- **第二個 backbone**：`"extra_dino_models": ["facebook/dinov2-small"]` 會用 DINOv2 再算一次真實感與條件指標（`kid_dinov2-small/`），摘要裡多一張「換 backbone 的穩健性檢查」表。範例設定檔預設開著；主表和第二張表的排名不一致時，那個結論不要下。
- **可以續跑**：中斷後再送一次，已完成的步驟會跳過；`FORCE=1`（或 `--force`）全部重做。加新模型時只會補跑新模型的生成和去干擾物，三種評估會重跑，因為所有模型要放在同一次評估裡比。
- **先看會跑什麼**：`--dry_run` 只印出每一步的指令，不執行。
- **時間**：每個模型、每個 split，K = 8 時生成加去干擾物約 5–15 分鐘（refiner 較久），三種評估合計約 5–10 分鐘。

## 評估外部模型（不在這個 repo 的模型）

只要輸出圖檔就能評：每個 split 的每一幀生成一張（或 K 張），檔名用 GT 的檔名，再加入設定檔：

```json
{"name": "my_model", "images": {"test": {"dir": "path/to/test_images", "suffix": ".png"}}}
```

- K 張的命名用 `"suffix": "_k{k}.png", "k": 8`。
- 圖會轉成灰階並縮到 128×128。如果輸出不是 CLAHE 後的亮度範圍，加 `"clahe": true`。
- 也可以自己轉：`python scripts/eval/images_to_npz.py --image_dir ... --suffix ... --out my.npz`。npz 格式見 `src/metrics/samples_io.py`。

## 手動逐步執行（除錯用）

`run_eval_suite.py` 就是依序呼叫下面這些腳本，`--dry_run` 會印出它實際用的參數：

```
generate_eval_samples.py / images_to_npz.py  → samples/<name>.npz
inpaint_generated_samples.py                 → <name>_clean.npz（去干擾物）、<name>_eq.npz（再加 --add_gt_mask）
eval_dinov3_kid.py      （吃 _eq）           → kid/dinov3_kid.json、progression.png
eval_damage_texture.py  （吃 _clean）        → texture/damage_texture.json
eval_temporal.py        （吃 _clean）        → temporal/temporal.json、temporal.png
summarize_eval.py                            → summary.md
```

- 時序 refiner（`use_refiner_residual`）不用另外處理，`generate_eval_samples.py` 會自動改成依時間整條推進。
- 要比較時序模型時，加一組單幀模型的 `"shared_noise": true` 當對照（見 [temporal.md](temporal.md)）。

## 版本

目前的評估協定是 **eval-v1**（寫在每次執行的 `suite_config.json`）。參考集、骨幹模型、像素門檻、σ、patch 數任何一項改變就要升版；不同版本的數字不能直接比較。主 backbone 換成 DINOv2 時，真實感與條件指標要標成 eval-v1（dinov2-small），只和同樣用 DINOv2 的結果比；損傷型態和時序指標不用 backbone，不受影響。

2026-10-28 起，DINOv3 指標的主結果是 **eval-v1 + JPEG q75**：參考集和其他設定不變，只在抽特徵前把生成圖做一次 JPEG q75。`suite_config.json` 上仍寫 eval-v1，要看模型設定裡的 `jpeg_quality`（`dinov3_kid.json` 也會記）。對齊前後的 KID、鄰居對比、演進 ratio 不能並列；損傷型態和時序指標不受影響。

## 文件

| 文件 | 內容 |
|---|---|
| [protocol.md](protocol.md) | 共同的評估設定：資料切分、參考集（含 JPEG 對齊）、前處理、npz 格式、不確定度、打亂檢查、可比性規則 |
| [interference.md](interference.md) | 干擾物偵測器與生成圖的補圖 |
| [distribution.md](distribution.md) | DINOv3 KID 系列、偽影、背圖檢查；試過但不採用的設計 |
| [condition.md](condition.md) | 條件指標：線性 probe、GT 鄰居對比、損傷演進 |
| [damage_texture.md](damage_texture.md) | 損傷型態的像素統計：灰霧、白塊、裂縫與訓練柱的重疊 |
| [temporal.md](temporal.md) | 時序一致性：閃爍、色調跳動、損傷與裂縫的保留；共用雜訊對照 |
| [oracle.md](oracle.md) | Oracle 版本（GT 的 DINOv3 特徵當條件）：目的、對照組、循環偏差、怎麼讀 |
| [condition_g.md](condition_g.md) | 遲滯迴圈條件與 GRU 隱狀態 g：影像柱在資料集的位置、g 的組成、生成器眼中的 CTR1 / R307、Δg、條件端建議 |
| [results.md](results.md) | 目前的結果（JPEG 對齊後）：base、refiner、V106C、v11appr、cGAN；後半保留對齊前的舊結果（gn、base0920、V9、oracle 等） |
| [references.md](references.md) | 參考文獻，依指標分類 |
| [research_log.md](research_log.md) | 原始研究筆記，記錄了每一步怎麼走到現在的設計。**裡面較早的數字有些已被更正**，以 results.md 為準 |

## 程式

| 檔案 | 作用 |
|---|---|
| `src/metrics/dino_mmd.py` | DINOv3 特徵、分組 U-statistic、jackknife、鄰居對比、演進曲線 |
| `src/metrics/interference.py` | 干擾物分割器 |
| `src/metrics/damage_texture.py` | 像素分類、裂縫圖、容忍度 F1 |
| `scripts/eval/generate_eval_samples.py` | 生成評估樣本 npz |
| `scripts/eval/inpaint_generated_samples.py` | 偵測並補掉生成圖的干擾物 |
| `scripts/eval/train_interference_detector.py` | 訓練干擾物偵測器 |
| `scripts/eval/eval_dinov3_kid.py` | DINOv3 指標主腳本 |
| `scripts/eval/eval_damage_texture.py` | 損傷型態指標 |
| `src/metrics/temporal.py`、`scripts/eval/eval_temporal.py` | 時序一致性指標 |
| `scripts/eval/dinov3_kid_sanity.py` | 指標本身的失真單調性檢查 |
| `scripts/eval/run_eval_suite.py`、`job/evaluation/eval_suite.sh` | 一條指令跑完整套評估 |
| `scripts/eval/summarize_eval.py` | 三種評估結果 → 一份 `summary.md` |
| `scripts/eval/images_to_npz.py`、`src/metrics/samples_io.py` | 外部模型的圖檔 → 評估用 npz；npz 格式 |
| `scripts/eval/pack_eval_assets.sh` | 打包不在 git 裡的參考集與偵測器 |
| `scripts/viz/export_drift_view_panel_from_images.py` | 把任何模型的圖排成 drift × DI 的雙面 panel，供目視比較 |
