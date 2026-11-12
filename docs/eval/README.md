# 損傷影像生成的評估

這個資料夾說明如何客觀評估 RC 柱損傷影像生成器，涵蓋四件事：

- 生成圖像不像真的；
- 有沒有跟著條件（遲滯、DI、drift）畫出對的損傷；
- 是不是照搬訓練柱；
- 失敗時是哪一種失敗。

所有指標都針對這個資料的限制設計：

- test 只有一根柱（CTR1，124 張）。
- 相鄰幀幾乎相同。
- GT 上有實驗器材（干擾物），生成器不會畫。

## 文件

| 文件 | 內容 |
|---|---|
| [metrics.md](metrics.md) | **定案的指標**：每個指標的問題、算法、刻度、怎麼讀；共同設定（切分、影像域、不確定度） |
| [results.md](results.md) | **目前的結果**：每個版本一個 checkpoint，加上 cGAN |
| [archive/](archive/) | 指標的設計過程與舊結果，見[下方](#封存的文件) |

## 安裝

1. **程式與資料**：clone 這個 repo。`Data/DamagePattern/` 的 `label.csv`、切分檔、SAM 遮罩在 git 裡；GT 原始照片（`images/`）沒有進 git，要另外放進去。
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
   - 每幀的剝落比例標籤 `Data/DamagePattern/anylabel_r_spall.csv`（人工剝落多邊形，`src/metrics/spall_labels.py`）；舊的 `r_spall.csv` 也一起打包，只用來重現舊數字（`--r_spall_csv old`）；
   - 干擾物偵測器 `results/InterferenceDetector/v1/best.pt` 與 `backbone_config.json`；
   - 去干擾物用的 CFM 補圖模型（V9 `selfcondnoDI` 的 `args.json` 和 `last.ckpt`，和做參考集的是同一個）；
   - 補圖模型條件需要的 GRU extractor `phase1_extractor/`。

   維護者用 `bash scripts/eval/pack_eval_assets.sh` 產生它。偵測器的 DINOv3 權重已經在 `best.pt` 裡，所以載入偵測器不需要 Hugging Face 權限。

4. **指標用的 backbone**，二選一：
   - **DINOv3**（預設，和既有數字可比）：`facebook/dinov3-vits16-pretrain-lvd1689m` 在 Hugging Face 需要申請，權限綁帳號。先到模型頁面同意授權，再執行 `huggingface-cli login`。
   - **DINOv2**（不用申請）：在設定檔寫 `"dino_model": "facebook/dinov2-small"`。模型排名和主要結論跟 DINOv3 一致（[archive/distribution.md](archive/distribution.md#backbone-的選擇)），但數字不能和 DINOv3 的直接比較。

   第一次執行時會下載到快取，之後 job 預設 `HF_HUB_OFFLINE=1` 離線執行；第一次要用 `qsub -v HF_HUB_OFFLINE=0`。登入的 token 過期時，連公開模型也會下載失敗，`huggingface-cli logout` 或重新登入即可。
5. **GRU extractor**（`./phase1_extractor`）已經在壓縮檔裡：去干擾物和用本 repo 的模型生成都要用它。

## 執行

主結果（[results.md](results.md)）用這兩個 job 產生：

```bash
qsub job/evaluation/eval_cgan_clahe.sh   # cGAN 轉到參考集的影像域（只需一次）
qsub job/evaluation/eval_final.sh        # suite（test + val）+ 抄襲與多樣性（test）
```

評估新模型的步驟：

1. 在 suite 設定檔（例如複製 `job/evaluation/configs/eval_suite_final.json`）加一個模型。格式見 `scripts/eval/run_eval_suite.py` 開頭的說明，每個模型可以是下面三種之一：
   - 本 repo 的 CFM run：填 `result_dir` 和 `checkpoint`；
   - 一個存好的圖檔資料夾：填 `images`；
   - 一個做好的 npz：填 `npz`。
2. **影像域要和參考集一致**（[metrics.md](metrics.md#參考集與影像域)）：
   - 本 repo 的模型輸出已經是 CLAHE 域，加 `"jpeg_quality": 75`。
   - 輸出原始照片影像域的模型（例如 cGAN），加 `"clahe": true` 和 `"jpeg_quality": 75`。這類模型通常會畫干擾物，設定檔最上層的 `"remove_interference"` 要開著（預設就是開的）。
3. 執行 suite：

   ```bash
   CUDA_VISIBLE_DEVICES=0 python scripts/eval/run_eval_suite.py --config <config.json>
   # 篩選用，只跑 test 的 KID（樣本還是會生成）：
   python scripts/eval/run_eval_suite.py --config <config.json> --quick
   ```

   結果在 `<out_dir>/<split>/summary.md`，各步驟的 log 在 `<out_dir>/logs/`。同一份模型清單再跑會跳過。DINOv3 特徵存在 `results/eval_suite/feature_cache`，已經算過的圖不會再抽一次。
4. 抄襲與多樣性指標不在 suite 裡，另外跑 `scripts/eval/eval_diversity.py`（參數見 `job/evaluation/eval_final.sh`）。輸入是 suite 產生的 npz，每個模型要給同樣的 `--jpeg`；需要 CLAHE 的模型列在 `--clahe`。

說明：

- **開始前會先檢查**：參考集、偵測器、backbone 權限、checkpoint 都在不在。缺什麼會直接說，不會跑到一半才失敗。
- **可以續跑**：中斷後再送一次，已完成而且模型清單沒變的步驟會跳過；`--force` 全部重做。模型清單變了，三種評估會重跑，因為所有模型要放在同一次評估裡比。生成和去干擾物只補還沒有的。
- **篩選**：`--quick`（job 用 `QUICK=1`）只跑 test 的 KID。要定案數字時再跑完整的 test + val、損傷型態和時序。
- **先看會跑什麼**：`--dry_run` 只印出每一步的指令，不執行。
- **時間**：新的 checkpoint 生成是主要的時間，每個模型、每個 split，K = 8 約 2–15 分鐘。指標本身，KID 在特徵已快取時是幾秒；損傷型態和時序各約半分鐘一個 split。完整一輪（兩個 split）大約 3 分鐘，不含生成。

### 外部模型（不在這個 repo 的模型）

只要輸出圖檔就能評：每個 split 的每一幀生成一張（或 K 張），檔名用 GT 的檔名，再加入設定檔：

```json
{"name": "my_model", "images": {"test": {"dir": "path/to/test_images", "suffix": ".png"}}, "clahe": true, "jpeg_quality": 75}
```

- K 張的命名用 `"suffix": "_k{k}.png", "k": 8`。
- 圖會轉成灰階並縮到 128×128。`"clahe": true` 用的是資料管線本身的 CLAHE（`dino_mmd.clahe_uint8`），不是 cv2 的。
- 也可以自己轉：`python scripts/eval/images_to_npz.py --image_dir ... --suffix ... --clahe --out my.npz`；已經有 npz 時用 `--from_npz`。npz 格式見 `src/metrics/samples_io.py`。

### 手動逐步執行（除錯用）

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
- 要比較時序模型時，加一組單幀模型的 `"shared_noise": true` 當對照（見 [archive/temporal.md](archive/temporal.md)）。

## 版本

評估協定是 **eval-v1**（寫在每次執行的 `suite_config.json`），2026-10-28 起 DINOv3 指標加上 JPEG q75 對齊，2026-11-06 起外部模型要先轉到 CLAHE 影像域。參考集、backbone、像素門檻、σ、patch 數、影像域任何一項改變，數字就不能和舊的並列。

## 封存的文件

`archive/` 保留指標的設計過程，數字以 [results.md](results.md) 為準。其中的 cGAN 數字是在沒有 CLAHE 的影像域算的，已不成立。

| 文件 | 內容 |
|---|---|
| [protocol.md](archive/protocol.md) | 共同設定的完整版：JPEG 對齊的發現過程、npz 格式、打亂檢查、可比性規則 |
| [distribution.md](archive/distribution.md) | KID 系列的估計量、backbone 選擇、試過但不採用的設計 |
| [condition.md](archive/condition.md) | 線性 probe、GT 鄰居對比、損傷演進的完整說明 |
| [damage_texture.md](archive/damage_texture.md) | 像素分類、裂縫、綜合偏差的完整說明 |
| [temporal.md](archive/temporal.md) | 時序指標與共用雜訊對照 |
| [interference.md](archive/interference.md) | 干擾物偵測器與生成圖的補圖 |
| [oracle.md](archive/oracle.md) | Oracle 版本（GT 的 DINOv3 特徵當條件）與循環偏差 |
| [condition_g.md](archive/condition_g.md) | 遲滯迴圈條件與 GRU 隱狀態 g 的分析 |
| [results.md](archive/results.md) | 2026-10-29 的結果，含 JPEG 對齊前後、gn、base0920、V9、oracle |
| [references.md](archive/references.md) | 參考文獻，依指標分類 |
| [research_log.md](archive/research_log.md) | 原始研究筆記；較早的數字有些已被更正 |

## 程式

| 檔案 | 作用 |
|---|---|
| `src/metrics/dino_mmd.py` | DINOv3 特徵、分組 U-statistic、jackknife、鄰居對比、演進曲線、`clahe_uint8` |
| `src/metrics/interference.py` | 干擾物分割器 |
| `src/metrics/damage_texture.py` | 像素分類、裂縫圖、容忍度 F1 |
| `src/metrics/temporal.py`、`scripts/eval/eval_temporal.py` | 時序指標 |
| `scripts/eval/generate_eval_samples.py` | 生成評估樣本 npz |
| `scripts/eval/inpaint_generated_samples.py` | 偵測並補掉生成圖的干擾物 |
| `scripts/eval/eval_dinov3_kid.py` | KID、鄰居對比、演進 |
| `scripts/eval/eval_damage_texture.py` | 損傷面積與綜合偏差 |
| `scripts/eval/eval_diversity.py` | 抄襲率、spread/skill、同條件內 1 − mask IoU |
| `scripts/eval/run_eval_suite.py`、`scripts/eval/summarize_eval.py` | 一條指令跑完 suite；結果 → `summary.md` |
| `scripts/eval/images_to_npz.py`、`src/metrics/samples_io.py` | 外部模型的圖檔 → 評估用 npz；npz 格式 |
| `scripts/eval/pack_eval_assets.sh` | 打包不在 git 裡的參考集與偵測器 |
| `scripts/viz/export_drift_view_panel_from_images.py` | 把任何模型的圖排成 drift × DI 的雙面 panel，供目視比較 |
