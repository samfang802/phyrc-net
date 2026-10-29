# 時序生成

一個速度網路。單幀和時序是同一條路徑的兩端，用 `s` 分開。真實照片永遠是終點。生成圖只當起點。推論時若像素飽和，用凍結的單幀模型重畫，不在壞圖上加一個小速度。

這份接在 `PAIRED_BRIDGE.md` 和 `FLOWEDIT_TEMPORAL.md` 後面。步數 11135 的 corrector 試跑已經做過，結論寫在推論那一節。

## 最終版 

條件仍是這一幀的 g、方向、Patch 的 XY、鋼筋遮罩。DI、柱編號、靜態設計參數不進生成器。M 不當「哪裡可以改」的開關。

記真實這一幀為 `x`，起點圖為 `x_src`。

```text
s  大部分在 0 附近，一條尾巴到 1
x0 = (1 - s) * x_src + s * ε
z_t = (1 - t) * x0 + t * x
v   = x - x0
```

`t = 0` 是起點，`t = 1` 是真實這一幀。`s = 0` 是從上一幀走。`s = 1` 時 `x0` 是高斯，路徑和單幀 OT 相同。

`x_src` 大多數是真實上一幀。較小的機率改成凍結單幀 G 畫的圖，這時 `s` 用小的。沒有上一幀的樣本直接 `s = 1`。

每個樣本的速度 MSE 除以該樣本目標速度的均方（梯度裡停掉這個除數）。`s → 1` 的目標尺度大約是 1，`s → 0` 的目標是兩張照片的差。不除的話，尾巴幾個百分點就占掉大部分梯度，小速度學不動。

推論：

1. 第一幀用凍結的單幀權重 `chjqbp9m` 從噪聲畫。聯合模型的 `s = 1` 尾巴還沒畫出同樣的灰格子之前，不換。
2. 後面每一幀先用橋，`s = 0`，從上一幀的輸出積分到這一幀的 g。
3. 飽和才把 `s` 調高：貼到黑或白的像素變多，或三個通道分開。不用整張平均亮度。高 DI 的真實剝落本來就更暗。
4. 調高時一次進到訓練看過的那一段，並用凍結的單幀模型重畫。不要只調到 0.2，壞圖還在起點裡。
5. 格子用第一幀的高頻輕輕混回來，不是整段換成第一幀。step 11135 把高頻全部換掉之後，黑白塊沒了，剝落變成糊的低頻暗塊。

六十步的小誤差仍會累加。這個版本沒有一項力把裂縫收成真實的那一條。它做的是：起點落在訓練看過的圖上，飽和時整段重畫，格子不要每幀重抽。

## 訓練細節

同一根柱、同一個方向，在該分割裡按時間排。這一幀是終點。分割裡它前面最近的一張是真實上一幀。中間若有幀在別的分割，這一對會跳過。

凍結單幀圖先用 `chjqbp9m` 離線畫好存下來，不在每個 step 裡跑 ODE。它只替換 `x_src`。終點仍是真實這一幀。畫的是上一幀那個時刻的 g，讓起點像推論時「單幀模型交出的上一張」，而不是這一幀的另一個樣本。

`s` 的抽法：大部分質量在 `[0, 0.3]`，其餘均勻到 1。推論的主路徑是 `s = 0`，所以質量留在小的一端。尾巴的工作是讓同一個損失裡還有單幀路徑。

損失：

```text
L = mean( ||v_pred - v||^2 / stopgrad(mean(v^2)) )
```

分母按樣本算。min-SNR 是為「從高斯走到圖」設計的，兩端都會被壓低。混了大小兩種速度之後，先用這個除法把每個樣本的貢獻拉平，再決定要不要留 min-SNR。

g 的 10% dropout 若打開，只作用在 G，不作用在別的頭。現在這條沒有 M。

## 已經跑過的 corrector

step 11135 的橋權重，高 DI 6×6，test，CTR1。每一幀橋走完後以 `s = 0.35` 交回 `chjqbp9m`，高頻全部換成第一幀。

圖：`results/DamagePatternGeneration/CFMADM/2026-10-07_18-22-12_paired_bridge/inference_samples/overview_di_3_high_6x6_selfcondOFF_step_11135_ode30_bridge_corr0p35_mg.png`

六列平均灰階約 142–151，通道差約 3.7，沒有彩邊。格子還在，底部暗區會隨時刻變寬。暗區是糊的，因為第一幀的高頻把 corrector 的裂縫蓋掉了。所以最終版的錨是輕混，不是替換。

## 論文

### 兩端都是圖

**BBDM**（Bo Li、Kaitao Xue、Bin Liu、Yu-Kun Lai，CVPR 2023，arXiv:2205.07680）。圖像翻譯做成布朗橋，兩端都是圖，中間不落到標準高斯。前向是 `(1-m) x + m y` 再加方差 `δ = 2 m (1-m)`，中點方差 0.5。他們在 VQGAN latent 裡做，用來翻譯差很遠的兩個域。

取兩端釘住。不取中點大方差，也不進 VQGAN。前後幀是同一根柱的照片，中點再加一大團噪聲，舊裂縫會被洗掉。

**I²SB**（Guan-Horng Liu、Arash Vahdat、De-An Huang、Evangelos A. Theodorou、Weili Nie、Anima Anandkumar，ICML 2023，arXiv:2302.05872）。成對圖之間的 Schrödinger bridge。邊際在邊界對給定時有解析形式，訓練不必先把整條橋模擬出來。論文做修復：退化圖裡的結構留在橋上。

取「不要先洗成高斯再重畫」。不取他們的分數模型和修復任務的實作。速度仍是一個向量。

### 噪聲等級把單幀和時序放在一起

**Diffusion Forcing**（Boyuan Chen、Diego Martí Monsó、Yilun Du、Max Simchowitz、Russ Tedrake、Vincent Sitzmann，NeurIPS 2024）。每個 token 獨立抽一個噪聲等級，再一起去噪。噪聲是部分遮住：0 是乾淨歷史，滿噪聲是這幀等於沒有歷史。因果模型因此可以在不把過去全部打成噪聲的情況下預測下一幀。他們能把序列滾過訓練長度，別的方法在那裡會散。噪聲方案他們寫明是任務相關的，時間上很相關的資料若每幀都獨立滿噪聲，訊噪比會太低。

`s` 的尾巴就是這個噪聲等級。質量留在小 `s`，因為相鄰兩幀很像，不是每幀都該重畫。

**History-Guided Video Diffusion**（Song 等人，arXiv:2502.06764）。把 Diffusion Forcing 的獨立噪聲等級接到非因果的視頻 Transformer。歷史和要生成的幀是同一條輸入，微調時每幀一個噪聲等級，抽樣時可以把一段歷史再遮住。條件引導的無條件分支就是歷史全部加噪。

取「歷史的噪聲等級可以在抽樣時再調」。推論裡把飽和的上一幀調高 `s`，是這件事。重畫在尾巴學穩之前仍用凍結單幀權重，不把還沒學好的聯合網路當成這個引導器。

**Rolling Diffusion**（David Ruhe、Jonathan Heek、Tim Salimans、Emiel Hoogeboom，ICML 2024，arXiv:2402.09470）。滑窗裡越後面的幀噪聲越大，去噪時第一幀先乾淨，後面仍帶噪聲，再把窗子往後移。Kinetics-600 上，時間變化劇烈、一次跳很多幀時比普通擴散好；變化慢、序列短時兩者差不多。

不取這條固定滑窗。我們相鄰幀差很小，獨立抽一個 `s` 就夠。

**EDM**（Tero Karras、Miika Aittala、Timo Aila、Samuli Laine，NeurIPS 2022，arXiv:2206.00364）。他們把不同噪聲等級的損失權重拉開：有效權重是 `λ(σ) c_out(σ)^2`，設 `λ = 1 / c_out^2`，讓各等級一開始的損失量級接近。訓練時的噪聲分布則集中在真正影響畫面的中段，不是均勻鋪滿。

取「不同尺度的目標不要直接平均」。我們不搬他們的 precondition。做法是每個樣本除以自己目標速度的均方。

### 生成圖只當起點

**DAgger**（Stéphane Ross、Geoffrey Gordon、J. Andrew Bagnell，AISTATS 2011）。模仿學習裡，訓練只看專家走過的狀態，測試卻走在自己的狀態上，誤差會沿時間累加。DAgger 用目前的策略去採狀態，專家仍給這些狀態上的動作，再和舊資料合在一起訓。

凍結單幀圖換掉 `x_src`、終點仍是真實這一幀，就是這一手。專家是真實照片。不在 15000 step 裡用橋自己滾出狀態，那要每步跑整段 ODE。

**Self Forcing**（Xun Huang、Zhengqi Li、Guande He、Mingyuan Zhou、Eli Shechtman，NeurIPS 2025，arXiv:2506.08009）。自回歸視頻擴散在訓練時就按推論的方式滾：每一幀條件是自己已經生成的幀，再用整段影片的分布匹配當損失。他們用少步模型和梯度截斷，避免每幀都回傳完整去噪的梯度。論文的訓練是後訓練｀，幾十到幾百步，不是從頭的長循環。

這是把 DAgger 做到「橋自己走出的壞圖」上。最終版不放進 15000 step。飽和圖若以後仍對不上，再考慮一小段這種後訓練。

**Self-Consuming Generative Models Go MAD**（Sina Alemohammad 等人，ICLR 2024，arXiv:2307.01850）。生成圖再當成下一代的訓練目標，品質或多樣性會逐代下降。他們叫 Model Autophagy Disorder。真實資料若不夠新鮮，合成資料再多也補不回來。少量合成資料混進真實資料有時無妨，合成資料變成主體就會壞。

所以單幀圖只當 `x_src`。終點永遠是真實這一幀。

### 把偏掉的圖拉回來

**Score-Based Generative Modeling through SDEs**（Yang Song、Jascha Sohl-Dickstein、Diederik P. Kingma、Abhishek Kumar、Stefano Ermon、Ben Poole，ICLR 2021）。前向 SDE 把資料打成噪聲，反向 SDE 靠分數（對數密度的梯度）走回來。Predictor 先按 SDE 或 ODE 走一步，corrector 再用 Langevin 把樣本拉回這個時刻的邊際。他們寫明 corrector 是在修離散積分累積的誤差。純 ODE 和 SDE 的邊際可以相同，但 ODE 沒有這個拉回。

我們的橋目前是純 ODE。速度指錯時，像素留在軌跡上。corrector 的離散版是：把圖加回一部分噪聲，再用一個會走向柱子分布的模型走到 `t = 1`。那個模型在尾巴學穩前是 `chjqbp9m`。

**Stochastic Interpolants**（Michael S. Albergo、Nicholas M. Boffi、Eric Vanden-Eijnden，JMLR 2025，arXiv:2303.08797）。同一條插值可以寫成 ODE，也可以寫成帶可調擴散係數的 SDE。速度是條件期望，分數是另一個二次損失。他們證明：回歸漂移就控得住 SDE 的似然；ODE 還得把 Fisher divergence 壓下去。插值在優化耦合時可以收成兩個分布之間的 Schrödinger bridge。

取「只有速度的 ODE 沒有回復力」。不另訓一個分數頭。加噪再去噪就是在用單幀模型當這個分數的替身。

**FreeDoM**（Jiwen Yu、Yinhuai Wang、Chen Zhao、Bernard Ghanem、Jian Zhang，ICCV 2023）。不重訓。用一個和時間無關的能量衡量「現在這張圖離條件多遠」，採樣時把能量的梯度加進更新。能量可以來自現成網路。

這裡的能量很窄：三個通道不要分開，貼到 0 或 1 的像素不要變多。不用整張平均亮度，避免把真實的暗剝落拉亮。

**StreamingT2V**（Roberto Henschel 等人，CVPR 2025，arXiv:2403.14773）。自回歸長影片會忘掉第一段。短記憶是前一個片段的注意力。長記憶是 appearance preservation：把第一段的特徵一直注進後面的交叉注意力，場景和物件身份才留得住。

格子不隨時間移動。第一幀的高頻就是這個長記憶。step 11135 證明全部替換會把裂縫抹成低頻暗塊，所以只輕混。

**DPS**（Hyungjin Chung、Jeongsol Kim、Michael T. Mccann、Marc L. Klasky、Jong Chul Ye，ICLR 2023）。擴散後驗採樣。每步用乾淨圖的估計去算觀測誤差的梯度，把樣本拉向滿足 `A(x) = y` 的地方。終點未知時，這是布朗橋那種 `(終點 - 現在) / 剩餘時間` 的替代，但必須有觀測算子。

我們沒有像素級的觀測。g 不是一張圖。這條不用。

### 看過、不放進最終版

**FlowEdit**（Kulikov、Kleiner、Huberman-Spiegelglas、Michaeli，ICCV 2025，arXiv:2412.08629）。同一張噪聲上，兩個條件的速度相減，乾淨圖只加這個差。背景對得上就不動。我們的裂縫沒有固定位置，相減把對不齊的筆畫平均成糊影。高 DI 尤其明顯。

**Delta Denoising Score**（Hertz、Aberman、Cohen-Or，ICCV 2023）。擴散模型上用兩次分數的差做編輯。和 FlowEdit 同一類，我們的模型是 flow。

**SDEdit**（Meng 等人，ICLR 2022）。上一幀加噪再積分。噪聲不夠時，壞圖還在起點裡；噪聲太大時，每幀重畫。最終版的 corrector 是它的形狀，但只在飽和時開，而且重畫用的是凍結單幀模型。

**Differential Diffusion**（Levin、Fried）。每格不同的編輯強度。強度若來自現在的 M，M 的 sigmoid 大多過不了 0.5，序列會停住。

**MasaCtrl**（Mingdeng Cao 等人，ICCV 2023，arXiv:2304.08465）。這一幀出 query，上一幀出 key 和 value，把紋理鎖在來源圖上。他們用在去噪中後段和解碼器的自注意力。我們的自注意力只在最底層，鎖不住裂縫。

**TokenFlow**（Michal Geyer、Omer Bar-Tal、Shai Bagon、Tali Dekel，ICLR 2024，arXiv:2307.10373）。先有一支影片，用它的特徵對應把編輯後的 token 送到各幀。我們是在生成這支影片，沒有一支已經對齊的原片。

**Pyramidal Flow Matching**（Jin 等人，ICLR 2025）。整個視頻模型重訓。這一步不上。

**RePaint**（Lugmayr 等人，CVPR 2022）和 **Restora-Flow**（Hadzic 等人，WACV 2026）。硬遮罩把已知區貼回去，再用重採樣或額外前向拉回流形。我們沒有可靠的硬遮罩。corrector 用的是加噪再去噪，不是貼遮罩。

**BAgger**（Po 等人，CVPR 2026）。先讓模型走出長序列，再拿來當修正樣本。每一輪都是完整 rollout。和 Self Forcing 一樣，不放進 15000 step。