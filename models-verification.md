# Kaggle GPU Space — 四個模型核實（2026-11-09）

等電話驗證嗰陣先做咗嘅準備：逐個模型核實 repo、版本、大細、T4 上跑唔跑得郁。
結論：Qwen3-8B 同 Real-ESRGAN 完全冇問題；Wan 2.2-5B 用 offload 跑得郁但慢；
**Qwen-Image-2.1 全精度 BF16 放唔入 Kaggle，要用量化版**。

## Kaggle 環境硬限制（已核實，2026-11-06／10-31 資料）
- GPU：T4 x2，兩張獨立 16GB 顯存（唔係一張 32GB），4 CPU、29GB RAM
- Session 最長 12 小時，閒置 20 分鐘斷線；每週約 30 GPU 小時
- `/kaggle/working` 儲存上限 **20GB**（session 之後保留）；以外係 scratch，session 完即刪（實測約 1.1TB 空間）
- Private dataset 另外計：每個 dataset 上限 200GB，總共 200GB —— 大權重可以做成 private dataset attach，唔佔 20GB

## 1. Qwen-Image-2.1（出圖＋改圖）
- Repo：`Qwen/Qwen-Image-2.1`（Hugging Face，2026-09-20 發佈）
- 7B visual generation component，32 層 single-stream DiT，BF16
- **全 repo 約 33.1GB**：transformer 約 14.2GB（9.97＋4.26GB 兩 shard）＋ Qwen3-VL 8B text encoder 約 16GB ＋ VAE
- 問題：BF16 全套 33GB ＞ 20GB working 上限；單張 16GB T4 都裝唔晒 transformer＋encoder
- 出路（起 notebook 嗰陣揀）：FP8／GGUF 量化版（已有社群量化 variant；DiT 有 Q8_0／Q6_K／Q4_K_M GGUF，text encoder 有 Q4_K_M＋mmproj），或者 Turbo checkpoint（8 steps，2026-10-09 出）；
  權重放 scratch 即場下載（每次 session 重新下載），或者做 private dataset attach
- 牌照：Qwen Research License（非商業用），Andrew 個人用冇問題
- Diffusers：`QwenImage21Pipeline`，要 `transformers>=5.17`＋最新 diffusers git 版

## 2. Wan 2.2-5B（做短片）
- 用 TI2V-5B（統一 T2V＋I2V）：`Wan-AI/Wan2.2-TI2V-5B-Diffusers`，Apache 2.0
- BF16 約 10.5GB，另加 UMT5-XXL text encoder（11B，BF16 約 22GB —— 可以放 CPU 跑 `t5_cpu`，但都要下載）
- T4 上：480p、短片（5 秒內），diffusers＋`offload_model True`＋`convert_model_dtype` 跑得郁；720p 5 秒片喺 4090 都要約 9 分鐘，T4 會更慢，要預時間
- 輕量出路：GGUF Q4_K_M 全套約 6.1GB（transformer＋encoder＋VAE），但嗰條路係 ComfyUI／GGUF loader，唔係 diffusers；先試 diffusers offload，唔得先轉 GGUF
- 唔好搞錯：冇 1.3B Wan 2.2（嗰個係 Wan 2.1）；A14B 係 27B MoE，唔啱

## 3. Qwen3-8B（本地大模型對話）
- 官方 AWQ 4-bit：`Qwen/Qwen3-8B-AWQ`，約 5GB，Apache 2.0
- 全精度 BF16 係 15.26GB，AWQ 版 4.7GB 左右，16GB T4 輕鬆跑，連 8K context 都夠位
- 用 transformers＋autoawq 或者 vLLM；唔使 GGUF

## 4. Real-ESRGAN（放大修復舊圖到 4x）
- 權重 `RealESRGAN_x4plus.pth`，約 64MB（16.7M 參數），官方 GitHub releases v0.1.0
- `https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth`
- `pip install realesrgan`＋權重就得，CPU 都跑得郁，完全冇大細問題

## 起 notebook 嗰陣嘅計劃
1. 電話驗證完成 → 開 notebook，accelerator 揀 GPU T4 x2，internet ON
2. 四個模型分開做 Gradio tab；權重下載去 scratch（唔使留喺 /kaggle/working），或者一次過做 private dataset 長用
3. Qwen-Image-2.1 先試 FP8／量化版；Wan 2.2 先試 diffusers offload 480p
4. 每個 tab 逐個實測出到嘢先交貨

## Notebook 已寫好（2026-11-09）
- 檔案：`~/workspace/kaggle-gpu-space/kaggle-gpu-space.ipynb`（13 格：環境檢查、安裝、工具、四個模型 loader、四個煙霧測試、Gradio）
- 出圖用 `unsloth/Qwen-Image-2.1-FP8`（transformer FP8 7.12GB＋text encoder FP8 9.39GB＋VAE，合共約 17GB；`torch_dtype=torch.float8_e4m3fn`＋`device_map='balanced'`；失敗 fallback 官方 BF16＋CPU offload）
- 短片用 `Wan-AI/Wan2.2-TI2V-5B-Diffusers`＋`WanPipeline`＋`enable_model_cpu_offload()`，480p、33 格、25 steps（官方例用 `pipe.to(device)`，T4 16GB 一定要改 offload）
- 對話用 `Qwen/Qwen3-8B-AWQ`（transformers＋autoawq，`enable_thinking=False`）
- 放大用 `RealESRGAN_x4plus.pth`（GitHub releases v0.1.0 直鏈，64MB）
- 模型逐個 lazy load，用 `_LOADED`＋`unload_all()` 管理顯存；Gradio `share=True` 出公開連結
- **2026-11-09 11:43 監察發現**：煙霧測試 1（Qwen3-8B-AWQ）兩次 Run All 都失敗，`ImportError: Loading an AWQ quantized model requires gptqmodel`——新版 transformers 讀 AWQ 要 gptqmodel，唔係 autoawq。已將 `gptqmodel` 加落安裝格，喺原有 notebook 改返嗰格再重跑（唔使重新 import 成個 notebook）。

## v3 修訂（2026-11-09 12:30，監察＋實查後）
- **監察睇錯 run**：12:19 嗰轉監察睇到嘅係 Kaggle import 嗰陣自動跑嘅 CPU 版（未掛 GPU，299.7 秒），唔係真正嗰個 GPU draft session（嗰個仲跑緊，cell 2a 裝緊套件）。以後監察要睇 draft session（editor /edit/run/...），唔好睇 import 自動出嘅 version run。
- **真正問題（CPU 版安裝 log 揭露，GPU 版用同一套安裝行一樣會中）**：pip 將 torch 由 2.11.0+cu130 升到 2.15.0（torchao 最新版／diffusers git main 逼升），搞爛三樣嘢：(1) gptqmodel 7.10.0 喺 torch 2.15 讀唔到 `_chunk_or_narrow_cat`；(2) diffusers git main＋transformers 5.20.2 唔夾，`Config() got an unexpected keyword argument 'deprecated'`（diffusers 同 torchao import 都係呢個錯）；(3) transformers 5.20.2 話 GPTQ device-map API 變咗。
- **v3 修正**：版本全部釘死——`diffusers==0.41.0`（2026-10-06 發佈，第一個有 Qwen-Image 2.1 嘅正式版）＋`transformers==5.17.0`（Qwen 官方文件指定嘅配對）；**唔裝 torchao**（筆記本程式碼根本冇用到佢，淨係佢會逼升 torch）；安裝後印 torch 版本驗證留喺 2.11。
- **basicsr／realesrgan 喺 Kaggle Python 3.13 裝唔到**：`--no-build-isolation` 都失敗（Preparing metadata 失敗，冇輸出），係 basicsr 1.4.2 舊 setup.py 同 Python 3.13 唔夾。v3 改用 notebook 自帶 RRDBNet 程式碼（同 basicsr 嗰個架構一樣，權重檔 RealESRGAN_x4plus.pth 直接 torch.load），唔使再裝呢兩個套件。

## v4 修訂（2026-11-09 14:27，安裝格卡死 113 分鐘後）
- v3 嘅安裝格喺 `Building wheel for autoawq ... done` 之後靜咗 113 分鐘，實查確認唔係畫面問題：pip 喺度靜靜 build gptqmodel 嘅 wheel（呢個環境冇預製 wheel，要由源碼編 CUDA，唔知幾時先完）。
- v4 決定：對話唔用 AWQ，改用 **bitsandbytes 4-bit（NF4）＋Qwen/Qwen3-8B 全精度**（下載約 15GB，顯存佔約 5GB）；autoawq／gptqmodel 一齊唔裝。安裝格而家全部係預製 wheel，唔使再等源碼 build。diffusers 0.41.0＋transformers 5.17.0 維持不變。
- 監察教訓：monitor 每次都要碌到安裝格輸出最底睇最後一行，唔好淨睇可見嗰行；v3 嗰轉 12:36 報過嘅「torch: 2.11.0+cpu」係 CPU 自動版嗰邊嘅輸出，唔係 draft session 嘅，實查已更正。
