# DeepSeek-OCR 検証手順とレポート

**最終更新**: 2025-10-30
**検証対象**: DeepSeek-OCR (フォーク版)

---

## 目次

1. [検証概要](#検証概要)
2. [システム要件](#システム要件)
3. [検証環境セットアップ](#検証環境セットアップ)
4. [検証手順](#検証手順)
5. [ベンチテスト実行](#ベンチテスト実行)
6. [結果記録](#結果記録)
7. [トラブルシューティング](#トラブルシューティング)

---

## 検証概要

このドキュメントは、DeepSeek-OCRの性能検証手順を段階的に説明します。

### 検証対象

- **リポジトリ**: https://github.com/deepseek-ai/DeepSeek-OCR (フォーク版)
- **モード**: Large, Gundam
- **実装方法**: vLLM, Transformers (HF Pipeline)
- **検証項目**: 精度、処理速度、VRAM使用率

### 検証パラメータ

| パラメータ | Large モード | Gundam モード |
|---|---|---|
| Base Size | 1280 | 1024 |
| Image Size | 1280 | 640 |
| Crop Mode | False | True |
| Min Crops | - | 2 |
| Max Crops | - | 6 |
| 説明 | 1280×1280を1ショット投入 | 640×640タイル複数枚+1024×1024グローバルビュー |
| 用途 | 一般的な文書 | 高密度・高解像度ページ（新聞・図面など） |

### 実装方法の違い

#### vLLM (推奨)
- **要件**: vllm 0.8.5+cu118, transformers>=4.51.1
- **特徴**: バッチ処理、ストリーミング対応、KVキャッシュ・ページドアテンション対応
- **性能**: 高速（Transformersより優位）
- **用途**: 本番環境、高スループット

#### Transformers (HF Pipeline)
- **要件**: transformers>=4.46.3
- **特徴**: シンプルな実装、互換性が高い
- **性能**: KVキャッシュ・ページドアテンションが弱い
- **用途**: 開発、PoC、メモリ制約が少ない環境

---

## システム要件

### GPU インスタンス推奨

| インスタンス | アーキテクチャ | コア | vCPU | ストレージ | GPU     | GPU memory | 料金         |
| ------------ | -------------- | ---- | ---- | ---------- | ------- | ---------- | ------------ |
| g5.xlarge    | x86_64         | 2    | 4    | 250GB      | A10G x1 | 22.4GiB    | 1.006 USD/h  |
| g6.xlarge    | x86_64         | 2    | 4    | 250GB      | L4 x1   | 22.4GiB    | 0.8048 USD/h |

### 推奨スペック

```
- GPU VRAM: 最小 24GB （推奨 24GB以上）
- システムメモリ: 32GB以上
- ストレージ: 100GB以上（モデルキャッシュ用）
- Python: 3.12
- CUDA: 11.8
```

### モデル構成

```
アーキテクチャ:
  ├── Vision Encoder (DeepEncoder, ~300-400M params)
  └── Decoder (DeepSeek-3B-MoE)
      └── 実際の推論時: ~0.5-0.6B params分のエキスパートのみ使用
```

---

## 検証環境セットアップ

### 前提条件

1. **AWS EC2インスタンスの起動**
   ```bash
   # g5.xlarge または g6.xlarge を起動
   # Ubuntu 22.04 または 24.04を使用
   # GPU ドライバが事前インストール済みであることを確認
   ```

2. **Docker と Docker Compose のインストール**
   ```bash
   # Ubuntu での例
   sudo apt-get update
   sudo apt-get install -y docker.io docker-compose
   sudo usermod -aG docker $USER
   newgrp docker
   ```

3. **NVIDIA Docker Runtime のインストール**
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
   sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

### リポジトリのセットアップ

```bash
# リポジトリのクローン
git clone <your-forked-repo-url> DeepSeek-OCR
cd DeepSeek-OCR

# ブランチの確認
git status

# 必要なファイルの配置確認
ls -la docker-compose.yml
ls -la benchmark.py
```

### テストPDFの準備

```bash
# サンプルPDFをアップロード（手動で準備）
# 以下の構造で配置:
# DeepSeek-OCR/
#   ├── test_pdfs/
#   │   └── sample.pdf  （テスト用PDFファイル）
#   └── benchmark_output/  （出力ディレクトリ）

mkdir -p test_pdfs benchmark_output
```

---

## 検証手順

### 手順1: 環境確認

```bash
# GPU が認識されていることを確認
nvidia-smi

# Docker コンテナが起動可能か確認
docker ps

# リポジトリ内容の確認
ls -la DeepSeek-OCR-master/
ls -la DeepSeek-OCR-master/DeepSeek-OCR-vllm/
ls -la DeepSeek-OCR-master/DeepSeek-OCR-hf/
```

### 手順2: Docker コンテナの起動

```bash
cd DeepSeek-OCR

# コンテナの起動
docker-compose up -d

# ログを確認
docker-compose logs -f deepseek-ocr

# コンテナ内でのセットアップを確認
docker-compose exec deepseek-ocr bash
```

### 手順3: ベンチテストスクリプトの実行

```bash
# コンテナ内で実行（またはホストから実行）
python3 benchmark.py

# または Docker コマンドで実行
docker-compose exec deepseek-ocr python3 benchmark.py
```

### 手順4: 結果の確認

```bash
# 出力ファイルの確認
ls -la benchmark_output/

# JSON 結果の確認
cat benchmark_output/benchmark_results_*.json

# CSV 結果の確認
cat benchmark_output/benchmark_results_*.csv

# サマリーレポートの確認
cat benchmark_output/benchmark_summary_*.md
```

---

## ベンチテスト実行

### ベンチテストスクリプト (benchmark.py) の仕様

#### 機能

1. **複数構成での自動テスト**
   - Large & Gundam モードの両モード
   - vLLM & Transformers の両実装
   - 計4パターンの組み合わせ

2. **計測項目**
   - モデル読み込み時間 (秒)
   - 推論時間 平均・標準偏差・最小・最大 (秒)
   - GPUメモリ使用率 (GB)
   - システム情報（CUDA バージョン、GPU デバイス名など）

3. **出力形式**
   - JSON形式: 詳細なメトリクスデータ
   - CSV形式: スプレッドシート用
   - Markdown: サマリーレポート

#### 実行例

```bash
# 基本実行
python3 benchmark.py

# 出力例:
# ================================================================================
# DeepSeek-OCR Benchmark
# ================================================================================
#
# System Information:
#   timestamp: 2025-10-30T12:34:56.789012
#   cuda_available: True
#   cuda_version: 11.8
#   pytorch_version: 2.1.0
#   device_name: NVIDIA A10G
#   device_capability: (8, 6)
#   gpu_memory_total_gb: 22.4
#
# Benchmarking large mode (Large mode (1280×1280, single shot)):
# ============================================================
#
#   [TRANSFORMERS]
#     ✓ Model load time: 2.34s
#     ✓ Inference time (avg): 5.67s ± 0.12s
#     ✓ Peak memory: 18.5GB
#
#   [VLLM]
#     ✓ Model load time: 1.89s
#     ✓ Inference time (avg): 3.45s ± 0.08s
#     ✓ Peak memory: 17.2GB
```

---

## 結果記録

### 結果テーブル (更新対象)

以下のテーブルにベンチテスト結果を記入してください：

#### 評価パラメータ

| インスタンス | モード | 実装方法 | 精度 | 速度 (秒) | VRAM使用率 (%) | 注記 |
|---|---|---|---|---|---|---|
| g5.xlarge | large | vllm | - | - | - | |
| g5.xlarge | large | transformers | - | - | - | |
| g5.xlarge | gundam | vllm | - | - | - | |
| g5.xlarge | gundam | transformers | - | - | - | |
| g6.xlarge | large | vllm | - | - | - | |
| g6.xlarge | large | transformers | - | - | - | |
| g6.xlarge | gundam | vllm | - | - | - | |
| g6.xlarge | gundam | transformers | - | - | - | |

#### 記入手順

1. **速度測定**
   ```bash
   # ベンチテストスクリプトの出力から "inference_time_avg" を参照
   # 単位: 秒
   ```

2. **精度測定**
   ```bash
   # テストPDFを処理して、OCR結果の精度を評価
   # (テキスト抽出の正確性、レイアウト保持度など)
   # 定性的評価: 高/中/低 、または定量的指標
   ```

3. **VRAM使用率**
   ```bash
   # peak_memory_gb / total_gpu_memory_gb × 100
   # または nvidia-smi での実測値
   ```

### 結果の保存

ベンチテスト実行後、自動生成されるファイル:

```
benchmark_output/
├── benchmark_results_YYYYMMDD_HHMMSS.json
├── benchmark_results_YYYYMMDD_HHMMSS.csv
└── benchmark_summary_YYYYMMDD_HHMMSS.md
```

これらのファイルを本ドキュメントに添付またはコピーしてください。

---

## トラブルシューティング

### よくあるエラーと対応

#### 1. CUDA メモリ不足エラー

```
RuntimeError: CUDA out of memory.
```

**原因**: GPU メモリが足りない

**対応**:
```bash
# config.py のパラメータを調整
MAX_CROPS = 4  # 6から4に減らす
MAX_CONCURRENCY = 50  # 100から50に減らす
```

#### 2. vLLM インポートエラー

```
ModuleNotFoundError: No module named 'vllm'
```

**原因**: vLLM がインストールされていない

**対応**:
```bash
pip install vllm==0.8.5 --extra-index-url https://download.pytorch.org/whl/cu118
```

#### 3. Transformers バージョンエラー

```
The model was trained using a newer version of transformers.
```

**原因**: Transformers バージョンが古い

**対応**:
```bash
pip install --upgrade transformers>=4.51.1
```

#### 4. Docker GPU アクセスエラー

```
Could not load dynamic library 'libcuda.so.1'
```

**原因**: NVIDIA Docker Runtime が正しく設定されていない

**対応**:
```bash
# Docker daemon.json を確認
cat /etc/docker/daemon.json

# 以下の設定が含まれていることを確認:
# "runtimes": {
#   "nvidia": {
#     "path": "nvidia-container-runtime",
#     "runtimeArgs": []
#   }
# }

# Docker を再起動
sudo systemctl restart docker
```

#### 5. モデルダウンロードエラー

```
ConnectionError: Failed to download model from Hugging Face Hub
```

**原因**: ネットワーク接続またはHF認証

**対応**:
```bash
# HF トークンを設定
huggingface-cli login

# または環境変数で指定
export HF_TOKEN=your_token_here

# キャッシュディレクトリを確認
ls -la ~/.cache/huggingface/
```

### ログ確認方法

```bash
# Docker コンテナのログ
docker-compose logs -f deepseek-ocr

# 詳細ログ (Python の場合)
python3 -u benchmark.py  # unbuffered output

# GPU ログ
watch -n 1 nvidia-smi
```

---

## 補足情報

### プロンプトテンプレート

公式で推奨されているプロンプト:

```python
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'

# その他の用途別プロンプト:
# 文書化: '<image>\n<|grounding|>Convert the document to markdown.'
# OCR: '<image>\n<|grounding|>OCR this image.'
# フリーOCR: '<image>\nFree OCR.'
# 図形解析: '<image>\nParse the figure.'
# 一般的な説明: '<image>\nDescribe this image in detail.'
# 領域検出: '<image>\nLocate <|ref|>xxxx<|/ref|> in the image.'
```

### リファレンス

- **DeepSeek-OCR 論文**: `DeepSeek_OCR_paper.pdf`
- **公式リポジトリ**: https://github.com/deepseek-ai/DeepSeek-OCR
- **モデル情報**: https://huggingface.co/deepseek-ai/DeepSeek-OCR

---

## チェックリスト

検証実施時の確認項目:

- [ ] インスタンスが起動している
- [ ] GPU ドライバが正しくインストールされている (`nvidia-smi` で確認)
- [ ] Docker & NVIDIA Docker Runtime が正常に動作している
- [ ] リポジトリがクローンされている
- [ ] `docker-compose.yml` が配置されている
- [ ] `benchmark.py` が配置されている
- [ ] テスト用 PDF ファイルが準備されている
- [ ] `benchmark_output` ディレクトリが作成されている
- [ ] Docker コンテナが起動している
- [ ] ベンチテストが実行完了した
- [ ] 結果ファイルが生成されている
- [ ] 結果を本ドキュメントに記録した

---

## まとめ

このドキュメントに従って、DeepSeek-OCR の検証を段階的に実施できます。

**次のステップ:**
1. AWS EC2 インスタンス (g5.xlarge または g6.xlarge) を起動
2. Docker 環境をセットアップ
3. 本リポジトリをクローン
4. `benchmark.py` を実行
5. 結果を記録し、本ドキュメントを更新

問題が発生した場合は、**トラブルシューティング** セクションを参照してください。
