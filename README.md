# AI Text Detector

Phát hiện văn bản do AI tạo ra vs con người viết.  
Kiến trúc: **RoBERTa / DeBERTa / DistilRoBERTa + LoRA + DANN (Domain-Adversarial Neural Network)**.

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Chạy trên Google Colab](#2-chạy-trên-google-colab)
3. [Chạy trên Kaggle](#3-chạy-trên-kaggle)
4. [Chuẩn bị dữ liệu](#4-chuẩn-bị-dữ-liệu)
5. [Cách sử dụng](#5-cách-sử-dụng)
6. [Điều chỉnh khi GPU yếu](#6-điều-chỉnh-khi-gpu-yếu)
7. [Cấu trúc thư mục](#7-cấu-trúc-thư-mục)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Tổng quan kiến trúc

```
Văn bản đầu vào
      │
      ▼
 Tokenizer (RoBERTa / DeBERTa / DistilRoBERTa)
      │
      ▼
 Transformer Encoder + LoRA
      │
   [CLS] embedding
      │
  ┌───┴───────────────────────────┐
  ▼                               ▼
class_head                  GRL → domain_head
Human / AI (nhãn chính)     Source / Target domain
```

- **LoRA**: Chỉ fine-tune ~2M / 125M tham số → tiết kiệm VRAM, training nhanh hơn.
- **DANN + GRL**: Buộc encoder học đặc trưng không phụ thuộc domain → generalize tốt hơn trên tập test.
- **Metric chính**: FPR @ TPR=95% ↓ — tỉ lệ nhầm văn bản Human thành AI khi bắt đúng 95% văn bản AI.

---

## 2. Chạy trên Google Colab

### Yêu cầu
- **GPU runtime**: T4 (miễn phí) hoặc A100 (Colab Pro).  
  Vào `Runtime → Change runtime type → GPU`.

### Bước 1 — Clone repo và cài thư viện

```python
# Clone project
!git clone https://github.com/NguyenTran2312/AI-text-detector.git
%cd AI-text-detector

# Cài tất cả thư viện
!pip install -q -r requirements.txt
```

> **Lưu ý T4 (15 GB VRAM):** Sau khi cài xong, bạn cần giảm batch size.  
> Xem [Mục 6 — Điều chỉnh khi GPU yếu](#6-điều-chỉnh-khi-gpu-yếu).

### Bước 2 — Mount Google Drive (để lưu checkpoint và kết quả)

```python
from google.colab import drive
drive.mount('/content/drive')
```

Tạo thư mục output trên Drive:

```python
import os
os.makedirs('/content/drive/MyDrive/ai-detector/checkpoints', exist_ok=True)
os.makedirs('/content/drive/MyDrive/ai-detector/plots',       exist_ok=True)
os.makedirs('/content/drive/MyDrive/ai-detector/submissions', exist_ok=True)
```

### Bước 3 — Cập nhật đường dẫn trong config

```python
# Mở file configs/config.py và sửa các dòng sau:
config_patch = """
TRAIN_PATH          = '/content/drive/MyDrive/data/cleaned_text_data.jsonl'
DEV_PATH            = '/content/drive/MyDrive/data/subtaskA_dev_monolingual.jsonl'
TEST_LABELED_PATH   = '/content/drive/MyDrive/data/subtaskA_monolingual_labeled.jsonl'
TEST_UNLABELED_PATH = '/content/drive/MyDrive/data/subtaskA_monolingual_unlabeled.jsonl'

CKPT_DIR  = '/content/drive/MyDrive/ai-detector/checkpoints'
PLOT_DIR  = '/content/drive/MyDrive/ai-detector/plots'
SUB_DIR   = '/content/drive/MyDrive/ai-detector/submissions'
"""
print("Sao chép các dòng trên vào configs/config.py")
```

Hoặc patch trực tiếp bằng code (tiện hơn khi dùng notebook):

```python
import sys
sys.path.insert(0, '/content/AI-text-detector')

from configs.config import CFG

CFG.TRAIN_PATH          = '/content/drive/MyDrive/data/cleaned_text_data.jsonl'
CFG.DEV_PATH            = '/content/drive/MyDrive/data/subtaskA_dev_monolingual.jsonl'
CFG.TEST_LABELED_PATH   = '/content/drive/MyDrive/data/subtaskA_monolingual_labeled.jsonl'
CFG.TEST_UNLABELED_PATH = '/content/drive/MyDrive/data/subtaskA_monolingual_unlabeled.jsonl'
CFG.CKPT_DIR            = '/content/drive/MyDrive/ai-detector/checkpoints'
CFG.PLOT_DIR            = '/content/drive/MyDrive/ai-detector/plots'
CFG.SUB_DIR             = '/content/drive/MyDrive/ai-detector/submissions'

# Giảm batch size cho T4
CFG.BATCH_SIZE  = 32
CFG.ACCUM_STEPS = 8     # effective batch = 32 × 8 = 256 (giữ nguyên)
CFG.NUM_WORKERS = 2
```

### Bước 4 — Đăng nhập W&B

```python
import wandb
wandb.login()   # Dán API key từ https://wandb.ai/authorize
```

Nếu không muốn dùng W&B:

```python
import os
os.environ["WANDB_MODE"] = "disabled"
```

### Bước 5 — Chạy

**Chạy toàn bộ 12 ablation runs:**

```python
from ablation import run_all
results = run_all()
```

**Hoặc chỉ chạy 1 run để thử nhanh:**

```python
from ablation import ABLATION_RUNS, run_single, seed_everything
from src.dataset import build_data, make_loaders
from transformers import AutoTokenizer

run_cfg       = ABLATION_RUNS[0]   # run1_baseline (RoBERTa, không DANN)
CFG.MODEL_NAME = run_cfg.model_name

tokenizer = AutoTokenizer.from_pretrained(run_cfg.model_name)
train_ds, val_ds, target_ds, dev_ds, test_ds, submit_ds = build_data(tokenizer)

result = run_single(
    run_cfg=run_cfg,
    train_ds=train_ds, val_ds=val_ds, target_ds=target_ds,
    dev_ds=dev_ds, test_ds=test_ds, submit_ds=submit_ds,
    tokenizer=tokenizer,
)
print(f"FPR @ TPR=95% : {result['fpr_at_95tpr']:.4f}")
print(f"F1 Macro      : {result['test_f1']:.4f}")
print(f"AUC-ROC       : {result['test_auc']:.4f}")
```

### Bước 6 — (Tùy chọn) Chạy EDA và tăng cường dữ liệu

```python
# EDA + baseline TF-IDF (chạy trước khi train để hiểu dữ liệu)
!python notebooks/eda.py

# Tăng cường dữ liệu từ PeerRead + Groq API
# Lấy API key miễn phí tại: https://console.groq.com → API Keys
import os
os.environ["GROQ_API_KEY"] = "gsk_xxxxxxxxxxxx"   # Thay bằng key thật

!python notebooks/data_augmentation.py
```

### Tránh mất tiến trình khi Colab ngắt kết nối

Colab T4 miễn phí giới hạn ~4–6 giờ liên tục. Để không mất công:

```python
# Checkpoint được tự động lưu lên Drive sau mỗi epoch.
# Khi session bị ngắt, chạy lại từ đầu — script sẽ tự detect
# file checkpoint "final" và bỏ qua run đã hoàn thành.

# Kiểm tra run nào đã xong:
from src.train import ckpt_path
import os

for run in ABLATION_RUNS:
    done = os.path.exists(ckpt_path(run.run_id, "final"))
    print(f"  {'✅' if done else '⬜'} {run.run_id}")
```

---

## 3. Chạy trên Kaggle

### Yêu cầu
- Bật GPU P100 (miễn phí) hoặc T4×2: `Settings → Accelerator → GPU`.
- Kaggle cho ~30 giờ GPU/tuần — đủ để chạy đầy đủ ablation study.

### Bước 1 — Thêm dataset dữ liệu

1. Tải file dữ liệu `.jsonl` lên Kaggle Datasets (hoặc dùng dataset public của cuộc thi).
2. Trong notebook: `+ Add Data` → tìm dataset → Add.
3. Dữ liệu sẽ xuất hiện tại `/kaggle/input/{dataset-name}/`.

### Bước 2 — Clone repo trong notebook

```python
!git clone https://github.com/NguyenTran2312/AI-text-detector.git /kaggle/working/AI-text-detector
%cd /kaggle/working/AI-text-detector

!pip install -q peft wandb polars lightgbm groq
# torch, transformers, sklearn đã có sẵn trên Kaggle
```

### Bước 3 — Patch config

```python
import sys
sys.path.insert(0, '/kaggle/working/AI-text-detector')

from configs.config import CFG

# Trỏ đến dataset đã add (thay {dataset-name} bằng tên thật)
CFG.TRAIN_PATH          = '/kaggle/input/{dataset-name}/cleaned_text_data.jsonl'
CFG.DEV_PATH            = '/kaggle/input/{dataset-name}/subtaskA_dev_monolingual.jsonl'
CFG.TEST_LABELED_PATH   = '/kaggle/input/{dataset-name}/subtaskA_monolingual_labeled.jsonl'
CFG.TEST_UNLABELED_PATH = '/kaggle/input/{dataset-name}/subtaskA_monolingual_unlabeled.jsonl'

# Output lưu trong /kaggle/working (được tự động lưu khi commit notebook)
CFG.CKPT_DIR = '/kaggle/working/checkpoints'
CFG.PLOT_DIR = '/kaggle/working/plots'
CFG.SUB_DIR  = '/kaggle/working/submissions'

import os
for d in [CFG.CKPT_DIR, CFG.PLOT_DIR, CFG.SUB_DIR]:
    os.makedirs(d, exist_ok=True)

# P100 có 16 GB VRAM — batch size 64 là an toàn
CFG.BATCH_SIZE  = 64
CFG.ACCUM_STEPS = 4    # effective batch = 64 × 4 = 256
CFG.NUM_WORKERS = 4
```

### Bước 4 — Thêm W&B API key qua Kaggle Secrets

1. Vào **Add-ons → Secrets** trong notebook.
2. Thêm secret tên `WANDB_API_KEY` với giá trị là API key từ [wandb.ai](https://wandb.ai/authorize).
3. Bật toggle **"Attach to notebook"**.

```python
from kaggle_secrets import UserSecretsClient
import wandb, os

secrets = UserSecretsClient()
os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
wandb.login(key=os.environ["WANDB_API_KEY"])
```

Nếu không muốn dùng W&B:

```python
os.environ["WANDB_MODE"] = "disabled"
```

### Bước 5 — Chạy

```python
from ablation import run_all
results = run_all()
```

### Bước 6 — Lưu file submission để nộp

Sau khi chạy xong, file CSV xuất hiện tại `/kaggle/working/submissions/`.  
Kaggle tự động lưu toàn bộ `/kaggle/working/` khi bạn **Save & Run All (Commit)**.

```python
# Xem file submission đã tạo
import os, glob
subs = glob.glob('/kaggle/working/submissions/*.csv')
for f in subs:
    print(f)

# Đọc kiểm tra nhanh
import pandas as pd
df_sub = pd.read_csv(subs[0])
print(df_sub.head())
print(f"Tổng predictions: {len(df_sub):,}")
print(df_sub['label'].value_counts())
```

### Dùng Groq API trên Kaggle (cho data_augmentation.py)

```python
secrets = UserSecretsClient()
os.environ["GROQ_API_KEY"] = secrets.get_secret("GROQ_API_KEY")

!python notebooks/data_augmentation.py
```

---

## 4. Chuẩn bị dữ liệu

### Cấu trúc file JSONL

Mỗi dòng là một JSON object:

```json
{"id": "abc123", "text": "Nội dung văn bản...", "label": 0, "source": "wikipedia", "model": "human"}
{"id": "def456", "text": "Nội dung văn bản...", "label": 1, "source": "reddit",    "model": "gpt-4"}
```

| Trường | Ý nghĩa |
|---|---|
| `text` | Nội dung văn bản cần phân loại |
| `label` | `0` = Human viết, `1` = AI tạo ra |
| `source` | Domain nguồn (wikipedia, reddit, arxiv, ...) |
| `model` | Tên model AI hoặc `"human"` |

### Tạo dữ liệu training từ đầu (tùy chọn)

Nếu chưa có `cleaned_text_data.jsonl`, chạy pipeline sau:

```python
# Bước 1: EDA + làm sạch dữ liệu gốc → tạo cleaned_text_data.jsonl
!python notebooks/eda.py

# Bước 2: Tăng cường thêm từ PeerRead + Groq (tùy chọn)
!python notebooks/data_augmentation.py
```

---

## 5. Cách sử dụng

### Chạy toàn bộ 12 ablation runs

```python
from ablation import run_all
results = run_all()
```

12 runs được nhóm theo backbone:

| Backbone | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---|---|---|---|
| RoBERTa-base | Baseline | DANN | DANN lr=1e-5 | DANN dropout=0.5 |
| DeBERTa-v3-base | Baseline | DANN | DANN lr=1e-5 | DANN dropout=0.5 |
| DistilRoBERTa-base | Baseline | DANN | DANN lr=1e-5 | DANN dropout=0.5 |

### Chạy một run cụ thể

```python
from ablation import ABLATION_RUNS

# Xem tất cả run
for i, r in enumerate(ABLATION_RUNS):
    print(f"[{i}] {r.run_id} — {r.description}")
```

```python
# Chọn run theo index và chạy
from ablation import run_single
from src.dataset import build_data
from transformers import AutoTokenizer
from configs.config import CFG

run_cfg        = ABLATION_RUNS[4]     # Ví dụ: run1_deberta_base
CFG.MODEL_NAME = run_cfg.model_name

tokenizer = AutoTokenizer.from_pretrained(run_cfg.model_name)
train_ds, val_ds, target_ds, dev_ds, test_ds, submit_ds = build_data(tokenizer)

result = run_single(
    run_cfg=run_cfg,
    train_ds=train_ds, val_ds=val_ds, target_ds=target_ds,
    dev_ds=dev_ds, test_ds=test_ds, submit_ds=submit_ds,
    tokenizer=tokenizer,
)
```

### Chỉ tính lại threshold (không train lại)

Khi đã có checkpoint, dùng `is_threshold_only=True` để chạy nhanh:

```python
result = run_single(
    run_cfg=run_cfg,
    ...,
    is_threshold_only=True,
)
```

### Xem kết quả sau khi chạy

```python
import json, pandas as pd

# Đọc kết quả đã lưu
with open(f"{CFG.PLOT_DIR}/ablation_results.json") as f:
    results = json.load(f)

df = pd.DataFrame(results)[["run_id", "use_dann", "lr", "fpr_at_95tpr", "test_f1", "test_auc"]]
df = df.sort_values("fpr_at_95tpr")   # Sắp xếp theo metric chính
print(df.to_string(index=False))
```

---

## 6. Điều chỉnh khi GPU yếu

Bảng tham chiếu theo VRAM:

| GPU | VRAM | BATCH_SIZE | ACCUM_STEPS | Ghi chú |
|---|---|---|---|---|
| A100 | 40 GB | 128 | 2 | Mặc định trong config |
| V100 / A10 | 16–24 GB | 64 | 4 | Colab Pro / Kaggle P100 |
| T4 | 15 GB | 32 | 8 | Colab free |
| RTX 3070 / 4060 | 8 GB | 16 | 16 | Local consumer GPU |

Cách áp dụng (patch trực tiếp trong notebook):

```python
from configs.config import CFG

# Ví dụ cho T4
CFG.BATCH_SIZE  = 32
CFG.ACCUM_STEPS = 8     # Giữ nguyên effective batch = BATCH_SIZE × ACCUM_STEPS = 256
CFG.NUM_WORKERS = 2     # T4 trên Colab thường chỉ có 2 CPU
CFG.MAX_LEN     = 128   # Giảm thêm nếu vẫn OOM (mặc định 256)
```

Nếu vẫn OOM với backbone lớn (DeBERTa), thử chạy DistilRoBERTa trước:

```python
# DistilRoBERTa nhỏ hơn RoBERTa ~40%, phù hợp GPU yếu
run_cfg = next(r for r in ABLATION_RUNS if "distil" in r.run_id)
```

---

## 7. Cấu trúc thư mục

```
AI-text-detector/
│
├── ablation.py              # Entry point: 12 ablation runs
├── cli.py                   # Giao diện terminal (dùng khi chạy local)
├── requirements.txt
│
├── configs/
│   └── config.py            # Tất cả hyperparameter và đường dẫn
│
├── src/
│   ├── model.py             # DANN_TextDetector + GRL + LoRA
│   ├── dataset.py           # Dataset, DataLoader, Collator
│   ├── train.py             # Training loop, evaluate, checkpoint, submission
│   ├── plots.py             # Learning curve, ablation summary, ROC
│   └── error_analysis.py    # Phân tích lỗi chi tiết
│
├── notebooks/
│   ├── eda.py               # EDA + baseline TF-IDF/LightGBM
│   └── data_augmentation.py # Tăng cường dữ liệu PeerRead + Groq API
│
└── outputs/                 # Tự tạo khi chạy
    ├── checkpoints/         # Model .pt
    ├── plots/               # PNG + ablation_results.json
    └── submissions/         # CSV nộp bài
```

---

## 8. Troubleshooting

**`CUDA Out of Memory`**  
→ Giảm `CFG.BATCH_SIZE` và tăng `CFG.ACCUM_STEPS` tương ứng. Xem [Mục 6](#6-điều-chỉnh-khi-gpu-yếu).

**`ModuleNotFoundError: No module named 'peft'`**  
→ Chạy: `!pip install peft`

**`FileNotFoundError` khi đọc dữ liệu**  
→ Kiểm tra lại đường dẫn đã patch vào `CFG` có đúng không.  
→ Trên Colab: chạy `!ls /content/drive/MyDrive/data/` để xác nhận file tồn tại.  
→ Trên Kaggle: chạy `!ls /kaggle/input/` để xem tên dataset thật.

**W&B báo lỗi authentication**  
→ Chạy lại `wandb.login()` hoặc đặt `os.environ["WANDB_MODE"] = "disabled"`.

**Colab ngắt kết nối giữa chừng**  
→ Checkpoint được lưu sau mỗi epoch lên Drive. Chạy lại `run_all()` — các run đã có file `final` checkpoint sẽ tự động bỏ qua.

**DeBERTa báo lỗi tokenizer**  
→ Chạy: `!pip install sentencepiece`

**Kaggle: notebook chạy xong nhưng không thấy file submission**  
→ Nhấn **Save & Run All (Commit)** để Kaggle lưu output. File sẽ xuất hiện trong tab **Output** của notebook version đó.
