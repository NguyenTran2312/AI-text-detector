# AI Text Detector — Đồ án Cuối kỳ Deep Learning 2026

Phát hiện văn bản do AI tạo ra (Human vs Machine) sử dụng **RoBERTa + LoRA + DANN** (Domain-Adversarial Neural Network).

---

## Cấu trúc dự án

```
ai-text-detector/
│
├── ablation.py                  # Entry point — chạy 4 ablation runs
│
├── configs/
│   └── config.py                # Cấu hình tập trung (paths, hyperparameters)
│
├── src/
│   ├── dataset.py               # Dataset, DataLoader, Collator
│   ├── model.py                 # DANN_TextDetector (RoBERTa + LoRA + GRL)
│   ├── train.py                 # Training loop, evaluate, checkpoint helpers
│   └── plots.py                 # Learning curves, ablation summary, ROC overlay
│
├── notebooks/
│   ├── eda.py                   # EDA + baseline ML (TF-IDF, LightGBM)
│   └── data_augmentation.py     # Thu thập PeerRead + generate AI text (Groq API)
│
├── outputs/
│   ├── plots/                   # Biểu đồ learning curves, ablation summary, ROC
│   ├── checkpoints/             # Model weights (.pt) cho mỗi run
│   └── submissions/             # File CSV predict trên test unlabeled
│
├── requirements.txt
└── README.md
```

---

## Thiết kế Ablation Study (4 runs)

| Run | Mô tả | DANN | lr | dropout | warmup |
|-----|-------|------|----|---------|--------|
| `run1_baseline` | RoBERTa+LoRA, không DANN | ✗ | 2e-5 | 0.3 | 0.1 |
| `run2_dann_default` | DANN, config gốc | ✓ | 2e-5 | 0.3 | 0.1 |
| `run3_dann_lr_low` | DANN, lr thấp hơn | ✓ | 1e-5 | 0.3 | 0.1 |
| `run4_dann_dropout_warmup` | DANN, regularize mạnh hơn | ✓ | 2e-5 | 0.5 | 0.06 |

**Metric chính:** `FPR @ TPR=95%` trên target domain (test có label). Thấp hơn = tốt hơn.

---

## Hướng dẫn chạy trên Google Colab / Kaggle

### Bước 1 — Clone repo và cài thư viện

```bash
!git clone https://github.com/<your-username>/ai-text-detector.git
%cd ai-text-detector
!pip install -r requirements.txt -q
```

### Bước 2 — Mount Google Drive (Colab) hoặc thêm dataset (Kaggle)

```python
# Colab
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Kaggle — thêm dataset vào notebook settings, sau đó path là /kaggle/input/...
```

### Bước 3 — Cấu hình đường dẫn dữ liệu

Mở `configs/config.py` và chỉnh 4 đường dẫn sau:

```python
TRAIN_PATH          = "/content/drive/MyDrive/cleaned_text_data.jsonl"
DEV_PATH            = "/content/drive/MyDrive/SubtaskA/subtaskA_dev_monolingual.jsonl"
TEST_LABELED_PATH   = "/content/drive/MyDrive/subtaskA_monolingual_labeled.jsonl"
TEST_UNLABELED_PATH = "/content/drive/MyDrive/subtaskA_monolingual_unlabeled.jsonl"
```

### Bước 4 — Đăng nhập W&B (tùy chọn, có điểm cộng)

```python
import wandb
wandb.login()   # Nhập API key từ https://wandb.ai/authorize
```

### Bước 5 — Chuẩn bị dữ liệu (chạy 1 lần)

```bash
# Chạy EDA và xem phân phối dữ liệu
!python notebooks/eda.py

# Nếu cần tạo thêm dữ liệu AI từ Groq API
!python notebooks/data_augmentation.py
```

### Bước 6 — Chạy Ablation Study

```bash
!python ablation.py
```

Hoặc trong Jupyter/Colab cell:

```python
import sys
sys.path.insert(0, '/content/ai-text-detector')

from ablation import run_all
results = run_all()
```

### Chạy riêng từng run (để debug)

```python
from ablation import ABLATION_RUNS, run_single, seed_everything
from src.dataset import build_data, make_loaders
from transformers import AutoTokenizer
from configs.config import CFG

tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)
train_ds, val_ds, target_ds, dev_ds, test_ds, submit_ds = build_data(tokenizer)

# Chỉ chạy run 1
result = run_single(
    run_cfg   = ABLATION_RUNS[0],   # run1_baseline
    train_ds  = train_ds,
    val_ds    = val_ds,
    target_ds = target_ds,
    dev_ds    = dev_ds,
    test_ds   = test_ds,
    submit_ds = submit_ds,
    tokenizer = tokenizer,
)
```

---

## Output sau khi chạy xong

```
outputs/
├── plots/
│   ├── run1_baseline_curves.png          # Train/val loss + acc curve
│   ├── run2_dann_default_curves.png
│   ├── run3_dann_lr_low_curves.png
│   ├── run4_dann_dropout_warmup_curves.png
│   ├── ablation_summary.png              # Bar chart so sánh FPR/F1/AUC
│   ├── roc_curves_all.png               # ROC overlay 4 runs
│   └── ablation_results.json            # Bảng số liệu đầy đủ
│
├── checkpoints/
│   ├── run1_baseline_best.pt            # Best checkpoint theo val F1
│   ├── run1_baseline_epoch0.pt          # Checkpoint theo epoch (để debug)
│   ├── run1_baseline_final.pt           # Final checkpoint sau khi load best
│   └── ...                             # Tương tự cho run2, run3, run4
│
└── submissions/
    ├── submission_run1_baseline.csv
    ├── submission_run2_dann_default.csv
    ├── submission_run3_dann_lr_low.csv
    └── submission_run4_dann_dropout_warmup.csv
```

---

## Yêu cầu phần cứng

| Cấu hình | Ước tính thời gian / run |
|----------|--------------------------|
| GPU T4 (Colab free) | ~45–60 phút |
| GPU A100 (Colab Pro) | ~15–20 phút |
| CPU only | Không khuyến nghị |

> **Lưu ý:** Toàn bộ 4 runs tokenize dữ liệu **1 lần duy nhất** ở đầu để tiết kiệm thời gian.

---

## Thư viện chính

| Thư viện | Mục đích |
|----------|----------|
| `transformers` | RoBERTa pretrained model |
| `peft` | LoRA (Parameter-Efficient Fine-Tuning) |
| `torch` | Training loop, GPU computation |
| `wandb` | Experiment tracking (điểm cộng) |
| `scikit-learn` | Metrics: F1, AUC, ROC curve |
| `optuna` | HPO (dùng trong notebook gốc) |
| `lightgbm` | Baseline model trong EDA |
| `polars` | Load JSONL nhanh hơn pandas |

---

## Nhóm thực hiện

| Thành viên | MSSV | Nhiệm vụ |
|------------|------|----------|
| ... | ... | Data pipeline, EDA |
| ... | ... | Model architecture, Training loop |
| ... | ... | Ablation study, Error analysis |
| ... | ... | Báo cáo, Thuyết trình |
