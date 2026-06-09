# AI Text Detector

Phát hiện văn bản do AI tạo ra vs con người viết, sử dụng **RoBERTa / DeBERTa / DistilRoBERTa + LoRA + DANN (Domain-Adversarial Neural Network)**.

---

## Mục lục

1. [Kiến trúc tổng quan](#kiến-trúc-tổng-quan)
2. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
3. [Cài đặt môi trường](#cài-đặt-môi-trường)
4. [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
5. [Cách chạy](#cách-chạy)
6. [Kết quả Ablation Study](#kết-quả-ablation-study)
7. [Giải thích các thành phần kỹ thuật](#giải-thích-các-thành-phần-kỹ-thuật)
8. [Troubleshooting](#troubleshooting)

---

## Kiến trúc tổng quan

```
Văn bản đầu vào
      │
      ▼
 Tokenizer (RoBERTa / DeBERTa / DistilRoBERTa)
      │
      ▼
 Transformer Encoder + LoRA adapter
      │
   [CLS] token embedding
      │
  ┌───┴────────────────────────┐
  ▼                            ▼
class_head               GRL (đảo dấu gradient)
  │                            │
  ▼                            ▼
Human / AI              domain_head
(nhãn chính)         Source / Target domain
                     (chỉ dùng lúc training)
```

**DANN** (Domain-Adversarial Neural Network) giúp model học các đặc trưng **không phụ thuộc domain** — tức là hoạt động tốt trên cả dữ liệu train lẫn test có phân phối khác nhau.

**LoRA** (Low-Rank Adaptation) giảm số tham số cần fine-tune từ ~125M xuống ~2M, tiết kiệm VRAM và thời gian training đáng kể.

---

## Cấu trúc thư mục

```
AI-text-detector/
│
├── ablation.py              # Entry point: chạy toàn bộ 12 ablation runs
├── cli.py                   # Giao diện terminal để chạy 1 run đơn lẻ
├── requirements.txt         # Danh sách thư viện Python cần cài
│
├── configs/
│   └── config.py            # Tất cả hyperparameter và đường dẫn tập trung ở đây
│
├── src/
│   ├── model.py             # Kiến trúc DANN_TextDetector + GRL + LoRA
│   ├── dataset.py           # Dataset, DataLoader, Collator
│   ├── train.py             # Training loop, evaluate, checkpoint, submission
│   ├── plots.py             # Biểu đồ learning curve, ablation summary, ROC
│   └── error_analysis.py    # Phân tích lỗi chi tiết sau training
│
├── notebooks/
│   ├── eda.py               # Phân tích dữ liệu + baseline ML (TF-IDF + LightGBM)
│   └── data_augmentation.py # Tăng cường dữ liệu từ PeerRead + Groq API
│
└── outputs/                 # Tự động tạo khi chạy
    ├── checkpoints/         # File model .pt
    ├── plots/               # Biểu đồ PNG + ablation_results.json
    └── submissions/         # File CSV để nộp bài
```

---

## Cài đặt môi trường

### Yêu cầu phần cứng

| Cấu hình | VRAM | Ghi chú |
|---|---|---|
| Khuyến nghị | 24 GB+ (RTX 3090/4090) | BATCH_SIZE=128 mặc định |
| Tối thiểu | 8 GB (T4, RTX 3070) | Giảm BATCH_SIZE xuống 16–32 |
| CPU only | — | Chạy được nhưng rất chậm |

### Bước 1 — Tạo môi trường ảo

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
# hoặc: venv\Scripts\activate   # Windows
```

### Bước 2 — Cài thư viện

```bash
pip install -r requirements.txt
```

> **Lưu ý GPU:** Nếu dùng CUDA 12.x, cài torch phù hợp trước:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### Bước 3 — Đăng nhập Weights & Biases (W&B)

Project dùng W&B để theo dõi experiment. Lấy API key miễn phí tại [wandb.ai](https://wandb.ai/authorize).

```bash
wandb login
# Dán API key khi được hỏi
```

> Nếu không muốn dùng W&B, đặt biến môi trường `WANDB_MODE=disabled` trước khi chạy.

---

## Chuẩn bị dữ liệu

### Cấu trúc file dữ liệu

Mỗi file là định dạng **JSONL** (mỗi dòng một JSON object):

```json
{"id": "abc123", "text": "Nội dung văn bản...", "label": 0, "source": "wikipedia", "model": "human"}
{"id": "def456", "text": "Nội dung văn bản...", "label": 1, "source": "reddit", "model": "gpt-4"}
```

- `label = 0` → Human viết
- `label = 1` → AI tạo ra

### Đặt file dữ liệu

Cập nhật đường dẫn trong `configs/config.py`:

```python
TRAIN_PATH          = "/path/to/cleaned_text_data.jsonl"
DEV_PATH            = "/path/to/subtaskA_dev_monolingual.jsonl"
TEST_LABELED_PATH   = "/path/to/subtaskA_monolingual_labeled.jsonl"
TEST_UNLABELED_PATH = "/path/to/subtaskA_monolingual_unlabeled.jsonl"
```

### (Tùy chọn) Chạy EDA và augmentation trước

```bash
# Phân tích dữ liệu và tạo cleaned_text_data.jsonl
python notebooks/eda.py

# Tăng cường dữ liệu từ PeerRead (cần GROQ_API_KEY cho phần AI generation)
export GROQ_API_KEY=gsk_xxxxxxxxxxxx
python notebooks/data_augmentation.py
```

---

## Cách chạy

### Cách 1 — Chạy toàn bộ Ablation Study (12 runs)

```bash
python ablation.py
```

Script sẽ chạy tuần tự 12 runs với 3 backbone × 4 cấu hình. Sau khi xong, kết quả được lưu tại:
- `outputs/plots/ablation_results.json` — bảng số liệu
- `outputs/plots/ablation_summary.png` — biểu đồ so sánh
- `outputs/plots/roc_curves_all.png` — ROC overlay
- `outputs/submissions/submission_*.csv` — file nộp bài cho mỗi run

---

### Cách 2 — Chạy một run đơn lẻ qua CLI

#### Baseline (không DANN):
```bash
python cli.py \
  --model_name roberta-base \
  --run_id my_baseline
```

#### DANN với DeBERTa:
```bash
python cli.py \
  --model_name microsoft/deberta-v3-base \
  --use_dann \
  --use_dev_x15 \
  --lr 2e-5 \
  --run_id deberta_dann
```

#### Chỉ tính lại threshold từ checkpoint cũ (không train lại):
```bash
python cli.py \
  --run_id my_baseline \
  --is_threshold_only
```

#### Xem tất cả tham số:
```bash
python cli.py --help
```

---

### Cách 3 — Chạy trong Jupyter / Colab

```python
from ablation import run_all
results = run_all()
```

Hoặc chạy một run cụ thể:

```python
from ablation import ABLATION_RUNS, run_single
from src.dataset import build_data, make_loaders
from transformers import AutoTokenizer
from configs.config import CFG

# Chọn run muốn chạy (ví dụ run đầu tiên)
run_cfg = ABLATION_RUNS[0]
CFG.MODEL_NAME = run_cfg.model_name

tokenizer = AutoTokenizer.from_pretrained(run_cfg.model_name)
train_ds, val_ds, target_ds, dev_ds, test_ds, submit_ds = build_data(tokenizer)

result = run_single(
    run_cfg=run_cfg,
    train_ds=train_ds, val_ds=val_ds, target_ds=target_ds,
    dev_ds=dev_ds, test_ds=test_ds, submit_ds=submit_ds,
    tokenizer=tokenizer,
)
print(f"FPR @ TPR=95%: {result['fpr_at_95tpr']:.4f}")
```

---

### Điều chỉnh khi GPU yếu

Nếu gặp lỗi **CUDA Out of Memory**, sửa trong `configs/config.py`:

```python
BATCH_SIZE  = 32   # Giảm từ 128 xuống 32
ACCUM_STEPS = 8    # Tăng lên để giữ effective batch size = 32×8 = 256
NUM_WORKERS = 4    # Giảm số CPU worker
```

Hoặc truyền thẳng qua CLI:

```bash
python cli.py --model_name roberta-base --batch_size 32 --accum_steps 8
```

---

## Kết quả Ablation Study

Metric chính: **FPR @ TPR=95%** — tỷ lệ văn bản Human bị nhầm thành AI khi model phát hiện đúng 95% văn bản AI. **Thấp hơn = tốt hơn.**

| Run | Backbone | DANN | lr | FPR@95%↓ | F1↑ | AUC↑ |
|---|---|---|---|---|---|---|
| run1_baseline | RoBERTa-base | ✗ | 2e-5 | — | — | — |
| run2_dann_default | RoBERTa-base | ✓ | 2e-5 | — | — | — |
| run1_deberta_base | DeBERTa-v3-base | ✗ | 2e-5 | — | — | — |
| run2_deberta_dann | DeBERTa-v3-base | ✓ | 2e-5 | — | — | — |
| ... | ... | ... | ... | ... | ... | ... |

> Bảng sẽ được điền sau khi chạy xong. Kết quả chi tiết lưu tại `outputs/plots/ablation_results.json`.

---

## Giải thích các thành phần kỹ thuật

### DANN — Domain-Adversarial Neural Network

Model được huấn luyện đồng thời trên 2 mục tiêu đối nghịch nhau:
1. **Phân loại đúng** Human vs AI (class_head)
2. **Không thể phân biệt** domain train vs domain test (domain_head + GRL)

**Gradient Reversal Layer (GRL)** là chìa khóa: trong backward pass, GRL đảo dấu gradient trước khi truyền về encoder. Điều này buộc encoder học các đặc trưng "bất khả tri domain" — hoạt động tốt trên cả 2 domain.

### LoRA — Low-Rank Adaptation

Thay vì fine-tune toàn bộ 125M tham số của RoBERTa, LoRA chỉ thêm các ma trận hạng thấp nhỏ (rank=8) vào các layer attention. Số tham số trainable giảm từ ~125M xuống còn ~2M, giúp:
- Tiết kiệm VRAM đáng kể
- Training nhanh hơn
- Ít overfitting hơn

### Lambda Schedule

Hệ số GRL (lambda) không cố định mà tăng dần từ 0 → 1 theo công thức sigmoid trong suốt quá trình training. Đầu training, lambda ≈ 0 giúp model học phân loại trước. Dần dần lambda tăng lên, buộc model học domain-invariant features.

### Dev × 15

Tập dev (target domain) thường nhỏ hơn tập train nhiều lần. Để DANN hoạt động hiệu quả, tập target cần đủ lớn để cung cấp gradient ổn định. Nhân bản dev 15 lần giải quyết vấn đề này mà không cần thêm dữ liệu mới.

---

## Troubleshooting

**Lỗi: `CUDA Out of Memory`**
→ Giảm `BATCH_SIZE` và tăng `ACCUM_STEPS` tương ứng trong `config.py`.

**Lỗi: `ModuleNotFoundError: No module named 'peft'`**
→ Chạy `pip install peft` hoặc `pip install -r requirements.txt`.

**Lỗi: `FileNotFoundError` khi đọc dữ liệu**
→ Kiểm tra lại các đường dẫn `*_PATH` trong `configs/config.py`.

**W&B không log được**
→ Chạy `wandb login` trước, hoặc đặt `WANDB_MODE=disabled` để tắt W&B.

**Training quá chậm trên CPU**
→ Project được tối ưu cho GPU. Trên CPU, nên giảm `BATCH_SIZE=8`, `EPOCHS=1`, `MAX_LEN=128` để chạy thử.

**Model DeBERTa báo lỗi tokenizer**
→ Cài thêm: `pip install sentencepiece`
