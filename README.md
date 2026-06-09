```markdown
# AI Text Detector

Dự án phân loại và phát hiện văn bản do AI tạo ra (Machine-Generated Text) so với văn bản do con người viết (Human-Written). 
Kiến trúc cốt lõi sử dụng: **RoBERTa / DeBERTa / DistilRoBERTa + LoRA + DANN (Domain-Adversarial Neural Network)** để tối ưu hóa pipeline huấn luyện và tăng cường khả năng tổng quát hóa trên các miền dữ liệu (domains) khác nhau.

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Cài đặt & Chuẩn bị dữ liệu](#2-cài-đặt--chuẩn-bị-dữ-liệu)
3. [Chạy trên Google Colab / Kaggle](#3-chạy-trên-google-colab--kaggle)
4. [Hướng dẫn Huấn luyện (Training & Evaluation)](#4-hướng-dẫn-huấn-luyện-training--evaluation)
5. [Điều chỉnh khi GPU yếu](#5-điều-chỉnh-khi-gpu-yếu)
6. [Cấu trúc thư mục](#6-cấu-trúc-thư-mục)
7. [Troubleshooting](#7-troubleshooting)
8. [Tài liệu tham khảo](#8-tài-liệu-tham-khảo)

---

## 1. Tổng quan kiến trúc

```text
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

* **LoRA**: Chỉ fine-tune ~2M / 125M tham số → tiết kiệm VRAM, training nhanh hơn, dễ dàng xử lý các tập dữ liệu văn bản lớn.
* **DANN + GRL**: Buộc encoder học đặc trưng không phụ thuộc domain → generalize tốt hơn trên tập test.
* **Metric chính**: FPR @ TPR=95% ↓ — tỉ lệ nhận diện nhầm văn bản Human thành AI khi bắt đúng 95% văn bản AI.

---

## 2. Cài đặt & Chuẩn bị dữ liệu

### Clone repository

```bash
git clone [https://github.com/NguyenTran2312/AI-text-detector.git](https://github.com/NguyenTran2312/AI-text-detector.git)
cd AI-text-detector

```

### Cài đặt thư viện

```bash
pip install -r requirements.txt

```

Tải Dataset (Download Links)
Vui lòng tải các file dữ liệu dưới đây và đặt vào thư mục lưu trữ của bạn (ví dụ: trên Google Drive hoặc thư mục local) trước khi tiến hành huấn luyện:
Tải File train gốc (Original Train Data): 1HeCgnLuDoUHhP-2OsTSSC3FXRLVoI6OG

Tải File cleaned_text_data.jsonl: 1OglczhXUZmABpx_ASuA84TLt5yonO0aV

Tải File subtaskA_dev_monolingual.jsonl (Dev Set): 11rOHC43k8y2tgH1wY2nc43aAxg8yS-vY

Tải File subtaskA_monolingual_labeled.jsonl (Test Labeled): 1--wjFQtIp51OptVLMFsiOGrA7-62JvaV

Tải File subtaskA_monolingual_unlabeled.jsonl (Test Unlabeled): 1kaalGjfBLaUbtcnOdN6kCM2Cp-cEgbKH

### Cấu trúc file JSONL dữ liệu

Mỗi dòng trong tập dữ liệu là một JSON object:

```json
{"id": "abc123", "text": "Nội dung văn bản...", "label": 0, "source": "wikipedia", "model": "human"}
{"id": "def456", "text": "Nội dung văn bản...", "label": 1, "source": "reddit",    "model": "gpt-4"}

```

Nếu chưa có file `cleaned_text_data.jsonl`, bạn có thể chạy pipeline chuẩn bị dữ liệu:

```bash
# EDA + làm sạch dữ liệu gốc
python notebooks/eda.py

# Tăng cường thêm từ PeerRead + Groq (cần cấu hình GROQ_API_KEY)
python notebooks/data_augmentation.py

```

---

## 3. Chạy trên Google Colab / Kaggle

### Trên Google Colab (T4 / A100)

1. Mount Google Drive và tạo thư mục lưu trữ: `checkpoints/`, `plots/`, `submissions/`.
2. Cập nhật các biến đường dẫn trong `configs/config.py` trỏ về Google Drive của bạn (VD: `CFG.TRAIN_PATH = '/content/drive/MyDrive/data/cleaned_text_data.jsonl'`).
3. Tham khảo mục [Điều chỉnh GPU](https://www.google.com/search?q=%235-%C4%91i%E1%BB%81u-ch%E1%BB%89nh-khi-gpu-y%E1%BA%BFu) nếu dùng bản T4 miễn phí.

### Trên Kaggle (P100 / T4x2)

1. Upload dataset định dạng `.jsonl` vào Kaggle Datasets và chọn **Add Data**.
2. Patch lại `CFG` trong notebook để trỏ đường dẫn đọc dữ liệu về `/kaggle/input/{tên-dataset}/...` và đường dẫn lưu output về `/kaggle/working/`.
3. Bật Internet trong Kaggle Notebook để cài các package thiếu và dùng `WandB`.

---

## 4. Hướng dẫn Huấn luyện (Training & Evaluation)

Dự án cung cấp 2 phương thức chính để huấn luyện mô hình:

1. **Dùng `cli.py**`: Giao diện dòng lệnh linh hoạt để chạy thử nghiệm nhanh một cấu hình.
2. **Dùng `ablation.py**`: Chạy tự động toàn bộ 12 phiên bản thực nghiệm (Ablation Study) để so sánh chi tiết.

⚠️ **LƯU Ý QUAN TRỌNG KHI DÙNG DEBERTA:**

> Kiến trúc DeBERTa sử dụng một loại Tokenizer đặc biệt. Nếu bạn chạy các cấu hình sử dụng backbone `microsoft/deberta-v3-base`, bạn **bắt buộc phải cài đặt thêm thư viện `sentencepiece**`.
> Lệnh cài đặt: `pip install sentencepiece transformers`

### Cách 1: Chạy thực nghiệm đơn lẻ qua Terminal (`cli.py`)

Đây là cách tốt nhất khi bạn muốn debug, thử nghiệm nhanh một ý tưởng, hoặc khi tài nguyên GPU có hạn.

**Xem tất cả các tham số hỗ trợ:**

```bash
python cli.py --help

```

**Các lệnh phổ biến:**

* **Chạy Baseline thuần (Không DANN) với RoBERTa:**
```bash
python cli.py --model_name roberta-base --run_id run_roberta_baseline

```


* **Chạy cấu hình DANN đầy đủ với DeBERTa:**
```bash
python cli.py \
  --model_name microsoft/deberta-v3-base \
  --use_dann \
  --use_dev_x15 \
  --run_id run_deberta_dann_full \
  --description "DeBERTa với DANN và Devx15"

```


* **Chế độ Debug nhanh (Giảm Epoch & Batch Size tránh OOM):**
```bash
python cli.py --model_name distilroberta-base --epochs 1 --batch_size 16 --run_id test_run

```


* **Chế độ chỉ tính lại Threshold (Không train lại):**
Nếu đã có file checkpoint trong `outputs/checkpoints/`, bạn có thể chạy lại để xuất file submission và tính metrics:
```bash
python cli.py --run_id ten_run_da_train --is_threshold_only

```



### Cách 2: Chạy toàn bộ Ablation Study (`ablation.py`)

File này sẽ tự động chạy tuần tự 12 cấu hình thực nghiệm khác nhau (kết hợp giữa 3 backbone: RoBERTa, DeBERTa, DistilRoBERTa và 4 setting DANN/Baseline). Phù hợp khi bạn treo máy chạy pipeline qua đêm trên Kaggle.

```bash
python ablation.py

```

**Kết quả đầu ra của `ablation.py`:**

1. In bảng tổng sắp so sánh F1, AUC, và FPR@95% ra console. Đánh dấu model tốt nhất (`◀ BEST`).
2. Lưu file tổng hợp kết quả tại `outputs/plots/ablation_results.json`.
3. Vẽ và lưu biểu đồ Bar Chart so sánh và đường cong ROC Curve (ROC overlays) vào thư mục `outputs/plots/`.

---

## 5. Điều chỉnh khi GPU yếu

Bảng cấu hình tham khảo để tinh chỉnh trong `configs/config.py` hoặc qua tham số CLI `--batch_size`, `--accum_steps`:

| GPU | VRAM | BATCH_SIZE | ACCUM_STEPS | Ghi chú |
| --- | --- | --- | --- | --- |
| RTX 5090 | 40 GB | 128 | 2 | Mặc định trong config |
| V100 / A10 | 16–24 GB | 64 | 4 | Colab Pro / Kaggle P100 |
| T4 | 15 GB | 32 | 8 | Colab free |
| RTX 3070 / 4060 | 8 GB | 16 | 16 | Local consumer GPU |

*Lưu ý: Giữ nguyên Effective Batch Size (`BATCH_SIZE` × `ACCUM_STEPS` = 256) để bảo đảm hiệu quả huấn luyện.*

---

## 6. Cấu trúc thư mục

```text
AI-text-detector/
│
├── ablation.py              # Entry point: 12 ablation runs
├── cli.py                   # Giao diện terminal cho 1 run đơn lẻ
├── requirements.txt
│
├── configs/
│   └── config.py            # Chứa các hyperparameter và đường dẫn
│
├── src/
│   ├── model.py             # DANN_TextDetector + GRL + LoRA
│   ├── dataset.py           # Dataset, DataLoader, Collator
│   ├── train.py             # Training loop, evaluate, submission generator
│   ├── plots.py             # Script vẽ biểu đồ (Learning curve, ROC,...)
│   └── error_analysis.py    # Phân tích lỗi chi tiết
│
├── notebooks/
│   ├── eda.py               # Exploratory Data Analysis & baseline models
│   └── data_augmentation.py # Sinh thêm dữ liệu qua API Groq
│
└── outputs/                 # Thư mục sinh ra tự động
    ├── checkpoints/         # Model weights (.pt)
    ├── plots/               # Biểu đồ và file JSON kết quả
    └── submissions/         # File CSV dùng để nộp bài

```

---

## 7. Troubleshooting

* **`CUDA Out of Memory`**: Giảm `CFG.BATCH_SIZE` và tăng `CFG.ACCUM_STEPS` tương ứng. Nếu vẫn lỗi, thử dùng `distilroberta-base`.
* **`ModuleNotFoundError: No module named 'peft'`**: Chạy `pip install peft`.
* **W&B báo lỗi authentication**: Chạy lại `wandb.login()` hoặc tắt wandb bằng lệnh `os.environ["WANDB_MODE"] = "disabled"`.
* **Colab / Kaggle bị ngắt kết nối giữa chừng**: Quá trình train sẽ tự lưu checkpoint sau mỗi epoch. Khi kết nối lại và chạy tiếp, các lượt chạy đã hoàn thành (có file `final`) sẽ tự động được bỏ qua.
* **Không thấy file submission trên Kaggle**: Nhớ nhấn nút **Save & Run All (Commit)**. Khi notebook chạy xong 100%, file CSV sẽ xuất hiện trong thẻ **Output** của version đó.

---


```

```
