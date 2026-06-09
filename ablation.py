
# -*- coding: utf-8 -*-
# ablation.py
#
# File entry point chính của project — chạy toàn bộ Ablation Study.
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Ablation Study là gì?                                                  ║
# ║  Chạy nhiều phiên bản thực nghiệm (runs) với các cấu hình khác nhau    ║
# ║  để hiểu thành phần nào đóng góp nhiều nhất vào kết quả cuối cùng.     ║
# ║                                                                         ║
# ║  Project này so sánh 3 backbone × 4 cấu hình = 12 runs:                ║
# ║    Backbone: RoBERTa-base, DeBERTa-v3-base, DistilRoBERTa-base         ║
# ║    Cấu hình: Baseline / DANN / DANN+lr thấp / DANN+dropout+warmup      ║
# ║                                                                         ║
# ║  Metric chính: FPR @ TPR=95% — thấp hơn = tốt hơn                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Cách chạy:
#   python ablation.py                    — chạy toàn bộ 12 runs
#   from ablation import run_all          — gọi trong Colab/Jupyter

import json
import os
import random
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Đảm bảo Python tìm thấy các module trong thư mục gốc của project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import CFG
from src.dataset import make_loaders, TextDetectionDataset, InferenceDataset
from src.model import DANN_TextDetector
from src.train import (
    ckpt_path, save_checkpoint, load_checkpoint,
    train_one_epoch, evaluate,
    compute_fpr_at_tpr,
    generate_submission,
)
from src.plots import plot_ablation_summary, plot_roc_curves
from src.error_analysis import run_error_analysis
import warnings
warnings.filterwarnings("ignore", message="Failed to load")

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(CFG.PLOT_DIR, exist_ok=True)
os.makedirs(CFG.CKPT_DIR, exist_ok=True)
os.makedirs(CFG.SUB_DIR,  exist_ok=True)

# Tự động chọn GPU nếu có, không thì dùng CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# CẤU TRÚC DỮ LIỆU MỘT ABLATION RUN
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AblationConfig:
    """
    Đóng gói tất cả siêu tham số của một ablation run thành một object.
    Dùng @dataclass để tự động tạo __init__ và __repr__.

    Trường quan trọng:
      run_id       — tên định danh duy nhất (dùng làm tên file checkpoint)
      description  — mô tả ngắn để in ra console và log W&B
      model_name   — tên backbone trên HuggingFace (thay đổi được giữa các run)
      use_dann     — True: dùng domain adversarial; False: baseline thuần
      use_dev_x15  — True: nhân bản dev×15 cho DANN target loop
      history      — dict lưu loss/acc theo epoch sau khi train xong
    """
    run_id:       str
    description:  str
    model_name:   str     # Backbone: "roberta-base" / "microsoft/deberta-v3-base" / ...
    use_dann:     bool
    use_dev_x15:  bool
    lr:           float = 2e-5
    dropout:      float = 0.3
    warmup_ratio: float = 0.1
    history:      dict  = field(default_factory=dict)  # Được điền sau khi training


# ══════════════════════════════════════════════════════════════════════════════
# DANH SÁCH 12 ABLATION RUNS
# Nhóm theo backbone để dễ so sánh ảnh hưởng của từng thay đổi
# ══════════════════════════════════════════════════════════════════════════════

ABLATION_RUNS = [
    # ──────────────────────────────────────────────────────────────────────────
    # BACKBONE 1: RoBERTa-base (mô hình gốc, encoder-only, 125M params)
    # ──────────────────────────────────────────────────────────────────────────
    AblationConfig(
        run_id="run1_baseline",
        description="Baseline — RoBERTa+LoRA, không DANN, không dev×15",
        model_name="roberta-base",
        use_dann=False, use_dev_x15=False,   # Không có domain adaptation
        lr=2e-5, dropout=0.3, warmup_ratio=0.1
    ),
    AblationConfig(
        run_id="run2_dann_default",
        description="DANN — RoBERTa config gốc (lr=2e-5, dropout=0.3, warmup=0.1)",
        model_name="roberta-base",
        use_dann=True, use_dev_x15=True,     # Bật DANN + dev×15
        lr=2e-5, dropout=0.3, warmup_ratio=0.1
    ),
    AblationConfig(
        run_id="run3_dann_lr_low",
        description="DANN — RoBERTa lr=1e-5 (học chậm hơn, giảm thiểu quên kiến thức cũ)",
        model_name="roberta-base",
        use_dann=True, use_dev_x15=True,
        lr=1e-5,        # Learning rate thấp hơn → cập nhật weight nhỏ hơn mỗi bước
        dropout=0.3, warmup_ratio=0.1
    ),
    AblationConfig(
        run_id="run4_dann_dropout_warmup",
        description="DANN — RoBERTa dropout=0.5, warmup=0.06 (regularize mạnh hơn)",
        model_name="roberta-base",
        use_dann=True, use_dev_x15=True,
        lr=2e-5,
        dropout=0.5,       # Dropout cao hơn → chống overfitting mạnh hơn
        warmup_ratio=0.06  # Warmup ngắn hơn → learning rate tăng nhanh hơn
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # BACKBONE 2: DeBERTa-v3-base (kiến trúc Disentangled Attention, mạnh hơn)
    # DeBERTa tách biệt attention theo nội dung và vị trí → thường tốt hơn RoBERTa
    # ──────────────────────────────────────────────────────────────────────────
    AblationConfig(
        run_id="run1_deberta_base",
        description="Baseline — DeBERTa+LoRA, không DANN, không dev×15",
        model_name="microsoft/deberta-v3-base",
        use_dann=False, use_dev_x15=False,
        lr=2e-5, dropout=0.3, warmup_ratio=0.1
    ),
    AblationConfig(
        run_id="run2_deberta_dann",
        description="DANN — DeBERTa config gốc (lr=2e-5, dropout=0.3, warmup=0.1)",
        model_name="microsoft/deberta-v3-base",
        use_dann=True, use_dev_x15=True,
        lr=2e-5, dropout=0.3, warmup_ratio=0.1
    ),
    AblationConfig(
        run_id="run3_deberta_dann_lr_low",
        description="DANN — DeBERTa lr=1e-5 (học chậm, thích ứng miền tối ưu)",
        model_name="microsoft/deberta-v3-base",
        use_dann=True, use_dev_x15=True,
        lr=1e-5, dropout=0.3, warmup_ratio=0.1
    ),
    AblationConfig(
        run_id="run4_deberta_dann_dropout_warmup",
        description="DANN — DeBERTa dropout=0.5, warmup=0.06 (chống quá khớp miền nguồn)",
        model_name="microsoft/deberta-v3-base",
        use_dann=True, use_dev_x15=True,
        lr=2e-5, dropout=0.5, warmup_ratio=0.06
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # BACKBONE 3: DistilRoBERTa-base (phiên bản chưng cất, nhỏ hơn 40%, nhanh hơn 60%)
    # Phù hợp khi tài nguyên hạn chế hoặc cần inference nhanh
    # ──────────────────────────────────────────────────────────────────────────
    AblationConfig(
        run_id="run1_distil_baseline",
        description="Baseline — DistilRoBERTa+LoRA, không DANN, không dev×15",
        model_name="distilroberta-base",
        use_dann=False, use_dev_x15=False,
        lr=2e-5, dropout=0.3, warmup_ratio=0.1
    ),
    AblationConfig(
        run_id="run2_distil_dann_default",
        description="DANN — DistilRoBERTa config gốc (lr=2e-5, dropout=0.3, warmup=0.1)",
        model_name="distilroberta-base",
        use_dann=True, use_dev_x15=True,
        lr=2e-5, dropout=0.3, warmup_ratio=0.1
    ),
    AblationConfig(
        run_id="run3_distil_dann_lr_low",
        description="DANN — DistilRoBERTa lr=1e-5 (học chậm trên mạng nén nhẹ)",
        model_name="distilroberta-base",
        use_dann=True, use_dev_x15=True,
        lr=1e-5, dropout=0.3, warmup_ratio=0.1
    ),
    AblationConfig(
        run_id="run4_distil_dann_dropout_warmup",
        description="DANN — DistilRoBERTa dropout=0.5, warmup=0.06 (tăng cường tổng quát hóa)",
        model_name="distilroberta-base",
        use_dann=True, use_dev_x15=True,
        lr=2e-5, dropout=0.5, warmup_ratio=0.06
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# SEED EVERYTHING — Đảm bảo kết quả tái lặp được
# ══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int = CFG.SEED):
    """
    Đặt seed cho tất cả nguồn ngẫu nhiên trong chương trình.
    Cần thiết để kết quả giữa các lần chạy là như nhau.

    Các nguồn ngẫu nhiên cần kiểm soát:
      random      — shuffle dữ liệu Python thuần
      numpy       — các phép toán vector/matrix
      torch       — khởi tạo trọng số, dropout
      cuda        — các phép toán GPU
      PYTHONHASHSEED — hash của dict/set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ══════════════════════════════════════════════════════════════════════════════
# RUN SINGLE — Chạy một ablation run đơn lẻ
# ══════════════════════════════════════════════════════════════════════════════

def run_single(
    run_cfg: AblationConfig,
    train_ds, val_ds, target_ds, dev_ds, test_ds, submit_ds,
    tokenizer,
    is_threshold_only: bool  = False,  # Nếu True: chỉ tính lại threshold, không train
    pretrained_state: dict   = None,   # Weights từ run trước (để tái sử dụng)
    _df_val_raw              = None,   # DataFrame gốc để error analysis
    _df_dev_raw              = None,
    _df_test_labeled_raw     = None,
    _df_test_unlabeled_raw   = None,
) -> dict:
    """
    Thực hiện một ablation run hoàn chỉnh theo các bước:
      1. Tạo DataLoaders từ datasets
      2. Khởi tạo model (DANN_TextDetector với backbone và config tương ứng)
      3. Training loop qua CFG.EPOCHS epoch với early stopping
      4. Load best checkpoint và evaluate trên dev + test có nhãn
      5. Tạo file submission cho test không nhãn
      6. Chạy error analysis (nếu có raw DataFrames)
      7. Dọn dẹp model khỏi VRAM và trả về dict kết quả

    Trả về dict chứa đầy đủ metrics, thông tin cấu hình, và dữ liệu ROC
    để dùng trong plot_ablation_summary() và plot_roc_curves().
    """
    seed_everything()
    run_id = run_cfg.run_id

    print(f"\n{'='*70}\n  [{run_id}]  Backend: {run_cfg.model_name}\n  👉 {run_cfg.description}\n{'='*70}")

    # ── Bước 1: Tạo DataLoaders ───────────────────────────────────────────────
    train_loader, val_loader, target_loader, dev_loader, test_loader, submit_loader = make_loaders(
        train_ds, val_ds, target_ds, dev_ds, test_ds, submit_ds, tokenizer, run_cfg.use_dev_x15,
    )

    # ── Bước 2: Khởi tạo model ───────────────────────────────────────────────
    model = DANN_TextDetector(
        model_name=run_cfg.model_name,
        num_classes=CFG.NUM_CLASSES,
        num_domains=CFG.NUM_DOMAINS,
        dropout=run_cfg.dropout,
        use_lora=True,
        use_dann=run_cfg.use_dann,
    ).to(device)

    # ── Bước 3: Training hoặc load checkpoint ────────────────────────────────
    if is_threshold_only and pretrained_state is not None:
        # Chế độ chỉ tính lại threshold: load weights từ disk hoặc từ dict
        src_run_id = pretrained_state.get("source_run_id", "")
        disk = ckpt_path(src_run_id, "final")
        if os.path.exists(disk):
            print(f"  [CKPT] Load từ disk: {disk}")
            load_checkpoint(disk, model)
        else:
            model.load_state_dict(pretrained_state["state"])
    else:
        # ── Training từ đầu ──────────────────────────────────────────────────
        # Tính tổng số bước và bước warmup cho scheduler
        len_dl       = min(len(train_loader), len(target_loader)) if target_loader else len(train_loader)
        total_steps  = (len_dl // CFG.ACCUM_STEPS) * CFG.EPOCHS
        warmup_steps = int(total_steps * run_cfg.warmup_ratio)

        # AdamW: optimizer tiêu chuẩn cho fine-tuning transformer
        optimizer = torch.optim.AdamW(model.parameters(), lr=run_cfg.lr, weight_decay=CFG.WEIGHT_DECAY)
        # Linear schedule: learning rate tăng dần trong warmup, giảm dần sau đó
        scheduler = get_linear_schedule_with_warmup(optimizer,
                    num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        # GradScaler cho AMP: điều chỉnh tỷ lệ loss tự động để tránh underflow với float16
        scaler = torch.amp.GradScaler('cuda') if hasattr(torch, 'amp') else torch.cuda.amp.GradScaler()

        # Khởi tạo W&B run cho lần chạy này
        wrun = wandb.init(
            project=CFG.WANDB_PROJECT, entity=CFG.WANDB_ENTITY,
            name=run_id, group="ablation",
            config={
                "run_id": run_id, "use_dann": run_cfg.use_dann,
                "use_dev_x15": run_cfg.use_dev_x15, "lr": run_cfg.lr,
                "dropout": run_cfg.dropout, "warmup_ratio": run_cfg.warmup_ratio,
                "epochs": CFG.EPOCHS, "model": run_cfg.model_name,
            }, reinit=True
        )

        best_f1, best_state, no_improve, global_step = 0.0, None, 0, 0
        history = {}

        # ── Vòng lặp training theo epoch ─────────────────────────────────────
        for epoch in range(CFG.EPOCHS):
            (train_loss, train_cls_loss, train_dom_loss,
             train_f1, train_acc, global_step) = train_one_epoch(
                model, train_loader, target_loader, optimizer, scheduler, scaler,
                epoch, CFG.EPOCHS, global_step, run_cfg.use_dann, device,
            )
            val_f1, val_acc, val_loss, _, _ = evaluate(model, val_loader, device)

            # Lưu lịch sử để vẽ learning curve sau
            history[epoch] = {
                "train_loss": train_loss, "val_loss": val_loss,
                "train_acc": train_acc,   "val_acc": val_acc,
                "train_f1": train_f1,     "val_f1": val_f1
            }

            wandb.log({
                "epoch/train/loss": train_loss, "epoch/train/f1": train_f1,
                "epoch/val/loss": val_loss,     "epoch/val/f1": val_f1,
                "epoch": epoch
            })
            print(f"  Epoch {epoch+1}/{CFG.EPOCHS} | train_loss={train_loss:.4f} | val_F1={val_f1:.4f}")

            # ── Early stopping & checkpoint ───────────────────────────────────
            if val_f1 > best_f1:
                best_f1    = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
                save_checkpoint(model, optimizer, scheduler, epoch, val_f1, run_id, tag="best")
            else:
                no_improve += 1
                if no_improve >= CFG.PATIENCE:
                    print(f"  ⏹  Early stop tại epoch {epoch+1}")
                    break

        # Load lại best checkpoint trước khi evaluate
        run_cfg.history = history
        best_disk = ckpt_path(run_id, "best")
        if os.path.exists(best_disk):
            load_checkpoint(best_disk, model)
        # Lưu "final" để đánh dấu run đã hoàn thành
        save_checkpoint(model, optimizer, scheduler, max(history.keys()), best_f1, run_id, tag="final")
        wrun.finish()

    # ── Bước 4: Evaluate ─────────────────────────────────────────────────────
    print("  Evaluating trên dev gốc...")
    dev_f1, dev_acc, dev_loss, dev_probs, dev_labels = evaluate(model, dev_loader, device)
    dev_fpr, _ = compute_fpr_at_tpr(dev_probs, dev_labels, target_tpr=0.95)

    print("  Evaluating trên test có label...")
    test_f1, test_acc, test_loss, test_probs, test_labels = evaluate(model, test_loader, device)
    fpr_at_95, thr_95 = compute_fpr_at_tpr(test_probs, test_labels, target_tpr=0.95)
    auc = float(roc_auc_score(test_labels, test_probs))
    fpr_arr, tpr_arr, _ = roc_curve(test_labels, test_probs, pos_label=1)

    # ── Bước 5: Tạo file submission ──────────────────────────────────────────
    # Dùng ngưỡng tính được từ test có nhãn (đạt TPR=95%) để predict tập unlabeled
    generate_submission(model, submit_loader, submit_ds, thr_95, run_id, device)

    # ── Bước 6: Error analysis (nếu có raw DataFrames) ───────────────────────
    if _df_val_raw is not None and _df_dev_raw is not None:
        _ckpt = ckpt_path(run_id, "final")
        if os.path.exists(_ckpt):
            # Tạo model mới để error analysis (tránh conflict state)
            _model_ea = DANN_TextDetector(
                run_cfg.model_name, CFG.NUM_CLASSES, CFG.NUM_DOMAINS,
                dropout=run_cfg.dropout, use_lora=True, use_dann=run_cfg.use_dann
            ).to(device)
            load_checkpoint(_ckpt, _model_ea)
            run_error_analysis(
                run_id=run_id, model=_model_ea, tokenizer=tokenizer, device=device,
                df_val=_df_val_raw, df_dev=_df_dev_raw,
                df_test_labeled=_df_test_labeled_raw,
                df_test_unlabeled=_df_test_unlabeled_raw
            )
            del _model_ea
            torch.cuda.empty_cache()

    # ── Bước 7: Dọn dẹp VRAM và trả về kết quả ──────────────────────────────
    del model
    torch.cuda.empty_cache()

    return {
        # Thông tin run
        "run_id": run_id, "description": run_cfg.description,
        "use_dann": run_cfg.use_dann, "use_dev_x15": run_cfg.use_dev_x15,
        "lr": run_cfg.lr, "dropout": run_cfg.dropout, "warmup_ratio": run_cfg.warmup_ratio,
        # Metric trên dev (target domain, có nhãn)
        "dev_f1": round(dev_f1, 6), "dev_acc": round(dev_acc, 6), "dev_fpr_at_95": round(dev_fpr, 6),
        # Metric trên test (target domain, có nhãn) — metric chính
        "test_f1": round(test_f1, 6), "test_acc": round(test_acc, 6), "test_auc": round(auc, 6),
        "fpr_at_95tpr": round(fpr_at_95, 6),   # ← METRIC CHÍNH: thấp hơn = tốt hơn
        "threshold_95": round(thr_95, 6),       # Ngưỡng tương ứng dùng cho submission
        # Dữ liệu ROC để vẽ biểu đồ tổng hợp
        "roc_fpr": fpr_arr.tolist(), "roc_tpr": tpr_arr.tolist(),
        # Weights để tái sử dụng (nếu cần)
        "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()} if False else {},
    }


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL — Chạy toàn bộ 12 ablation runs tuần tự
# ══════════════════════════════════════════════════════════════════════════════

def run_all():
    """
    Hàm điều phối chạy toàn bộ ablation study:
      1. Đọc tất cả dữ liệu thô từ đĩa một lần duy nhất
      2. Với mỗi run trong ABLATION_RUNS:
         a. Cập nhật CFG.MODEL_NAME (patch global để các module khác dùng đúng tên)
         b. Tạo Tokenizer phù hợp với backbone
         c. Tokenize dữ liệu (mỗi run tokenize riêng vì tokenizer khác nhau)
         d. Gọi run_single()
      3. Sau khi xong tất cả runs: vẽ biểu đồ so sánh và lưu kết quả JSON

    Dữ liệu thô được đọc một lần → tiết kiệm I/O so với đọc trong mỗi run.
    """
    seed_everything()

    import polars as pl
    from sklearn.model_selection import train_test_split

    # ── Đọc dữ liệu thô một lần duy nhất ────────────────────────────────────
    print("📖 Đang nạp dữ liệu thô từ hệ thống...")
    _df_train_full = pl.read_ndjson(CFG.TRAIN_PATH).to_pandas()
    _df_train_full["domain_id"] = 0  # Source domain
    df_train_raw, df_val_raw = train_test_split(
        _df_train_full, test_size=CFG.VAL_SIZE, random_state=CFG.SEED,
        stratify=_df_train_full["label"]
    )

    df_dev_raw = pl.read_ndjson(CFG.DEV_PATH).to_pandas()
    df_dev_raw["domain_id"] = 1  # Target domain

    df_test_labeled_raw = pl.read_ndjson(CFG.TEST_LABELED_PATH).to_pandas()
    df_test_labeled_raw["domain_id"] = 1

    df_test_unlabeled_raw = pl.read_ndjson(CFG.TEST_UNLABELED_PATH).to_pandas()
    df_test_unlabeled_raw["domain_id"] = 1

    # Đảm bảo cột "source" tồn tại cho error analysis
    for df in [df_val_raw, df_dev_raw, df_test_labeled_raw]:
        if "source" not in df.columns:
            df["source"] = "unknown"

    all_results = []
    best_state  = None

    # ── Chạy từng run ─────────────────────────────────────────────────────────
    for run_cfg in ABLATION_RUNS:
        # QUAN TRỌNG: Cập nhật CFG.MODEL_NAME để các module (dataset, train, ...)
        # dùng đúng tên model của run hiện tại
        CFG.MODEL_NAME = run_cfg.model_name

        # Tạo tokenizer mới cho mỗi run (mỗi backbone có tokenizer khác nhau)
        tokenizer = AutoTokenizer.from_pretrained(run_cfg.model_name)

        # Nhân bản dev×15 hoặc giữ nguyên tùy cấu hình run
        df_dev_x15 = pd.concat([df_dev_raw] * 15, ignore_index=True) if run_cfg.use_dev_x15 else df_dev_raw

        # Tạo Dataset objects với tokenizer tương ứng
        train_ds  = TextDetectionDataset(df_train_raw,        tokenizer, CFG.MAX_LEN)
        val_ds    = TextDetectionDataset(df_val_raw,          tokenizer, CFG.MAX_LEN)
        target_ds = TextDetectionDataset(df_dev_x15,          tokenizer, CFG.MAX_LEN)
        dev_ds    = TextDetectionDataset(df_dev_raw,          tokenizer, CFG.MAX_LEN)
        test_ds   = TextDetectionDataset(df_test_labeled_raw, tokenizer, CFG.MAX_LEN)
        submit_ds = InferenceDataset(df_test_unlabeled_raw,   tokenizer, CFG.MAX_LEN)

        result = run_single(
            run_cfg=run_cfg,
            train_ds=train_ds, val_ds=val_ds, target_ds=target_ds,
            dev_ds=dev_ds, test_ds=test_ds, submit_ds=submit_ds,
            tokenizer=tokenizer,
            _df_val_raw=df_val_raw, _df_dev_raw=df_dev_raw,
            _df_test_labeled_raw=df_test_labeled_raw,
            _df_test_unlabeled_raw=df_test_unlabeled_raw,
        )
        all_results.append(result)

        # Theo dõi run có FPR thấp nhất để dùng làm điểm khởi đầu nếu cần
        if (best_state is None or
                result["fpr_at_95tpr"] < min(r["fpr_at_95tpr"] for r in all_results[:-1])
                if len(all_results) > 1 else True):
            best_state = {"source_run_id": result["run_id"], "state": result.get("model_state", {})}

    # ── Tổng kết: vẽ biểu đồ và lưu kết quả ─────────────────────────────────
    plot_ablation_summary(all_results)   # Bar chart so sánh FPR / F1 / AUC
    plot_roc_curves(all_results)         # ROC curve overlay tất cả runs
    _print_summary_table(all_results)    # Bảng số liệu ra console

    # Lưu kết quả dạng JSON (không lưu model_state vì quá nặng)
    save_path = os.path.join(CFG.PLOT_DIR, "ablation_results.json")
    save_results = [{k: v for k, v in r.items() if k not in ("model_state", "roc_fpr", "roc_tpr")}
                    for r in all_results]
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Kết quả đã lưu: {save_path}")
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY TABLE — In bảng tổng kết ra console
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary_table(results: list):
    """
    In bảng so sánh tất cả runs ra console với định dạng gọn gàng.
    Run có FPR thấp nhất được đánh dấu ◀ BEST.
    """
    print("\n" + "=" * 80 + "\n  ABLATION STUDY — KẾT QUẢ TỔNG HỢP\n" + "=" * 80)
    print(f"  {'Run ID':<28} {'DANN':^5} {'lr':^8} {'drop':^6} {'FPR@95%↓':^10} {'F1↑':^8} {'AUC↑':^8}")
    print("  " + "-" * 76)
    best_fpr = min(r["fpr_at_95tpr"] for r in results)
    for r in results:
        mark = " ◀ BEST" if r["fpr_at_95tpr"] == best_fpr else ""
        print(f"  {r['run_id']:<28} {'✓' if r['use_dann'] else '✗':^5} "
              f"{r['lr']:^8.0e} {r['dropout']:^6.1f} "
              f"{r['fpr_at_95tpr']:^10.4f} {r['test_f1']:^8.4f} {r['test_auc']:^8.4f}{mark}")
    print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Đăng nhập W&B trước khi chạy (cần API key từ https://wandb.ai/authorize)
    wandb.login()
    run_all()
