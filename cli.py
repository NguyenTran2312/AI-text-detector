# -*- coding: utf-8 -*-
# run_cli.py
import argparse
import os
import sys
import pandas as pd
import polars as pl
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# Nạp các module cốt lõi từ project
from configs.config import CFG
from src.dataset import TextDetectionDataset, InferenceDataset
from ablation import AblationConfig, run_single, seed_everything
import warnings
warnings.filterwarnings("ignore", message="Failed to load")

def parse_args():
    parser = argparse.ArgumentParser(description="🚀 CLI Điều khiển Huấn luyện Đơn lẻ AI Text Detector")
    
    # Các tham số cấu hình bắt buộc hoặc tùy chọn chính
    parser.add_argument("--model_name", type=str, default="roberta-base",
                        help="Tên backbone trên HuggingFace (e.g., roberta-base, microsoft/deberta-v3-base, distilroberta-base)")
    parser.add_argument("--run_id", type=str, default="custom_cli_run",
                        help="Tên định danh duy nhất cho lượt chạy này để lưu checkpoint")
    parser.add_argument("--description", type=str, default="Chạy đơn lẻ từ giao diện Terminal",
                        help="Mô tả ngắn gọn về mục đích lượt chạy")
    
    # Khởi tạo các cờ bật/tắt kiến trúc mạng
    parser.add_argument("--use_dann", action="store_true",
                        help="Bật cơ chế Domain Adaptation (DANN). Nếu không truyền mặc định là False")
    parser.add_argument("--use_dev_x15", action="store_true",
                        help="Nhân bản tập dữ liệu Target lên 15 lần để cân bằng tiến trình DANN")
    
    # Siêu tham số (Hyperparameters)
    parser.add_argument("--lr", type=float, default=2e-5, help="Tốc độ học (Learning Rate)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Tỷ lệ Dropout hệ thống")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Tỷ lệ Epoch dùng cho Warmup")
    
    # Ghi đè cấu hình cứng trong CFG (Tùy chọn nâng cao)
    parser.add_argument("--batch_size", type=int, default=None, help="Ghi đè BATCH_SIZE trong file config.py")
    parser.add_argument("--accum_steps", type=int, default=None, help="Ghi đè ACCUM_STEPS trong file config.py")
    parser.add_argument("--epochs", type=int, default=None, help="Ghi đè EPOCHS trong file config.py")
    parser.add_argument("--is_threshold_only", action="store_true",
                        help="Chỉ load checkpoint cũ để tính lại ngưỡng phân loại và phân tích lỗi, không train lại")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Ghi đè các cấu hình toàn cục vào lớp CFG dựa trên tham số terminal
    CFG.MODEL_NAME = args.model_name
    if args.batch_size is not None:
        CFG.BATCH_SIZE = args.batch_size
    if args.accum_steps is not None:
        CFG.ACCUM_STEPS = args.accum_steps
    if args.epochs is not None:
        CFG.EPOCHS = args.epochs
        
    seed_everything()

    # 2. Đóng gói tham số terminal vào đối tượng cấu hình AblationConfig
    run_cfg = AblationConfig(
        run_id       = args.run_id,
        description  = args.description,
        model_name   = args.model_name,
        use_dann     = args.use_dann,
        use_dev_x15  = args.use_dev_x15,
        lr           = args.lr,
        dropout      = args.dropout,
        warmup_ratio = args.warmup_ratio
    )

    # 3. Tiến hành đọc dữ liệu thô từ đĩa cứng bằng Polars
    print("📖 Đang nạp tập dữ liệu thô...")
    _df_train_full = pl.read_ndjson(CFG.TRAIN_PATH).to_pandas()
    _df_train_full["domain_id"] = 0
    df_train_raw, df_val_raw = train_test_split(_df_train_full, test_size=CFG.VAL_SIZE, random_state=CFG.SEED, stratify=_df_train_full["label"])
    
    df_dev_raw = pl.read_ndjson(CFG.DEV_PATH).to_pandas()
    df_dev_raw["domain_id"] = 1
    
    df_test_labeled_raw = pl.read_ndjson(CFG.TEST_LABELED_PATH).to_pandas()
    df_test_labeled_raw["domain_id"] = 1

    df_test_unlabeled_raw = pl.read_ndjson(CFG.TEST_UNLABELED_PATH).to_pandas()
    df_test_unlabeled_raw["domain_id"] = 1

    for df in [df_val_raw, df_dev_raw, df_test_labeled_raw]:
        if "source" not in df.columns: 
            df["source"] = "unknown"

    # 4. Tạo bộ Tokenizer chuẩn động và tokenize dữ liệu
    print(f"🪙 Khởi tạo bộ Tokenizer tương thích: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    df_dev_x15 = pd.concat([df_dev_raw] * 15, ignore_index=True) if args.use_dev_x15 else df_dev_raw

    train_ds  = TextDetectionDataset(df_train_raw,        tokenizer, CFG.MAX_LEN)
    val_ds    = TextDetectionDataset(df_val_raw,          tokenizer, CFG.MAX_LEN)
    target_ds = TextDetectionDataset(df_dev_x15,          tokenizer, CFG.MAX_LEN)
    dev_ds    = TextDetectionDataset(df_dev_raw,          tokenizer, CFG.MAX_LEN)
    test_ds   = TextDetectionDataset(df_test_labeled_raw, tokenizer, CFG.MAX_LEN)
    submit_ds = InferenceDataset(df_test_unlabeled_raw,   tokenizer, CFG.MAX_LEN)

    # 5. Kích hoạt tiến trình run_single tách biệt
    print("🚀 Đang khởi động tiến trình xử lý mạng Neural...")
    result = run_single(
        run_cfg=run_cfg,
        train_ds=train_ds, val_ds=val_ds, target_ds=target_ds,
        dev_ds=dev_ds, test_ds=test_ds, submit_ds=submit_ds,
        tokenizer=tokenizer,
        is_threshold_only=args.is_threshold_only,
        pretrained_state=None,
        _df_val_raw=df_val_raw,
        _df_dev_raw=df_dev_raw,
        _df_test_labeled_raw=df_test_labeled_raw,
        _df_test_unlabeled_raw=df_test_unlabeled_raw
    )

    print("\n" + "="*50 + "\n🎯 HOÀN THÀNH LƯỢT CHẠY CLI\n" + "="*50)
    print(f"  * Run ID    : {result['run_id']}")
    print(f"  * Test F1   : {result['test_f1']:.4f}")
    print(f"  * Test AUC  : {result['test_auc']:.4f}")
    print(f"  * FPR@95%   : {result['fpr_at_95tpr']:.4f}")

if __name__ == "__main__":
    main()
    
