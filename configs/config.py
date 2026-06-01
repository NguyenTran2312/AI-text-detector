# -*- coding: utf-8 -*-
# configs/config.py — Cấu hình tập trung cho toàn bộ project

import os
import torch

class CFG:
    # ── Đường dẫn dữ liệu ────────────────────────────────────────────────────
    TRAIN_PATH          = "/workspace/data/cleaned_text_data.jsonl"
    DEV_PATH            = "/workspace/data/subtaskA_dev_monolingual.jsonl"
    TEST_LABELED_PATH   = "/workspace/data/subtaskA_monolingual_labeled.jsonl"
    TEST_UNLABELED_PATH = "/workspace/data/subtaskA_monolingual_unlabeled.jsonl"

    # ── Model ─────────────────────────────────────────────────────────────────
    MODEL_NAME   = "roberta-base"
    MAX_LEN      = 256
    NUM_CLASSES  = 2   # 0: Human, 1: Machine
    NUM_DOMAINS  = 2   # 0: Source (train), 1: Target (dev)

    # ── Training (TỐI ƯU CHO RTX 5090 32GB VRAM) ─────────────────────────────
    # Thử nghiệm với 128. Nếu máy chạy êm và VRAM vẫn còn trống nhiều (xem bằng nvidia-smi),
    # bạn có thể mạnh dạn đẩy lên 256. Nếu báo lỗi "CUDA Out of Memory", hãy lùi về 128 hoặc 96.
    BATCH_SIZE  = 128  
    ACCUM_STEPS = 2
    EPOCHS       = 3
    PATIENCE     = 1
    VAL_SIZE     = 0.2
    SEED         = 42

    # ── Tối ưu phần cứng (16 vCPU) ────────────────────────────────────────────
    NUM_WORKERS = 12
    PIN_MEMORY  = True

    # ── Cố định (không tune trong ablation) ───────────────────────────────────
    WEIGHT_DECAY = 0.01
    GRAD_CLIP    = 1.0

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_EVERY_N_STEPS  = 50
    HIST_EVERY_N_EPOCH = 1

    # ── W&B ───────────────────────────────────────────────────────────────────
    WANDB_PROJECT = "ai-text-detector-ablation"
    WANDB_ENTITY  = None

    # ── Output dirs ───────────────────────────────────────────────────────────
    BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PLOT_DIR  = os.path.join(BASE_DIR, "outputs", "plots")
    CKPT_DIR  = os.path.join(BASE_DIR, "outputs", "checkpoints")
    SUB_DIR   = os.path.join(BASE_DIR, "outputs", "submissions")
