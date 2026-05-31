# -*- coding: utf-8 -*-
# configs/config.py — Cấu hình tập trung cho toàn bộ project

import os

class CFG:
    # ── Đường dẫn dữ liệu ────────────────────────────────────────────────────
    # Colab/Kaggle: chỉnh lại các path này cho phù hợp môi trường của bạn
    TRAIN_PATH          = "/workspace/cleaned_text_data.jsonl"
    DEV_PATH            = "/workspace/SubtaskA/subtaskA_dev_monolingual.jsonl"
    TEST_LABELED_PATH   = "/workspace/subtaskA_monolingual_labeled.jsonl"
    TEST_UNLABELED_PATH = "/workspace/subtaskA_monolingual_unlabeled.jsonl"

    # ── Model ─────────────────────────────────────────────────────────────────
    MODEL_NAME   = "roberta-base"
    MAX_LEN      = 256
    NUM_CLASSES  = 2   # 0: Human, 1: Machine
    NUM_DOMAINS  = 2   # 0: Source (train), 1: Target (dev)

    # ── Training ──────────────────────────────────────────────────────────────
    BATCH_SIZE   = 8
    EPOCHS       = 3
    PATIENCE     = 1
    ACCUM_STEPS  = 4
    VAL_SIZE     = 0.2
    SEED         = 42

    # ── Cố định (không tune trong ablation) ───────────────────────────────────
    WEIGHT_DECAY = 0.01
    GRAD_CLIP    = 1.0

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_EVERY_N_STEPS  = 50
    HIST_EVERY_N_EPOCH = 1

    # ── W&B ───────────────────────────────────────────────────────────────────
    WANDB_PROJECT = "ai-text-detector-ablation"
    WANDB_ENTITY  = None   # Điền username W&B của bạn nếu cần

    # ── Output dirs ───────────────────────────────────────────────────────────
    BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PLOT_DIR  = os.path.join(BASE_DIR, "outputs", "plots")
    CKPT_DIR  = os.path.join(BASE_DIR, "outputs", "checkpoints")
    SUB_DIR   = os.path.join(BASE_DIR, "outputs", "submissions")
