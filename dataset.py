# -*- coding: utf-8 -*-
# src/dataset.py — Dataset, Collator, DataLoader builder

import pandas as pd
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split

from configs.config import CFG


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_json(file_path: str) -> pd.DataFrame:
    return pl.read_ndjson(file_path).to_pandas()


# ── Datasets ──────────────────────────────────────────────────────────────────
class TextDetectionDataset(Dataset):
    """Dataset có label — dùng cho train / val / dev / test labeled."""
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        encodings = tokenizer(
            df["text"].tolist(),
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding=False,
        )
        self.input_ids      = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.class_labels   = df["label"].values
        self.domain_labels  = df["domain_id"].values

    def __len__(self):
        return len(self.class_labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels_class":   int(self.class_labels[idx]),
            "labels_domain":  int(self.domain_labels[idx]),
        }


class InferenceDataset(Dataset):
    """Dataset không có label — dùng cho file test submission."""
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        encodings = tokenizer(
            df["text"].tolist(),
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding=False,
        )
        self.input_ids      = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.ids            = df["id"].values

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


# ── Collator ──────────────────────────────────────────────────────────────────
class MultiTaskDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        class_labels  = [f.pop("labels_class")  for f in features]
        domain_labels = [f.pop("labels_domain") for f in features]
        batch = super().__call__(features)
        batch["labels_class"]  = torch.tensor(class_labels,  dtype=torch.long)
        batch["labels_domain"] = torch.tensor(domain_labels, dtype=torch.long)
        return batch


# ── Build all datasets (tokenize 1 lần) ──────────────────────────────────────
def build_data(tokenizer):
    """
    Load và tokenize toàn bộ dữ liệu 1 lần, tái sử dụng cho tất cả ablation runs.

    3 file nguồn:
      [1] TRAIN_PATH          → train + val (source domain, có label)
      [2] DEV_PATH            → DANN target ×15 + eval domain shift (có label)
      [3a] TEST_LABELED_PATH  → đo FPR@5%, F1, AUC (có label)
      [3b] TEST_UNLABELED_PATH→ generate submission (không có label)
    """
    # [1] Source
    df_train_full = load_json(CFG.TRAIN_PATH)
    df_train_full["domain_id"] = 0
    df_train, df_val = train_test_split(
        df_train_full,
        test_size=CFG.VAL_SIZE,
        random_state=CFG.SEED,
        stratify=df_train_full["label"],
    )

    # [2] Dev → DANN target ×15 + eval domain shift
    df_dev = load_json(CFG.DEV_PATH)
    df_dev["domain_id"] = 1
    df_dev_x15 = pd.concat([df_dev] * 15, ignore_index=True)
    df_dev_x15["domain_id"] = 1

    # [3a] Test có label
    df_test_labeled = load_json(CFG.TEST_LABELED_PATH)
    df_test_labeled["domain_id"] = 1

    # [3b] Test không label
    df_test_unlabeled = load_json(CFG.TEST_UNLABELED_PATH)

    print("=" * 55)
    print(f"  [1] Train (source)          : {len(df_train):>7,}")
    print(f"  [1] Val   (source)          : {len(df_val):>7,}")
    print(f"  [2] Dev target ×15 (DANN)   : {len(df_dev_x15):>7,}")
    print(f"  [2] Dev gốc (eval shift)    : {len(df_dev):>7,}")
    print(f"  [3a] Test có label (metric) : {len(df_test_labeled):>7,}")
    print(f"  [3b] Test unlabeled (sub.)  : {len(df_test_unlabeled):>7,}")
    print("=" * 55)

    print("\nTokenizing (1 lần cho toàn bộ ablation)...")
    train_ds  = TextDetectionDataset(df_train,        tokenizer, CFG.MAX_LEN)
    val_ds    = TextDetectionDataset(df_val,          tokenizer, CFG.MAX_LEN)
    target_ds = TextDetectionDataset(df_dev_x15,      tokenizer, CFG.MAX_LEN)
    dev_ds    = TextDetectionDataset(df_dev,          tokenizer, CFG.MAX_LEN)
    test_ds   = TextDetectionDataset(df_test_labeled, tokenizer, CFG.MAX_LEN)
    submit_ds = InferenceDataset(df_test_unlabeled,   tokenizer, CFG.MAX_LEN)
    print("Tokenization hoàn tất!\n")

    return train_ds, val_ds, target_ds, dev_ds, test_ds, submit_ds


def make_loaders(train_ds, val_ds, target_ds, dev_ds, test_ds, submit_ds,
                 tokenizer, use_dev_x15: bool):
    collate_fn     = MultiTaskDataCollator(tokenizer=tokenizer)
    inf_collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader  = DataLoader(train_ds,  batch_size=CFG.BATCH_SIZE,   shuffle=True,
                               num_workers=0, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    val_loader    = DataLoader(val_ds,    batch_size=CFG.BATCH_SIZE*2, shuffle=False,
                               num_workers=0, pin_memory=True, collate_fn=collate_fn)
    dev_loader    = DataLoader(dev_ds,    batch_size=CFG.BATCH_SIZE*2, shuffle=False,
                               num_workers=0, pin_memory=True, collate_fn=collate_fn)
    test_loader   = DataLoader(test_ds,   batch_size=CFG.BATCH_SIZE*2, shuffle=False,
                               num_workers=0, pin_memory=True, collate_fn=collate_fn)
    submit_loader = DataLoader(submit_ds, batch_size=CFG.BATCH_SIZE*2, shuffle=False,
                               num_workers=0, pin_memory=True, collate_fn=inf_collate_fn)
    target_loader = None
    if use_dev_x15:
        target_loader = DataLoader(target_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
                                   num_workers=0, pin_memory=True, collate_fn=collate_fn, drop_last=True)

    return train_loader, val_loader, target_loader, dev_loader, test_loader, submit_loader
