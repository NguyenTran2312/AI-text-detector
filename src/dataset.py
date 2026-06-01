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
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts          = df["text"].tolist()
        self.class_labels   = df["label"].values
        self.domain_labels  = df["domain_id"].values
        self.tokenizer      = tokenizer
        self.max_len        = max_len

    def __len__(self):
        return len(self.class_labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding=False,
        )
        return {
            "input_ids":      encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels_class":   int(self.class_labels[idx]),
            "labels_domain":  int(self.domain_labels[idx]),
        }

class InferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts     = df["text"].tolist()
        self.ids       = df["id"].values
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding=False,
        )
        return {
            "input_ids":      encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
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

# ── Build all datasets ────────────────────────────────────────────────────────
def build_data(tokenizer):
    df_train_full = load_json(CFG.TRAIN_PATH)
    df_train_full["domain_id"] = 0
    df_train, df_val = train_test_split(
        df_train_full, test_size=CFG.VAL_SIZE, random_state=CFG.SEED, stratify=df_train_full["label"]
    )

    df_dev = load_json(CFG.DEV_PATH)
    df_dev["domain_id"] = 1
    df_dev_x15 = pd.concat([df_dev] * 15, ignore_index=True)
    df_dev_x15["domain_id"] = 1

    df_test_labeled = load_json(CFG.TEST_LABELED_PATH)
    df_test_labeled["domain_id"] = 1

    df_test_unlabeled = load_json(CFG.TEST_UNLABELED_PATH)

    print("=" * 55)
    print(f"  [1] Train (source)          : {len(df_train):>7,}")
    print(f"  [1] Val   (source)          : {len(df_val):>7,}")
    print(f"  [2] Dev target ×15 (DANN)   : {len(df_dev_x15):>7,}")
    print(f"  [2] Dev gốc (eval shift)    : {len(df_dev):>7,}")
    print(f"  [3a] Test có label (metric) : {len(df_test_labeled):>7,}")
    print(f"  [3b] Test unlabeled (sub.)  : {len(df_test_unlabeled):>7,}")
    print("=" * 55)

    train_ds  = TextDetectionDataset(df_train,        tokenizer, CFG.MAX_LEN)
    val_ds    = TextDetectionDataset(df_val,          tokenizer, CFG.MAX_LEN)
    target_ds = TextDetectionDataset(df_dev_x15,      tokenizer, CFG.MAX_LEN)
    dev_ds    = TextDetectionDataset(df_dev,          tokenizer, CFG.MAX_LEN)
    test_ds   = TextDetectionDataset(df_test_labeled, tokenizer, CFG.MAX_LEN)
    submit_ds = InferenceDataset(df_test_unlabeled,   tokenizer, CFG.MAX_LEN)

    return train_ds, val_ds, target_ds, dev_ds, test_ds, submit_ds

# ── Make Loaders ──────────────────────────────────────────────────────────────
def make_loaders(train_ds, val_ds, target_ds, dev_ds, test_ds, submit_ds, tokenizer, use_dev_x15: bool):
    collate_fn     = MultiTaskDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    inf_collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    NUM_WORKERS = 12 

    train_loader  = DataLoader(train_ds,  batch_size=CFG.BATCH_SIZE,   shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    val_loader    = DataLoader(val_ds,    batch_size=CFG.BATCH_SIZE*2, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    dev_loader    = DataLoader(dev_ds,    batch_size=CFG.BATCH_SIZE*2, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    test_loader   = DataLoader(test_ds,   batch_size=CFG.BATCH_SIZE*2, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    submit_loader = DataLoader(submit_ds, batch_size=CFG.BATCH_SIZE*2, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True, collate_fn=inf_collate_fn)
                               
    target_loader = None
    if use_dev_x15:
        target_loader = DataLoader(target_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
                                   num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn, drop_last=True)

    return train_loader, val_loader, target_loader, dev_loader, test_loader, submit_loader
