# -*- coding: utf-8 -*-
# src/train.py — Training loop, Evaluate, Checkpoint helpers

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb

from sklearn.metrics import f1_score, accuracy_score, roc_curve, roc_auc_score

from configs.config import CFG
from src.model import compute_lambda


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def ckpt_path(run_id: str, tag: str = "best") -> str:
    return os.path.join(CFG.CKPT_DIR, f"{run_id}_{tag}.pt")


def save_checkpoint(model, optimizer, scheduler, epoch: int, val_f1: float,
                    run_id: str, tag: str = "best"):
    path = ckpt_path(run_id, tag)
    torch.save({
        "epoch":       epoch,
        "val_f1":      val_f1,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "sched_state": scheduler.state_dict(),
    }, path)
    print(f"  [CKPT] Saved {os.path.basename(path)} — epoch={epoch+1}, val_F1={val_f1:.4f}")


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer  and "optim_state" in ckpt: optimizer.load_state_dict(ckpt["optim_state"])
    if scheduler  and "sched_state" in ckpt: scheduler.load_state_dict(ckpt["sched_state"])
    print(f"  [CKPT] Loaded {os.path.basename(path)} — epoch={ckpt.get('epoch',0)+1}, val_F1={ckpt.get('val_f1',0):.4f}")
    return ckpt.get("epoch", 0), ckpt.get("val_f1", 0.0)


def run_already_done(run_id: str) -> bool:
    return os.path.exists(ckpt_path(run_id, "final"))


# ── Train one epoch ───────────────────────────────────────────────────────────
def train_one_epoch(model, source_loader, target_loader, optimizer, scheduler,
                    epoch: int, total_epochs: int, global_step: int, use_dann: bool,
                    device):
    model.train()
    ce_class  = nn.CrossEntropyLoss()
    ce_domain = nn.CrossEntropyLoss()

    total_loss = total_class_loss = total_domain_loss = 0.0
    all_preds, all_labels = [], []

    if target_loader is not None:
        len_dl      = min(len(source_loader), len(target_loader))
        target_iter = iter(target_loader)
    else:
        len_dl      = len(source_loader)
        target_iter = None

    source_iter = iter(source_loader)
    optimizer.zero_grad()

    for step in range(len_dl):
        src = next(source_iter)
        lambda_ = compute_lambda(epoch, step, len_dl, total_epochs) if use_dann else 0.0
        model.update_alpha(lambda_)

        src_ids  = src["input_ids"].to(device)
        src_mask = src["attention_mask"].to(device)
        src_cls  = src["labels_class"].to(device)
        src_dom  = src["labels_domain"].to(device)

        src_class_logits, src_domain_logits = model(src_ids, src_mask)
        loss_class = ce_class(src_class_logits, src_cls)
        loss       = loss_class
        loss_domain_val = 0.0

        if use_dann and src_domain_logits is not None:
            loss_domain_src = ce_domain(src_domain_logits, src_dom)
            if target_iter is not None:
                tgt = next(target_iter)
                tgt_ids  = tgt["input_ids"].to(device)
                tgt_mask = tgt["attention_mask"].to(device)
                tgt_dom  = tgt["labels_domain"].to(device)
                _, tgt_domain_logits = model(tgt_ids, tgt_mask)
                loss_domain = (loss_domain_src + ce_domain(tgt_domain_logits, tgt_dom)) / 2.0
            else:
                loss_domain = loss_domain_src
            loss = loss_class + lambda_ * loss_domain
            loss_domain_val = loss_domain.item()

        (loss / CFG.ACCUM_STEPS).backward()

        total_loss        += loss.item()
        total_class_loss  += loss_class.item()
        total_domain_loss += loss_domain_val
        all_preds.extend(src_class_logits.argmax(-1).detach().cpu().numpy())
        all_labels.extend(src_cls.cpu().numpy())

        if (step + 1) % CFG.ACCUM_STEPS == 0 or (step + 1) == len_dl:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if global_step % CFG.LOG_EVERY_N_STEPS == 0:
                wandb.log({
                    "step/loss_total"  : total_loss  / (step + 1),
                    "step/loss_class"  : total_class_loss  / (step + 1),
                    "step/loss_domain" : total_domain_loss / (step + 1),
                    "step/lr"          : scheduler.get_last_lr()[0],
                    "step/grl_alpha"   : lambda_,
                }, step=global_step)
            global_step += 1

    train_f1  = f1_score(all_labels, all_preds, average="macro")
    train_acc = accuracy_score(all_labels, all_preds)
    return (
        total_loss        / len_dl,
        total_class_loss  / len_dl,
        total_domain_loss / len_dl,
        train_f1, train_acc,
        global_step,
    )


# ── Evaluate ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    for batch in loader:
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["labels_class"].to(device)

        class_logits, _ = model(ids, mask)
        total_loss += ce(class_logits, labels).item()

        probs = torch.softmax(class_logits, -1)[:, 1].cpu().numpy()
        preds = class_logits.argmax(-1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    f1  = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)
    return f1, acc, total_loss / len(loader), np.array(all_probs), np.array(all_labels)


# ── FPR / Threshold helpers ───────────────────────────────────────────────────
def compute_fpr_at_tpr(probs, labels, target_tpr: float = 0.95):
    fpr_arr, tpr_arr, thresholds = roc_curve(labels, probs, pos_label=1)
    idx = np.argmin(np.abs(tpr_arr - target_tpr))
    return float(fpr_arr[idx]), float(thresholds[idx])


def find_optimal_threshold(probs, labels, target_fpr: float = 0.05):
    fpr_arr, tpr_arr, thresholds = roc_curve(labels, probs, pos_label=1)
    mask = fpr_arr <= target_fpr
    if not mask.any():
        return float(thresholds[0]), float(fpr_arr[0])
    best_idx = np.argmax(tpr_arr[mask])
    return float(thresholds[mask][best_idx]), float(fpr_arr[mask][best_idx])


# ── Generate submission ───────────────────────────────────────────────────────
@torch.no_grad()
def generate_submission(model, submit_loader, submit_ds, threshold: float,
                        run_id: str, device) -> pd.DataFrame:
    model.eval()
    all_preds = []
    for batch in submit_loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        class_logits, _ = model(ids, mask)
        probs = torch.softmax(class_logits, -1)[:, 1].cpu().numpy()
        all_preds.extend((probs >= threshold).astype(int))

    df_sub = pd.DataFrame({"id": submit_ds.ids, "label": all_preds})
    out_path = os.path.join(CFG.SUB_DIR, f"submission_{run_id}.csv")
    df_sub.to_csv(out_path, index=False)
    print(f"  [SUB] {out_path} — {len(all_preds)} predictions, threshold={threshold:.4f}")
    return df_sub
