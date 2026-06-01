# -*- coding: utf-8 -*-
# ablation.py — Entry point: chạy 4 ablation runs
#
# Cấu trúc:
#   Run 1 — Baseline : không DANN, lr=2e-5, dropout=0.3, warmup=0.1
#   Run 2 — DANN v1  : có DANN, lr=2e-5, dropout=0.3, warmup=0.1  (config gốc)
#   Run 3 — DANN v2  : có DANN, lr=1e-5, dropout=0.3, warmup=0.1  (lr thấp hơn)
#   Run 4 — DANN v3  : có DANN, lr=2e-5, dropout=0.5, warmup=0.06 (dropout cao + warmup ngắn)
#
# Tất cả runs dùng dev×15 làm DANN target (trừ baseline).
# Metric chính: FPR@TPR=95% trên test có label (target domain).
# ==============================================================================

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

# Thêm root vào sys.path để import được configs và src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import CFG
from src.dataset import build_data, make_loaders
from src.model import DANN_TextDetector
from src.train import (
    ckpt_path, save_checkpoint, load_checkpoint,
    train_one_epoch, evaluate,
    compute_fpr_at_tpr, find_optimal_threshold,
    generate_submission,
)
from src.plots import plot_learning_curves, plot_ablation_summary, plot_roc_curves
from src.error_analysis import run_error_analysis

os.makedirs(CFG.PLOT_DIR, exist_ok=True)
os.makedirs(CFG.CKPT_DIR, exist_ok=True)
os.makedirs(CFG.SUB_DIR,  exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class AblationConfig:
    run_id:       str
    description:  str
    use_dann:     bool
    use_dev_x15:  bool
    lr:           float = 2e-5
    dropout:      float = 0.3
    warmup_ratio: float = 0.1
    history:      dict  = field(default_factory=dict)

ABLATION_RUNS = [
    # ── Run 1: Baseline ─────────────────────────────────────────────────────
    AblationConfig(
        run_id      = "run1_baseline",
        description = "Baseline — RoBERTa+LoRA, không DANN, không dev×15",
        use_dann    = False,
        use_dev_x15 = False,
        lr          = 2e-5,
        dropout     = 0.3,
        warmup_ratio= 0.1,
    ),
    # ── Run 2: DANN — config gốc ─────────────────────────────────────────────
    AblationConfig(
        run_id      = "run2_dann_default",
        description = "DANN — config gốc (lr=2e-5, dropout=0.3, warmup=0.1)",
        use_dann    = True,
        use_dev_x15 = True,
        lr          = 2e-5,
        dropout     = 0.3,
        warmup_ratio= 0.1,
    ),
    # ── Run 3: DANN — lr thấp hơn ────────────────────────────────────────────
    AblationConfig(
        run_id      = "run3_dann_lr_low",
        description = "DANN — lr=1e-5 (học chậm hơn, ít catastrophic forgetting hơn)",
        use_dann    = True,
        use_dev_x15 = True,
        lr          = 1e-5,
        dropout     = 0.3,
        warmup_ratio= 0.1,
    ),
    # ── Run 4: DANN — dropout cao + warmup ngắn ──────────────────────────────
    AblationConfig(
        run_id      = "run4_dann_dropout_warmup",
        description = "DANN — dropout=0.5, warmup=0.06 (regularize mạnh hơn)",
        use_dann    = True,
        use_dev_x15 = True,
        lr          = 2e-5,
        dropout     = 0.5,
        warmup_ratio= 0.06,
    ),
]

def seed_everything(seed: int = CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# ── SINGLE RUN ──────────────────────────────────────────────────────────────
def run_single(
    run_cfg: AblationConfig,
    train_ds, val_ds, target_ds, dev_ds, test_ds, submit_ds,
    tokenizer,
    is_threshold_only: bool  = False,
    pretrained_state: dict   = None,
    _df_val_raw              = None,
    _df_dev_raw              = None,
    _df_test_raw             = None,
) -> dict:
    seed_everything()
    run_id = run_cfg.run_id

    print(f"\n{'='*70}")
    print(f"  [{run_id}]  {run_cfg.description}")
    print(f"  use_dann={run_cfg.use_dann} | use_dev_x15={run_cfg.use_dev_x15}")
    print(f"  lr={run_cfg.lr} | dropout={run_cfg.dropout} | warmup={run_cfg.warmup_ratio}")
    print(f"{'='*70}")

    train_loader, val_loader, target_loader, dev_loader, test_loader, submit_loader = make_loaders(
        train_ds, val_ds, target_ds, dev_ds, test_ds, submit_ds,
        tokenizer, run_cfg.use_dev_x15,
    )

    model = DANN_TextDetector(
        model_name  = CFG.MODEL_NAME,
        num_classes = CFG.NUM_CLASSES,
        num_domains = CFG.NUM_DOMAINS,
        dropout     = run_cfg.dropout,
        use_lora    = True,
        use_dann    = run_cfg.use_dann,
    ).to(device)

    if is_threshold_only and pretrained_state is not None:
        src_run_id = pretrained_state.get("source_run_id", "")
        disk = ckpt_path(src_run_id, "final")
        if os.path.exists(disk):
            print(f"  [CKPT] Load từ disk: {disk}")
            load_checkpoint(disk, model)
        else:
            model.load_state_dict(pretrained_state["state"])
    else:
        len_dl       = min(len(train_loader), len(target_loader)) if target_loader else len(train_loader)
        total_steps  = (len_dl // CFG.ACCUM_STEPS) * CFG.EPOCHS
        warmup_steps = int(total_steps * run_cfg.warmup_ratio)

        optimizer = torch.optim.AdamW(model.parameters(), lr=run_cfg.lr, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        scaler = torch.amp.GradScaler('cuda') if hasattr(torch, 'amp') else torch.cuda.amp.GradScaler()

        wrun = wandb.init(
            project = CFG.WANDB_PROJECT, entity = CFG.WANDB_ENTITY,
            name = run_id, group = "ablation", config = {
                "run_id": run_id, "use_dann": run_cfg.use_dann, "use_dev_x15": run_cfg.use_dev_x15,
                "lr": run_cfg.lr, "dropout": run_cfg.dropout, "warmup_ratio": run_cfg.warmup_ratio,
                "epochs": CFG.EPOCHS, "model": CFG.MODEL_NAME,
            }, reinit=True,
        )

        best_f1, best_state, no_improve, global_step = 0.0, None, 0, 0
        history = {}

        for epoch in range(CFG.EPOCHS):
            (train_loss, train_cls_loss, train_dom_loss,
             train_f1, train_acc, global_step) = train_one_epoch(
                model, train_loader, target_loader, optimizer, scheduler, scaler,
                epoch, CFG.EPOCHS, global_step, run_cfg.use_dann, device,
            )

            val_f1, val_acc, val_loss, _, _ = evaluate(model, val_loader, device)

            history[epoch] = {
                "train_loss": train_loss, "val_loss": val_loss,
                "train_acc":  train_acc,  "val_acc":  val_acc,
                "train_f1":   train_f1,   "val_f1":   val_f1,
            }

            wandb.log({
                "epoch/train/loss": train_loss, "epoch/train/f1": train_f1,
                "epoch/val/loss": val_loss, "epoch/val/f1": val_f1, "epoch": epoch,
            })

            print(f"  Epoch {epoch+1}/{CFG.EPOCHS} | train_loss={train_loss:.4f} | val_F1={val_f1:.4f}")

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

        run_cfg.history = history
        best_disk = ckpt_path(run_id, "best")
        if os.path.exists(best_disk): load_checkpoint(best_disk, model)
        save_checkpoint(model, optimizer, scheduler, max(history.keys()), best_f1, run_id, tag="final")

        plot_path = plot_learning_curves(run_cfg)
        if plot_path: wandb.log({"curves": wandb.Image(plot_path)})
        wrun.finish()

    print("  Evaluating trên dev gốc...")
    dev_f1, dev_acc, dev_loss, dev_probs, dev_labels = evaluate(model, dev_loader, device)
    dev_fpr, _ = compute_fpr_at_tpr(dev_probs, dev_labels, target_tpr=0.95)

    print("  Evaluating trên test có label...")
    test_f1, test_acc, test_loss, test_probs, test_labels = evaluate(model, test_loader, device)
    fpr_at_95, thr_95 = compute_fpr_at_tpr(test_probs, test_labels, target_tpr=0.95)
    auc = float(roc_auc_score(test_labels, test_probs))
    fpr_arr, tpr_arr, _ = roc_curve(test_labels, test_probs, pos_label=1)

    generate_submission(model, submit_loader, submit_ds, thr_95, run_id, device)

    result = {
        "run_id": run_id, "description": run_cfg.description, "use_dann": run_cfg.use_dann,
        "use_dev_x15": run_cfg.use_dev_x15, "lr": run_cfg.lr, "dropout": run_cfg.dropout, "warmup_ratio": run_cfg.warmup_ratio,
        "dev_f1": round(dev_f1, 6), "dev_acc": round(dev_acc, 6), "dev_fpr_at_95": round(dev_fpr, 6),
        "test_f1": round(test_f1, 6), "test_acc": round(test_acc, 6), "test_auc": round(auc, 6),
        "fpr_at_95tpr": round(fpr_at_95, 6), "threshold_95": round(thr_95, 6),
        "roc_fpr": fpr_arr.tolist(), "roc_tpr": tpr_arr.tolist(),
        "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
    }

    # ── SỬA LỖI TẠI ĐÂY: Error Analysis ─────────────────────────────────────
    if _df_val_raw is not None and _df_dev_raw is not None:
        _ckpt = ckpt_path(run_id, "final")
        if os.path.exists(_ckpt):
            _model_ea = DANN_TextDetector(
                CFG.MODEL_NAME, CFG.NUM_CLASSES, CFG.NUM_DOMAINS,
                dropout=run_cfg.dropout, use_lora=True, use_dann=run_cfg.use_dann,
            ).to(device)
            load_checkpoint(_ckpt, _model_ea)
            run_error_analysis(
                run_id=run_id, model=_model_ea, tokenizer=tokenizer, device=device,
                df_val=_df_val_raw, df_dev=_df_dev_raw, df_test=_df_test_raw,
            )
            del _model_ea
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return result

# ── MAIN RUN ALL ─────────────────────────────────────────────────────────────
def run_all():
    seed_everything()
    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)
    train_ds, val_ds, target_ds, dev_ds, test_ds, submit_ds = build_data(tokenizer)

    import polars as pl
    from sklearn.model_selection import train_test_split

    _df_train_full = pl.read_ndjson(CFG.TRAIN_PATH).to_pandas()
    _df_train_full["domain_id"] = 0
    _, df_val_raw = train_test_split(
        _df_train_full, test_size=CFG.VAL_SIZE, random_state=CFG.SEED, stratify=_df_train_full["label"]
    )
    
    # ĐÃ SỬA LỖI: Gán thêm cột domain_id = 1 cho dev thô và test thô
    df_dev_raw  = pl.read_ndjson(CFG.DEV_PATH).to_pandas()
    df_dev_raw["domain_id"] = 1 # <-- THÊM DÒNG NÀY
    
    df_test_raw = pl.read_ndjson(CFG.TEST_LABELED_PATH).to_pandas()
    df_test_raw["domain_id"] = 1 # <-- THÊM DÒNG NÀY

    for df in [df_val_raw, df_dev_raw, df_test_raw]:
        if "source" not in df.columns: df["source"] = "unknown"

    all_results = []
    best_state  = None

    for run_cfg in ABLATION_RUNS:
        result = run_single(
            run_cfg=run_cfg, train_ds=train_ds, val_ds=val_ds, target_ds=target_ds,
            dev_ds=dev_ds, test_ds=test_ds, submit_ds=submit_ds, tokenizer=tokenizer,
            _df_val_raw=df_val_raw, _df_dev_raw=df_dev_raw, _df_test_raw=df_test_raw,
        )
        all_results.append(result)

        if best_state is None or result["fpr_at_95tpr"] < min(r["fpr_at_95tpr"] for r in all_results[:-1]) if len(all_results) > 1 else True:
            best_state = {"source_run_id": result["run_id"], "state": result["model_state"]}

    plot_ablation_summary(all_results)
    plot_roc_curves(all_results)
    _print_summary_table(all_results)

    save_path = os.path.join(CFG.PLOT_DIR, "ablation_results.json")
    save_results = [{k: v for k, v in r.items() if k not in ("model_state", "roc_fpr", "roc_tpr")} for r in all_results]
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    
    return all_results

def _print_summary_table(results: list):
    print("\n" + "=" * 80 + "\n  ABLATION STUDY — KẾT QUẢ TỔNG HỢP\n" + "=" * 80)
    print(f"  {'Run ID':<28} {'DANN':^5} {'lr':^8} {'drop':^6} {'FPR@95%↓':^10} {'F1↑':^8} {'AUC↑':^8}")
    print("  " + "-" * 76)
    best_fpr = min(r["fpr_at_95tpr"] for r in results)
    for r in results:
        mark = " ◀ BEST" if r["fpr_at_95tpr"] == best_fpr else ""
        print(f"  {r['run_id']:<28} {'✓' if r['use_dann'] else '✗':^5} {r['lr']:^8.0e} {r['dropout']:^6.1f} {r['fpr_at_95tpr']:^10.4f} {r['test_f1']:^8.4f} {r['test_auc']:^8.4f}{mark}")
    print("=" * 80)

if __name__ == "__main__":
    wandb.login()
    run_all()
