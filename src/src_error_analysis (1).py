# -*- coding: utf-8 -*-
# src/error_analysis.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

from configs.config import CFG

LABEL_NAMES = {0: "Human", 1: "Machine"}
BLUE   = "#4C72B0"; ORANGE = "#DD8452"; GREEN  = "#55A868"; RED    = "#C44E52"; PURPLE = "#8172B2"; GRAY   = "#8C8C8C"
DARK_PARAMS = {
    "figure.facecolor": "#0F0F0F", "axes.facecolor":  "#1A1A1A", "axes.edgecolor":   "#444",
    "axes.labelcolor": "#DDD",     "xtick.color":      "#AAA",    "ytick.color":     "#AAA",
    "text.color":       "#EEE",    "grid.color":      "#333",    "grid.linewidth":   0.5,
}

@torch.no_grad()
def get_predictions(model, df: pd.DataFrame, tokenizer, device, batch_size: int = 32) -> pd.DataFrame:
    from src.dataset import TextDetectionDataset, MultiTaskDataCollator
    model.eval()
    is_labeled = "label" in df.columns
    
    df_loader = df.copy()
    if not is_labeled: df_loader["label"] = 0
    if "domain_id" not in df_loader.columns: df_loader["domain_id"] = 1

    collate_fn = MultiTaskDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    dataset    = TextDetectionDataset(df_loader, tokenizer, CFG.MAX_LEN)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    ce = nn.CrossEntropyLoss(reduction="none")
    all_probs, all_preds, all_losses = [], [], []

    for batch in loader:
        input_ids, mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels_class"].to(device)
        logits, _ = model(input_ids, mask)
        probs  = torch.softmax(logits, dim=-1)
        all_probs.extend(probs.float().cpu().numpy())
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        if is_labeled:
            all_losses.extend(ce(logits, labels).float().cpu().numpy())

    res = df.copy().reset_index(drop=True)
    probs_arr = np.array(all_probs)
    res["prob_human"], res["prob_machine"], res["pred"], res["confidence"] = probs_arr[:, 0], probs_arr[:, 1], all_preds, probs_arr.max(axis=1)
    res["pred_name"], res["text_len"] = res["pred"].map(LABEL_NAMES), res["text"].str.split().str.len()
    
    if is_labeled:
        res["loss"], res["correct"], res["label_name"] = all_losses, res["pred"] == res["label"], res["label"].map(LABEL_NAMES)
        def _et(r):
            if r["label"] == 1: return "TP" if r["pred"] == 1 else "FN"
            return "FP" if r["pred"] == 1 else "TN"
        res["error_type"] = res.apply(_et, axis=1)
    return res

def print_overview(df_res: pd.DataFrame, name: str):
    if "label" not in df_res.columns: return
    print(f"\n{'='*50}\n📊 OVERVIEW — {name}\n{'='*50}")
    print(f"Total: {len(df_res):,} | Acc: {df_res['correct'].mean()*100:.2f}%")
    print(classification_report(df_res["label"], df_res["pred"], target_names=["Human","Machine"], digits=4))

def plot_error_analysis(df_val: pd.DataFrame, df_dev: pd.DataFrame, save_path: str):
    plt.rcParams.update(DARK_PARAMS)
    fig = plt.figure(figsize=(20, 24), facecolor="#0F0F0F")
    fig.suptitle("Error Analysis (Validation vs Dev Set)", fontsize=18, fontweight="bold", color="white", y=0.97)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.42, wspace=0.35)
    sources = sorted(list(set(df_val["source"].unique()) | set(df_dev["source"].unique())))
    src_colors = dict(zip(sources, [BLUE, ORANGE, GREEN, RED, PURPLE, GRAY][:len(sources)]))

    # Row 0: CMs & ROC
    ax = fig.add_subplot(gs[0, 0]); sns.heatmap(confusion_matrix(df_val["label"], df_val["pred"]), annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False, xticklabels=["Human","Machine"], yticklabels=["Human","Machine"])
    ax.set_title("Confusion Matrix — Val Set"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax = fig.add_subplot(gs[0, 1]); sns.heatmap(confusion_matrix(df_dev["label"], df_dev["pred"]), annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False, xticklabels=["Human","Machine"], yticklabels=["Human","Machine"])
    ax.set_title("Confusion Matrix — Dev Set (bloomz)"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax = fig.add_subplot(gs[0, 2])
    for df, n, c in [(df_val, "Val", BLUE), (df_dev, "Dev", ORANGE)]:
        fpr, tpr, _ = roc_curve(df["label"], df["prob_machine"])
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{n} (AUC={auc(fpr,tpr):.4f})")
    ax.plot([0,1],[0,1],"w--",alpha=0.3); ax.set_title("ROC Curve"); ax.legend(); ax.grid(True, alpha=0.1)

    # Row 1: F1 per Source & AI Model
    ax = fig.add_subplot(gs[1, 0])
    v_f1 = {s: f1_score(g["label"], g["pred"], average="macro") for s, g in df_val.groupby("source")}
    ax.bar(v_f1.keys(), v_f1.values(), color=[src_colors.get(s, GRAY) for s in v_f1], width=0.5)
    ax.set_title("F1 per Source — Val"); ax.set_ylim(0, 1.1); ax.yaxis.grid(True, alpha=0.1)
    ax = fig.add_subplot(gs[1, 1])
    d_f1 = {s: f1_score(g["label"], g["pred"], average="macro") for s, g in df_dev.groupby("source")}
    ax.bar(d_f1.keys(), d_f1.values(), color=[src_colors.get(s, GRAY) for s in d_f1], width=0.5)
    ax.set_title("F1 per Source — Dev"); ax.set_ylim(0, 1.1); ax.yaxis.grid(True, alpha=0.1)
    ax = fig.add_subplot(gs[1, 2])
    m_f1 = {m: f1_score(g["label"], g["pred"], average="macro") for m, g in df_dev.groupby("model")}
    ax.bar(m_f1.keys(), m_f1.values(), color=[RED if k=="bloomz" else GREEN for k in m_f1], width=0.4)
    ax.set_title("F1 per AI Model — Dev"); ax.set_ylim(0, 1.1); ax.yaxis.grid(True, alpha=0.1)

    # Row 2: Conf, Loss, Len
    bins = np.linspace(0.5, 1.0, 25)
    ax = fig.add_subplot(gs[2, 0]); ax.hist(df_dev[df_dev["correct"]]["confidence"], bins=bins, color=GREEN, alpha=0.6, density=True, label="Correct"); ax.hist(df_dev[~df_dev["correct"]]["confidence"], bins=bins, color=RED, alpha=0.6, density=True, label="Wrong")
    ax.set_title("Confidence — Dev Set"); ax.legend()
    ax = fig.add_subplot(gs[2, 1])
    for et, col in [("TN", BLUE), ("TP", GREEN), ("FP", ORANGE), ("FN", RED)]:
        sub = df_dev[df_dev["error_type"] == et]
        if len(sub) > 0: ax.hist(sub["loss"].clip(upper=5), bins=25, color=col, alpha=0.5, label=et, density=True)
    ax.set_title("Loss by Error Type — Dev"); ax.legend()
    ax = fig.add_subplot(gs[2, 2])
    bins_len = [0, 50, 100, 200, 300, 500, 1000, 99999]; labels_len = ["<50", "50-100", "100-200", "200-300", "300-500", "500-1k", ">1k"]
    df_dev["len_bin"] = pd.cut(df_dev["text_len"], bins=bins_len, labels=labels_len)
    bin_acc = df_dev.groupby("len_bin", observed=True)["correct"].mean()
    ax.bar(bin_acc.index, bin_acc.values, color=PURPLE, width=0.6); ax.axhline(0.5, color=RED, ls=":")
    ax.set_title("Accuracy by Text Length — Dev"); ax.set_ylim(0, 1.1)

    # Row 3: Top Errors, PR, Calib
    ax = fig.add_subplot(gs[3, 0])
    fp_sub, fn_sub = df_dev[df_dev["error_type"]=="FP"]["source"].value_counts(), df_dev[df_dev["error_type"]=="FN"]["source"].value_counts()
    x = np.arange(len(sources)); w = 0.35
    ax.bar(x - w/2, [fp_sub.get(s, 0) for s in sources], w, label="FP", color=ORANGE, alpha=0.8)
    ax.bar(x + w/2, [fn_sub.get(s, 0) for s in sources], w, label="FN", color=RED, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(sources, rotation=15); ax.set_title("High-Conf Errors by Source — Dev"); ax.legend()
    ax = fig.add_subplot(gs[3, 1])
    for df, n, col in [(df_val, "Val", BLUE), (df_dev, "Dev", ORANGE)]:
        prec, rec, _ = precision_recall_curve(df["label"], df["prob_machine"])
        ax.plot(rec, prec, color=col, lw=2, label=f"{n} (AUC={auc(rec, prec):.4f})")
    ax.set_title("Precision-Recall Curve"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.legend()
    ax = fig.add_subplot(gs[3, 2])
    for df, n, col in [(df_val, "Val", BLUE), (df_dev, "Dev", ORANGE)]:
        df = df.copy(); df["prob_bin"] = pd.cut(df["prob_machine"], bins=np.linspace(0, 1, 11))
        cal = df.groupby("prob_bin", observed=True).apply(lambda g: pd.Series({"mp": g["prob_machine"].mean(), "ar": (g["label"]==1).mean()})).dropna()
        ax.plot(cal["mp"], cal["ar"], "o-", color=col, lw=2, ms=4, label=n)
    ax.plot([0, 1], [0, 1], "w--", alpha=0.3); ax.set_title("Calibration Plot"); ax.legend()

    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="#0F0F0F"); plt.close()

def plot_labeled_test_analysis(df_test: pd.DataFrame, save_path: str):
    """Đồ thị gọn gàng cho TEST LABELED (OUTFOX): Tránh cột rỗng, tập trung vào CM và text len"""
    plt.rcParams.update(DARK_PARAMS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0F0F0F")
    fig.suptitle("Supervised Domain Analysis — Labeled Test Set (OUTFOX)", fontsize=13, fontweight="bold", color="white", y=1.02)
    
    # Panel 1: Matrix nhầm lẫn
    ax = axes[0]
    sns.heatmap(confusion_matrix(df_test["label"], df_test["pred"]), annot=True, fmt="d", cmap="Oranges", ax=ax, cbar=False, xticklabels=["Human","Machine"], yticklabels=["Human","Machine"])
    ax.set_title("Confusion Matrix (Test Labeled)", fontsize=11, pad=8)
    
    # Panel 2: Accuracy theo độ dài text
    ax = axes[1]
    bins_len = [0, 50, 100, 200, 300, 500, 1000, 99999]; labels_len = ["<50", "50-100", "100-200", "200-300", "300-500", "500-1k", ">1k"]
    df_test["len_bin"] = pd.cut(df_test["text_len"], bins=bins_len, labels=labels_len)
    bin_acc = df_test.groupby("len_bin", observed=True)["correct"].mean()
    ax.bar(bin_acc.index, bin_acc.values, color=ORANGE, edgecolor="#333", width=0.6)
    ax.axhline(0.5, color=RED, ls=":")
    ax.set_title("Accuracy by Text Length (Test Labeled)", fontsize=11, pad=8)
    ax.set_ylim(0, 1.1)
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0F0F0F"); plt.close()

def plot_unlabeled_test_analysis(df_test: pd.DataFrame, save_path: str):
    """Đồ thị tối giản cho TEST UNLABELED (Submission Set)"""
    plt.rcParams.update(DARK_PARAMS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0F0F0F")
    fig.suptitle("Unsupervised Domain Analysis — Unlabeled Test Set (Submission)", fontsize=13, fontweight="bold", color="white", y=1.02)

    # Panel 1: Phân phối điểm P(Machine)
    ax = axes[0]; ax.hist(df_test["prob_machine"], bins=40, color=PURPLE, alpha=0.7)
    ax.axvline(0.5, color=RED, ls="--")
    num_m = (df_test["prob_machine"] >= 0.5).sum(); pct_m = num_m / len(df_test) * 100
    ax.set_title("Predicted Probability Distribution", fontsize=11, pad=8)
    ax.legend(fontsize=9, title=f"Predicted Machine:\n{num_m:,} ({pct_m:.1f}%)")

    # Panel 2: Độ dài câu theo dự đoán
    ax = axes[1]
    for lbl, col, name in [(0, GREEN, "Pred Human"), (1, ORANGE, "Pred Machine")]:
        sub = df_test[df_test["pred"] == lbl]
        if len(sub) > 0: sns.kdeplot(sub["text_len"].clip(upper=800), ax=ax, color=col, shade=True, label=f"{name} (avg={sub['text_len'].mean():.1f})")
    ax.set_title("Text Length Distribution by Prediction", fontsize=11, pad=8)
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0F0F0F"); plt.close()

def run_error_analysis(run_id: str, model, tokenizer, device, df_val: pd.DataFrame, df_dev: pd.DataFrame, df_test_labeled: pd.DataFrame = None, df_test_unlabeled: pd.DataFrame = None):
    print(f"\n{'='*60}\n🔍 EXPERT ERROR ANALYSIS WORKFLOW — {run_id}\n{'='*60}")
    
    df_val_res = get_predictions(model, df_val, tokenizer, device)
    df_dev_res = get_predictions(model, df_dev, tokenizer, device)
    print_overview(df_val_res, "Validation Set")
    print_overview(df_dev_res, "Dev Set (Bloomz)")
    
    grid_path = os.path.join(CFG.PLOT_DIR, f"{run_id}_error_analysis.png")
    plot_error_analysis(df_val_res, df_dev_res, grid_path)
    
    import wandb
    w_dict = {f"{run_id}/plots/val_dev_error_grid": wandb.Image(grid_path)} if wandb.run else {}

    if df_test_labeled is not None:
        print("  Analyzing Labeled Test set...")
        df_tl_res = get_predictions(model, df_test_labeled, tokenizer, device)
        print_overview(df_tl_res, "Test Labeled Set (OUTFOX)")
        p_tl = os.path.join(CFG.PLOT_DIR, f"{run_id}_test_labeled_analysis.png")
        plot_labeled_test_analysis(df_tl_res, p_tl)
        if wandb.run: w_dict[f"{run_id}/plots/test_labeled_analysis"] = wandb.Image(p_tl)

    if df_test_unlabeled is not None:
        print("  Analyzing Unlabeled Test set...")
        df_tu_res = get_predictions(model, df_test_unlabeled, tokenizer, device)
        p_tu = os.path.join(CFG.PLOT_DIR, f"{run_id}_test_unlabeled_analysis.png")
        plot_unlabeled_test_analysis(df_tu_res, p_tu)
        if wandb.run: w_dict[f"{run_id}/plots/test_unlabeled_analysis"] = wandb.Image(p_tu)

    if wandb.run and w_dict:
        wandb.log(w_dict)
        print("  [W&B] Clean analysis graphs logged successfully.")
    return {"status": "success"}
