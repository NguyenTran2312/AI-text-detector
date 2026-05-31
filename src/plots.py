# -*- coding: utf-8 -*-
# src/plots.py — Learning curves, ablation summary, ROC overlay

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.config import CFG


def plot_learning_curves(run_cfg) -> str:
    """Train/Val Loss + Train/Val Acc per run. Trả về path file PNG."""
    history = run_cfg.history
    if not history:
        return ""

    epochs      = sorted(history.keys())
    epoch_nums  = [e + 1 for e in epochs]
    train_losses = [history[e]["train_loss"] for e in epochs]
    val_losses   = [history[e]["val_loss"]   for e in epochs]
    train_accs   = [history[e]["train_acc"]  for e in epochs]
    val_accs     = [history[e]["val_acc"]    for e in epochs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{run_cfg.run_id}\n{run_cfg.description}", fontsize=9, y=1.02)

    # Loss
    ax = axes[0]
    ax.plot(epoch_nums, train_losses, "o-",  color="#2196F3", lw=2, label="Train Loss")
    ax.plot(epoch_nums, val_losses,   "s--", color="#F44336", lw=2, label="Val Loss")
    best_e = epoch_nums[int(np.argmin(val_losses))]
    ax.axvline(x=best_e, color="#FF9800", ls=":", alpha=0.7, label=f"Best epoch={best_e}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss Curve")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xticks(epoch_nums)

    # Accuracy
    ax = axes[1]
    ax.plot(epoch_nums, train_accs, "o-",  color="#2196F3", lw=2, label="Train Acc")
    ax.plot(epoch_nums, val_accs,   "s--", color="#F44336", lw=2, label="Val Acc")
    ax.set_ylim([max(0, min(train_accs + val_accs) - 0.05), 1.0])
    for i, e in enumerate(epoch_nums):
        gap = train_accs[i] - val_accs[i]
        if gap > 0.02:
            ax.annotate(f"Δ={gap:.3f}", xy=(e, val_accs[i]),
                        xytext=(e + 0.05, val_accs[i] - 0.02), fontsize=7, color="#9C27B0")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("Accuracy Curve")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xticks(epoch_nums)

    plt.tight_layout()
    out = os.path.join(CFG.PLOT_DIR, f"{run_cfg.run_id}_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [PLOT] {out}")
    return out


def plot_ablation_summary(all_results: list) -> str:
    """Bar chart so sánh FPR@95% / F1 / AUC của tất cả runs."""
    run_ids = [r["run_id"]       for r in all_results]
    fprs    = [r["fpr_at_95tpr"] for r in all_results]
    f1s     = [r["test_f1"]      for r in all_results]
    aucs    = [r["test_auc"]     for r in all_results]

    colors = ["#2196F3","#4CAF50","#FF9800","#F44336","#9C27B0","#00BCD4","#795548",
              "#E91E63","#009688","#FF5722","#607D8B"][:len(run_ids)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Ablation Study — So sánh tất cả Runs (Target Domain)", fontsize=12)

    for ax, values, title, lower_better in zip(
        axes,
        [fprs, f1s, aucs],
        ["FPR @ TPR=95% ↓", "F1 Macro (Test) ↑", "AUC-ROC (Test) ↑"],
        [True, False, False],
    ):
        bars = ax.bar(range(len(run_ids)), values, color=colors, edgecolor="white", lw=0.8)
        ax.set_xticks(range(len(run_ids)))
        ax.set_xticklabels(run_ids, rotation=35, ha="right", fontsize=7)
        ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.3)

        best_idx = int(np.argmin(values)) if lower_better else int(np.argmax(values))
        bars[best_idx].set_edgecolor("#FFD700"); bars[best_idx].set_linewidth(3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=6)

    plt.tight_layout()
    out = os.path.join(CFG.PLOT_DIR, "ablation_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [PLOT] {out}")
    return out


def plot_roc_curves(all_results: list) -> str:
    """Overlay ROC curves của tất cả runs."""
    colors = ["#2196F3","#4CAF50","#FF9800","#F44336","#9C27B0","#00BCD4","#795548",
              "#E91E63","#009688","#FF5722","#607D8B"]

    fig, ax = plt.subplots(figsize=(8, 7))
    for r, c in zip(all_results, colors):
        if not r.get("roc_fpr"):
            continue
        ax.plot(r["roc_fpr"], r["roc_tpr"], color=c, lw=1.8,
                label=f"{r['run_id']} (AUC={r['test_auc']:.4f})")

    ax.axvline(x=0.05, color="red", ls="--", alpha=0.5, label="FPR=5% target")
    ax.plot([0,1],[0,1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Ablation Runs (Target Domain)")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(True, alpha=0.2)

    out = os.path.join(CFG.PLOT_DIR, "roc_curves_all.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [PLOT] {out}")
    return out
