# -*- coding: utf-8 -*-
# src/error_analysis.py
# ==============================================================================
# Error Analysis Module
# Chạy sau mỗi ablation run, phân tích:
#   1. Confusion matrix (val + test)
#   2. Score distribution — human vs machine, đúng vs sai
#   3. Threshold sweep — tìm threshold tối ưu cho F1
#   4. FP/FN samples — các mẫu bị dự đoán sai với confidence cao nhất
#   5. F1 per source — nguồn nào model yếu nhất
#   6. Accuracy by text length — model yếu với text ngắn/dài?
#   7. Calibration plot — model có calibrated không?
#   8. Score distribution shift — DEV vs TEST (phát hiện domain shift)
# ==============================================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, accuracy_score,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
)

from configs.config import CFG


# ==============================================================================
# HELPER — Label encoder
# ==============================================================================
CLASS_NAMES = ["Human", "Machine"]


# ==============================================================================
# 1. CONFUSION MATRIX
# ==============================================================================
def plot_confusion_matrix(ax, labels, preds, title: str):
    cm     = confusion_matrix(labels, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    annot  = np.array([
        [f"{v:,}\n({p:.1f}%)" for v, p in zip(row_v, row_p)]
        for row_v, row_p in zip(cm, cm_pct)
    ])
    sns.heatmap(cm, annot=annot, fmt="s", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")


# ==============================================================================
# 2. SCORE DISTRIBUTION
# ==============================================================================
def plot_score_distribution(ax, probs, labels, title: str, threshold: float = 0.5):
    for lbl, color, name in [(0, "#4CAF50", "Human"), (1, "#F44336", "Machine")]:
        mask = labels == lbl
        ax.hist(probs[mask], bins=50, alpha=0.65, color=color,
                label=f"{name} (n={mask.sum():,})", density=True, edgecolor="none")
    ax.axvline(x=threshold, color="#FF9800", ls="--", lw=1.5,
               label=f"thr={threshold:.3f}")
    ax.axvline(x=0.5, color="gray", ls=":", lw=1, alpha=0.7, label="thr=0.5")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("P(Machine)"); ax.set_ylabel("Density")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


# ==============================================================================
# 3. THRESHOLD SWEEP
# ==============================================================================
def threshold_sweep(probs, labels, thresholds=None):
    """Trả về DataFrame với F1/Acc/FP/FN tại mỗi threshold."""
    if thresholds is None:
        thresholds = np.arange(0.10, 0.96, 0.05)
    rows = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 2),
            "f1_macro":  round(f1_score(labels, preds, average="macro",  zero_division=0), 4),
            "f1_machine":round(f1_score(labels, preds, pos_label=1,      zero_division=0), 4),
            "accuracy":  round(accuracy_score(labels, preds), 4),
            "fp":        int(((labels == 0) & (preds == 1)).sum()),
            "fn":        int(((labels == 1) & (preds == 0)).sum()),
        })
    return pd.DataFrame(rows)


def plot_threshold_sweep(ax, sweep_df: pd.DataFrame, title: str):
    t   = sweep_df["threshold"]
    ax.plot(t, sweep_df["f1_macro"],   "o-", color="#2196F3", lw=2,  label="F1 Macro")
    ax.plot(t, sweep_df["f1_machine"], "s--",color="#9C27B0", lw=1.5,label="F1 Machine")
    ax.plot(t, sweep_df["accuracy"],   "^-", color="#4CAF50", lw=1.5,label="Accuracy")

    best_idx = sweep_df["f1_macro"].idxmax()
    best_t   = sweep_df.loc[best_idx, "threshold"]
    best_f1  = sweep_df.loc[best_idx, "f1_macro"]
    ax.axvline(x=best_t, color="#FF9800", ls="--", lw=1.5,
               label=f"Best thr={best_t:.2f} (F1={best_f1:.4f})")

    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlim(0.05, 1.0); ax.set_ylim(0.0, 1.05)

    # Annotate FP/FN trên trục phụ
    ax2 = ax.twinx()
    ax2.plot(t, sweep_df["fp"], "x--", color="#F44336", alpha=0.5, lw=1, label="FP")
    ax2.plot(t, sweep_df["fn"], "+--", color="#FF9800", alpha=0.5, lw=1, label="FN")
    ax2.set_ylabel("Error Count", fontsize=8)
    ax2.tick_params(axis="y", labelsize=7)
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(list(ax.get_legend().legend_handles) + lines2,
              [t.get_text() for t in ax.get_legend().get_texts()] + labels2,
              fontsize=7, loc="lower left")


# ==============================================================================
# 4. FP / FN SAMPLE ANALYSIS
# ==============================================================================
def get_fp_fn_samples(texts, labels, preds, probs, top_n: int = 10):
    """Trả về top-N FP và FN theo confidence cao nhất."""
    df = pd.DataFrame({
        "text":  texts,
        "true":  labels,
        "pred":  preds,
        "prob":  probs,
    })
    fp = (df[(df["true"]==0) & (df["pred"]==1)]
            .nlargest(top_n, "prob")
            .reset_index(drop=True))
    fn = (df[(df["true"]==1) & (df["pred"]==0)]
            .nsmallest(top_n, "prob")
            .reset_index(drop=True))
    return fp, fn


def print_fp_fn_samples(fp: pd.DataFrame, fn: pd.DataFrame,
                         title: str = "", n_print: int = 5):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  FP/FN ANALYSIS  {title}")
    print(sep)

    print(f"\n⚠  Top {n_print} FALSE POSITIVES — Human bị nhầm là Machine")
    print("   (Model tự tin đây là AI nhưng thực ra là Human)\n")
    for _, r in fp.head(n_print).iterrows():
        print(f"  [P(Machine)={r['prob']:.4f}]")
        print(f"  {r['text'][:250].strip()}...")
        print()

    print(f"\n⚠  Top {n_print} FALSE NEGATIVES — Machine bị nhầm là Human")
    print("   (Model tự tin đây là Human nhưng thực ra là AI)\n")
    for _, r in fn.head(n_print).iterrows():
        print(f"  [P(Machine)={r['prob']:.4f}]")
        print(f"  {r['text'][:250].strip()}...")
        print()


# ==============================================================================
# 5. F1 PER SOURCE
# ==============================================================================
def f1_per_source(df_test: pd.DataFrame, preds, probs) -> pd.DataFrame:
    """Tính F1/Acc/FP/FN theo từng source."""
    df = df_test.copy()
    df["pred"] = preds
    df["prob"] = probs

    rows = []
    for src, grp in df.groupby("source"):
        f1  = f1_score(grp["label"], grp["pred"], average="macro", zero_division=0)
        acc = accuracy_score(grp["label"], grp["pred"])
        fp  = int(((grp["label"]==0) & (grp["pred"]==1)).sum())
        fn  = int(((grp["label"]==1) & (grp["pred"]==0)).sum())
        rows.append({"source": src, "f1": round(f1,4), "acc": round(acc,4),
                     "fp": fp, "fn": fn, "n": len(grp)})
    return pd.DataFrame(rows).sort_values("f1")


def plot_f1_per_source(ax, df_src: pd.DataFrame, title: str):
    colors = ["#F44336" if f < 0.75 else "#FF9800" if f < 0.85 else "#4CAF50"
              for f in df_src["f1"]]
    bars = ax.barh(df_src["source"], df_src["f1"], color=colors,
                   edgecolor="white", linewidth=0.8)
    ax.axvline(x=df_src["f1"].mean(), color="#2196F3", ls="--", lw=1.5,
               label=f"Mean F1={df_src['f1'].mean():.4f}")
    ax.set_xlim(0, 1.05)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("F1 Macro"); ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
    for bar, (_, r) in zip(bars, df_src.iterrows()):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{r['f1']:.3f}  (FP={r['fp']}, FN={r['fn']})",
                va="center", fontsize=7)


# ==============================================================================
# 6. ACCURACY BY TEXT LENGTH
# ==============================================================================
def plot_acc_by_length(ax, df_test: pd.DataFrame, preds, title: str):
    df = df_test.copy()
    df["pred"]       = preds
    df["word_count"] = df["text"].str.split().str.len()
    df["correct"]    = (df["label"] == df["pred"]).astype(int)

    bins   = [0, 50, 100, 200, 300, 500, 1000, 99999]
    labels = ["<50","50-100","100-200","200-300","300-500","500-1k",">1k"]
    df["length_bin"] = pd.cut(df["word_count"], bins=bins, labels=labels)

    stats = df.groupby("length_bin", observed=True).agg(
        acc=("correct", "mean"),
        n  =("correct", "count")
    ).reset_index()

    color_map = ["#F44336" if a < 0.75 else "#FF9800" if a < 0.85 else "#4CAF50"
                 for a in stats["acc"]]
    bars = ax.bar(stats["length_bin"].astype(str), stats["acc"],
                  color=color_map, edgecolor="white", linewidth=0.8)
    ax2 = ax.twinx()
    ax2.plot(range(len(stats)), stats["n"], "o--", color="#607D8B",
             lw=1.5, ms=5, label="n samples")
    ax2.set_ylabel("# Samples", fontsize=8); ax2.tick_params(axis="y", labelsize=7)

    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Word Count Bin"); ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, stats["acc"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", fontsize=7)


# ==============================================================================
# 7. CALIBRATION PLOT
# ==============================================================================
def plot_calibration(ax, probs, labels, title: str, n_bins: int = 10):
    """Reliability diagram: predicted probability vs actual fraction positive."""
    bins      = np.linspace(0, 1, n_bins + 1)
    bin_ids   = np.digitize(probs, bins[1:-1])
    bin_means = [probs[bin_ids == b].mean() if (bin_ids == b).sum() > 0 else np.nan
                 for b in range(n_bins)]
    bin_fracs = [labels[bin_ids == b].mean() if (bin_ids == b).sum() > 0 else np.nan
                 for b in range(n_bins)]
    bin_means = np.array(bin_means)
    bin_fracs = np.array(bin_fracs)
    valid = ~(np.isnan(bin_means) | np.isnan(bin_fracs))

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.plot(bin_means[valid], bin_fracs[valid], "o-", color="#2196F3",
            lw=2, ms=6, label="Model calibration")
    ax.fill_between(bin_means[valid], bin_fracs[valid],
                    bin_means[valid], alpha=0.15, color="#2196F3")
    ax.set_xlabel("Mean Predicted P(Machine)")
    ax.set_ylabel("Actual Fraction of Machine")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


# ==============================================================================
# 8. SCORE SHIFT — DEV vs TEST
# ==============================================================================
def plot_score_shift(ax, dev_probs, dev_labels, test_probs, test_labels, title: str):
    """
    Overlay score distributions của DEV và TEST để phát hiện domain shift.
    Nếu đường TEST dịch trái/phải so với DEV → threshold cần điều chỉnh.
    """
    for probs, labels, split, ls in [
        (dev_probs,  dev_labels,  "DEV",  "-"),
        (test_probs, test_labels, "TEST", "--"),
    ]:
        for lbl, color, name in [(0, "#4CAF50", "Human"), (1, "#F44336", "Machine")]:
            mask = labels == lbl
            if mask.sum() == 0:
                continue
            ax.hist(probs[mask], bins=40, alpha=0.4, density=True,
                    color=color, linestyle=ls, edgecolor="none",
                    label=f"{split}-{name} (n={mask.sum():,})")

    ax.set_xlabel("P(Machine)"); ax.set_ylabel("Density")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)


# ==============================================================================
# MAIN — run_error_analysis()
# ==============================================================================
def run_error_analysis(
    run_id:      str,
    # Val set (source domain)
    val_texts:   list,
    val_labels:  np.ndarray,
    val_probs:   np.ndarray,
    # Dev set (target domain — bloomz)
    dev_df:      pd.DataFrame,
    dev_labels:  np.ndarray,
    dev_probs:   np.ndarray,
    # Test set (target domain — full, có OUTFOX/GPT-4)
    test_df:     pd.DataFrame,
    test_labels: np.ndarray,
    test_probs:  np.ndarray,
):
    """
    Chạy toàn bộ error analysis cho 1 run.
    Lưu figure PNG và JSON report vào PLOT_DIR.

    Parameters
    ----------
    val_texts   : list[str] — text gốc của val set (để hiển thị FP/FN)
    val_labels  : array nhãn thật val
    val_probs   : P(Machine) trên val
    dev_df      : DataFrame dev (cần cột 'text', 'label', 'source')
    dev_labels  : array nhãn thật dev
    dev_probs   : P(Machine) trên dev
    test_df     : DataFrame test (cần cột 'text', 'label', 'source')
    test_labels : array nhãn thật test
    test_probs  : P(Machine) trên test
    """
    print(f"\n{'='*65}")
    print(f"  ERROR ANALYSIS — {run_id}")
    print(f"{'='*65}")

    # ── Tìm threshold tối ưu từ val set ──────────────────────────────────────
    val_sweep  = threshold_sweep(val_probs,  val_labels)
    dev_sweep  = threshold_sweep(dev_probs,  dev_labels)
    test_sweep = threshold_sweep(test_probs, test_labels)

    best_thr_val  = val_sweep.loc[val_sweep["f1_macro"].idxmax(),  "threshold"]
    best_thr_dev  = dev_sweep.loc[dev_sweep["f1_macro"].idxmax(),  "threshold"]
    best_thr_test = test_sweep.loc[test_sweep["f1_macro"].idxmax(),"threshold"]

    print(f"\n  Threshold tối ưu (theo F1 Macro):")
    print(f"    Val  : {best_thr_val:.2f}  → F1={val_sweep['f1_macro'].max():.4f}")
    print(f"    Dev  : {best_thr_dev:.2f}  → F1={dev_sweep['f1_macro'].max():.4f}")
    print(f"    Test : {best_thr_test:.2f} → F1={test_sweep['f1_macro'].max():.4f}")

    # Dùng best_thr_val làm threshold chính (không dùng test label để chọn thr)
    thr = best_thr_val

    val_preds  = (val_probs  >= thr).astype(int)
    dev_preds  = (dev_probs  >= thr).astype(int)
    test_preds = (test_probs >= thr).astype(int)

    # ── Classification reports ────────────────────────────────────────────────
    print(f"\n  Classification Report — VAL (thr={thr:.2f})")
    print(classification_report(val_labels,  val_preds,
                                 target_names=CLASS_NAMES, digits=4))
    print(f"  Classification Report — DEV (thr={thr:.2f})")
    print(classification_report(dev_labels,  dev_preds,
                                 target_names=CLASS_NAMES, digits=4))
    print(f"  Classification Report — TEST (thr={thr:.2f})")
    print(classification_report(test_labels, test_preds,
                                 target_names=CLASS_NAMES, digits=4))

    # ── F1 per source ─────────────────────────────────────────────────────────
    df_src_dev  = f1_per_source(dev_df,  dev_preds,  dev_probs)
    df_src_test = f1_per_source(test_df, test_preds, test_probs)

    print("\n  F1 per Source — DEV:")
    print(df_src_dev.to_string(index=False))
    print("\n  F1 per Source — TEST:")
    print(df_src_test.to_string(index=False))

    # ── FP/FN samples ─────────────────────────────────────────────────────────
    fp_val, fn_val   = get_fp_fn_samples(val_texts,
                                          val_labels, val_preds, val_probs)
    fp_test, fn_test = get_fp_fn_samples(test_df["text"].tolist(),
                                          test_labels, test_preds, test_probs)
    print_fp_fn_samples(fp_val,  fn_val,  title=f"VAL  (thr={thr:.2f})", n_print=3)
    print_fp_fn_samples(fp_test, fn_test, title=f"TEST (thr={thr:.2f})", n_print=3)

    # ── Vẽ figure tổng hợp (4×3 = 12 subplots) ───────────────────────────────
    fig = plt.figure(figsize=(22, 28))
    fig.suptitle(f"Error Analysis — {run_id}  (val_thr={thr:.2f})",
                 fontsize=14, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Row 0: Confusion matrices
    plot_confusion_matrix(fig.add_subplot(gs[0, 0]),
                          val_labels,  val_preds,  "Confusion Matrix — Val")
    plot_confusion_matrix(fig.add_subplot(gs[0, 1]),
                          dev_labels,  dev_preds,  "Confusion Matrix — Dev (bloomz)")
    plot_confusion_matrix(fig.add_subplot(gs[0, 2]),
                          test_labels, test_preds, "Confusion Matrix — Test (OUTFOX+GPT4)")

    # Row 1: Score distributions
    plot_score_distribution(fig.add_subplot(gs[1, 0]),
                            val_probs,  val_labels,  "Score Dist — Val",  thr)
    plot_score_distribution(fig.add_subplot(gs[1, 1]),
                            dev_probs,  dev_labels,  "Score Dist — Dev",  thr)
    plot_score_distribution(fig.add_subplot(gs[1, 2]),
                            test_probs, test_labels, "Score Dist — Test", thr)

    # Row 2: Threshold sweeps
    plot_threshold_sweep(fig.add_subplot(gs[2, 0]), val_sweep,  "Threshold Sweep — Val")
    plot_threshold_sweep(fig.add_subplot(gs[2, 1]), dev_sweep,  "Threshold Sweep — Dev")
    plot_threshold_sweep(fig.add_subplot(gs[2, 2]), test_sweep, "Threshold Sweep — Test")

    # Row 3: F1/source + Acc/length + Calibration + Score shift
    plot_f1_per_source(fig.add_subplot(gs[3, 0]), df_src_test, "F1 per Source — Test")
    plot_acc_by_length(fig.add_subplot(gs[3, 1]), test_df, test_preds,
                       "Accuracy by Text Length — Test")
    plot_calibration(fig.add_subplot(gs[3, 2]), test_probs, test_labels,
                     "Calibration Plot — Test")

    out_fig = os.path.join(CFG.PLOT_DIR, f"{run_id}_error_analysis.png")
    plt.savefig(out_fig, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  [PLOT] {out_fig}")

    # ── Figure phụ: DEV vs TEST score shift ───────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    plot_score_shift(ax2, dev_probs, dev_labels, test_probs, test_labels,
                     f"Score Distribution Shift — DEV vs TEST ({run_id})")
    out_shift = os.path.join(CFG.PLOT_DIR, f"{run_id}_score_shift.png")
    plt.savefig(out_shift, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {out_shift}")

    # ── Lưu JSON report ───────────────────────────────────────────────────────
    report = {
        "run_id":          run_id,
        "best_threshold":  {
            "from_val":  float(best_thr_val),
            "from_dev":  float(best_thr_dev),
            "from_test": float(best_thr_test),
            "used":      float(thr),
        },
        "val": {
            "f1_macro":  round(f1_score(val_labels,  val_preds,  average="macro"), 4),
            "accuracy":  round(accuracy_score(val_labels,  val_preds), 4),
            "fp":        int(((val_labels==0)  & (val_preds==1)).sum()),
            "fn":        int(((val_labels==1)  & (val_preds==0)).sum()),
        },
        "dev": {
            "f1_macro":      round(f1_score(dev_labels,  dev_preds,  average="macro"), 4),
            "accuracy":      round(accuracy_score(dev_labels,  dev_preds), 4),
            "fp":            int(((dev_labels==0)  & (dev_preds==1)).sum()),
            "fn":            int(((dev_labels==1)  & (dev_preds==0)).sum()),
            "f1_per_source": df_src_dev.to_dict("records"),
        },
        "test": {
            "f1_macro":      round(f1_score(test_labels, test_preds, average="macro"), 4),
            "accuracy":      round(accuracy_score(test_labels, test_preds), 4),
            "fp":            int(((test_labels==0) & (test_preds==1)).sum()),
            "fn":            int(((test_labels==1) & (test_preds==0)).sum()),
            "f1_per_source": df_src_test.to_dict("records"),
        },
        "top_fp_val":  fp_val.head(5)[["prob","text"]].assign(
                           text=fp_val.head(5)["text"].str[:300]).to_dict("records"),
        "top_fn_val":  fn_val.head(5)[["prob","text"]].assign(
                           text=fn_val.head(5)["text"].str[:300]).to_dict("records"),
        "top_fp_test": fp_test.head(5)[["prob","text"]].assign(
                           text=fp_test.head(5)["text"].str[:300]).to_dict("records"),
        "top_fn_test": fn_test.head(5)[["prob","text"]].assign(
                           text=fn_test.head(5)["text"].str[:300]).to_dict("records"),
    }

    out_json = os.path.join(CFG.PLOT_DIR, f"{run_id}_error_analysis.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  [JSON] {out_json}")

    return report
