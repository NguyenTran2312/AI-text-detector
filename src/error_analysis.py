# -*- coding: utf-8 -*-
# src/error_analysis.py
# ==============================================================================
# ERROR ANALYSIS — DANN Text Detector
# Adapted từ code gốc của nhóm, tích hợp vào pipeline ablation study
# ==============================================================================
import os
import json
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
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, roc_curve, auc,
    precision_recall_curve,
    precision_score, recall_score,
)
import warnings
warnings.filterwarnings("ignore")

from configs.config import CFG

LABEL_NAMES = {0: "Human", 1: "Machine"}

# Dark theme palette
BLUE   = "#4C72B0"; ORANGE = "#DD8452"
GREEN  = "#55A868"; RED    = "#C44E52"
PURPLE = "#8172B2"; GRAY   = "#8C8C8C"

DARK_PARAMS = {
    "figure.facecolor": "#0F0F0F", "axes.facecolor":  "#1A1A1A",
    "axes.edgecolor":   "#444",    "axes.labelcolor": "#DDD",
    "xtick.color":      "#AAA",    "ytick.color":     "#AAA",
    "text.color":       "#EEE",    "grid.color":      "#333",
    "grid.linewidth":   0.5,
}


# ==============================================================================
# SECTION 1 — THU THẬP PREDICTIONS CHI TIẾT
# ==============================================================================
@torch.no_grad()
def get_predictions(model, df: pd.DataFrame, tokenizer, device,
                    batch_size: int = 32) -> pd.DataFrame:
    """
    Trả về DataFrame với các cột:
      pred, prob_human, prob_machine, correct,
      confidence, loss, error_type, text_len
    """
    from src.dataset import TextDetectionDataset, MultiTaskDataCollator

    model.eval()
    collate_fn = MultiTaskDataCollator(tokenizer=tokenizer)
    dataset    = TextDetectionDataset(df, tokenizer, CFG.MAX_LEN)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
    ce = nn.CrossEntropyLoss(reduction="none")

    all_probs, all_preds, all_losses = [], [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_class   = batch["labels_class"].to(device)

        class_logits, _ = model(input_ids, attention_mask)
        probs  = torch.softmax(class_logits, dim=-1)
        preds  = class_logits.argmax(dim=-1)
        losses = ce(class_logits, labels_class)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_losses.extend(losses.cpu().numpy())

    result              = df.copy().reset_index(drop=True)
    probs_arr           = np.array(all_probs)
    result["prob_human"]   = probs_arr[:, 0]
    result["prob_machine"] = probs_arr[:, 1]
    result["pred"]         = all_preds
    result["loss"]         = all_losses
    result["correct"]      = result["pred"] == result["label"]
    result["confidence"]   = probs_arr.max(axis=1)
    result["pred_name"]    = result["pred"].map(LABEL_NAMES)
    result["label_name"]   = result["label"].map(LABEL_NAMES)
    result["text_len"]     = result["text"].str.split().str.len()

    def _error_type(row):
        if row["label"] == 1 and row["pred"] == 1: return "TP"
        if row["label"] == 0 and row["pred"] == 0: return "TN"
        if row["label"] == 0 and row["pred"] == 1: return "FP"
        if row["label"] == 1 and row["pred"] == 0: return "FN"
    result["error_type"] = result.apply(_error_type, axis=1)

    return result


# ==============================================================================
# SECTION 2 — PHÂN TÍCH TỔNG QUAN (text)
# ==============================================================================
def print_overview(df_result: pd.DataFrame, split_name: str):
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS — {split_name}")
    print(f"{'='*60}")
    print(f"Total samples : {len(df_result):,}")
    print(f"Correct       : {df_result['correct'].sum():,} ({df_result['correct'].mean()*100:.2f}%)")
    print(f"Wrong         : {(~df_result['correct']).sum():,} ({(~df_result['correct']).mean()*100:.2f}%)")

    print("\nError breakdown:")
    for etype, count in df_result["error_type"].value_counts().items():
        print(f"  {etype}: {count:,} ({count/len(df_result)*100:.2f}%)")

    print("\nClassification Report:")
    print(classification_report(df_result["label"], df_result["pred"],
                                 target_names=["Human","Machine"], digits=4))

    print("F1 per Source:")
    for src, grp in df_result.groupby("source"):
        f1  = f1_score(grp["label"], grp["pred"], average="macro")
        acc = (grp["pred"] == grp["label"]).mean()
        fp  = int(((grp["label"]==0) & (grp["pred"]==1)).sum())
        fn  = int(((grp["label"]==1) & (grp["pred"]==0)).sum())
        print(f"  {src:12s} | F1={f1:.4f} | Acc={acc:.4f} | FP={fp} | FN={fn}")

    if "model" in df_result.columns:
        print("\nF1 per AI Model:")
        for mdl, grp in df_result.groupby("model"):
            f1  = f1_score(grp["label"], grp["pred"], average="macro")
            acc = (grp["pred"] == grp["label"]).mean()
            print(f"  {mdl:14s} | F1={f1:.4f} | Acc={acc:.4f} | n={len(grp)}")


# ==============================================================================
# SECTION 3 — VẼ BIỂU ĐỒ TỔNG HỢP (dark theme)
# ==============================================================================
def plot_error_analysis(df_val_res: pd.DataFrame,
                        df_dev_res: pd.DataFrame,
                        df_test_res: pd.DataFrame = None,
                        save_path: str = None):
    """
    Vẽ 12-panel figure. Nếu có df_test_res thì thêm vào ROC và Calibration.
    """
    if save_path is None:
        save_path = os.path.join(CFG.PLOT_DIR, "error_analysis.png")

    plt.rcParams.update(DARK_PARAMS)

    sources = sorted(df_val_res["source"].unique())
    src_colors = dict(zip(sources, [BLUE, ORANGE, GREEN, RED, PURPLE, GRAY]))

    fig = plt.figure(figsize=(22, 26), facecolor="#0F0F0F")
    fig.suptitle("Error Analysis — DANN Text Detector", fontsize=20,
                 fontweight="bold", color="white", y=0.98)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.48, wspace=0.38)

    # ── ROW 0: Confusion matrix val + dev ────────────────────────────────────
    datasets_cm = [(df_val_res, "Val Set"), (df_dev_res, "Dev Set (bloomz — unseen)")]
    if df_test_res is not None:
        datasets_cm = [(df_val_res, "Val Set"),
                       (df_dev_res, "Dev Set (bloomz)"),
                       (df_test_res, "Test Set (OUTFOX+GPT4)")]
    for col, (df_r, title) in enumerate(datasets_cm[:3]):
        ax = fig.add_subplot(gs[0, col])
        cm = confusion_matrix(df_r["label"], df_r["pred"])
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        annot  = np.array([[f"{v}\n({p:.1f}%)" for v, p in zip(rv, rp)]
                           for rv, rp in zip(cm, cm_pct)])
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=ax,
                    xticklabels=["Human","Machine"],
                    yticklabels=["Human","Machine"],
                    linewidths=0.5, linecolor="#333",
                    cbar_kws={"shrink": 0.8}, annot_kws={"size": 11})
        ax.set_title(f"Confusion Matrix — {title}", fontsize=11, pad=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    # ── ROW 0 col 2 (nếu không có test): ROC val vs dev ──────────────────────
    if df_test_res is None:
        ax = fig.add_subplot(gs[0, 2])
        for df_r, label, color in [(df_val_res,"Val",BLUE),(df_dev_res,"Dev",ORANGE)]:
            fpr, tpr, _ = roc_curve(df_r["label"], df_r["prob_machine"])
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{label} (AUC={auc(fpr,tpr):.4f})")
        ax.plot([0,1],[0,1],"--",color=GRAY,lw=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC Curve — Val vs Dev", fontsize=11, pad=8)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── ROW 1: F1 per source ─────────────────────────────────────────────────
    for col, (df_r, title) in enumerate([
        (df_val_res, "Val"), (df_dev_res, "Dev")
    ]):
        ax = fig.add_subplot(gs[1, col])
        src_f1 = {src: f1_score(g["label"], g["pred"], average="macro")
                  for src, g in df_r.groupby("source")}
        colors = [src_colors.get(s, GRAY) for s in src_f1]
        bars = ax.bar(src_f1.keys(), src_f1.values(), color=colors,
                      edgecolor="#333", width=0.6)
        ax.axhline(y=df_r["correct"].mean(), color=RED, ls="--", lw=1.2,
                   label=f"Overall Acc={df_r['correct'].mean():.3f}")
        for bar, v in zip(bars, src_f1.values()):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.3f}", ha="center", fontsize=9, color="#EEE")
        ax.set_title(f"F1 per Source — {title}", fontsize=11, pad=8)
        ax.set_ylabel("Macro F1"); ax.set_ylim(0, 1.1)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)

    # ── ROW 1 col 2: F1 per source — Test (nếu có) hoặc F1/model dev ─────────
    ax = fig.add_subplot(gs[1, 2])
    if df_test_res is not None:
        src_f1 = {src: f1_score(g["label"], g["pred"], average="macro")
                  for src, g in df_test_res.groupby("source")}
        colors = [src_colors.get(s, GRAY) for s in src_f1]
        bars = ax.bar(src_f1.keys(), src_f1.values(), color=colors,
                      edgecolor="#333", width=0.6)
        for bar, v in zip(bars, src_f1.values()):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.3f}", ha="center", fontsize=9, color="#EEE")
        ax.set_title("F1 per Source — Test (OUTFOX+GPT4)", fontsize=11, pad=8)
        ax.set_ylabel("Macro F1"); ax.set_ylim(0, 1.1)
        ax.tick_params(axis="x", rotation=20)
        ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
    elif "model" in df_dev_res.columns:
        mdl_f1 = {mdl: f1_score(g["label"], g["pred"], average="macro")
                  for mdl, g in df_dev_res.groupby("model")}
        colors_m = [GREEN if m == "human" else RED for m in mdl_f1]
        bars = ax.bar(mdl_f1.keys(), mdl_f1.values(), color=colors_m,
                      edgecolor="#333", width=0.5)
        for bar, v in zip(bars, mdl_f1.values()):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.3f}", ha="center", fontsize=10, color="#EEE")
        ax.set_title("F1 per AI Model — Dev\n(bloomz=unseen, human=seen)",
                     fontsize=11, pad=8)
        ax.set_ylabel("Macro F1"); ax.set_ylim(0, 1.1)
        ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)

    # ── ROW 2: Confidence + Loss + Text length ────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    bins = np.linspace(0.5, 1.0, 30)
    df_ea = df_test_res if df_test_res is not None else df_dev_res
    ea_label = "Test" if df_test_res is not None else "Dev"
    ax.hist(df_ea[df_ea["correct"]]["confidence"],  bins=bins, alpha=0.7,
            color=GREEN, label="Correct", density=True)
    ax.hist(df_ea[~df_ea["correct"]]["confidence"], bins=bins, alpha=0.7,
            color=RED,   label="Wrong",   density=True)
    ax.set_title(f"Confidence — Correct vs Wrong\n({ea_label} Set)", fontsize=11, pad=8)
    ax.set_xlabel("Max Probability"); ax.set_ylabel("Density")
    ax.legend(fontsize=9); ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[2, 1])
    etype_colors = {"TP": GREEN, "TN": BLUE, "FP": ORANGE, "FN": RED}
    for etype, grp in df_ea.groupby("error_type"):
        ax.hist(grp["loss"].clip(upper=5), bins=40, alpha=0.6,
                color=etype_colors.get(etype, GRAY),
                label=f"{etype} (n={len(grp)})", density=True)
    ax.set_title(f"Loss Distribution by Error Type\n({ea_label} Set)", fontsize=11, pad=8)
    ax.set_xlabel("Cross-entropy Loss"); ax.set_ylabel("Density")
    ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[2, 2])
    bins_len = [0, 50, 100, 200, 300, 500, 1000, 99999]
    labels_len = ["<50","50-100","100-200","200-300","300-500","500-1k",">1k"]
    df_ea = df_ea.copy()
    df_ea["len_bin"] = pd.cut(df_ea["text_len"], bins=bins_len, labels=labels_len)
    bin_acc = df_ea.groupby("len_bin", observed=True)["correct"].mean()
    bin_cnt = df_ea.groupby("len_bin", observed=True).size()
    bars = ax.bar(bin_acc.index, bin_acc.values, color=PURPLE,
                  edgecolor="#333", width=0.7)
    ax2 = ax.twinx()
    ax2.plot(range(len(bin_cnt)), bin_cnt.values, "o--",
             color=ORANGE, lw=1.5, ms=5, label="Sample count")
    ax2.set_ylabel("Sample count", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    for bar, v in zip(bars, bin_acc.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{v:.2f}", ha="center", fontsize=8, color="#EEE")
    ax.axhline(y=0.5, color=RED, ls="--", lw=1, alpha=0.5)
    ax.set_title(f"Accuracy by Text Length — {ea_label}", fontsize=11, pad=8)
    ax.set_xlabel("Word count bin"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1); ax.tick_params(axis="x", rotation=20)
    ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)

    # ── ROW 3: Top errors + PR curve + Calibration ───────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    df_plot = df_test_res if df_test_res is not None else df_dev_res
    fp_df = df_plot[df_plot["error_type"]=="FP"].nlargest(20, "confidence")
    fn_df = df_plot[df_plot["error_type"]=="FN"].nlargest(20, "confidence")
    fp_src = fp_df["source"].value_counts()
    fn_src = fn_df["source"].value_counts()
    x = np.arange(len(sources)); w = 0.35
    ax.bar(x-w/2, [fp_src.get(s,0) for s in sources], w,
           label="FP (Human→Machine)", color=ORANGE, alpha=0.85)
    ax.bar(x+w/2, [fn_src.get(s,0) for s in sources], w,
           label="FN (Machine→Human)", color=RED, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(sources, rotation=20)
    ax.set_title(f"Top-20 High-Conf Errors by Source\n({ea_label})", fontsize=11, pad=8)
    ax.set_ylabel("Count"); ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[3, 1])
    plot_pairs = [(df_val_res,"Val",BLUE),(df_dev_res,"Dev",ORANGE)]
    if df_test_res is not None:
        plot_pairs.append((df_test_res,"Test",GREEN))
    for df_r, label, color in plot_pairs:
        prec, rec, _ = precision_recall_curve(df_r["label"], df_r["prob_machine"])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, color=color, lw=2, label=f"{label} (AUC={pr_auc:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=11, pad=8)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(0,1); ax.set_ylim(0,1.05)

    ax = fig.add_subplot(gs[3, 2])
    for df_r, label, color in plot_pairs:
        df_r = df_r.copy()
        df_r["prob_bin"] = pd.cut(df_r["prob_machine"], bins=np.linspace(0,1,11))
        cal = df_r.groupby("prob_bin", observed=True).apply(
            lambda g: pd.Series({
                "mean_prob":      g["prob_machine"].mean(),
                "actual_pos_rate":(g["label"]==1).mean(),
            })
        ).dropna()
        ax.plot(cal["mean_prob"], cal["actual_pos_rate"], "o-",
                color=color, lw=2, ms=5, label=label)
    ax.plot([0,1],[0,1],"--",color=GRAY,lw=1,label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability (Machine)")
    ax.set_ylabel("Actual Positive Rate")
    ax.set_title("Calibration Plot\n(gần đường chéo = model tốt)", fontsize=11, pad=8)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(0,1); ax.set_ylim(0,1.05)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0F0F0F")
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"\n  [PLOT] {save_path}")


# ==============================================================================
# SECTION 4 — IN MẪU LỖI ĐỊNH TÍNH
# ==============================================================================
def show_error_samples(df_result: pd.DataFrame, error_type: str = "FN",
                       n: int = 5, min_confidence: float = 0.8):
    samples = (df_result[df_result["error_type"] == error_type]
               .query(f"confidence >= {min_confidence}")
               .nlargest(n, "confidence"))

    print(f"\n{'='*70}")
    print(f"Top {n} {error_type} samples (confidence ≥ {min_confidence})")
    print("⚠ Machine text predicted as Human — model bị đánh lừa"
          if error_type == "FN" else
          "⚠ Human text predicted as Machine — false alarm")
    print(f"{'='*70}")

    for i, row in samples.iterrows():
        src = row.get("source", "N/A")
        mdl = row.get("model",  "N/A")
        print(f"\n[{i}] Source: {src} | Model: {mdl} | "
              f"Confidence: {row['confidence']:.4f} | Loss: {row['loss']:.4f}")
        print(f"     True: {row['label_name']} → Pred: {row['pred_name']}")
        print(f"     Text: {str(row['text'])[:300].replace(chr(10),' ')}...")
        print("-" * 70)


# ==============================================================================
# SECTION 5 — THRESHOLD ANALYSIS
# ==============================================================================
def threshold_analysis(df_result: pd.DataFrame, split_name: str) -> float:
    print(f"\n{'='*60}")
    print(f"THRESHOLD ANALYSIS — {split_name}")
    print(f"{'='*60}")
    header = f"{'Threshold':>10} | {'F1':>7} | {'Acc':>7} | {'Precision':>10} | {'Recall':>8} | {'FP':>6} | {'FN':>6}"
    print(header); print("-" * 70)

    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.30, 0.81, 0.05):
        preds = (df_result["prob_machine"] >= thresh).astype(int)
        f1    = f1_score(df_result["label"], preds, average="macro", zero_division=0)
        acc   = (preds == df_result["label"]).mean()
        fp    = int(((df_result["label"]==0) & (preds==1)).sum())
        fn    = int(((df_result["label"]==1) & (preds==0)).sum())
        prec  = precision_score(df_result["label"], preds, zero_division=0)
        rec   = recall_score(df_result["label"],    preds, zero_division=0)
        mark  = " ← best" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
        print(f"  {thresh:.2f}     | {f1:.4f} | {acc:.4f} | {prec:.4f}     | {rec:.4f}   | {fp:6d} | {fn:6d}{mark}")

    print(f"\nBest threshold: {best_thresh:.2f} → F1={best_f1:.4f}")
    return best_thresh


# ==============================================================================
# SECTION 6 — SCORE SHIFT: DEV vs TEST
# ==============================================================================
def plot_score_shift(df_dev_res: pd.DataFrame, df_test_res: pd.DataFrame,
                     run_id: str):
    """Visualize score distribution shift giữa DEV và TEST."""
    plt.rcParams.update(DARK_PARAMS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0F0F0F")
    fig.suptitle(f"Score Distribution Shift — DEV vs TEST ({run_id})",
                 fontsize=12, color="white")

    for ax, df_r, title in [
        (axes[0], df_dev_res,  "DEV  (bloomz)"),
        (axes[1], df_test_res, "TEST (OUTFOX+GPT4)"),
    ]:
        for lbl, color, name in [(0,GREEN,"Human"),(1,RED,"Machine")]:
            mask = df_r["label"] == lbl
            ax.hist(df_r[mask]["prob_machine"], bins=50, alpha=0.65,
                    color=color, label=f"{name} (n={mask.sum():,})",
                    density=True, edgecolor="none")
        ax.set_title(title, fontsize=10); ax.set_xlabel("P(Machine)")
        ax.set_ylabel("Density"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    out = os.path.join(CFG.PLOT_DIR, f"{run_id}_score_shift.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#0F0F0F")
    plt.rcParams.update(plt.rcParamsDefault)
    plt.close()
    print(f"  [PLOT] {out}")


# ==============================================================================
# MAIN — run_error_analysis() — gọi từ ablation.py
# ==============================================================================
def run_error_analysis(run_id: str, model, tokenizer, device,
                       df_val: pd.DataFrame,
                       df_dev: pd.DataFrame,
                       df_test: pd.DataFrame = None):
    """
    Entry point chính. Nhận model + 3 DataFrame, chạy toàn bộ pipeline.

    Parameters
    ----------
    run_id   : tên run (dùng để đặt tên file output)
    model    : DANN_TextDetector đã load weights
    tokenizer: AutoTokenizer
    device   : torch.device
    df_val   : DataFrame val set (cần cột: text, label, source)
    df_dev   : DataFrame dev set (cần cột: text, label, source, model)
    df_test  : DataFrame test labeled (cần cột: text, label, source) — optional
    """
    os.makedirs(CFG.PLOT_DIR, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  ERROR ANALYSIS — {run_id}")
    print(f"{'='*65}")

    # 1. Thu thập predictions
    print("  Collecting predictions...")
    df_val_res  = get_predictions(model, df_val,  tokenizer, device)
    df_dev_res  = get_predictions(model, df_dev,  tokenizer, device)
    df_test_res = get_predictions(model, df_test, tokenizer, device) if df_test is not None else None

    # 2. Overview text
    print_overview(df_val_res,  f"VAL  SET — {run_id}")
    print_overview(df_dev_res,  f"DEV  SET — {run_id}")
    if df_test_res is not None:
        print_overview(df_test_res, f"TEST SET — {run_id}")

    # 3. Threshold analysis
    best_val  = threshold_analysis(df_val_res,  "Val")
    best_dev  = threshold_analysis(df_dev_res,  "Dev")
    best_test = threshold_analysis(df_test_res, "Test") if df_test_res is not None else None

    # 4. Mẫu lỗi định tính — dùng dev (hoặc test nếu có)
    df_ea = df_test_res if df_test_res is not None else df_dev_res
    show_error_samples(df_ea, error_type="FN", n=5, min_confidence=0.85)
    show_error_samples(df_ea, error_type="FP", n=5, min_confidence=0.85)

    # 5. Vẽ figure tổng hợp
    plot_error_analysis(
        df_val_res, df_dev_res, df_test_res,
        save_path=os.path.join(CFG.PLOT_DIR, f"{run_id}_error_analysis.png"),
    )

    # 6. Score shift plot (nếu có test)
    if df_test_res is not None:
        plot_score_shift(df_dev_res, df_test_res, run_id)

    # 7. Log lên W&B nếu có run active
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({
                f"{run_id}/ea/val_best_threshold" : float(best_val),
                f"{run_id}/ea/dev_best_threshold" : float(best_dev),
            })
            if best_test is not None:
                wandb.log({f"{run_id}/ea/test_best_threshold": float(best_test)})
            ea_plot = os.path.join(CFG.PLOT_DIR, f"{run_id}_error_analysis.png")
            wandb.log({f"{run_id}/ea/chart": wandb.Image(ea_plot)})
            print("  [W&B] Error analysis logged")
    except Exception:
        pass

    return {
        "best_threshold_val":  float(best_val),
        "best_threshold_dev":  float(best_dev),
        "best_threshold_test": float(best_test) if best_test else None,
    }
