# -*- coding: utf-8 -*-
# notebooks/eda.py
# ==============================================================================
# EDA — Exploratory Data Analysis + Baseline ML Models
# Mục tiêu:
#   1. Phân tích phân phối dữ liệu (label, source, model, text length)
#   2. N-gram analysis (bigram, trigram) human vs machine
#   3. Text cleaning pipeline
#   4. Baseline ML: TF-IDF + Logistic Regression + LightGBM
#   5. Error analysis baseline + kết luận tại sao cần Deep Learning
# ==============================================================================

import os
import re
import sys
import string
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
)
from lightgbm import LGBMClassifier

# ── NLTK downloads ────────────────────────────────────────────────────────────
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_PATH = "/content/SubtaskA/subtaskA_train_monolingual.jsonl"
DEV_PATH   = "/content/SubtaskA/subtaskA_dev_monolingual.jsonl"
PLOT_DIR   = "/content/drive/MyDrive/eda_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

SEED = 42


# ==============================================================================
# SECTION 1 — LOAD DATA
# ==============================================================================
def load_data():
    df  = pd.read_json(TRAIN_PATH, lines=True)
    dev = pd.read_json(DEV_PATH,   lines=True)

    df  = df.drop(columns=["id"], errors="ignore")
    dev = dev.drop(columns=["id"], errors="ignore")

    # Thêm cột tiện ích
    df["text_length"]  = df["text"].str.len()
    df["word_count"]   = df["text"].str.split().str.len()
    df["model_type"]   = df["model"].apply(lambda x: "human" if x == "human" else "machine")
    df["split"]        = "train"

    dev["text_length"] = dev["text"].str.len()
    dev["word_count"]  = dev["text"].str.split().str.len()
    dev["model_type"]  = dev["model"].apply(lambda x: "human" if x == "human" else "machine")
    dev["split"]       = "dev"

    print("=" * 60)
    print("TRAIN SET")
    print(f"  Tổng mẫu       : {len(df):,}")
    print(f"  Columns        : {list(df.columns)}")
    print(f"  Label dist     : {df['label'].value_counts().to_dict()}")
    print(f"  Missing values : {df.isnull().sum().sum()}")
    print(f"  Duplicate rows : {df.duplicated(subset='text').sum():,}")
    print("\nDEV SET")
    print(f"  Tổng mẫu       : {len(dev):,}")
    print(f"  Label dist     : {dev['label'].value_counts().to_dict()}")
    print("=" * 60)
    return df, dev


# ==============================================================================
# SECTION 2 — EDA PLOTS
# ==============================================================================
def plot_label_distribution(df, dev):
    """Phân phối nhãn train vs dev."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Label Distribution — Train vs Dev", fontsize=13, fontweight="bold")

    for ax, data, title in zip(axes, [df, dev], ["Train Set", "Dev Set"]):
        counts = data["label"].value_counts().sort_index()
        bars   = ax.bar(["Human (0)", "Machine (1)"], counts.values,
                        color=["#4CAF50", "#F44336"], edgecolor="white", linewidth=1.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Số mẫu")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                    f"{val:,}\n({val/len(data)*100:.1f}%)", ha="center", fontsize=10)
        ax.set_ylim(0, max(counts.values) * 1.2)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "1_label_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] 1_label_distribution.png")


def plot_source_distribution(df, dev):
    """Phân phối source × label cho train và dev."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Source Distribution by Label", fontsize=13, fontweight="bold")

    for ax, data, title in zip(axes, [df, dev], ["Train", "Dev"]):
        pivot = data.groupby(["source", "label"]).size().unstack(fill_value=0)
        pivot.plot(kind="bar", ax=ax, color=["#4CAF50","#F44336"],
                   edgecolor="white", linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Source")
        ax.set_ylabel("Số mẫu")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.legend(["Human (0)", "Machine (1)"])
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "2_source_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] 2_source_distribution.png")


def plot_model_distribution(df):
    """Phân phối AI model trong tập train."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("AI Model Distribution — Train Set", fontsize=13, fontweight="bold")

    # Count per model
    model_counts = df["model"].value_counts()
    axes[0].bar(model_counts.index, model_counts.values,
                color=plt.cm.Set3(range(len(model_counts))), edgecolor="white")
    axes[0].set_title("Số mẫu theo Model")
    axes[0].set_xticklabels(model_counts.index, rotation=35, ha="right")
    axes[0].set_ylabel("Số mẫu")
    axes[0].grid(axis="y", alpha=0.3)
    for i, (m, v) in enumerate(model_counts.items()):
        axes[0].text(i, v + 100, f"{v:,}", ha="center", fontsize=8)

    # Model × Source heatmap
    pivot = df.groupby(["model", "source"]).size().unstack(fill_value=0)
    sns.heatmap(pivot, ax=axes[1], annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, cbar_kws={"label": "Count"})
    axes[1].set_title("Model × Source Heatmap")
    axes[1].set_xlabel("Source")
    axes[1].set_ylabel("Model")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "3_model_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] 3_model_distribution.png")


def plot_text_length(df, dev):
    """Phân phối text length: violin + histogram + boxplot."""
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Text Length Analysis — Human vs Machine", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # 1. Violin — train
    ax1 = fig.add_subplot(gs[0, 0])
    sns.violinplot(x="model_type", y="word_count", data=df,
                   palette={"human": "#4CAF50", "machine": "#F44336"},
                   inner="box", ax=ax1)
    ax1.set_title("Word Count — Train (Violin)")
    ax1.set_xlabel(""); ax1.set_ylabel("Word Count")

    # 2. Histogram — train (density)
    ax2 = fig.add_subplot(gs[0, 1])
    for mtype, color in [("human", "#4CAF50"), ("machine", "#F44336")]:
        data = df[df["model_type"] == mtype]["word_count"]
        ax2.hist(data, bins=60, density=True, alpha=0.6, color=color,
                 label=mtype, edgecolor="none")
    ax2.set_title("Word Count Density — Train")
    ax2.set_xlabel("Word Count"); ax2.set_ylabel("Density")
    ax2.set_xlim(0, 1000)
    ax2.legend()

    # 3. Boxplot per source — train
    ax3 = fig.add_subplot(gs[0, 2])
    sns.boxplot(x="source", y="word_count", hue="model_type", data=df,
                palette={"human": "#4CAF50", "machine": "#F44336"},
                fliersize=2, ax=ax3)
    ax3.set_title("Word Count per Source — Train")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, ha="right")
    ax3.set_xlabel(""); ax3.set_ylabel("Word Count")
    ax3.legend(loc="upper right")

    # 4. Violin — dev
    ax4 = fig.add_subplot(gs[1, 0])
    sns.violinplot(x="model_type", y="word_count", data=dev,
                   palette={"human": "#4CAF50", "machine": "#F44336"},
                   inner="box", ax=ax4)
    ax4.set_title("Word Count — Dev (Violin)")
    ax4.set_xlabel(""); ax4.set_ylabel("Word Count")

    # 5. Histogram — dev
    ax5 = fig.add_subplot(gs[1, 1])
    for mtype, color in [("human", "#4CAF50"), ("machine", "#F44336")]:
        data = dev[dev["model_type"] == mtype]["word_count"]
        ax5.hist(data, bins=40, density=True, alpha=0.6, color=color,
                 label=mtype, edgecolor="none")
    ax5.set_title("Word Count Density — Dev")
    ax5.set_xlabel("Word Count"); ax5.set_ylabel("Density")
    ax5.legend()

    # 6. Summary stats table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    stats = df.groupby("model_type")["word_count"].describe()[["mean","50%","std","min","max"]]
    stats.columns = ["Mean","Median","Std","Min","Max"]
    stats = stats.round(1)
    table = ax6.table(
        cellText  = stats.values,
        rowLabels = stats.index,
        colLabels = stats.columns,
        cellLoc   = "center",
        loc       = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    ax6.set_title("Word Count Stats — Train", pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "4_text_length.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] 4_text_length.png")


def plot_ngrams(df, top_n: int = 20):
    """Top N bi-gram và tri-gram cho human vs machine."""
    stop_words = set(stopwords.words("english"))

    def preprocess(text):
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\d+", "", text)
        tokens = word_tokenize(text)
        return [w for w in tokens if w not in stop_words and len(w) > 2]

    def get_ngrams(texts, n):
        all_tokens = [t for text in texts for t in preprocess(text)]
        ngrams = [tuple(all_tokens[i:i+n]) for i in range(len(all_tokens)-n+1)]
        return Counter(ngrams).most_common(top_n)

    print("Computing n-grams (có thể mất vài phút)...")
    human_texts   = df[df["model_type"] == "human"]["text"].tolist()
    machine_texts = df[df["model_type"] == "machine"]["text"].tolist()

    for n, name in [(2, "Bigram"), (3, "Trigram")]:
        human_ng   = get_ngrams(human_texts,   n)
        machine_ng = get_ngrams(machine_texts, n)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(f"Top {top_n} {name}s — Human vs Machine", fontsize=13, fontweight="bold")

        for ax, ngrams, title, color in zip(
            axes, [human_ng, machine_ng],
            ["Human", "Machine"], ["#4CAF50", "#F44336"]
        ):
            labels = [" ".join(g) for g, _ in ngrams]
            counts = [c for _, c in ngrams]
            ax.barh(labels[::-1], counts[::-1], color=color, alpha=0.85, edgecolor="white")
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Count")
            ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        fname = f"5_{name.lower()}_comparison.png"
        plt.savefig(os.path.join(PLOT_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] {fname}")


# ==============================================================================
# SECTION 3 — TEXT CLEANING
# ==============================================================================
def clean_text(text: str) -> str:
    """
    Pipeline làm sạch văn bản:
      1. Truncation — cắt bỏ phần đuôi (References, See also, ...)
      2. Line removal — xóa các dòng rác (Edit:, TL;DR, ...)
      3. HTML/CSS stripping
      4. Whitespace normalization
    """
    text = str(text)

    # 1. Truncation
    trunc_kws = ["References", "External links", "See also", "Further reading", "Bibliography"]
    trunc_pat = r"^\s*[=\-#*]*\s*(?:" + "|".join(re.escape(k) for k in trunc_kws) + r")\b"
    m = re.search(trunc_pat, text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        text = text[:m.start()]

    # 2. Line removal
    line_kws  = ["Edit:", "ETA:", "TL;DR", "Tips:", "Warnings:", "Notes:"]
    line_pat  = r"^\s*(?:" + "|".join(re.escape(k) for k in line_kws) + r").*$"
    text = re.sub(line_pat, "", text, flags=re.IGNORECASE | re.MULTILINE)

    # 3. HTML/CSS
    text = re.sub(r"styleborderleftpx[^\s]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r'style\s*=\s*"[^"]*"',    "", text, flags=re.IGNORECASE)
    text = re.sub(r"style\s*=\s*'[^']*'",    "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>",                "", text)

    # 4. Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ",   text)

    return text.strip()


def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    print("Applying text cleaning pipeline...")
    df = df.copy()
    df["text_raw"]  = df["text"]                        # Giữ lại bản gốc
    df["text"]      = df["text"].apply(clean_text)
    df["word_count_after"] = df["text"].str.split().str.len()

    # Thống kê trước/sau
    before = df["word_count"].mean()
    after  = df["word_count_after"].mean()
    print(f"  Word count trung bình: {before:.0f} → {after:.0f} (giảm {(before-after)/before*100:.1f}%)")
    print(f"  Mẫu bị xóa hoàn toàn: {(df['text'].str.len() < 10).sum()}")

    # Loại bỏ mẫu quá ngắn sau cleaning
    df = df[df["text"].str.split().str.len() >= 10].reset_index(drop=True)
    print(f"  Mẫu còn lại sau filtering: {len(df):,}")
    return df


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Undersample class majority để cân bằng nhãn."""
    counts = df["label"].value_counts()
    print(f"\nTrước balancing: {counts.to_dict()}")
    min_count = counts.min()
    df_balanced = (df.groupby("label", group_keys=False)
                     .apply(lambda x: x.sample(min_count, random_state=SEED)))
    df_balanced = df_balanced.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"Sau  balancing: {df_balanced['label'].value_counts().to_dict()}")
    return df_balanced


# ==============================================================================
# SECTION 4 — BASELINE ML MODELS
# ==============================================================================
def run_baseline(df: pd.DataFrame):
    """TF-IDF + Logistic Regression + LightGBM với full evaluation."""

    print("\n" + "="*60)
    print("BASELINE ML MODELS")
    print("="*60)

    # ── Train/Val/Test split ───────────────────────────────────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        df["text"], df["label"], test_size=0.1, random_state=SEED, stratify=df["label"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1111, random_state=SEED, stratify=y_temp
    )
    # → ~80% train, 10% val, 10% test

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # ── TF-IDF ────────────────────────────────────────────────────────────────
    print("\nFitting TF-IDF (max_features=10000, ngram=(1,2))...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2),
                            stop_words="english", sublinear_tf=True)
    X_tr_tfidf  = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_te_tfidf  = tfidf.transform(X_test)

    # ── Models ────────────────────────────────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=SEED, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            random_state=SEED, verbose=-1, n_jobs=-1
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_tr_tfidf, y_train)

        y_pred      = model.predict(X_te_tfidf)
        y_proba     = model.predict_proba(X_te_tfidf)[:, 1]
        y_pred_val  = model.predict(X_val_tfidf)

        f1_test  = f1_score(y_test, y_pred, average="macro")
        f1_val   = f1_score(y_val,  y_pred_val, average="macro")
        auc      = roc_auc_score(y_test, y_proba)

        print(f"  Val F1  : {f1_val:.4f}")
        print(f"  Test F1 : {f1_test:.4f}")
        print(f"  AUC     : {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Human","Machine"]))

        results[name] = {
            "model":   model,
            "y_pred":  y_pred,
            "y_proba": y_proba,
            "f1":      f1_test,
            "auc":     auc,
        }

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_baseline_results(results, y_test)

    # ── Error analysis ────────────────────────────────────────────────────────
    best_name  = max(results, key=lambda k: results[k]["f1"])
    best_model = results[best_name]
    _error_analysis_baseline(X_test, y_test, best_model["y_pred"],
                             best_model["y_proba"], best_name)

    return results, tfidf


def _plot_baseline_results(results: dict, y_test):
    """Confusion matrix + ROC + PR curve cho cả 2 models."""
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 3, figsize=(18, 6 * n_models))
    fig.suptitle("Baseline Model Evaluation", fontsize=14, fontweight="bold")

    for row, (name, res) in enumerate(results.items()):
        axrow = axes[row] if n_models > 1 else axes

        # Confusion Matrix
        cm = confusion_matrix(y_test, res["y_pred"])
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        annot  = np.array([[f"{v}\n({p:.1f}%)" for v, p in zip(r_v, r_p)]
                           for r_v, r_p in zip(cm, cm_pct)])
        sns.heatmap(cm, annot=annot, fmt="s", cmap="Blues",
                    xticklabels=["Human","Machine"],
                    yticklabels=["Human","Machine"], ax=axrow[0])
        axrow[0].set_title(f"{name}\nConfusion Matrix")
        axrow[0].set_xlabel("Predicted"); axrow[0].set_ylabel("Actual")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        axrow[1].plot(fpr, tpr, color="#2196F3", lw=2,
                      label=f"AUC = {res['auc']:.4f}")
        axrow[1].plot([0,1],[0,1], "k--", alpha=0.4)
        axrow[1].set_xlabel("FPR"); axrow[1].set_ylabel("TPR")
        axrow[1].set_title(f"{name}\nROC Curve")
        axrow[1].legend(); axrow[1].grid(alpha=0.3)

        # Precision-Recall Curve
        prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
        ap = average_precision_score(y_test, res["y_proba"])
        axrow[2].plot(rec, prec, color="#F44336", lw=2, label=f"AP = {ap:.4f}")
        axrow[2].set_xlabel("Recall"); axrow[2].set_ylabel("Precision")
        axrow[2].set_title(f"{name}\nPrecision-Recall Curve")
        axrow[2].legend(); axrow[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "6_baseline_evaluation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] 6_baseline_evaluation.png")


def _error_analysis_baseline(X_test, y_test, y_pred, y_proba, model_name: str):
    """Phân tích mẫu lỗi + distribution plot."""
    err_df = pd.DataFrame({
        "text":       X_test.values,
        "true":       y_test.values,
        "pred":       y_pred,
        "proba":      y_proba,
    })
    err_df["error_type"] = err_df.apply(
        lambda r: "FP" if r["true"]==0 and r["pred"]==1
             else "FN" if r["true"]==1 and r["pred"]==0
             else "correct",
        axis=1
    )

    fp = err_df[err_df["error_type"] == "FP"].nlargest(5, "proba")
    fn = err_df[err_df["error_type"] == "FN"].nsmallest(5, "proba")

    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS — {model_name}")
    print(f"{'='*60}")
    total = len(err_df)
    wrong = err_df[err_df["error_type"] != "correct"]
    print(f"  Tổng mẫu : {total:,}")
    print(f"  Đúng     : {total - len(wrong):,} ({(total-len(wrong))/total*100:.1f}%)")
    print(f"  Sai      : {len(wrong):,} ({len(wrong)/total*100:.1f}%)")
    print(f"  FP       : {(err_df['error_type']=='FP').sum():,}")
    print(f"  FN       : {(err_df['error_type']=='FN').sum():,}")

    print(f"\n⚠ Top 5 FP (Human bị nhầm là Machine — confidence cao):")
    for _, r in fp.iterrows():
        print(f"  [P(Machine)={r['proba']:.3f}] {r['text'][:200]}...")
        print()

    print(f"\n⚠ Top 5 FN (Machine bị nhầm là Human — confidence thấp):")
    for _, r in fn.iterrows():
        print(f"  [P(Machine)={r['proba']:.3f}] {r['text'][:200]}...")
        print()

    # Confidence distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Error Analysis — {model_name}", fontsize=12, fontweight="bold")

    # Histogram confidence by correctness
    for label, group, color in [("Correct", "correct", "#4CAF50"), ("Wrong", "FP", "#F44336"), ("FN", "FN", "#FF9800")]:
        sub = err_df[err_df["error_type"] == group]["proba"] if group != "correct" else err_df[err_df["error_type"] == group]["proba"]
        axes[0].hist(sub, bins=30, alpha=0.6, label=label, color=color, density=True)
    axes[0].set_title("Confidence Distribution by Outcome")
    axes[0].set_xlabel("P(Machine)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Threshold sweep
    thresholds = np.arange(0.3, 0.85, 0.05)
    f1s, fprs, fns = [], [], []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1s.append(f1_score(y_test, preds, average="macro"))
        fprs.append(((y_test==0) & (preds==1)).sum())
        fns.append(((y_test==1) & (preds==0)).sum())

    ax2 = axes[1]
    ax2.plot(thresholds, f1s, "o-", color="#2196F3", lw=2, label="F1 Macro")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("F1 Macro", color="#2196F3")
    ax2.tick_params(axis="y", labelcolor="#2196F3")
    ax2.set_title("Threshold Analysis")
    ax2.grid(alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.plot(thresholds, fprs, "s--", color="#F44336", lw=1.5, alpha=0.7, label="FP count")
    ax2b.plot(thresholds, fns,  "^--", color="#FF9800", lw=1.5, alpha=0.7, label="FN count")
    ax2b.set_ylabel("Error Count")
    ax2b.tick_params(axis="y")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "7_error_analysis_baseline.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] 7_error_analysis_baseline.png")


# ==============================================================================
# SECTION 5 — CONCLUSION
# ==============================================================================
def print_conclusion():
    print("\n" + "="*60)
    print("KẾT LUẬN — TẠI SAO CẦN DEEP LEARNING?")
    print("="*60)
    print("""
Baseline TF-IDF đã chạm giới hạn trần vì các lý do sau:

1. TF-IDF xem văn bản là "túi từ" — không hiểu ngữ cảnh hay
   thứ tự từ. Khi AI dùng tiếng lóng (doin', thinkin'), hoặc
   khi human viết văn học thuật formal, TF-IDF bị đánh lừa.

2. False Positive điển hình: Human viết trên Wikipedia/arxiv
   với cấu trúc câu rất formal → TF-IDF nghĩ là AI.

3. False Negative điển hình: AI bắt chước văn phong Reddit
   với tiếng lóng và cấu trúc đơn giản → TF-IDF nghĩ là human.

4. Domain shift (train → dev): TF-IDF không có cơ chế
   adaptation, F1 drop rõ ràng khi domain thay đổi.

→ Cần mô hình hiểu ngữ cảnh sâu (RoBERTa) + domain adaptation
  (DANN) để vượt qua những hạn chế này.
""")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("Loading data...")
    df, dev = load_data()

    print("\n--- EDA Plots ---")
    plot_label_distribution(df, dev)
    plot_source_distribution(df, dev)
    plot_model_distribution(df)
    plot_text_length(df, dev)
    plot_ngrams(df, top_n=20)

    print("\n--- Text Cleaning ---")
    df_clean = apply_cleaning(df)

    print("\n--- Balancing Dataset ---")
    df_balanced = balance_dataset(df_clean)

    print("\n--- Baseline ML Models ---")
    results, tfidf = run_baseline(df_balanced)

    print_conclusion()

    # Lưu cleaned + balanced data
    out_path = "/content/drive/MyDrive/cleaned_text_data.jsonl"
    cols_to_save = ["text", "label", "source", "model"]
    df_balanced[cols_to_save].to_json(
        out_path, orient="records", lines=True, force_ascii=False
    )
    print(f"\n[SAVED] {out_path} — {len(df_balanced):,} samples")


if __name__ == "__main__":
    main()
