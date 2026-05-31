# -*- coding: utf-8 -*-
# notebooks/data_augmentation.py
# ==============================================================================
# Data Augmentation Pipeline
# Mục tiêu: Tạo thêm dữ liệu để mở rộng tập train
#
# Nguồn dữ liệu:
#   [A] Human reviews: PeerRead (GitHub — không cần HuggingFace token)
#   [B] Machine reviews: Groq API (llama3-70b-8192) với 3 persona khác nhau
#
# Output: merged vào cleaned_text_data.jsonl cùng với tập train gốc
# ==============================================================================

import os
import re
import sys
import json
import glob
import time
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# ── Install nếu chạy trên Colab ──────────────────────────────────────────────
try:
    from groq import Groq
except ImportError:
    os.system("pip install groq -qq")
    from groq import Groq

# ── Paths & Config ────────────────────────────────────────────────────────────
TRAIN_PATH     = "/content/SubtaskA/subtaskA_train_monolingual.jsonl"
PEERREAD_DIR   = "/content/PeerRead"
OUTPUT_PATH    = "/content/drive/MyDrive/cleaned_text_data.jsonl"
PLOT_DIR       = "/content/drive/MyDrive/augmentation_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

SEED               = 42
MAX_HUMAN_SAMPLES  = 500        # Số mẫu human từ PeerRead
MAX_WORDS_PER_TEXT = 1500       # Giới hạn độ dài
MIN_WORDS_PER_TEXT = 50         # Loại bỏ text quá ngắn
GROQ_MODEL         = "llama3-70b-8192"
GROQ_MAX_TOKENS    = 800
API_DELAY_SEC      = 0.6        # Tránh rate limit

random.seed(SEED)
np.random.seed(SEED)


# ==============================================================================
# SECTION A — HUMAN DATA từ PeerRead
# ==============================================================================
def clone_peerread():
    """Clone PeerRead từ GitHub nếu chưa có."""
    if not os.path.exists(PEERREAD_DIR):
        print("Cloning PeerRead from GitHub...")
        os.system("git clone https://github.com/allenai/PeerRead.git /content/PeerRead")
    else:
        print("PeerRead đã tồn tại, bỏ qua clone.")


def extract_human_reviews() -> pd.DataFrame:
    """
    Trích xuất human peer reviews từ các file JSON của PeerRead.
    Cấu trúc: PeerRead/data/{conference}/{split}/reviews/*.json
    Mỗi file JSON chứa field 'reviews' là list các dict có key 'comments'.
    """
    print("\n--- [A] Trích xuất Human Reviews từ PeerRead ---")

    review_files = glob.glob(os.path.join(PEERREAD_DIR, "data", "*", "*", "reviews", "*.json"))
    print(f"  Tìm thấy {len(review_files)} file review JSON")

    reviews = []
    for fpath in review_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "reviews" not in data:
                continue
            for review in data["reviews"]:
                text = review.get("comments", "").strip()
                if not text:
                    continue
                words = text.split()
                if len(words) < MIN_WORDS_PER_TEXT:
                    continue
                # Giới hạn độ dài
                text = " ".join(words[:MAX_WORDS_PER_TEXT])
                reviews.append(text)
        except Exception:
            continue

    print(f"  Trích xuất được {len(reviews):,} reviews hợp lệ")

    # Lấy ngẫu nhiên MAX_HUMAN_SAMPLES mẫu
    if len(reviews) > MAX_HUMAN_SAMPLES:
        reviews = random.sample(reviews, MAX_HUMAN_SAMPLES)

    df_human = pd.DataFrame({
        "text"  : reviews,
        "label" : 0,
        "source": "peerread",
        "model" : "human",
    })

    print(f"  Số mẫu human sau sampling: {len(df_human):,}")
    return df_human


# ==============================================================================
# SECTION B — MACHINE DATA từ Groq API
# ==============================================================================
# 3 persona để tạo đa dạng văn phong AI
REVIEWER_PERSONAS = [
    {
        "role":        "harsh_critic",
        "label":       "Harsh Critic",
        "instruction": (
            "Write a harsh and critical academic peer review of the following abstract. "
            "Focus on weaknesses, lack of novelty, methodological flaws, and poor presentation. "
            "Be direct and unsparing. ONLY provide the review text itself — "
            "no greetings, no sign-off, no meta-commentary. "
            "Target length: 100–250 words."
        ),
    },
    {
        "role":        "constructive",
        "label":       "Constructive Reviewer",
        "instruction": (
            "Write a detailed and constructive academic peer review of the following abstract. "
            "Balance strengths and weaknesses, offer specific improvement suggestions. "
            "Maintain a professional tone. ONLY provide the review text itself — "
            "no greetings, no sign-off. "
            "Target length: 200–400 words."
        ),
    },
    {
        "role":        "brief",
        "label":       "Brief Reviewer",
        "instruction": (
            "Write a brief academic peer review of the following abstract. "
            "Mention one or two key strengths and one or two limitations. "
            "Keep it concise. ONLY provide the review text itself — no greetings, no sign-off. "
            "Target length: 50–120 words."
        ),
    },
]


def clean_ai_output(text: str) -> str:
    """
    Loại bỏ boilerplate AI thường thêm vào đầu/cuối bài review.
    """
    text = str(text).strip()

    # Các cụm mở đầu phổ biến của AI
    openers = [
        r"^(Here is|Here's|Below is|I've provided|I have provided|"
        r"As requested,|As an AI)[^\.]*[\.\:]\s*",
        r"^(Academic Review|Peer Review|Review of the Abstract):?\s*\n*",
        r"^(This is|The following is) an? (academic )?review[^\.]*\.\s*",
        r"^Abstract to review:\s*```.*?```\s*",
        r"^```.*?```\s*",
    ]
    for pat in openers:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL).strip()

    # Cắt ở dấu câu cuối cùng nếu text bị cụt
    last = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last > len(text) * 0.5:   # Chỉ cắt nếu có ý nghĩa
        text = text[:last + 1]

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def generate_machine_reviews(abstracts: list, client, n_per_persona: int) -> pd.DataFrame:
    """
    Sinh AI reviews từ Groq API với 3 persona.
    Trả về DataFrame với columns: text, label, source, model.
    """
    print(f"\n--- [B] Sinh Machine Reviews qua Groq API ---")
    print(f"  Persona: {[p['label'] for p in REVIEWER_PERSONAS]}")
    print(f"  Mỗi persona: {n_per_persona} mẫu → tổng {n_per_persona * len(REVIEWER_PERSONAS)} mẫu")

    machine_data = []
    abstract_pool = abstracts.copy()
    random.shuffle(abstract_pool)
    abs_idx = 0

    for persona in REVIEWER_PERSONAS:
        success = 0
        print(f"\n  Generating [{persona['label']}]...")

        pbar = tqdm(total=n_per_persona, desc=f"  {persona['label']}")
        while success < n_per_persona:
            if abs_idx >= len(abstract_pool):
                print(f"  Hết abstract! Dừng ở {success} mẫu.")
                break

            abstract = abstract_pool[abs_idx]
            abs_idx += 1

            prompt = (
                f"{persona['instruction']}\n\n"
                f"Abstract:\n\"\"\"\n{abstract}\n\"\"\""
            )

            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model      = GROQ_MODEL,
                    temperature= 0.8,
                    max_tokens = GROQ_MAX_TOKENS,
                )
                raw_text  = response.choices[0].message.content
                clean_txt = clean_ai_output(raw_text)

                # Lọc output quá ngắn hoặc chứa lỗi API
                if len(clean_txt.split()) < MIN_WORDS_PER_TEXT:
                    continue

                machine_data.append({
                    "text"  : clean_txt,
                    "label" : 1,
                    "source": "peerread",
                    "model" : f"llama3-70b-{persona['role']}",
                })
                success += 1
                pbar.update(1)
                time.sleep(API_DELAY_SEC)

            except Exception as e:
                print(f"  [API Error] {e} — retrying in 3s...")
                time.sleep(3)
                continue

        pbar.close()

    df_machine = pd.DataFrame(machine_data)
    print(f"\n  Tổng mẫu machine sinh được: {len(df_machine):,}")
    return df_machine


def extract_abstracts_from_peerread() -> list:
    """Lấy abstract từ PeerRead để làm prompt cho Groq."""
    abstract_files = glob.glob(os.path.join(PEERREAD_DIR, "data", "*", "*", "*.json"))
    abstracts = []
    for fpath in abstract_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            abstract = data.get("abstract", "").strip()
            if abstract and len(abstract.split()) > 30:
                abstracts.append(abstract)
        except Exception:
            continue
    print(f"  Tìm thấy {len(abstracts):,} abstract từ PeerRead")
    return abstracts


# ==============================================================================
# SECTION C — MERGE & VALIDATE
# ==============================================================================
def merge_datasets(df_train_orig, df_human, df_machine) -> pd.DataFrame:
    """Gộp tất cả dữ liệu, loại bỏ duplicate, kiểm tra chất lượng."""
    print("\n--- [C] Merge & Validate ---")

    combined = pd.concat([df_train_orig, df_human, df_machine], ignore_index=True)
    print(f"  Trước dedup: {len(combined):,}")

    # Loại bỏ duplicate theo text (giữ first)
    combined = combined.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"  Sau  dedup : {len(combined):,}")

    # Loại bỏ text quá ngắn
    combined["word_count"] = combined["text"].str.split().str.len()
    combined = combined[combined["word_count"] >= MIN_WORDS_PER_TEXT].reset_index(drop=True)
    print(f"  Sau  length filter: {len(combined):,}")

    # Label distribution
    print("\n  Label distribution:")
    print(combined["label"].value_counts().to_frame("count").assign(
        pct=lambda x: (x["count"] / len(combined) * 100).round(1)
    ))

    print("\n  Source distribution:")
    print(combined.groupby(["source", "label"]).size().unstack(fill_value=0))

    return combined


def plot_augmentation_summary(df_before, df_after):
    """So sánh phân phối trước và sau augmentation."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Data Augmentation — Trước vs Sau", fontsize=13, fontweight="bold")

    # Label distribution
    for ax, data, title in zip(
        [axes[0], None], [df_before, df_after], ["Trước (train gốc)", "Sau (train + augmented)"]
    ):
        pass

    # Side-by-side label count
    labels_before = df_before["label"].value_counts().sort_index()
    labels_after  = df_after["label"].value_counts().sort_index()

    x = np.arange(2)
    w = 0.35
    axes[0].bar(x - w/2, labels_before.values, w, label="Trước", color="#2196F3", alpha=0.8)
    axes[0].bar(x + w/2, labels_after.values,  w, label="Sau",   color="#4CAF50", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Human (0)", "Machine (1)"])
    axes[0].set_title("Label Distribution")
    axes[0].set_ylabel("Số mẫu")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Source distribution after
    source_counts = df_after.groupby(["source", "label"]).size().unstack(fill_value=0)
    source_counts.plot(kind="bar", ax=axes[1],
                       color=["#4CAF50","#F44336"], edgecolor="white")
    axes[1].set_title("Source Distribution (Sau augmentation)")
    axes[1].set_xlabel("")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha="right")
    axes[1].legend(["Human","Machine"])
    axes[1].grid(axis="y", alpha=0.3)

    # Word count distribution before vs after (machine only)
    df_before["word_count"] = df_before["text"].str.split().str.len()
    df_after["word_count"]  = df_after["text"].str.split().str.len()
    for df, label, color in [
        (df_before[df_before["label"]==1], "Machine (trước)", "#F44336"),
        (df_after[df_after["label"]==1],   "Machine (sau)",   "#FF9800"),
    ]:
        axes[2].hist(df["word_count"].clip(0, 800), bins=50, alpha=0.6,
                     label=label, color=color, density=True, edgecolor="none")
    axes[2].set_title("Machine Text Length (Trước vs Sau)")
    axes[2].set_xlabel("Word Count"); axes[2].set_ylabel("Density")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "augmentation_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {out}")


# ==============================================================================
# SECTION D — TEXT CLEANING (dùng lại từ EDA)
# ==============================================================================
def clean_text(text: str) -> str:
    text = str(text)
    trunc_kws = ["References", "External links", "See also", "Further reading", "Bibliography"]
    trunc_pat = r"^\s*[=\-#*]*\s*(?:" + "|".join(re.escape(k) for k in trunc_kws) + r")\b"
    m = re.search(trunc_pat, text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        text = text[:m.start()]
    line_kws = ["Edit:", "ETA:", "TL;DR", "Tips:", "Warnings:", "Notes:"]
    line_pat = r"^\s*(?:" + "|".join(re.escape(k) for k in line_kws) + r").*$"
    text = re.sub(line_pat, "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"styleborderleftpx[^\s]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r'style\s*=\s*"[^"]*"',    "", text, flags=re.IGNORECASE)
    text = re.sub(r"style\s*=\s*'[^']*'",    "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>",                "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def deep_clean_ai_review(text: str) -> str:
    """Dọn dẹp sâu cho văn bản do AI sinh — loại boilerplate và placeholder."""
    patterns = [
        r"(?i)(I must say,\s*)?(I am|I'm)\s*(underwhelmed|astonished|surprised)[^\.]*\.\s*",
        r"(?i)(Paper Title|Authors):\s*\[Insert.*?\].*\n*",
        r"\[Insert.*?\]",
        r"(?i)^(Summary|Overall Assessment|Overall Impression|Strengths|"
        r"Weaknesses and Suggestions for Improvement|Weaknesses and Suggestions|"
        r"Minor Comments|Conclusion|Recommendation|Specific Comments|"
        r"Actionable Suggestions):\s*\n*",
        r"\bPeer Review\b",
        r"^[ \t]*(?:\d+\.|\*|\-)\s*(?:[^:\n]+:\s*)?",
    ]
    text = str(text)
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE | re.MULTILINE)
    last = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last > 0:
        text = text[:last + 1]
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    # ── Bước 1: Clone PeerRead ────────────────────────────────────────────────
    clone_peerread()

    # ── Bước 2: Load tập train gốc ───────────────────────────────────────────
    print("\nLoading original train set...")
    df_train = pd.read_json(TRAIN_PATH, lines=True)
    df_train = df_train.drop(columns=["id"], errors="ignore")
    df_train["text"] = df_train["text"].apply(clean_text)
    df_train = df_train[df_train["text"].str.split().str.len() >= MIN_WORDS_PER_TEXT]
    print(f"  Train gốc: {len(df_train):,} mẫu")

    # ── Bước 3: Human data từ PeerRead ───────────────────────────────────────
    df_human = extract_human_reviews()
    df_human["text"] = df_human["text"].apply(clean_text)

    # ── Bước 4: Machine data từ Groq API ─────────────────────────────────────
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    if not GROQ_API_KEY:
        print("\n[WARN] GROQ_API_KEY không tìm thấy.")
        print("  Colab: Secrets → GROQ_API_KEY")
        print("  Kaggle: Add-ons → Secrets → GROQ_API_KEY")
        print("  Bỏ qua bước sinh machine data.\n")
        df_machine = pd.DataFrame(columns=["text","label","source","model"])
    else:
        client    = Groq(api_key=GROQ_API_KEY)
        abstracts = extract_abstracts_from_peerread()

        # Số mẫu mỗi persona = human_samples / 3 để balance
        n_per_persona = max(1, len(df_human) // len(REVIEWER_PERSONAS))
        df_machine = generate_machine_reviews(abstracts, client, n_per_persona)
        df_machine["text"] = df_machine["text"].apply(deep_clean_ai_review)

    # ── Bước 5: Merge & Validate ──────────────────────────────────────────────
    df_combined = merge_datasets(df_train, df_human, df_machine)

    # ── Bước 6: Plot summary ──────────────────────────────────────────────────
    plot_augmentation_summary(df_train, df_combined)

    # ── Bước 7: Lưu output ───────────────────────────────────────────────────
    cols = ["text", "label", "source", "model"]
    # Đảm bảo cột model tồn tại
    if "model" not in df_combined.columns:
        df_combined["model"] = "unknown"

    df_combined[cols].to_json(
        OUTPUT_PATH, orient="records", lines=True, force_ascii=False
    )
    print(f"\n[SAVED] {OUTPUT_PATH}")
    print(f"  Tổng mẫu: {len(df_combined):,}")
    print(f"  Label 0 (Human)  : {(df_combined['label']==0).sum():,}")
    print(f"  Label 1 (Machine): {(df_combined['label']==1).sum():,}")


if __name__ == "__main__":
    main()
