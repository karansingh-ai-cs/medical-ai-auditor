"""
exploration.py
--------------
Analysis notebook (runnable as a Python script).
Explores the labeled dataset, visualizes label distributions,
checks labeling function agreement, and runs baseline analysis.

Run cell by cell in any IDE, or as a full script:
  python notebooks/exploration.py

Convert to Jupyter notebook (optional):
  pip install jupytext
  jupytext --to notebook notebooks/exploration.py
  jupyter notebook notebooks/exploration.ipynb
"""

# %% [markdown]
# # Medical AI Auditor — Data Exploration
# This notebook explores the labeled dataset and validates our weak supervision pipeline.

# %% Setup
import os, sys
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

# %% Load dataset
df = pd.read_csv("../data/labeled_dataset.csv")
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nSample:\n{df[['question','ai_response','label']].head(3).to_string()}")

# %% [markdown]
# ## 1. Label Distribution

# %% Label distribution
label_counts = df["label"].value_counts()
total = len(df)

LABEL_COLORS = {
    "no_issue":             "#1D9E75",
    "incomplete_reasoning": "#BA7517",
    "missing_causal_link":  "#534AB7",
    "dangerous_omission":   "#D85A30",
    "critical_failure":     "#A32D2D",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

# Bar chart
colors = [LABEL_COLORS[l] for l in label_counts.index]
bars = ax1.barh(label_counts.index, label_counts.values, color=colors, alpha=0.88, height=0.55)
for bar, v in zip(bars, label_counts.values):
    ax1.text(v + 10, bar.get_y() + bar.get_height()/2,
             f"{v:,}  ({100*v/total:.1f}%)", va="center", fontsize=9)
ax1.set_xlabel("Count")
ax1.set_title("Label Distribution")
ax1.invert_yaxis()
ax1.spines[["top","right"]].set_visible(False)

# Pie chart
ax2.pie(label_counts.values, labels=label_counts.index,
        colors=colors, autopct="%1.1f%%", startangle=140,
        pctdistance=0.85, labeldistance=1.05,
        textprops={"fontsize": 9})
ax2.set_title("Label Proportions")

plt.suptitle("Label Distribution in Labeled Dataset", fontsize=12)
plt.tight_layout()
plt.savefig("../experiments/label_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: experiments/label_distribution.png")

# %% [markdown]
# ## 2. Response Length Analysis by Label

# %% Response length vs label
df["response_word_count"] = df["ai_response"].str.split().str.len()
df["question_word_count"] = df["question"].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Boxplot: response length by label
order = ["no_issue", "incomplete_reasoning", "missing_causal_link",
         "dangerous_omission", "critical_failure"]
palette = {k: v for k, v in LABEL_COLORS.items()}

sns.boxplot(data=df, y="label", x="response_word_count", order=order,
            palette=palette, ax=axes[0], orient="h")
axes[0].set_title("Response Word Count by Label")
axes[0].set_xlabel("Word count")
axes[0].spines[["top","right"]].set_visible(False)

# Violin plot
sns.violinplot(data=df, y="label", x="response_word_count", order=order,
               palette=palette, ax=axes[1], orient="h", inner="quartile")
axes[1].set_title("Response Word Count Distribution (Violin)")
axes[1].set_xlabel("Word count")
axes[1].spines[["top","right"]].set_visible(False)

plt.suptitle("Response Length vs Label", fontsize=12)
plt.tight_layout()
plt.savefig("../experiments/response_length_by_label.png", dpi=150, bbox_inches="tight")
plt.show()

# Print stats
print("\nMean response word count by label:")
print(df.groupby("label")["response_word_count"].mean().round(1).to_string())

# %% [markdown]
# ## 3. Labeling Function Analysis
# Check how often each LF fires, and inter-LF agreement.

# %% LF analysis
from src.detection.weak_labeler import label_dataset, LABELING_FUNCTIONS

# Get verbose labels (includes which LF fired)
df_verbose = label_dataset(df[["question", "ai_response"]].copy(), verbose=True)

# Plot LF firing counts
fired_counts = df_verbose["fired_by"].value_counts()

fig, ax = plt.subplots(figsize=(10, 4))
colors_lf = plt.cm.Set2(np.linspace(0, 1, len(fired_counts)))
bars = ax.bar(range(len(fired_counts)), fired_counts.values,
              color=colors_lf, alpha=0.88)

for bar, (lf, count) in zip(bars, fired_counts.items()):
    pct = 100 * count / total
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)

ax.set_xticks(range(len(fired_counts)))
ax.set_xticklabels(fired_counts.index, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Samples labeled by this LF")
ax.set_title("Labeling Function Firing Counts")
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("../experiments/lf_firing_counts.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Keyword Coverage Analysis
# What fraction of responses contain safety/causal keywords?

# %% Keyword coverage
from src.knowledge.medical_keywords import SAFETY_REFERRAL_PHRASES, CAUSAL_CONNECTORS

def coverage_rate(column: str, keywords: list) -> pd.Series:
    return df[column].str.lower().apply(
        lambda text: any(kw in text for kw in keywords)
    )

df["has_safety_phrase"] = coverage_rate("ai_response", SAFETY_REFERRAL_PHRASES)
df["has_causal_connector"] = coverage_rate("ai_response", CAUSAL_CONNECTORS)

print("Safety phrase present by label:")
print(df.groupby("label")["has_safety_phrase"].mean().round(3).to_string())

print("\nCausal connector present by label:")
print(df.groupby("label")["has_causal_connector"].mean().round(3).to_string())

# Heatmap
coverage_df = df.groupby("label")[["has_safety_phrase","has_causal_connector"]].mean()
coverage_df.columns = ["Safety phrase rate", "Causal connector rate"]

fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(coverage_df, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=0, vmax=1, linewidths=0.5, ax=ax)
ax.set_title("Keyword Coverage by Label")
plt.tight_layout()
plt.savefig("../experiments/keyword_coverage.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Most Common Words per Label (Simple TF-IDF-like analysis)

# %% Word frequency per label
from collections import Counter
import re

STOPWORDS = {"the","a","an","is","it","to","of","in","and","or","for",
             "be","was","are","that","this","with","have","you","your",
             "not","if","on","at","by","but","can","do","has","as","my"}

def top_words(texts: list, n: int = 12) -> list:
    words = []
    for text in texts:
        words.extend(re.findall(r'\b[a-z]{4,}\b', text.lower()))
    filtered = [w for w in words if w not in STOPWORDS]
    return Counter(filtered).most_common(n)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, label in enumerate(order):
    subset = df[df["label"] == label]["ai_response"].tolist()
    top    = top_words(subset)
    words, counts = zip(*top) if top else ([], [])
    color = LABEL_COLORS[label]
    axes[i].barh(words[::-1], counts[::-1], color=color, alpha=0.85)
    axes[i].set_title(f"{label}\n(n={len(subset):,})", fontsize=9)
    axes[i].spines[["top","right"]].set_visible(False)

axes[5].axis("off")  # hide empty subplot
plt.suptitle("Top Words in AI Responses by Label", fontsize=12)
plt.tight_layout()
plt.savefig("../experiments/top_words_by_label.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Prompt Type vs Label Distribution
# Shows how the generation strategy seeded label diversity.

# %% Prompt type analysis
if "prompt_type" in df.columns:
    cross = pd.crosstab(df["prompt_type"], df["label"], normalize="index")
    fig, ax = plt.subplots(figsize=(10, 4))
    cross.plot(kind="bar", ax=ax, color=list(LABEL_COLORS.values()), alpha=0.85)
    ax.set_title("Label Distribution by Prompt Type (row-normalised)")
    ax.set_xlabel("Prompt type used for generation")
    ax.set_ylabel("Proportion")
    ax.legend(loc="upper right", fontsize=8)
    ax.tick_params(axis="x", rotation=15)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("../experiments/prompt_type_vs_label.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nPrompt type → label distribution:")
    print(cross.round(3).to_string())
else:
    print("No 'prompt_type' column found — skipping this section.")

# %% [markdown]
# ## 7. Sample Examples per Label
# Manual inspection of a few examples from each label.

# %% Inspect examples
pd.set_option("display.max_colwidth", 80)

for label in order:
    print(f"\n{'═'*65}")
    print(f"  {label.upper()}")
    print(f"{'═'*65}")
    sample = df[df["label"] == label][["question","ai_response"]].head(2)
    for _, row in sample.iterrows():
        print(f"  Q: {row['question'][:70]}")
        print(f"  R: {row['ai_response'][:70]}")
        print()

# %% Summary statistics
print("\n" + "═"*50)
print("  DATASET SUMMARY")
print("═"*50)
print(f"  Total samples         : {len(df):,}")
print(f"  Unique questions      : {df['question'].nunique():,}")
print(f"  Mean response length  : {df['response_word_count'].mean():.1f} words")
print(f"  Min response length   : {df['response_word_count'].min()} words")
print(f"  Max response length   : {df['response_word_count'].max()} words")
print(f"  Has safety phrase     : {df['has_safety_phrase'].mean():.1%}")
print(f"  Has causal connector  : {df['has_causal_connector'].mean():.1%}")
print("═"*50)
