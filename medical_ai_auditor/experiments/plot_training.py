"""
plot_training.py
----------------
Read training history JSON files and produce publication-ready plots.

Generates:
  experiments/training_curves.png — loss, accuracy, macro-F1 per epoch
  experiments/bert_vs_biobert.png — side-by-side BERT vs BioBERT comparison

Usage:
  python experiments/plot_training.py
  python experiments/plot_training.py --bert experiments/history_bert.json
                                      --bio  experiments/history_biobert.json
"""

import os
import sys
import json
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PALETTE = {
    "bert":        "#534AB7",
    "biobert":     "#1D9E75",
    "train_loss":  "#E24B4A",
    "val_acc":     "#534AB7",
    "val_f1":      "#1D9E75",
    "grid":        "#e8e8e8",
}


def load_history(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def smooth(values: list, alpha: float = 0.3) -> list:
    """Exponential moving average smoothing."""
    result, last = [], values[0]
    for v in values:
        last = alpha * v + (1 - alpha) * last
        result.append(last)
    return result


def plot_single_run(history: dict, model_tag: str, save_path: str):
    """
    Three-panel plot: train loss / val accuracy / val macro-F1 vs. epoch.
    """
    epochs   = list(range(1, len(history["train_loss"]) + 1))
    loss     = history["train_loss"]
    acc      = history["val_accuracy"]
    f1       = history["val_macro_f1"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f"Training Curves — {model_tag.upper()}", fontsize=12, fontweight="normal")

    data_series = [
        (axes[0], loss, "Train Loss",        PALETTE["train_loss"], "Loss"),
        (axes[1], acc,  "Val Accuracy",      PALETTE["val_acc"],    "Accuracy"),
        (axes[2], f1,   "Val Macro-F1",      PALETTE["val_f1"],     "Macro-F1"),
    ]

    for ax, values, label, color, ylabel in data_series:
        ax.plot(epochs, values, "o--", color=color, linewidth=1.5,
                markersize=5, label=label, alpha=0.6)
        ax.plot(epochs, smooth(values), "-", color=color, linewidth=2.2,
                label=f"{label} (smoothed)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.set_xticks(epochs)
        ax.yaxis.grid(True, linestyle="--", color=PALETTE["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)

        # Annotate best epoch
        best_idx = (np.argmin(values) if "Loss" in ylabel else np.argmax(values))
        best_val = values[best_idx]
        ax.annotate(
            f"best: {best_val:.4f}",
            xy=(epochs[best_idx], best_val),
            xytext=(5, 8), textcoords="offset points",
            fontsize=8, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_comparison(bert_history: dict, bio_history: dict, save_path: str):
    """
    Side-by-side comparison of BERT vs BioBERT training curves.
    Two metrics shown: val_accuracy and val_macro_f1.
    """
    epochs_bert = list(range(1, len(bert_history["train_loss"]) + 1))
    epochs_bio  = list(range(1, len(bio_history["train_loss"])  + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("BERT vs BioBERT — Training Comparison", fontsize=12)

    for ax, metric, ylabel in [
        (axes[0], "val_accuracy", "Validation Accuracy"),
        (axes[1], "val_macro_f1", "Validation Macro-F1"),
    ]:
        bert_vals = bert_history[metric]
        bio_vals  = bio_history[metric]

        ax.plot(epochs_bert, bert_vals, "o-", color=PALETTE["bert"],
                label="BERT", linewidth=1.8, markersize=5)
        ax.plot(epochs_bio,  bio_vals,  "s-", color=PALETTE["biobert"],
                label="BioBERT", linewidth=1.8, markersize=5)

        # Shade region where BioBERT leads
        min_len = min(len(bert_vals), len(bio_vals))
        x_range = list(range(1, min_len + 1))
        ax.fill_between(
            x_range,
            bert_vals[:min_len],
            bio_vals[:min_len],
            where=[b > a for a, b in zip(bert_vals[:min_len], bio_vals[:min_len])],
            alpha=0.12, color=PALETTE["biobert"], label="BioBERT advantage"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(metric.replace("_", " ").title())
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, linestyle="--", color=PALETTE["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_label_distribution(csv_path: str, save_path: str):
    """Bar chart of label distribution in the labeled dataset."""
    import pandas as pd

    df      = pd.read_csv(csv_path)
    counts  = df["label"].value_counts()
    labels  = counts.index.tolist()
    values  = counts.values.tolist()
    total   = sum(values)

    LABEL_COLORS = {
        "no_issue":             "#1D9E75",
        "incomplete_reasoning": "#BA7517",
        "missing_causal_link":  "#534AB7",
        "dangerous_omission":   "#D85A30",
        "critical_failure":     "#A32D2D",
    }
    colors = [LABEL_COLORS.get(l, "#888780") for l in labels]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(labels, values, color=colors, alpha=0.88, height=0.55)

    # Percentage labels on bars
    for bar, v in zip(bars, values):
        pct = 100 * v / total
        ax.text(v + 15, bar.get_y() + bar.get_height() / 2,
                f"{v:,}  ({pct:.1f}%)", va="center", fontsize=9)

    ax.set_xlabel("Number of samples")
    ax.set_title("Label Distribution in Labeled Dataset", fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", color=PALETTE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.invert_yaxis()  # most common at top

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert",  default="experiments/history_bert.json")
    parser.add_argument("--bio",   default="experiments/history_biobert.json")
    parser.add_argument("--data",  default="data/labeled_dataset.csv")
    args = parser.parse_args()

    os.makedirs("experiments", exist_ok=True)

    # Single-run plots
    if os.path.exists(args.bert):
        h = load_history(args.bert)
        plot_single_run(h, "bert", "experiments/training_curves_bert.png")
    else:
        logger.warning(f"BERT history not found: {args.bert}")

    if os.path.exists(args.bio):
        h = load_history(args.bio)
        plot_single_run(h, "biobert", "experiments/training_curves_biobert.png")
    else:
        logger.warning(f"BioBERT history not found: {args.bio}")

    # Comparison plot
    if os.path.exists(args.bert) and os.path.exists(args.bio):
        plot_comparison(
            load_history(args.bert),
            load_history(args.bio),
            "experiments/bert_vs_biobert.png",
        )

    # Label distribution
    if os.path.exists(args.data):
        plot_label_distribution(args.data, "experiments/label_distribution.png")
    else:
        logger.warning(f"Data file not found: {args.data}. Skipping label plot.")


if __name__ == "__main__":
    main()
