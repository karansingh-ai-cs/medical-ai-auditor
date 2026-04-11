"""
evaluate_all.py
---------------
Evaluate and compare all three systems:
  1. Rule-Based Baseline
  2. ML-Only (BERT or BioBERT)
  3. Hybrid System

Outputs:
  - Console: per-class classification report + macro metrics
  - evaluation/cm_rule.png         — confusion matrix (rule-based)
  - evaluation/cm_ml.png           — confusion matrix (ML-only)
  - evaluation/cm_hybrid.png       — confusion matrix (hybrid)
  - evaluation/comparison.png      — side-by-side accuracy/F1 bar chart
  - evaluation/error_analysis.csv  — misclassified examples
  - evaluation/results_summary.json — all numbers in one place
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
)

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

LABEL_NAMES = [
    "no_issue", "incomplete_reasoning", "missing_causal_link",
    "dangerous_omission", "critical_failure"
]

OUTPUT_DIR = "evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Prediction getters ────────────────────────────────────────────────────────

def get_rule_preds(df: pd.DataFrame) -> list[str]:
    from src.detection.weak_labeler import detect_label
    return [detect_label(r["question"], r["ai_response"]) for _, r in df.iterrows()]


def get_ml_preds(df: pd.DataFrame, model, tokenizer, device: str = "cpu") -> list[str]:
    from src.models.bert_classifier import predict_batch
    preds = predict_batch(
        model, tokenizer,
        df["question"].tolist(),
        df["ai_response"].tolist(),
        device=device,
    )
    return [p["label"] for p in preds]


def get_hybrid_preds(df: pd.DataFrame, auditor) -> list[str]:
    results = auditor.audit_batch(
        df["question"].tolist(),
        df["ai_response"].tolist(),
    )
    return [r["final_label"] for r in results]


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "macro_f1":  round(f1_score(y_true, y_pred, average="macro",
                         labels=LABEL_NAMES, zero_division=0), 4),
        "macro_p":   round(precision_score(y_true, y_pred, average="macro",
                         labels=LABEL_NAMES, zero_division=0), 4),
        "macro_r":   round(recall_score(y_true, y_pred, average="macro",
                         labels=LABEL_NAMES, zero_division=0), 4),
        "report":    classification_report(
                         y_true, y_pred,
                         labels=LABEL_NAMES,
                         target_names=LABEL_NAMES,
                         zero_division=0, output_dict=True),
    }


def print_report(name: str, metrics: dict):
    print(f"\n{'═'*60}")
    print(f"  System: {name}")
    print(f"{'═'*60}")
    report = classification_report(
        metrics["_y_true"], metrics["_y_pred"],
        labels=LABEL_NAMES,
        target_names=LABEL_NAMES, zero_division=0
    )
    print(report)
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Macro-F1 : {metrics['macro_f1']:.4f}")
    print(f"  Macro-P  : {metrics['macro_p']:.4f}")
    print(f"  Macro-R  : {metrics['macro_r']:.4f}")


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title: str, save_path: str):
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_NAMES)

    # Normalise rows to percentages
    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    short_labels = ["no_issue", "incomplete", "missing_causal", "dangerous", "critical"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="normal")

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", ax=axes[0],
                xticklabels=short_labels, yticklabels=short_labels,
                cmap="Blues", linewidths=0.3, linecolor="#e0e0e0")
    axes[0].set_title("Counts")
    axes[0].set_ylabel("True label")
    axes[0].set_xlabel("Predicted label")
    axes[0].tick_params(axis="x", rotation=30)

    # Normalised
    sns.heatmap(cm_norm, annot=True, fmt=".2f", ax=axes[1],
                xticklabels=short_labels, yticklabels=short_labels,
                cmap="Greens", vmin=0, vmax=1, linewidths=0.3, linecolor="#e0e0e0")
    axes[1].set_title("Row-normalised (recall per class)")
    axes[1].set_ylabel("True label")
    axes[1].set_xlabel("Predicted label")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")


# ── Comparison bar chart ──────────────────────────────────────────────────────

def plot_comparison(results: dict):
    """Side-by-side bar chart: accuracy + macro-F1 for all systems."""
    systems = list(results.keys())
    accs    = [results[s]["accuracy"]  for s in systems]
    f1s     = [results[s]["macro_f1"]  for s in systems]
    precs   = [results[s]["macro_p"]   for s in systems]
    recalls = [results[s]["macro_r"]   for s in systems]

    x     = np.arange(len(systems))
    width = 0.2

    COLORS = ["#1D9E75", "#534AB7", "#D85A30", "#BA7517"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5*width, accs,    width, label="Accuracy",  color=COLORS[0], alpha=0.85)
    ax.bar(x - 0.5*width, f1s,     width, label="Macro-F1",  color=COLORS[1], alpha=0.85)
    ax.bar(x + 0.5*width, precs,   width, label="Macro-P",   color=COLORS[2], alpha=0.85)
    ax.bar(x + 1.5*width, recalls, width, label="Macro-R",   color=COLORS[3], alpha=0.85)

    for i, (a, f, p, r) in enumerate(zip(accs, f1s, precs, recalls)):
        for j, v in enumerate([a, f, p, r]):
            ax.text(i + (j-1.5)*width, v + 0.008, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("System Comparison: Rule-based vs ML-only vs Hybrid", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    save_path = os.path.join(OUTPUT_DIR, "comparison.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")


# ── Per-class F1 radar-style bar ──────────────────────────────────────────────

def plot_per_class_f1(all_results: dict):
    """Grouped bar chart of per-class F1 across systems."""
    # Generate colors dynamically based on number of systems
    ALL_COLORS = ["#1D9E75", "#534AB7", "#D85A30", "#BA7517", "#A32D2D", "#185FA5"]
    n_systems  = len(all_results)
    colors     = ALL_COLORS[:n_systems]

    x       = np.arange(len(LABEL_NAMES))
    width   = 0.8 / n_systems
    offsets = np.linspace(-(n_systems-1)*width/2, (n_systems-1)*width/2, n_systems)

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (sname, res) in enumerate(all_results.items()):
        f1s = [
            res["report"].get(label, {}).get("f1-score", 0)
            for label in LABEL_NAMES
        ]
        ax.bar(x + offsets[i], f1s, width * 0.9,
               label=sname, color=colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1-score")
    ax.set_title("Per-class F1: System Comparison", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    save_path = os.path.join(OUTPUT_DIR, "per_class_f1.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")


# ── Error analysis ────────────────────────────────────────────────────────────

def error_analysis(
    df: pd.DataFrame,
    y_true: list[str],
    y_pred: list[str],
    system_name: str,
    n: int = 50,
):
    """
    Find and save misclassified examples.
    Particularly flag cases where a dangerous/critical case was mislabeled as safe.
    """
    errors = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            row = df.iloc[i]
            errors.append({
                "system":         system_name,
                "question":       row["question"][:150],
                "ai_response":    row["ai_response"][:150],
                "true_label":     true,
                "pred_label":     pred,
                "is_safety_miss": (
                    true in {"dangerous_omission", "critical_failure"}
                    and pred in {"no_issue", "incomplete_reasoning"}
                ),
            })

    err_df = pd.DataFrame(errors)
    safety_misses = err_df["is_safety_miss"].sum() if len(err_df) else 0

    print(f"\n  [{system_name}] Errors: {len(errors)} / {len(y_true)}  "
          f"| Safety misses (dangerous → safe): {safety_misses}")

    save_path = os.path.join(OUTPUT_DIR, f"errors_{system_name.lower().replace(' ','_')}.csv")
    err_df.head(n).to_csv(save_path, index=False)
    logger.info(f"Saved error analysis: {save_path}")

    return err_df


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation(
    test_csv:         str   = "data/labeled_dataset.csv",
    checkpoint_bert:  str   = "experiments/best_bert.pt",
    checkpoint_bio:   str   = "experiments/best_biobert.pt",
    bert_model:       str   = "bert-base-uncased",
    bio_model:        str   = "dmis-lab/biobert-base-cased-v1.2",
    test_size:        int   = 500,
    device:           str   = "cpu",
    skip_ml:          bool  = False,   # set True for fast rule-only testing
):
    """
    Run complete evaluation across all three systems.
    """
    # ── Load test data ─────────────────────────────────────────────────────────
    df = pd.read_csv(test_csv)
    test_df = df.sample(min(test_size, len(df)), random_state=42).reset_index(drop=True)
    y_true  = test_df["label"].tolist()
    logger.info(f"Test set: {len(test_df)} samples")

    all_results = {}

    # ── 1. Rule-Based ──────────────────────────────────────────────────────────
    logger.info("Evaluating rule-based system...")
    rule_preds = get_rule_preds(test_df)
    rule_metrics = compute_metrics(y_true, rule_preds)
    rule_metrics["_y_true"] = y_true
    rule_metrics["_y_pred"] = rule_preds
    all_results["Rule-Based"] = rule_metrics
    print_report("Rule-Based", rule_metrics)
    plot_confusion_matrix(y_true, rule_preds, "Rule-Based System",
                          os.path.join(OUTPUT_DIR, "cm_rule.png"))
    error_analysis(test_df, y_true, rule_preds, "Rule-Based")

    if not skip_ml:
        # ── 2. ML-Only (BERT) ─────────────────────────────────────────────────
        if os.path.exists(checkpoint_bert):
            logger.info("Evaluating ML-only (BERT)...")
            from src.models.bert_classifier import load_from_checkpoint
            from src.detection.hybrid_auditor import HybridAuditor

            bert_model_obj, bert_tok = load_from_checkpoint(
                checkpoint_bert, bert_model, device=device)
            ml_preds = get_ml_preds(test_df, bert_model_obj, bert_tok, device)
            ml_metrics = compute_metrics(y_true, ml_preds)
            ml_metrics["_y_true"] = y_true
            ml_metrics["_y_pred"] = ml_preds
            all_results["ML-Only (BERT)"] = ml_metrics
            print_report("ML-Only (BERT)", ml_metrics)
            plot_confusion_matrix(y_true, ml_preds, "ML-Only BERT",
                                  os.path.join(OUTPUT_DIR, "cm_bert.png"))

            # ── 3. Hybrid (BERT) ─────────────────────────────────────────────
            logger.info("Evaluating hybrid (BERT)...")
            hybrid = HybridAuditor(bert_model_obj, bert_tok, device=device)
            hyb_preds = get_hybrid_preds(test_df, hybrid)
            hyb_metrics = compute_metrics(y_true, hyb_preds)
            hyb_metrics["_y_true"] = y_true
            hyb_metrics["_y_pred"] = hyb_preds
            all_results["Hybrid (BERT)"] = hyb_metrics
            print_report("Hybrid (BERT)", hyb_metrics)
            plot_confusion_matrix(y_true, hyb_preds, "Hybrid System (BERT)",
                                  os.path.join(OUTPUT_DIR, "cm_hybrid_bert.png"))
            error_analysis(test_df, y_true, hyb_preds, "Hybrid-BERT")

        # ── 4. ML-Only (BioBERT) + Hybrid ────────────────────────────────────
        if os.path.exists(checkpoint_bio):
            logger.info("Evaluating ML-only (BioBERT)...")
            from src.models.bert_classifier import load_from_checkpoint
            from src.detection.hybrid_auditor import HybridAuditor

            bio_model_obj, bio_tok = load_from_checkpoint(
                checkpoint_bio, bio_model, device=device)
            bio_preds = get_ml_preds(test_df, bio_model_obj, bio_tok, device)
            bio_metrics = compute_metrics(y_true, bio_preds)
            bio_metrics["_y_true"] = y_true
            bio_metrics["_y_pred"] = bio_preds
            all_results["ML-Only (BioBERT)"] = bio_metrics
            print_report("ML-Only (BioBERT)", bio_metrics)
            plot_confusion_matrix(y_true, bio_preds, "ML-Only BioBERT",
                                  os.path.join(OUTPUT_DIR, "cm_biobert.png"))

            logger.info("Evaluating hybrid (BioBERT)...")
            bio_hybrid   = HybridAuditor(bio_model_obj, bio_tok, device=device)
            bioh_preds   = get_hybrid_preds(test_df, bio_hybrid)
            bioh_metrics = compute_metrics(y_true, bioh_preds)
            bioh_metrics["_y_true"] = y_true
            bioh_metrics["_y_pred"] = bioh_preds
            all_results["Hybrid (BioBERT)"] = bioh_metrics
            print_report("Hybrid (BioBERT)", bioh_metrics)
            plot_confusion_matrix(y_true, bioh_preds, "Hybrid System (BioBERT)",
                                  os.path.join(OUTPUT_DIR, "cm_hybrid_biobert.png"))
            error_analysis(test_df, y_true, bioh_preds, "Hybrid-BioBERT")

    # ── Charts ─────────────────────────────────────────────────────────────────
    # Strip internal keys before saving
    saveable = {
        k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        for k, v in all_results.items()
    }
    plot_comparison(saveable)
    plot_per_class_f1(saveable)

    # ── Save JSON summary ──────────────────────────────────────────────────────
    summary_path = os.path.join(OUTPUT_DIR, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(saveable, f, indent=2)
    logger.info(f"Saved: {summary_path}")

    # ── Print final table ──────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  FINAL COMPARISON TABLE")
    print("═"*60)
    print(f"  {'System':<25} {'Acc':>6} {'F1':>6} {'P':>6} {'R':>6}")
    print(f"  {'─'*25} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    for sname, res in saveable.items():
        print(f"  {sname:<25} {res['accuracy']:>6.4f} {res['macro_f1']:>6.4f} "
              f"{res['macro_p']:>6.4f} {res['macro_r']:>6.4f}")
    print("═"*60 + "\n")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", default="data/labeled_dataset.csv")
    parser.add_argument("--bert_ckpt", default="experiments/best_bert.pt")
    parser.add_argument("--bio_ckpt",  default="experiments/best_biobert.pt")
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--rules_only", action="store_true")
    args = parser.parse_args()

    run_evaluation(
        test_csv        = args.test_csv,
        checkpoint_bert = args.bert_ckpt,
        checkpoint_bio  = args.bio_ckpt,
        device          = args.device,
        test_size       = args.test_size,
        skip_ml         = args.rules_only,
    )
