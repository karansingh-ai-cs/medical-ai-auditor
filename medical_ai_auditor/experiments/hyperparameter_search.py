"""
hyperparameter_search.py
------------------------
Grid search over key hyperparameters.
Tries every combination, logs results to experiments/hparam_results.csv,
and prints the best configuration at the end.

Tested hyperparameters:
  - learning_rate:  [1e-5, 2e-5, 3e-5]
  - batch_size:     [8, 16]
  - dropout:        [0.1, 0.3]
  - freeze_layers:  [0, 4]

Total combinations: 3 × 2 × 2 × 2 = 24
Estimated time: ~4 min per combo on GPU, ~25 min on CPU (2 epochs each)

Usage:
  python experiments/hyperparameter_search.py
  python experiments/hyperparameter_search.py --device cuda --epochs 2
"""

import os
import sys
import csv
import time
import itertools
import logging
import argparse

import torch
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ── Search grid ────────────────────────────────────────────────────────────────
GRID = {
    "learning_rate": [1e-5, 2e-5, 3e-5],
    "batch_size":    [8, 16],
    "dropout":       [0.1, 0.3],
    "freeze_layers": [0, 4],
}

RESULTS_PATH = "experiments/hparam_results.csv"
RESULT_COLS  = [
    "run_id", "learning_rate", "batch_size", "dropout", "freeze_layers",
    "val_accuracy", "val_macro_f1", "val_macro_p", "val_macro_r",
    "train_time_sec", "best_epoch",
]


def run_single(
    train_df, val_df,
    model_name:    str,
    learning_rate: float,
    batch_size:    int,
    dropout:       float,
    freeze_layers: int,
    epochs:        int,
    max_len:       int,
    device:        str,
    run_id:        int,
) -> dict:
    """Train + validate one hyperparameter combination. Returns metrics dict."""
    from src.models.bert_classifier import load_model
    from src.models.train import train

    logger.info(f"\nRun {run_id} │ lr={learning_rate} bs={batch_size} "
                f"dropout={dropout} freeze={freeze_layers}")

    model, tokenizer = load_model(
        model_name, num_labels=5, dropout=dropout, freeze_bert_layers=freeze_layers
    )

    t0 = time.time()
    _, history = train(
        model=model, tokenizer=tokenizer,
        train_df=train_df, val_df=val_df,
        epochs=epochs, batch_size=batch_size, lr=learning_rate,
        max_len=max_len, save_dir="experiments",
        model_tag=f"hparam_run{run_id}",
        device=device, patience=1,   # aggressive early stop for speed
    )
    elapsed = round(time.time() - t0, 1)

    best_f1    = max(history["val_macro_f1"])
    best_epoch = history["val_macro_f1"].index(best_f1) + 1
    best_acc   = history["val_accuracy"][best_epoch - 1]

    # Clean up checkpoint to save disk space
    ckpt = f"experiments/hparam_run{run_id}.pt"
    if os.path.exists(ckpt):
        os.remove(ckpt)

    return {
        "run_id":          run_id,
        "learning_rate":   learning_rate,
        "batch_size":      batch_size,
        "dropout":         dropout,
        "freeze_layers":   freeze_layers,
        "val_accuracy":    best_acc,
        "val_macro_f1":    best_f1,
        "val_macro_p":     max(history.get("val_macro_p", [0])),
        "val_macro_r":     max(history.get("val_macro_r", [0])),
        "train_time_sec":  elapsed,
        "best_epoch":      best_epoch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="data/labeled_dataset.csv")
    parser.add_argument("--model",   default="bert-base-uncased")
    parser.add_argument("--epochs",  type=int, default=2)
    parser.add_argument("--max_len", type=int, default=256)   # shorter for speed
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_runs", type=int, default=24,
                        help="Max combinations to try (for quick testing)")
    args = parser.parse_args()

    from src.models.dataset import split_dataset

    # Load & split data once
    logger.info(f"Loading data: {args.data}")
    df = pd.read_csv(args.data)
    train_df, val_df, _ = split_dataset(df)

    # Generate all combinations
    keys    = list(GRID.keys())
    values  = list(GRID.values())
    combos  = list(itertools.product(*values))[:args.max_runs]
    total   = len(combos)

    logger.info(f"Grid search: {total} combinations, {args.epochs} epochs each")

    # Resume from existing results
    done_runs = set()
    all_results = []
    if os.path.exists(RESULTS_PATH):
        existing = pd.read_csv(RESULTS_PATH)
        all_results = existing.to_dict("records")
        done_runs   = set(existing["run_id"].tolist())
        logger.info(f"Resuming — {len(done_runs)} runs already done.")

    # CSV writer (append mode)
    write_header = not os.path.exists(RESULTS_PATH)
    csv_file = open(RESULTS_PATH, "a", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=RESULT_COLS)
    if write_header:
        writer.writeheader()

    # Run grid search
    for i, combo in enumerate(combos, 1):
        if i in done_runs:
            continue
        params = dict(zip(keys, combo))
        try:
            result = run_single(
                train_df=train_df, val_df=val_df,
                model_name=args.model,
                epochs=args.epochs, max_len=args.max_len, device=args.device,
                run_id=i, **params,
            )
            all_results.append(result)
            writer.writerow(result)
            csv_file.flush()
            logger.info(f"  Run {i}/{total} done — val_macro_f1={result['val_macro_f1']:.4f}")
        except Exception as e:
            logger.error(f"Run {i} failed: {e}")

    csv_file.close()

    # ── Summary ────────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results).sort_values("val_macro_f1", ascending=False)

    print("\n" + "═"*70)
    print("  HYPERPARAMETER SEARCH RESULTS (top 5 by val macro-F1)")
    print("═"*70)
    print(results_df[["run_id","learning_rate","batch_size","dropout",
                       "freeze_layers","val_macro_f1","val_accuracy"]].head(5).to_string(index=False))

    best = results_df.iloc[0]
    print(f"\n  Best config (run #{int(best['run_id'])}):")
    print(f"    learning_rate : {best['learning_rate']}")
    print(f"    batch_size    : {int(best['batch_size'])}")
    print(f"    dropout       : {best['dropout']}")
    print(f"    freeze_layers : {int(best['freeze_layers'])}")
    print(f"    val_macro_f1  : {best['val_macro_f1']:.4f}")
    print(f"\n  Full results saved: {RESULTS_PATH}")
    print("═"*70)

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        for ax, param in zip(axes, ["learning_rate", "batch_size", "dropout"]):
            grouped = results_df.groupby(param)["val_macro_f1"].mean()
            ax.bar([str(k) for k in grouped.index], grouped.values,
                   color="#534AB7", alpha=0.85)
            ax.set_title(f"Effect of {param}")
            ax.set_ylabel("Mean val macro-F1")
            ax.set_ylim(0, 1)
            ax.spines[["top","right"]].set_visible(False)

        plt.suptitle("Hyperparameter Search — Effect on Validation Macro-F1", fontsize=11)
        plt.tight_layout()
        plot_path = "experiments/hparam_search.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {plot_path}")
    except Exception as e:
        logger.warning(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
