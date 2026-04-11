"""
run_pipeline.py
---------------
Master script: runs the entire pipeline end-to-end.

Steps:
  1. Load MedQuAD data
  2. Generate AI responses (template or OpenAI)
  3. Apply weak supervision labels
  4. Train BERT classifier
  5. (Optional) Train BioBERT classifier
  6. Run evaluation on all systems

Usage:
  # Full run with template responses (no API key needed):
  python experiments/run_pipeline.py

  # With OpenAI responses:
  OPENAI_API_KEY=sk-... python experiments/run_pipeline.py --openai

  # Skip data generation (reuse existing data/labeled_dataset.csv):
  python experiments/run_pipeline.py --skip_data

  # Train BioBERT instead of BERT:
  python experiments/run_pipeline.py --model biobert

  # Use GPU:
  python experiments/run_pipeline.py --device cuda
"""

import os
import sys
import argparse
import logging
import torch
import pandas as pd

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Medical AI Auditor — Full Pipeline")
    p.add_argument("--openai",       action="store_true", help="Use OpenAI for response generation")
    p.add_argument("--skip_data",    action="store_true", help="Skip data loading & labeling")
    p.add_argument("--skip_train",   action="store_true", help="Skip training, go straight to eval")
    p.add_argument("--model",        default="bert",      help="bert | biobert | clinicalbert")
    p.add_argument("--epochs",       type=int, default=4)
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--max_samples",  type=int, default=5000, help="Max Q&A pairs to use")
    p.add_argument("--max_len",      type=int, default=512)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--eval_only_rules", action="store_true", help="Evaluate rule-based only (fast)")
    return p.parse_args()


MODEL_NAMES = {
    "bert":         "bert-base-uncased",
    "biobert":      "dmis-lab/biobert-base-cased-v1.2",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "pubmedbert":   "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
}

# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    model_hf_name = MODEL_NAMES.get(args.model, args.model)
    tag           = args.model

    # BioBERT needs lower learning rate — it diverges at 2e-5
    if args.model in ("biobert", "clinicalbert", "pubmedbert") and args.lr == 2e-5:
        args.lr = 1e-5
        print(f"  Note: Using lr=1e-5 for {args.model} (2e-5 causes divergence)")

    print("\n" + "═"*65)
    print("  Medical AI Reliability Auditor — Pipeline")
    print("═"*65)
    print(f"  Model   : {model_hf_name}")
    print(f"  Device  : {args.device}")
    print(f"  Epochs  : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print("═"*65 + "\n")

    # ════════════════════════════════════════════════════════
    # STEP 1: Load data
    # ════════════════════════════════════════════════════════
    labeled_path = "data/labeled_dataset.csv"

    if not args.skip_data:
        from src.data_pipeline.load_data import build_combined_dataset
        from src.data_pipeline.generate_responses import generate_responses
        from src.detection.weak_labeler import label_dataset, print_label_stats

        logger.info("Step 1/4 — Loading real medical Q&A data...")
        raw_df = build_combined_dataset(
            use_real_data=True,
            medquad_max=args.max_samples,
            save_path="data/raw_questions.csv",
        )

        # If real data loaded successfully, ai_response column already exists
        # (ChatDoctor/HealthCareMagic have real responses built-in)
        # Only run generation if we only have questions (MedQuAD fallback)
        if "ai_response" in raw_df.columns:
            logger.info("Step 2/4 — Real AI responses already loaded. Skipping generation.")
            response_df = raw_df.copy()
            response_df.to_csv("data/raw_dataset.csv", index=False)
        else:
            logger.info("Step 2/4 — Generating template responses (fallback)...")
            response_df = generate_responses(
                questions  = raw_df["question"].tolist(),
                use_openai = args.openai,
                save_path  = "data/raw_dataset.csv",
            )

        logger.info("Step 3/4 — Applying weak supervision labels...")
        labeled_df = label_dataset(response_df, verbose=True)
        print_label_stats(labeled_df)

        # ── Balance: ensure no_issue class has enough samples ─────────────────
        # Use MedQuAD reference answers as "complete" responses → they get no_issue
        from src.detection.weak_labeler import detect_label
        no_issue_count = (labeled_df['label'] == 'no_issue').sum()
        target_no_issue = int(len(labeled_df) * 0.20)  # aim for 20%

        if no_issue_count < target_no_issue:
            logger.info(f"no_issue samples: {no_issue_count} — adding reference answers to balance...")
            raw_path = "data/raw_questions.csv"
            if os.path.exists(raw_path):
                raw_df = pd.read_csv(raw_path)
                if 'reference_answer' in raw_df.columns:
                    # Use reference answers — these are complete, expert responses
                    ref_df = raw_df[['question','reference_answer']].dropna()
                    ref_df = ref_df.rename(columns={'reference_answer':'ai_response'})
                    ref_df['prompt_type'] = 'reference'
                    # Only keep ones that actually get no_issue label
                    ref_df['label'] = ref_df.apply(
                        lambda r: detect_label(r['question'], r['ai_response']), axis=1
                    )
                    ref_no_issue = ref_df[ref_df['label'] == 'no_issue']
                    need = target_no_issue - no_issue_count
                    to_add = ref_no_issue.head(need)
                    if len(to_add) > 0:
                        from src.knowledge.medical_keywords import LABEL_TO_ID
                        to_add = to_add.copy()
                        to_add['label_id'] = to_add['label'].map(LABEL_TO_ID)
                        labeled_df = pd.concat([labeled_df, to_add], ignore_index=True)
                        labeled_df = labeled_df.sample(frac=1, random_state=42).reset_index(drop=True)
                        logger.info(f"Added {len(to_add)} no_issue reference samples")

        print_label_stats(labeled_df)
        labeled_df.to_csv(labeled_path, index=False)
        logger.info(f"Saved labeled dataset: {labeled_path}")
    else:
        logger.info(f"Skipping data step. Loading: {labeled_path}")
        if not os.path.exists(labeled_path):
            logger.error(f"File not found: {labeled_path}. Run without --skip_data first.")
            sys.exit(1)

    # ════════════════════════════════════════════════════════
    # STEP 2: Train
    # ════════════════════════════════════════════════════════
    checkpoint = f"experiments/best_{tag}.pt"

    if not args.skip_train:
        from src.models.bert_classifier import load_model
        from src.models.dataset import split_dataset
        from src.models.train import train

        logger.info(f"Step 4/4 — Training {args.model} classifier...")
        df = pd.read_csv(labeled_path)

        train_df, val_df, test_df = split_dataset(df, train_ratio=0.80, val_ratio=0.10)
        test_df.to_csv("data/test_split.csv", index=False)  # save for later use

        model, tokenizer = load_model(model_hf_name, num_labels=5)

        trained_model, history = train(
            model      = model,
            tokenizer  = tokenizer,
            train_df   = train_df,
            val_df     = val_df,
            epochs     = args.epochs,
            batch_size = args.batch_size,
            lr         = args.lr,
            max_len    = args.max_len,
            save_dir   = "experiments",
            model_tag  = tag,
            device     = args.device,
        )

        print("\nTraining history (macro-F1 per epoch):")
        for ep, f1 in enumerate(history["val_macro_f1"], 1):
            bar = "█" * int(f1 * 40)
            print(f"  Epoch {ep}: {f1:.4f}  {bar}")
    else:
        logger.info("Skipping training.")

    # ════════════════════════════════════════════════════════
    # STEP 3: Evaluate
    # ════════════════════════════════════════════════════════
    logger.info("Running evaluation...")
    test_csv = "data/test_split.csv" if os.path.exists("data/test_split.csv") else labeled_path

    from evaluation.evaluate_all import run_evaluation
    run_evaluation(
        test_csv        = test_csv,
        checkpoint_bert = checkpoint if tag == "bert" else "experiments/best_bert.pt",
        checkpoint_bio  = checkpoint if tag == "biobert" else "experiments/best_biobert.pt",
        device          = args.device,
        test_size       = 500,
        skip_ml         = args.eval_only_rules,
    )

    print("\n✓ Pipeline complete.")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Evaluation : evaluation/results_summary.json")
    print(f"  Charts     : evaluation/comparison.png\n")


if __name__ == "__main__":
    main()
