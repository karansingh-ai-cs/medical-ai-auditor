"""
train.py
--------
Full training pipeline for the Medical AI Auditor classifier.

Features:
  - Weighted loss for class imbalance
  - Linear LR warmup + decay
  - Early stopping on validation Macro-F1
  - Gradient clipping
  - Per-epoch metrics logging
  - Best model checkpoint saving
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import (
    classification_report, accuracy_score,
    f1_score, precision_score, recall_score,
)
from collections import Counter

logger = logging.getLogger(__name__)

LABEL_NAMES = [
    "no_issue", "incomplete_reasoning", "missing_causal_link",
    "dangerous_omission", "critical_failure"
]


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(labels: list[int], num_classes: int, device: str) -> torch.Tensor:
    """
    Compute inverse-frequency class weights to handle label imbalance.
    Passed to CrossEntropyLoss as the `weight` parameter.
    """
    counts = Counter(labels)
    total  = len(labels)
    weights = torch.zeros(num_classes)
    for cls in range(num_classes):
        count = counts.get(cls, 1)
        weights[cls] = total / (num_classes * count)
    logger.info(f"Class weights: {weights.tolist()}")
    return weights.to(device)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, criterion, device, max_grad_norm=1.0):
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids   = batch["input_ids"].to(device)
        attn_mask   = batch["attention_mask"].to(device)
        token_types = batch["token_type_ids"].to(device)
        labels      = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_types,
        )
        loss = criterion(outputs.logits, labels)
        loss.backward()

        # Clip gradients to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch(model, loader, device) -> dict:
    """Run evaluation. Returns dict with accuracy, f1, precision, recall, and per-class report."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attn_mask)
            preds   = outputs.logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(
        all_labels, all_preds,
        labels=list(range(5)),
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy":  accuracy_score(all_labels, all_preds),
        "macro_f1":  f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "macro_p":   precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "macro_r":   recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "report":    report,
        "preds":     all_preds,
        "labels":    all_labels,
    }


# ── Main train function ───────────────────────────────────────────────────────

def train(
    model,
    tokenizer,
    train_df,
    val_df,
    # Hyperparameters
    epochs:          int   = 4,
    batch_size:      int   = 16,
    lr:              float = 2e-5,
    weight_decay:    float = 0.01,
    warmup_ratio:    float = 0.10,
    max_len:         int   = 512,
    use_class_weights: bool = True,
    use_sampler:     bool  = False,
    # Early stopping
    patience:        int   = 2,
    # Output
    save_dir:        str   = "experiments",
    model_tag:       str   = "bert",
    device:          str   = "cpu",
) -> tuple:
    """
    Full training pipeline.

    Args:
        model:        Pre-loaded BERT/BioBERT model.
        tokenizer:    Matching tokenizer.
        train_df:     Training DataFrame (columns: question, ai_response, label_id).
        val_df:       Validation DataFrame.
        ...           (see hyperparameter args above)

    Returns:
        (trained_model, training_history_dict)
    """
    from src.models.dataset import make_dataset, build_weighted_sampler

    os.makedirs(save_dir, exist_ok=True)
    model.to(device)

    # ── Datasets & Loaders ────────────────────────────────────────────────────
    train_ds = make_dataset(train_df, tokenizer, max_len=max_len, augment=True)
    val_ds   = make_dataset(val_df,   tokenizer, max_len=max_len, augment=False)

    if use_sampler:
        sampler      = build_weighted_sampler(train_df["label_id"].tolist(), 5)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size * 2)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer    = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Loss function ─────────────────────────────────────────────────────────
    if use_class_weights:
        weights   = compute_class_weights(train_df["label_id"].tolist(), 5, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_f1     = 0.0
    patience_ctr = 0
    history     = {"train_loss": [], "val_accuracy": [], "val_macro_f1": []}
    checkpoint  = os.path.join(save_dir, f"best_{model_tag}.pt")

    print(f"\n{'─'*60}")
    print(f"  Training: {model_tag} | {epochs} epochs | device={device}")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"{'─'*60}")

    for epoch in range(1, epochs + 1):
        # Train
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)

        # Validate
        metrics  = eval_epoch(model, val_loader, device)
        val_f1   = metrics["macro_f1"]
        val_acc  = metrics["accuracy"]

        history["train_loss"].append(round(avg_loss, 4))
        history["val_accuracy"].append(round(val_acc, 4))
        history["val_macro_f1"].append(round(val_f1, 4))

        print(f"  Epoch {epoch}/{epochs} │ loss={avg_loss:.4f} │ "
              f"acc={val_acc:.4f} │ macro-F1={val_f1:.4f}")

        # Per-class F1
        for label in LABEL_NAMES:
            f1 = metrics["report"][label]["f1-score"]
            print(f"    {label:<28} F1={f1:.3f}")

        # Save best
        if val_f1 > best_f1:
            best_f1     = val_f1
            patience_ctr = 0
            torch.save(model.state_dict(), checkpoint)
            print(f"  ✓ Saved best model (macro-F1={best_f1:.4f}) → {checkpoint}")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping after {epoch} epochs (patience={patience})")
                break

    # Save training history
    hist_path = os.path.join(save_dir, f"history_{model_tag}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest macro-F1: {best_f1:.4f}")
    print(f"Checkpoint:    {checkpoint}")
    print(f"History:       {hist_path}\n")

    # Reload best weights
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model, history
