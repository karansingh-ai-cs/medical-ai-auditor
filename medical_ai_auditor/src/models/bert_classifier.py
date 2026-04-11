"""
bert_classifier.py
------------------
Model factory for BERT and BioBERT sequence classifiers.

Supported model names:
  BERT:     "bert-base-uncased"
  BioBERT:  "dmis-lab/biobert-base-cased-v1.2"
  PubMedBERT: "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
  ClinicalBERT: "emilyalsentzer/Bio_ClinicalBERT"
"""

import torch
import torch.nn as nn
from transformers import (
    BertForSequenceClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
)
import logging

logger = logging.getLogger(__name__)

# ── Supported models ───────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "bert":          "bert-base-uncased",
    "biobert":       "dmis-lab/biobert-base-cased-v1.2",
    "pubmedbert":    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "clinicalbert":  "emilyalsentzer/Bio_ClinicalBERT",
}


def load_model(
    model_name: str = "bert-base-uncased",
    num_labels: int = 5,
    dropout:    float = 0.3,
    freeze_bert_layers: int = 0,
) -> tuple:
    """
    Load a BERT-family model + tokenizer for sequence classification.

    Args:
        model_name:          HuggingFace model ID or short alias from MODEL_REGISTRY.
        num_labels:          Number of output classes (default 5).
        dropout:             Dropout probability (applied to hidden + attention).
        freeze_bert_layers:  Number of BERT encoder layers to freeze (0 = none).
                             Useful when fine-tuning on small datasets.

    Returns:
        (model, tokenizer) tuple.
    """
    # Resolve alias if needed
    resolved_name = MODEL_REGISTRY.get(model_name, model_name)
    logger.info(f"Loading model: {resolved_name}  (num_labels={num_labels})")

    model = AutoModelForSequenceClassification.from_pretrained(
        resolved_name,
        num_labels=num_labels,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        ignore_mismatched_sizes=True,   # handles BioBERT → custom head
    )

    tokenizer = AutoTokenizer.from_pretrained(resolved_name)

    # Optionally freeze early BERT layers (helps regularise on small data)
    if freeze_bert_layers > 0:
        _freeze_encoder_layers(model, freeze_bert_layers)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    return model, tokenizer


def _freeze_encoder_layers(model, num_layers: int):
    """Freeze the first `num_layers` transformer encoder layers."""
    try:
        encoder_layers = model.bert.encoder.layer
        for i in range(min(num_layers, len(encoder_layers))):
            for param in encoder_layers[i].parameters():
                param.requires_grad = False
        logger.info(f"Froze first {num_layers} encoder layers.")
    except AttributeError:
        logger.warning("Could not freeze encoder layers — model structure may differ.")


def load_from_checkpoint(
    checkpoint_path: str,
    model_name: str = "bert-base-uncased",
    num_labels: int = 5,
    device: str = "cpu",
) -> tuple:
    """Load a trained model from a saved .pt checkpoint."""
    model, tokenizer = load_model(model_name, num_labels)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    return model, tokenizer


# ── Inference helpers ─────────────────────────────────────────────────────────

def predict_single(
    model, tokenizer, question: str, response: str,
    max_len: int = 512, device: str = "cpu"
) -> dict:
    """
    Run inference on a single (question, response) pair.
    Returns predicted label name + class probabilities.
    """
    from src.knowledge.medical_keywords import ID_TO_LABEL

    model.eval()
    enc = tokenizer(
        question, response,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(
            enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits

    probs    = torch.softmax(logits, dim=-1).squeeze().cpu()
    pred_idx = probs.argmax().item()

    return {
        "label":      ID_TO_LABEL[pred_idx],
        "label_id":   pred_idx,
        "confidence": round(probs[pred_idx].item(), 4),
        "probabilities": {
            ID_TO_LABEL[i]: round(p.item(), 4)
            for i, p in enumerate(probs)
        },
    }


def predict_batch(
    model, tokenizer, questions: list[str], responses: list[str],
    batch_size: int = 32, max_len: int = 512, device: str = "cpu"
) -> list[dict]:
    """Run batch inference. Returns list of prediction dicts."""
    from src.knowledge.medical_keywords import ID_TO_LABEL
    from tqdm import tqdm

    model.eval()
    results = []

    for start in tqdm(range(0, len(questions), batch_size), desc="Predicting"):
        batch_q = questions[start:start + batch_size]
        batch_r = responses[start:start + batch_size]

        enc = tokenizer(
            batch_q, batch_r,
            max_length=max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        # Move ALL tensors to the correct device
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(
                enc["input_ids"],
                attention_mask=enc["attention_mask"],
            ).logits

        probs   = torch.softmax(logits, dim=-1).cpu()
        indices = probs.argmax(dim=-1)

        for i in range(len(batch_q)):
            results.append({
                "label":      ID_TO_LABEL[indices[i].item()],
                "label_id":   indices[i].item(),
                "confidence": round(probs[i][indices[i]].item(), 4),
            })

    return results
