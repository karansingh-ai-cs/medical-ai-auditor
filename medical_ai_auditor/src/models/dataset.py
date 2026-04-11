"""
dataset.py
----------
PyTorch Dataset for (question, ai_response) → safety label classification.
Input format: [CLS] question [SEP] ai_response [SEP]
"""

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MedicalAuditDataset(Dataset):
    """
    Dataset for medical AI response auditing.

    Each item is a tokenized pair of (question, ai_response)
    formatted for BERT-style models.
    """

    def __init__(
        self,
        questions:  list[str],
        responses:  list[str],
        labels:     list[int],
        tokenizer,
        max_len:    int = 512,
        augment:    bool = False,
    ):
        assert len(questions) == len(responses) == len(labels), \
            "questions, responses, and labels must have the same length"

        self.questions  = questions
        self.responses  = responses
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_len    = max_len
        self.augment    = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        question = self.questions[idx]
        response = self.responses[idx]
        label    = self.labels[idx]

        # Optional simple augmentation: randomly truncate response
        if self.augment and torch.rand(1).item() > 0.8:
            words    = response.split()
            cutoff   = max(5, int(len(words) * 0.7))
            response = " ".join(words[:cutoff])

        encoding = self.tokenizer(
            question,
            response,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )

        # token_type_ids: 0 for question tokens, 1 for response tokens
        token_type_ids = encoding.get(
            "token_type_ids",
            torch.zeros(self.max_len, dtype=torch.long)
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),       # (max_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (max_len,)
            "token_type_ids": token_type_ids.squeeze(0),              # (max_len,)
            "labels":         torch.tensor(label, dtype=torch.long),
        }

    @property
    def label_counts(self) -> dict:
        """Return count of each label in the dataset."""
        from collections import Counter
        return dict(Counter(self.labels))


def build_weighted_sampler(labels: list[int], num_classes: int) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler to handle class imbalance.
    Each class is sampled proportionally to 1/class_count.
    """
    from collections import Counter
    counts       = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[lbl] for lbl in labels]
    return WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )


def split_dataset(
    df,
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
    seed:        int   = 42,
) -> tuple:
    """
    Stratified split into train / val / test DataFrames.
    Returns (train_df, val_df, test_df).
    """
    from sklearn.model_selection import train_test_split

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, train_size=train_ratio,
        stratify=df["label_id"], random_state=seed
    )
    # Second split: val vs test
    val_ratio_adj = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_ratio_adj,
        stratify=temp_df["label_id"], random_state=seed
    )

    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_dataset(df, tokenizer, max_len: int = 512, augment: bool = False) -> MedicalAuditDataset:
    """Convenience function: DataFrame → MedicalAuditDataset."""
    return MedicalAuditDataset(
        questions = df["question"].tolist(),
        responses = df["ai_response"].tolist(),
        labels    = df["label_id"].tolist(),
        tokenizer = tokenizer,
        max_len   = max_len,
        augment   = augment,
    )
