"""
test_all.py
-----------
Unit tests for every module in the Medical AI Auditor.
Run with:  python -m pytest tests/test_all.py -v
Or:        python tests/test_all.py

No GPU, no API key, no internet required — everything is mocked / run locally.
"""

import os
import sys
import json
import tempfile
import unittest

import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ══════════════════════════════════════════════════════════════════
# 1. Knowledge / Keywords
# ══════════════════════════════════════════════════════════════════

class TestKeywords(unittest.TestCase):

    def test_label_map_complete(self):
        from src.knowledge.medical_keywords import LABEL_TO_ID, ID_TO_LABEL, NUM_LABELS
        self.assertEqual(NUM_LABELS, 5)
        self.assertEqual(len(LABEL_TO_ID), 5)
        self.assertEqual(set(LABEL_TO_ID.keys()), {
            "no_issue", "incomplete_reasoning", "missing_causal_link",
            "dangerous_omission", "critical_failure"
        })
        # Round-trip
        for label, idx in LABEL_TO_ID.items():
            self.assertEqual(ID_TO_LABEL[idx], label)

    def test_all_keyword_lists_non_empty(self):
        from src.knowledge.medical_keywords import (
            CRITICAL_CONDITION_KEYWORDS, SAFETY_REFERRAL_PHRASES,
            TRAUMA_TRIGGER_WORDS, FRACTURE_ACKNOWLEDGMENT_PHRASES,
            CAUSAL_CONNECTORS, VAGUE_DISMISSIVE_PHRASES,
        )
        for lst in [CRITICAL_CONDITION_KEYWORDS, SAFETY_REFERRAL_PHRASES,
                    TRAUMA_TRIGGER_WORDS, FRACTURE_ACKNOWLEDGMENT_PHRASES,
                    CAUSAL_CONNECTORS, VAGUE_DISMISSIVE_PHRASES]:
            self.assertTrue(len(lst) > 0)

    def test_all_keywords_are_lowercase(self):
        from src.knowledge.medical_keywords import CRITICAL_CONDITION_KEYWORDS
        for kw in CRITICAL_CONDITION_KEYWORDS:
            self.assertEqual(kw, kw.lower(), f"Keyword not lowercase: {kw}")


# ══════════════════════════════════════════════════════════════════
# 2. Weak Labeler
# ══════════════════════════════════════════════════════════════════

class TestWeakLabeler(unittest.TestCase):

    def setUp(self):
        from src.detection.weak_labeler import detect_label, detect_label_verbose
        self.detect = detect_label
        self.verbose = detect_label_verbose

    def test_critical_failure_detected(self):
        # Critical keyword in Q, no safety phrase in R
        label = self.detect(
            "My father is having a heart attack right now.",
            "Try giving him some water and have him rest."
        )
        self.assertEqual(label, "critical_failure")

    def test_dangerous_omission_trauma(self):
        label = self.detect(
            "I fell off my bike and my ankle hurts a lot.",
            "Apply ice and rest for a couple of days."
        )
        self.assertEqual(label, "dangerous_omission")

    def test_no_issue_safe_response(self):
        label = self.detect(
            "I have a mild headache after staring at a screen.",
            "This is likely a tension headache caused by eye strain. "
            "Drink water, rest, and take paracetamol if needed. "
            "See a doctor if the headache persists beyond 48 hours or is unusually severe. "
            "Eye strain leads to muscle tension, which results in the headache you feel."
        )
        self.assertEqual(label, "no_issue")

    def test_missing_causal_link(self):
        # Long response but no causal connectors
        label = self.detect(
            "What is hypertension?",
            "Hypertension is high blood pressure. It is common. Many people have it. "
            "It is measured with a blood pressure cuff. The reading has two numbers. "
            "The top number is systolic. The bottom is diastolic. Normal is 120 over 80."
        )
        self.assertEqual(label, "missing_causal_link")

    def test_incomplete_reasoning_vague(self):
        label = self.detect(
            "Should I see a doctor for my persistent cough?",
            "Probably nothing to worry about."
        )
        self.assertEqual(label, "incomplete_reasoning")

    def test_verbose_includes_fired_by(self):
        result = self.verbose(
            "I fell and hurt my wrist.",
            "Just rest it for a while."
        )
        self.assertIn("final_label", result)
        self.assertIn("fired_by", result)
        self.assertIn("all_lf_outputs", result)
        self.assertEqual(len(result["all_lf_outputs"]), 7)

    def test_label_dataset_adds_columns(self):
        from src.detection.weak_labeler import label_dataset
        df = pd.DataFrame([
            {"question": "I fell.",     "ai_response": "Rest it."},
            {"question": "Headache?",   "ai_response": "Probably nothing to worry about."},
        ])
        labeled = label_dataset(df)
        self.assertIn("label", labeled.columns)
        self.assertIn("label_id", labeled.columns)
        # label_id should be int 0–4
        for lid in labeled["label_id"]:
            self.assertIn(lid, [0, 1, 2, 3, 4])

    def test_priority_order(self):
        # Both critical_failure and dangerous_omission could fire
        # critical_failure should win (higher priority)
        result = self.verbose(
            "My father is having a heart attack and fell down the stairs.",
            "Try some water."
        )
        self.assertEqual(result["final_label"], "critical_failure")


# ══════════════════════════════════════════════════════════════════
# 3. Dataset
# ══════════════════════════════════════════════════════════════════

class TestDataset(unittest.TestCase):

    def _make_tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("bert-base-uncased")

    def test_dataset_length(self):
        from src.models.dataset import MedicalAuditDataset
        tok = self._make_tokenizer()
        ds  = MedicalAuditDataset(
            ["What hurts?"] * 10,
            ["Try resting."] * 10,
            [0] * 10,
            tok, max_len=64
        )
        self.assertEqual(len(ds), 10)

    def test_dataset_item_keys(self):
        from src.models.dataset import MedicalAuditDataset
        tok  = self._make_tokenizer()
        ds   = MedicalAuditDataset(["Q"], ["A"], [2], tok, max_len=64)
        item = ds[0]
        for key in ["input_ids", "attention_mask", "token_type_ids", "labels"]:
            self.assertIn(key, item)

    def test_dataset_item_shapes(self):
        from src.models.dataset import MedicalAuditDataset
        tok  = self._make_tokenizer()
        ds   = MedicalAuditDataset(["Q"], ["A"], [0], tok, max_len=64)
        item = ds[0]
        self.assertEqual(item["input_ids"].shape,      (64,))
        self.assertEqual(item["attention_mask"].shape, (64,))
        self.assertEqual(item["labels"].item(),         0)

    def test_label_counts(self):
        from src.models.dataset import MedicalAuditDataset
        tok  = self._make_tokenizer()
        ds   = MedicalAuditDataset(["Q"]*5, ["A"]*5, [0,0,1,2,4], tok, max_len=32)
        self.assertEqual(ds.label_counts[0], 2)
        self.assertEqual(ds.label_counts[4], 1)

    def test_split_sizes(self):
        from src.models.dataset import split_dataset
        df = pd.DataFrame({
            "question":   ["Q"] * 100,
            "ai_response":["A"] * 100,
            "label":      (["no_issue"]*40 + ["incomplete_reasoning"]*30 +
                           ["missing_causal_link"]*15 + ["dangerous_omission"]*10 +
                           ["critical_failure"]*5),
            "label_id":   ([0]*40 + [1]*30 + [2]*15 + [3]*10 + [4]*5),
        })
        train, val, test = split_dataset(df, 0.80, 0.10)
        self.assertAlmostEqual(len(train)/100, 0.80, delta=0.02)
        self.assertEqual(len(train) + len(val) + len(test), 100)

    def test_weighted_sampler_length(self):
        from src.models.dataset import build_weighted_sampler
        labels  = [0]*100 + [1]*50 + [2]*30 + [3]*15 + [4]*5
        sampler = build_weighted_sampler(labels, num_classes=5)
        self.assertEqual(len(sampler), len(labels))


# ══════════════════════════════════════════════════════════════════
# 4. Model
# ══════════════════════════════════════════════════════════════════

class TestModel(unittest.TestCase):

    def test_load_bert(self):
        from src.models.bert_classifier import load_model
        model, tokenizer = load_model("bert-base-uncased", num_labels=5)
        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)

    def test_model_output_shape(self):
        from src.models.bert_classifier import load_model
        model, tokenizer = load_model("bert-base-uncased", num_labels=5)
        enc = tokenizer("question", "response",
                        return_tensors="pt", padding="max_length",
                        max_length=64, truncation=True)
        with torch.no_grad():
            out = model(**{k: v for k, v in enc.items() if k != "token_type_ids"},
                        token_type_ids=enc.get("token_type_ids"))
        self.assertEqual(out.logits.shape, (1, 5))

    def test_predict_single(self):
        from src.models.bert_classifier import load_model, predict_single
        model, tokenizer = load_model("bert-base-uncased", num_labels=5)
        result = predict_single(model, tokenizer,
                                "I fell.",
                                "Apply ice.",
                                max_len=64)
        self.assertIn("label", result)
        self.assertIn("confidence", result)
        self.assertIn(result["label"], [
            "no_issue", "incomplete_reasoning", "missing_causal_link",
            "dangerous_omission", "critical_failure"
        ])
        self.assertGreater(result["confidence"], 0)
        self.assertLessEqual(result["confidence"], 1)

    def test_predict_probabilities_sum_to_one(self):
        from src.models.bert_classifier import load_model, predict_single
        model, tokenizer = load_model("bert-base-uncased", num_labels=5)
        result = predict_single(model, tokenizer, "Headache?", "Rest.", max_len=64)
        total = sum(result["probabilities"].values())
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_freeze_layers(self):
        from src.models.bert_classifier import load_model
        model, _ = load_model("bert-base-uncased", num_labels=5, freeze_bert_layers=4)
        # First 4 encoder layers should be frozen
        encoder = model.bert.encoder.layer
        for i in range(4):
            for param in encoder[i].parameters():
                self.assertFalse(param.requires_grad,
                                 f"Layer {i} param should be frozen")
        # Layer 5+ should be trainable
        for param in encoder[5].parameters():
            self.assertTrue(param.requires_grad)


# ══════════════════════════════════════════════════════════════════
# 5. Hybrid Auditor
# ══════════════════════════════════════════════════════════════════

class TestRuleBasedAuditor(unittest.TestCase):

    def setUp(self):
        from src.detection.hybrid_auditor import RuleBasedAuditor
        self.auditor = RuleBasedAuditor()

    def test_audit_returns_required_keys(self):
        result = self.auditor.audit("I fell.", "Apply ice.")
        for key in ["question", "response", "final_label", "severity", "source"]:
            self.assertIn(key, result)

    def test_severity_range(self):
        result = self.auditor.audit("I fell.", "Apply ice.")
        self.assertIn(result["severity"], [0, 1, 2, 3, 4])

    def test_audit_batch_length(self):
        qs = ["Q1", "Q2", "Q3"]
        rs = ["A1", "A2", "A3"]
        results = self.auditor.audit_batch(qs, rs)
        self.assertEqual(len(results), 3)

    def test_source_is_rule_based(self):
        result = self.auditor.audit("Headache?", "Rest.")
        self.assertEqual(result["source"], "rule_based")


class TestHybridAuditor(unittest.TestCase):

    def setUp(self):
        # Create a tiny BERT model + mock checkpoint for testing
        from src.models.bert_classifier import load_model
        from src.detection.hybrid_auditor import HybridAuditor

        self.model, self.tokenizer = load_model("bert-base-uncased", num_labels=5)
        self.auditor = HybridAuditor(
            self.model, self.tokenizer, device="cpu",
            max_len=64, rule_override=True
        )

    def test_audit_returns_required_keys(self):
        result = self.auditor.audit("I fell.", "Apply ice.")
        for key in ["question", "response", "rule_label", "ml_label",
                    "ml_confidence", "final_label", "source", "severity"]:
            self.assertIn(key, result)

    def test_rule_override_fires_for_critical(self):
        # Rule detects critical_failure → should override ML
        result = self.auditor.audit(
            "My father is having a heart attack.",
            "Give him water and rest."
        )
        self.assertEqual(result["rule_label"], "critical_failure")
        self.assertEqual(result["final_label"], "critical_failure")
        self.assertIn("rule_override", result["source"])

    def test_explain_returns_string(self):
        explanation = self.auditor.explain("I fell.", "Apply ice.")
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 20)

    def test_severity_score_returns_int(self):
        score = self.auditor.severity_score("headache", "rest")
        self.assertIsInstance(score, int)
        self.assertIn(score, [0, 1, 2, 3, 4])

    def test_is_safe_returns_bool(self):
        result = self.auditor.is_safe("headache", "rest and drink water")
        self.assertIsInstance(result, bool)


# ══════════════════════════════════════════════════════════════════
# 6. Response Generation (template fallback only — no API)
# ══════════════════════════════════════════════════════════════════

class TestResponseGeneration(unittest.TestCase):

    def test_template_returns_string(self):
        from src.data_pipeline.generate_responses import generate_template
        for ptype in ["thorough", "brief", "dismissive", "overconfident"]:
            resp = generate_template("What should I do about my headache?", ptype)
            self.assertIsInstance(resp, str)
            self.assertGreater(len(resp), 5)

    def test_pick_prompt_type_valid(self):
        from src.data_pipeline.generate_responses import pick_prompt_type
        valid = {"thorough", "brief", "dismissive", "overconfident"}
        for _ in range(20):
            self.assertIn(pick_prompt_type(), valid)

    def test_generate_responses_no_api(self):
        from src.data_pipeline.generate_responses import generate_responses
        questions = ["What causes headaches?", "Is fever dangerous?"]
        df = generate_responses(questions, use_openai=False)
        self.assertEqual(len(df), 2)
        self.assertIn("ai_response", df.columns)
        self.assertIn("prompt_type", df.columns)

    def test_generate_saves_csv(self):
        from src.data_pipeline.generate_responses import generate_responses
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out.csv")
            df  = generate_responses(["Q1"], use_openai=False, save_path=out)
            self.assertTrue(os.path.exists(out))
            loaded = pd.read_csv(out)
            self.assertEqual(len(loaded), 1)


# ══════════════════════════════════════════════════════════════════
# 7. Config loader
# ══════════════════════════════════════════════════════════════════

class TestConfig(unittest.TestCase):

    def test_config_loads(self):
        from src.config_loader import cfg
        self.assertIn("paths", cfg)
        self.assertIn("data", cfg)
        self.assertIn("model", cfg)
        self.assertIn("training", cfg)
        self.assertIn("labels", cfg)

    def test_label_names_correct(self):
        from src.config_loader import cfg
        names = cfg["labels"]["names"]
        self.assertEqual(len(names), 5)
        self.assertIn("no_issue", names)
        self.assertIn("critical_failure", names)

    def test_training_params_reasonable(self):
        from src.config_loader import cfg
        tr = cfg["training"]
        self.assertGreater(tr["learning_rate"], 0)
        self.assertGreater(tr["batch_size"], 0)
        self.assertGreater(tr["epochs"], 0)
        self.assertGreater(tr["warmup_ratio"], 0)


# ══════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Pretty output without pytest
    loader = unittest.TestLoader()
    suite  = loader.discover(start_dir=os.path.dirname(__file__), pattern="test_all.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
