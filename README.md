# Medical AI Reliability Auditor

> Detecting missing critical reasoning in AI-generated medical responses.

---

## What this project does

When a patient asks an AI "I fell and my wrist hurts — what should I do?", the AI might say "apply ice and rest" — technically not wrong, but **dangerously incomplete** (no mention of fracture risk, no referral for X-ray).

This system **audits** AI-generated medical responses and classifies them into:

| Label | Meaning |
|-------|---------|
| `no_issue` | Response is complete and safe |
| `incomplete_reasoning` | Too short, vague, or missing steps |
| `missing_causal_link` | Facts given but no reasoning connecting them |
| `dangerous_omission` | Missing a critical warning or risk |
| `critical_failure` | Could cause direct harm — emergency not mentioned |

---

## Architecture

```
Medical Query
    ↓
LLM Response (generated)
    ↓
┌─────────────────────────────────────┐
│           Auditor System            │
│  ┌──────────────────────────────┐   │
│  │   Rule-Based Heuristics (7)  │   │
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │   BERT / BioBERT Classifier  │   │
│  │  [CLS] question [SEP] resp   │   │
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │   Hybrid Override Logic      │   │
│  │   (rule > ML for critical)   │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
    ↓
Output Label
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run full pipeline (no API key needed — uses template responses)

```bash
python experiments/run_pipeline.py
```

This will:
- Download MedQuAD from HuggingFace (~5 min first time)
- Generate template-based AI responses (instant, no key needed)
- Apply weak supervision labels
- Train BERT classifier (4 epochs)
- Run evaluation on all 3 systems

### 3. Run with OpenAI (better quality responses)

```bash
export OPENAI_API_KEY=sk-your-key-here
python experiments/run_pipeline.py --openai
```

### 4. Train BioBERT (recommended for better results)

```bash
python experiments/run_pipeline.py --model biobert
```

### 5. Evaluate rules only (no GPU, fast test)

```bash
python experiments/run_pipeline.py --skip_train --eval_only_rules
```

---

## Project Structure

```
medical_ai_auditor/
├── data/
│   ├── raw_questions.csv         ← loaded from MedQuAD
│   ├── raw_dataset.csv           ← questions + AI responses
│   ├── labeled_dataset.csv       ← with weak supervision labels
│   └── test_split.csv            ← held-out test set
│
├── src/
│   ├── knowledge/
│   │   └── medical_keywords.py   ← all keyword lists
│   ├── data_pipeline/
│   │   ├── load_data.py          ← MedQuAD + PubMed loaders
│   │   └── generate_responses.py ← LLM response generation
│   ├── detection/
│   │   ├── weak_labeler.py       ← 7 labeling functions
│   │   └── hybrid_auditor.py     ← full hybrid system
│   └── models/
│       ├── bert_classifier.py    ← model factory (BERT/BioBERT)
│       ├── dataset.py            ← PyTorch Dataset
│       └── train.py              ← training loop
│
├── evaluation/
│   └── evaluate_all.py           ← compare all 3 systems
│
└── experiments/
    └── run_pipeline.py           ← master entry point
```

---

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `bert` | `bert`, `biobert`, `clinicalbert`, `pubmedbert` |
| `--epochs` | `4` | Training epochs |
| `--batch_size` | `16` | Batch size (reduce to 8 if OOM on GPU) |
| `--lr` | `2e-5` | Learning rate |
| `--max_samples` | `5000` | How many MedQuAD questions to use |
| `--device` | auto | `cpu` or `cuda` |
| `--openai` | False | Use OpenAI API for response generation |
| `--skip_data` | False | Skip data step, use existing CSV |
| `--skip_train` | False | Skip training, evaluate existing checkpoint |

---

## Using the Auditor as a library

```python
from src.detection.hybrid_auditor import HybridAuditor, RuleBasedAuditor

# Quick rule-based check (no model needed)
auditor = RuleBasedAuditor()
result = auditor.audit(
    question="I fell down the stairs and my wrist hurts a lot.",
    response="Apply ice and rest for a few days."
)
print(result["final_label"])  # → dangerous_omission

# Full hybrid auditor (after training)
hybrid = HybridAuditor.from_checkpoint(
    checkpoint_path="experiments/best_bert.pt",
    model_name="bert-base-uncased",
)
result = hybrid.audit(question, response)
print(result["final_label"])
print(hybrid.explain(question, response))
```

---

## Expected Results (NOT REAL IT'S EXPECTED)

| System | Accuracy | Macro-F1 |
|--------|----------|----------|
| Rule-Based | ~0.62 | ~0.55 |
| ML-Only (BERT) | ~0.74 | ~0.70 |
| ML-Only (BioBERT) | ~0.79 | ~0.76 |
| Hybrid (BERT) | ~0.77 | ~0.73 |
| **Hybrid (BioBERT)** | **~0.82** | **~0.79** |

---

## Citation

If you use this project in academic work:

```
@misc{medical_ai_auditor_,
  title  = {Medical AI Reliability Auditor: Detecting Missing Critical Reasoning
            in AI-Generated Medical Responses},
  author = {[KARAN SINGH]},
  year   = {2026},
}
```

Key references: BERT [Devlin et al., 2019], BioBERT [Lee et al., 2020],
MedQuAD [Ben Abacha & Demner-Fushman, 2019], Snorkel [Ratner et al., 2017].

---

## Limitations & Future Work

- Weak supervision labels are heuristic — not clinically validated
- The model audits completeness, not factual correctness
- Future: integrate SNOMED-CT ontology for richer clinical reasoning checks
- Future: real-time API wrapper for medical chatbot deployment
