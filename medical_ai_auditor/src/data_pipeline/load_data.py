"""
load_data.py
------------
Load REAL medical Q&A data from free HuggingFace datasets.

Priority order (best to good):
  1. ChatDoctor        — real patient Qs + ChatGPT responses  ← BEST
  2. HealthCareMagic   — real patient Qs + doctor responses
  3. iCliniq           — real doctor responses
  4. MedQuAD           — NIH Q&A (fallback)

Why ChatDoctor is best for this project:
  The 'output' field IS a real ChatGPT response to a real patient question.
  Some are complete and safe → no_issue
  Some miss critical warnings → dangerous_omission / critical_failure
  This is exactly the distribution we want to audit.
"""

import os
import logging
import pandas as pd
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Individual loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_chatdoctor(max_samples: int = 3000) -> pd.DataFrame:
    """
    lavita/ChatDoctor-HealthCareMagic-100k
    input  = real patient question
    output = ChatGPT generated response   <-- this is what we audit
    """
    from datasets import load_dataset
    logger.info(f"Loading ChatDoctor (n={max_samples})...")
    ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
    df = pd.DataFrame(ds).rename(columns={"input": "question", "output": "ai_response"})
    df = _clean(df, max_samples, "chatdoctor")
    logger.info(f"  ChatDoctor: {len(df)} rows")
    return df


def load_healthcaremagic(max_samples: int = 1000) -> pd.DataFrame:
    """
    medalpaca/medical_meadow_healthcaremagic
    Real doctor responses — tend to be complete → mostly no_issue labels.
    Good for balancing the dataset.
    """
    from datasets import load_dataset
    logger.info(f"Loading HealthCareMagic (n={max_samples})...")
    try:
        ds = load_dataset("medalpaca/medical_meadow_healthcaremagic", split="train")
        df = pd.DataFrame(ds)

        # The 'input' field has an instruction prefix — strip it
        def strip_prefix(text: str) -> str:
            markers = ["patient:", "question:", "###input:", "input:"]
            tl = text.lower()
            for m in markers:
                idx = tl.find(m)
                if idx != -1:
                    return text[idx + len(m):].strip()
            # fallback: take last sentence-like chunk after newline
            parts = text.split("\n")
            return parts[-1].strip() if parts else text[:300]

        df["question"]    = df["input"].apply(strip_prefix)
        df["ai_response"] = df["output"].str.strip()
        df = _clean(df, max_samples, "healthcaremagic")
        logger.info(f"  HealthCareMagic: {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"  HealthCareMagic failed ({e}) — skipping")
        return pd.DataFrame(columns=["question", "ai_response", "source"])


def load_icliniq(max_samples: int = 500) -> pd.DataFrame:
    """
    medalpaca/medical_meadow_icliniq
    Real doctor Q&A — high quality complete answers.
    """
    from datasets import load_dataset
    logger.info(f"Loading iCliniq (n={max_samples})...")
    try:
        ds = load_dataset("medalpaca/medical_meadow_icliniq", split="train")
        df = pd.DataFrame(ds).rename(columns={"input": "question", "output": "ai_response"})
        df = _clean(df, max_samples, "icliniq")
        logger.info(f"  iCliniq: {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"  iCliniq failed ({e}) — skipping")
        return pd.DataFrame(columns=["question", "ai_response", "source"])


def load_medquad(max_samples: int = 1000) -> pd.DataFrame:
    """
    keivalya/MedQuad-MedicalQnADataset
    NIH reference Q&A. Use reference answer as the 'ai_response'.
    Expert answers → mostly no_issue labels → helps balance the dataset.
    """
    from datasets import load_dataset
    logger.info(f"Loading MedQuAD (n={max_samples})...")
    try:
        ds = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
        df = pd.DataFrame(ds)
        rename = {}
        for col in df.columns:
            if "question" in col.lower(): rename[col] = "question"
            elif "answer"  in col.lower(): rename[col] = "ai_response"
        df = df.rename(columns=rename)
        df = _clean(df, max_samples, "medquad_reference")
        logger.info(f"  MedQuAD: {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"  MedQuAD failed ({e}) — skipping")
        return pd.DataFrame(columns=["question", "ai_response", "source"])


def _clean(df: pd.DataFrame, max_samples: int, source_name: str) -> pd.DataFrame:
    """Standard cleaning applied to every loader."""
    df = df[["question", "ai_response"]].dropna().copy()
    df["question"]    = df["question"].str.strip()
    df["ai_response"] = df["ai_response"].str.strip()
    df = df[df["question"].str.len()    > 20]
    df = df[df["ai_response"].str.len() > 40]
    df = df[df["question"].str.len()    < 1500]
    df["source"] = source_name
    return df.head(max_samples).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Combined builder
# ─────────────────────────────────────────────────────────────────────────────

def build_real_dataset(
    chatdoctor_n:      int = 3000,
    healthcaremagic_n: int = 1000,
    icliniq_n:         int = 500,
    medquad_n:         int = 500,
    save_path:         Optional[str] = None,
) -> pd.DataFrame:
    """
    Build the combined dataset from all real sources.

    Mix rationale:
      ChatDoctor (3000)      — real ChatGPT responses, varied quality → all 5 labels
      HealthCareMagic (1000) — real doctor responses → mostly no_issue
      iCliniq (500)          — real doctor responses → mostly no_issue
      MedQuAD (500)          — NIH expert answers   → mostly no_issue

    The doctor/reference responses balance the no_issue class
    without any synthetic generation needed.
    """
    frames = []
    for loader, n in [
        (load_chatdoctor,      chatdoctor_n),
        (load_healthcaremagic, healthcaremagic_n),
        (load_icliniq,         icliniq_n),
        (load_medquad,         medquad_n),
    ]:
        if n > 0:
            try:
                df = loader(n)
                if len(df) > 0:
                    frames.append(df)
            except Exception as e:
                logger.warning(f"Loader failed: {e}")

    if not frames:
        raise RuntimeError("All data sources failed. Check your internet connection.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="question")
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Summary
    logger.info(f"\n{'─'*45}")
    logger.info(f"  Combined dataset: {len(combined)} samples")
    for src, cnt in combined["source"].value_counts().items():
        bar = "█" * int(20 * cnt / len(combined))
        logger.info(f"  {src:<28} {cnt:>5}  {bar}")
    logger.info(f"{'─'*45}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        combined.to_csv(save_path, index=False)
        logger.info(f"Saved: {save_path}")

    return combined


# Backward-compat wrapper used by run_pipeline.py
def build_combined_dataset(
    use_real_data:     bool = True,
    medquad_max:       int  = 5000,
    save_path:         Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    if use_real_data:
        try:
            return build_real_dataset(
                chatdoctor_n=3000,
                healthcaremagic_n=800,
                icliniq_n=400,
                medquad_n=400,
                save_path=save_path,
            )
        except Exception as e:
            logger.warning(f"Real data failed ({e}), falling back to MedQuAD only")

    df = load_medquad(max_samples=medquad_max)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = build_real_dataset(
        chatdoctor_n=200, healthcaremagic_n=100,
        icliniq_n=50, medquad_n=50,
        save_path="data/raw_questions.csv",
    )
    print(f"\nSample:\n{df[['question','ai_response','source']].head(3).to_string()}")
