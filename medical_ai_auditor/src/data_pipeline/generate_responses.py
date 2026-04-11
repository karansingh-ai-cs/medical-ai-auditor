"""
generate_responses.py
---------------------
Generate AI responses for medical questions using an LLM API.
Supports OpenAI (gpt-3.5-turbo) and a free fallback using a local template.

Strategy:
  - 40% of prompts use a "thorough" system message  → likely no_issue
  - 35% use a "brief" system message                → likely incomplete
  - 15% use a "dismissive" system message           → likely dangerous_omission
  - 10% use an "overconfident" system message       → likely critical_failure
This seeding helps the weak labeler produce diverse labels.
"""

import os
import time
import json
import random
import logging
from typing import Optional
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── System prompt variants ────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "thorough": (
        "You are a knowledgeable medical AI assistant. "
        "Provide a thorough, safety-conscious answer. "
        "Always mention when to seek emergency care if relevant. "
        "Explain the reasoning behind your advice."
    ),
    "brief": (
        "You are a medical AI assistant. "
        "Give a brief, practical answer in 2-3 sentences. "
        "Keep it simple and direct."
    ),
    "dismissive": (
        "You are a casual health advisor. "
        "Give a short, reassuring answer. "
        "Minimise concern. Don't suggest hospital visits unless absolutely necessary."
    ),
    "overconfident": (
        "You are a medical AI. "
        "Give a confident definitive answer. "
        "Do not hedge or recommend seeing a doctor. "
        "Provide a direct treatment recommendation."
    ),
}

PROMPT_WEIGHTS = {
    "thorough":     0.40,
    "brief":        0.35,
    "dismissive":   0.15,
    "overconfident":0.10,
}


def pick_prompt_type() -> str:
    """Randomly select a prompt type according to distribution weights."""
    types   = list(PROMPT_WEIGHTS.keys())
    weights = list(PROMPT_WEIGHTS.values())
    return random.choices(types, weights=weights, k=1)[0]


# ── OpenAI generation ─────────────────────────────────────────────────────────

def generate_openai(
    question: str,
    prompt_type: str = "brief",
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 250,
    temperature: float = 0.7,
) -> str:
    """
    Generate response using OpenAI API.
    Set OPENAI_API_KEY in environment before calling.
    """
    try:
        import openai
    except ImportError:
        raise ImportError("Run: pip install openai")

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    system_msg = SYSTEM_PROMPTS[prompt_type]

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": question},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


# ── Template-based fallback (no API key needed) ───────────────────────────────

TEMPLATE_RESPONSES = {
    # ── Thorough: long + causal connectors + safety referral ─────────────────
    # These are designed to pass ALL labeling functions → label = no_issue
    "thorough": [
        "This symptom can be serious because it may indicate an underlying condition "
        "that requires medical evaluation. The discomfort you describe often results from "
        "inflammation or injury to surrounding tissue, which leads to pain and swelling. "
        "Consequently, I strongly recommend consulting a healthcare provider or visiting "
        "an emergency room if symptoms are severe or worsening. Early diagnosis leads to "
        "better outcomes. Do not ignore persistent symptoms, as they may suggest a more "
        "significant issue that requires prompt attention.",

        "Based on your description, this condition is likely caused by one of several "
        "possible underlying factors. Because symptoms like these can indicate anything "
        "from a minor issue to a serious medical problem, it is important to seek "
        "evaluation from a qualified physician. The reason this matters is that early "
        "treatment leads to faster recovery and prevents complications. If you experience "
        "severe pain, difficulty breathing, or loss of consciousness, call 911 immediately. "
        "A proper diagnosis results in a targeted treatment plan suited to your situation.",

        "The symptoms you are describing may indicate an injury or condition that requires "
        "medical attention. This is important because untreated conditions can lead to "
        "complications such as chronic pain or further injury. Due to the nature of these "
        "symptoms, a healthcare provider should evaluate you, as they may recommend imaging "
        "such as an X-ray to rule out a fracture or other structural damage. Consequently, "
        "please see a doctor promptly. If symptoms worsen suddenly, go to the emergency room.",

        "There are several possible causes for what you are experiencing, ranging from "
        "minor muscle strain to more serious conditions. Because an accurate diagnosis "
        "requires physical examination, I strongly recommend seeing a doctor or visiting "
        "urgent care. Treatment effectiveness depends on identifying the root cause, which "
        "is why self-diagnosis is not sufficient here. If the pain is severe, stems from "
        "a fall or impact, or is associated with numbness, consult a physician immediately "
        "as this could indicate a fracture or nerve injury that requires prompt treatment.",
    ],

    # ── Brief: short responses → label = incomplete_reasoning ────────────────
    "brief": [
        "Apply ice to reduce swelling and rest the affected area. "
        "Take over-the-counter pain relief if needed.",

        "This is usually not serious. Rest, stay hydrated, and monitor your symptoms. "
        "See a doctor if things don't improve in a few days.",

        "Try to rest and avoid strenuous activity. "
        "If symptoms persist, consult your doctor.",

        "Rest the area and take an anti-inflammatory medication. "
        "Monitor for any changes over the next 24 hours.",
    ],

    # ── Dismissive: vague phrases → label = incomplete_reasoning ─────────────
    "dismissive": [
        "Most likely nothing to worry about. "
        "Give it a couple of days and see how it goes.",

        "This is probably just minor irritation. "
        "Rest up and you should be fine.",

        "Don't stress about it — these things usually resolve on their own. "
        "Just take it easy for a while.",

        "Probably nothing serious. Just monitor it and rest.",
    ],

    # ── Overconfident: no referral + definitive claim → critical_failure ─────
    "overconfident": [
        "This is definitely muscle strain. "
        "Take ibuprofen every 6 hours and apply heat. No need to see a doctor.",

        "You have a common cold. "
        "Take paracetamol and rest. There is no need for medical attention.",

        "This sounds like a minor sprain. "
        "Wrap it with a bandage and stay off it for two days.",

        "This is just stress. Try relaxing more and getting better sleep. "
        "No medical attention is necessary.",
    ],
}


def generate_template(question: str, prompt_type: str) -> str:
    """Generate a template-based response (no API key required)."""
    options = TEMPLATE_RESPONSES[prompt_type]
    return random.choice(options).strip()


# ── Main generation pipeline ──────────────────────────────────────────────────

def generate_responses(
    questions: list[str],
    use_openai: bool = False,
    model: str = "gpt-3.5-turbo",
    rate_limit_delay: float = 0.5,
    save_path: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate AI responses for a list of medical questions.

    Args:
        questions:         List of question strings.
        use_openai:        If True, call OpenAI API. Else use template fallback.
        model:             OpenAI model name.
        rate_limit_delay:  Seconds between API calls.
        save_path:         Where to save the output CSV.
        resume_from:       Path to partially-completed CSV (skips already-done questions).

    Returns:
        DataFrame with columns: question, ai_response, prompt_type
    """
    # Load partial results if resuming
    done_questions = set()
    existing_rows  = []
    if resume_from and os.path.exists(resume_from):
        existing_df   = pd.read_csv(resume_from)
        done_questions = set(existing_df["question"].tolist())
        existing_rows  = existing_df.to_dict("records")
        logger.info(f"Resuming — {len(done_questions)} questions already done.")

    todo = [q for q in questions if q not in done_questions]
    logger.info(f"Generating responses for {len(todo)} questions "
                f"({'OpenAI' if use_openai else 'template fallback'})...")

    new_rows = []
    for q in tqdm(todo, desc="Generating"):
        prompt_type = pick_prompt_type()
        try:
            if use_openai:
                response = generate_openai(q, prompt_type=prompt_type, model=model)
                time.sleep(rate_limit_delay)
            else:
                response = generate_template(q, prompt_type=prompt_type)
        except Exception as e:
            logger.warning(f"Generation error: {e}. Using template fallback.")
            response = generate_template(q, prompt_type="brief")

        new_rows.append({
            "question":    q,
            "ai_response": response,
            "prompt_type": prompt_type,
        })

        # Checkpoint every 100 rows
        if save_path and len(new_rows) % 100 == 0:
            _checkpoint(existing_rows + new_rows, save_path)

    all_rows = existing_rows + new_rows
    df = pd.DataFrame(all_rows)

    if save_path:
        df.to_csv(save_path, index=False)
        logger.info(f"Saved {len(df)} responses to {save_path}")

    return df


def _checkpoint(rows: list, path: str):
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info(f"Checkpoint saved: {len(rows)} rows → {path}")


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate AI responses for medical questions")
    parser.add_argument("--input",  default="data/raw_questions.csv")
    parser.add_argument("--output", default="data/raw_dataset.csv")
    parser.add_argument("--openai", action="store_true",
                        help="Use OpenAI API (requires OPENAI_API_KEY env var)")
    parser.add_argument("--max",    type=int, default=5000)
    args = parser.parse_args()

    df_questions = pd.read_csv(args.input)
    questions    = df_questions["question"].dropna().unique().tolist()[:args.max]

    result = generate_responses(
        questions  = questions,
        use_openai = args.openai,
        save_path  = args.output,
    )
    print(f"\nSample output:")
    print(result[["question", "ai_response", "prompt_type"]].head(3).to_string())
