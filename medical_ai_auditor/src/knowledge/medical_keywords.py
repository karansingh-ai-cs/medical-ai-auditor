"""
medical_keywords.py
-------------------
Centralised keyword lists used by the rule-based heuristics and weak labeler.
Expand these lists to improve recall on specific medical domains.
"""

# ── Emergency / life-threatening conditions ───────────────────────────────────
CRITICAL_CONDITION_KEYWORDS = [
    # Cardiovascular
    "heart attack", "myocardial infarction", "cardiac arrest", "arrhythmia",
    "ventricular fibrillation", "atrial fibrillation", "chest pain", "aortic dissection",
    "pulmonary embolism", "deep vein thrombosis", "dvt",
    # Neurological
    "stroke", "tia", "transient ischemic attack", "seizure", "convulsion",
    "meningitis", "encephalitis", "intracranial hemorrhage", "subarachnoid hemorrhage",
    "brain bleed", "loss of consciousness", "syncope",
    # Respiratory
    "respiratory failure", "anaphylaxis", "anaphylactic shock", "asthma attack",
    "pneumothorax", "tension pneumothorax", "airway obstruction",
    # Abdominal / surgical
    "appendicitis", "ruptured appendix", "bowel obstruction", "intestinal perforation",
    "ectopic pregnancy", "ruptured ectopic", "internal bleeding", "sepsis",
    "septic shock", "peritonitis",
    # Trauma
    "hemorrhage", "severe bleeding", "uncontrolled bleeding", "spinal injury",
    "traumatic brain injury", "tbi", "skull fracture",
    # Toxicological
    "overdose", "poisoning", "toxic ingestion", "carbon monoxide",
    # Other
    "diabetic ketoacidosis", "dka", "hypoglycemia", "severe hypoglycemia",
    "hyperglycemia", "sickle cell crisis",
]

# ── Safety / referral phrases expected in a complete response ─────────────────
SAFETY_REFERRAL_PHRASES = [
    # Emergency
    "call 911", "call emergency", "go to the er", "go to the emergency room",
    "emergency room", "emergency department", "emergency care",
    "seek immediate", "seek emergency", "urgent care",
    # Medical professional
    "see a doctor", "see a physician", "consult a doctor", "consult a physician",
    "consult a specialist", "medical attention", "healthcare provider",
    "medical professional", "visit a doctor", "contact your doctor",
    # Diagnostic
    "get an x-ray", "get an mri", "get a ct scan", "imaging", "blood test",
    "laboratory test", "diagnostic test",
    # Monitoring
    "monitor closely", "watch for signs", "watch for symptoms",
    "if symptoms worsen", "if it gets worse", "if you experience",
]

# ── Trauma / injury trigger words ─────────────────────────────────────────────
TRAUMA_TRIGGER_WORDS = [
    "fell", "fall", "fallen", "tripped", "slipped",
    "hit", "struck", "blow", "impact",
    "crash", "collision", "accident", "car accident",
    "injured", "injury", "trauma", "wound",
    "broke", "broken", "fracture",
    "twisted", "sprained",
]

# ── Fracture / bone injury acknowledgment ────────────────────────────────────
FRACTURE_ACKNOWLEDGMENT_PHRASES = [
    "fracture", "broken bone", "break", "crack", "hairline",
    "x-ray", "xray", "radiograph", "imaging", "orthopedic",
    "orthopedist", "bone scan", "splint", "cast", "immobilize",
]

# ── Causal / reasoning connector words ───────────────────────────────────────
CAUSAL_CONNECTORS = [
    "because", "due to", "caused by", "leads to", "resulting in",
    "which means", "therefore", "as a result", "this can cause",
    "may indicate", "could suggest", "is a sign of", "is caused by",
    "can lead to", "results from", "stems from", "indicates",
    "suggests", "implies", "is associated with", "is linked to",
    "in order to", "so that", "consequently", "thus", "hence",
]

# ── Vague / dismissive phrases that signal incomplete reasoning ───────────────
VAGUE_DISMISSIVE_PHRASES = [
    "rest and relax", "it might get better", "see how it goes",
    "probably nothing", "just monitor it", "wait and see",
    "should be fine", "nothing to worry about", "don't worry about it",
    "most likely not serious", "usually gets better on its own",
    "just take it easy", "drink water and rest",
]

# ── Drug / medication safety keywords ────────────────────────────────────────
MEDICATION_SAFETY_KEYWORDS = [
    "overdose", "drug interaction", "contraindicated", "allergy",
    "allergic reaction", "side effect", "adverse effect",
    "do not take", "avoid taking", "consult before taking",
    "prescription required", "prescription only",
]

# ── Pediatric / vulnerable population flags ───────────────────────────────────
PEDIATRIC_KEYWORDS = [
    "child", "children", "infant", "baby", "toddler", "newborn",
    "pediatric", "adolescent", "teenager", "minor",
]

# ── Pregnancy-related red flags ───────────────────────────────────────────────
PREGNANCY_KEYWORDS = [
    "pregnant", "pregnancy", "prenatal", "trimester", "fetus",
    "fetal", "obstetric", "obstetrician", "ectopic",
]

# Label IDs mapping
LABEL_TO_ID = {
    "no_issue":               0,
    "incomplete_reasoning":   1,
    "missing_causal_link":    2,
    "dangerous_omission":     3,
    "critical_failure":       4,
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

NUM_LABELS = len(LABEL_TO_ID)
