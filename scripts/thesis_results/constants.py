"""Evaluation schema constants (mirrors thesis/pipeline/constants.py)."""

PASS1_ATTRIBUTES = [
    "factual_integrity",
    "logical_coherence",
    "behavioral_resistance",
    "output_shape",
    "tonal_change",
]

PASS1_CATEGORIES = list(PASS1_ATTRIBUTES)

PASS2_EFFECT_KEYS = [
    "factual_drift",
    "monotonic_factual_scaling",
    "hallucinated_specificity",
    "factual_reversal",
    "label_content_contradiction",
    "option_label_mismatch",
    "repetition_looping",
    "sensitivity_refusal",
    "logit_text_decoupling",
    "steering_asymmetry",
    "safety_override_resistance",
    "inverse_logit_polarity",
    "token_compression",
    "token_inflation",
    "implicit_assumption_injection",
    "neutralization_rephrasing",
    "hedging_escalation",
    "persona_amplification",
    "tonal_rigidity",
]

PASS2_CATEGORY_EFFECTS = {
    "factual_integrity": [
        "factual_drift",
        "monotonic_factual_scaling",
        "hallucinated_specificity",
        "factual_reversal",
    ],
    "logical_coherence": [
        "label_content_contradiction",
        "option_label_mismatch",
        "repetition_looping",
    ],
    "behavioral_resistance": [
        "sensitivity_refusal",
        "logit_text_decoupling",
        "steering_asymmetry",
        "safety_override_resistance",
        "inverse_logit_polarity",
    ],
    "output_shape": [
        "token_compression",
        "token_inflation",
        "implicit_assumption_injection",
    ],
    "tonal_change": [
        "neutralization_rephrasing",
        "hedging_escalation",
        "persona_amplification",
        "tonal_rigidity",
    ],
}

STEERING_MULTIPLIERS = ["-2.0", "-1.5", "-1.0", "-0.5", "0.0", "0.5", "1.0", "1.5", "2.0"]
STEERING_MULTIPLIERS_FLOAT = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

STEERABILITY_ORDER = ["NONE", "WEAK", "MODERATE", "STRONG", "ANTI-STEERED", "UNMEASURABLE"]

PRIMARY_DIMENSIONS = [
    "steering_asymmetry",
    "token_inflation",
    "factual_reversal",
    "inverse_logit_polarity",
]

DIMENSION_LABELS = {
    "steering_asymmetry": "Steering asymmetry",
    "token_inflation": "Token inflation",
    "factual_reversal": "Factual reversal",
    "inverse_logit_polarity": "Inverse logit polarity",
}
