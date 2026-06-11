PASS1_ATTRIBUTES = [
    "factual_integrity",
    "logical_coherence",
    "behavioral_resistance",
    "output_shape",
    "tonal_change",
]

# Labels interpreted as "no effect / no anomaly present"
NONE_KEYWORDS = {"NONE", "FALSE", "NO", "NOT_PRESENT"}

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
