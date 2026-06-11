"""
Logit trend hints and oracle labels for inverse_logit_polarity auditing.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

STEERING_KEYS = ["-2.0", "-1.5", "-1.0", "-0.5", "0.0", "+0.5", "+1.0", "+1.5", "+2.0"]
ALPHAS_NEG = [-2.0, -1.5, -1.0, -0.5]
ALPHAS_POS = [0.5, 1.0, 1.5, 2.0]
REQUIRED_HINT_KEYS = ("0.0", "+2.0", "-2.0")


def _parse_logit(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def steering_key_to_logit_col(key: str, prefix: str = "logit_") -> str:
    """Map steering key (-2.0, +0.5) to CSV column (logit_-2.0, logit_0.5)."""
    return f"{prefix}{key.lstrip('+')}"


def logits_from_row(row: Mapping[str, Any], prefix: str = "logit_") -> dict[str, float]:
    """Build a logits_map from CSV row columns like logit_0.0, logit_2.0."""
    out: dict[str, float] = {}
    for key in STEERING_KEYS:
        col = steering_key_to_logit_col(key, prefix)
        if col in row:
            v = _parse_logit(row.get(col))
            if v is not None:
                out[key] = v
    return out


def compute_logit_trend_hint(logits_map: Mapping[str, Any]) -> str:
    """
    Endpoint comparison hint for the Pass 2 judge (±2.0 vs 0.0).
    Returns INSUFFICIENT_DATA if baseline or endpoint logits are missing.
    """
    missing = [k for k in REQUIRED_HINT_KEYS if _parse_logit(logits_map.get(k)) is None]
    if missing:
        return "INSUFFICIENT_DATA (missing logits for: " + ", ".join(missing) + ")"

    b = _parse_logit(logits_map["0.0"])
    p2 = _parse_logit(logits_map["+2.0"])
    n2 = _parse_logit(logits_map["-2.0"])

    pos_trend = "INCREASING (EXPECTED)" if p2 >= b else "DECREASING (INVERSE)"
    neg_trend = "DECREASING (EXPECTED)" if n2 <= b else "INCREASING (INVERSE)"
    return (
        f"Positive Strengths (+2.0 vs 0.0): {pos_trend} | "
        f"Negative Strengths (-2.0 vs 0.0): {neg_trend}"
    )


def compute_oracle_labels(logits_map: Mapping[str, Any]) -> dict[str, Any]:
    """
    Oracle tiers for inverse_logit_polarity validation.
    Returns dict with boolean flags and metadata; incomplete rows have has_data=False.
    """
    b = _parse_logit(logits_map.get("0.0"))
    p2 = _parse_logit(logits_map.get("+2.0"))
    n2 = _parse_logit(logits_map.get("-2.0"))

    if b is None or p2 is None or n2 is None:
        return {"has_data": False}

    pos_ep_inv = p2 < b
    neg_ep_inv = n2 > b
    either_ep_inv = pos_ep_inv or neg_ep_inv
    both_ep_inv = pos_ep_inv and neg_ep_inv

    pos_any_inv = any(
        (v := _parse_logit(logits_map.get(f"+{a:.1f}"))) is not None and v < b
        for a in ALPHAS_POS
    )
    neg_any_inv = any(
        (v := _parse_logit(logits_map.get(f"{a:.1f}"))) is not None and v > b
        for a in ALPHAS_NEG
    )

    any_dir_inv = pos_any_inv or neg_any_inv

    xs, ys = [], []
    for a in ALPHAS_NEG + [0.0] + ALPHAS_POS:
        key = f"{a:.1f}" if a <= 0 else f"+{a:.1f}"
        v = _parse_logit(logits_map.get(key))
        if v is not None:
            xs.append(a)
            ys.append(v)

    slope_inv = False
    slope = None
    if len(xs) >= 3:
        slope = float(np.polyfit(xs, ys, 1)[0])
        slope_inv = slope < 0

    max_ep_delta = max(max(0.0, b - p2), max(0.0, n2 - b))

    l05 = _parse_logit(logits_map.get("+0.5"))
    l10 = _parse_logit(logits_map.get("+1.0"))
    collapse_at_pos_2 = (
        pos_ep_inv
        and l05 is not None
        and l10 is not None
        and l05 > b
        and l10 > b
    )

    pos_any_not_ep = pos_any_inv and not pos_ep_inv
    neg_any_not_ep = neg_any_inv and not neg_ep_inv
    nonmono_mid_inv = either_ep_inv and (pos_any_not_ep or neg_any_not_ep)

    return {
        "has_data": True,
        "baseline": b,
        "logit_p2": p2,
        "logit_n2": n2,
        "pos_ep_inv": pos_ep_inv,
        "neg_ep_inv": neg_ep_inv,
        "either_ep_inv": either_ep_inv,
        "both_ep_inv": both_ep_inv,
        "pos_any_inv": pos_any_inv,
        "neg_any_inv": neg_any_inv,
        "any_dir_inv": any_dir_inv,
        "slope_inv": slope_inv,
        "slope": slope,
        "max_ep_delta": max_ep_delta,
        "collapse_at_pos_2": collapse_at_pos_2,
        "nonmono_mid_inv": nonmono_mid_inv,
        "hint": compute_logit_trend_hint(logits_map),
    }
