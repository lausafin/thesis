"""Shared helpers for thesis result regeneration."""

from __future__ import annotations

import pandas as pd

from .constants import PASS1_CATEGORIES


def is_flagged(val) -> bool:
    if pd.isna(val):
        return False
    return str(val).upper().strip() in {"FLAGGED", "YES", "TRUE", "1"}


def effect_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).startswith("judge_effect_")]


def add_pass1_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for cat in PASS1_CATEGORIES:
        col = f"judge_{cat}"
        if col in out.columns:
            out[f"flag_{cat}"] = out[col].apply(is_flagged)
        else:
            out[f"flag_{cat}"] = False
    out["n_categories_flagged"] = out[[f"flag_{c}" for c in PASS1_CATEGORIES]].sum(axis=1)
    out["any_category_flagged"] = out["n_categories_flagged"] > 0
    return out


def severity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = effect_columns(df)
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
