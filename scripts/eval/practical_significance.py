#!/usr/bin/env python3
"""
Practical-significance analysis for Pass-2 Wilcoxon results.

Classifies dataset×dimension tests into primary vs exploratory tiers and
exports sensitivity tables over mean-severity and prevalence thresholds.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.thesis_results.config import DATA_DIR  # noqa: E402

DEFAULT_IAA_FILE = DATA_DIR / "iaa/agreement_metrics.csv"

MEAN_THRESHOLDS = [0.2, 0.5, 1.0]
PCT_THRESHOLDS = [1.0, 5.0, 10.0]

PRIMARY_MEAN = 0.5
PRIMARY_PCT = 10.0
PRIMARY_IAA = 0.65

LOGIT_AUDIT_DIMENSIONS = {"inverse_logit_polarity", "inverse logit polarity"}


def _dim_key(name: str) -> str:
    return str(name).replace(" ", "_").lower()


def load_iaa_alpha(iaa_path: Path | None = None) -> dict[str, float]:
    path = iaa_path or DEFAULT_IAA_FILE
    if not path.exists():
        return {}
    iaa = pd.read_csv(path)
    pass2 = iaa[iaa["Pass"] == 2]
    return {
        str(row["key"]).lower(): float(row["Krippendorff_Ordinal"])
        for _, row in pass2.iterrows()
        if pd.notna(row.get("Krippendorff_Ordinal"))
    }


def add_cohens_d(results_df: pd.DataFrame) -> pd.DataFrame:
    df = results_df.copy()
    if "cohens_d" in df.columns:
        return df
    std = pd.to_numeric(df["std_severity"], errors="coerce")
    mean = pd.to_numeric(df["mean_severity"], errors="coerce")
    df["cohens_d"] = np.where(std > 0, mean / std, np.nan).round(4)
    return df


def classify_tier(row: pd.Series, iaa_map: dict[str, float]) -> str:
    mean = row.get("mean_severity", 0)
    pct = row.get("pct_nonzero", 0)
    sig = bool(row.get("sig_bh", False))
    dim_raw = row.get("dimension_raw") or _dim_key(row.get("dimension", ""))
    dim_key = _dim_key(dim_raw)
    iaa = iaa_map.get(dim_key, np.nan)
    has_validation = (
        (pd.notna(iaa) and iaa >= PRIMARY_IAA)
        or dim_key in LOGIT_AUDIT_DIMENSIONS
    )
    if (
        sig
        and mean >= PRIMARY_MEAN
        and pct >= PRIMARY_PCT
        and has_validation
    ):
        return "primary"
    if sig:
        return "exploratory"
    return "not_significant"


def confidence_label(iaa: float) -> str:
    if pd.isna(iaa):
        return "unknown"
    if iaa >= 0.65:
        return "high"
    if iaa >= 0.55:
        return "moderate"
    return "low"


def build_sensitivity_grid(results_df: pd.DataFrame) -> pd.DataFrame:
    sig = results_df[results_df["sig_bh"] == True].copy()
    rows = []
    for mean_thr in MEAN_THRESHOLDS:
        for pct_thr in PCT_THRESHOLDS:
            mask = (sig["mean_severity"] >= mean_thr) & (sig["pct_nonzero"] >= pct_thr)
            rows.append({
                "mean_threshold": mean_thr,
                "pct_nonzero_threshold": pct_thr,
                "n_bh_significant": len(sig),
                "n_meeting_thresholds": int(mask.sum()),
                "pct_of_bh_significant": round(mask.sum() / len(sig) * 100, 1) if len(sig) else 0,
            })
    return pd.DataFrame(rows)


def build_dimension_tiers(results_df: pd.DataFrame, iaa_map: dict[str, float]) -> pd.DataFrame:
    df = results_df.copy()
    if "dimension_raw" not in df.columns:
        df["dimension_raw"] = df["dimension"].map(_dim_key)

    df["iaa_alpha"] = df["dimension_raw"].map(lambda d: iaa_map.get(_dim_key(d), np.nan))
    df["confidence"] = df["iaa_alpha"].map(confidence_label)
    df["logit_audit_validated"] = df["dimension_raw"].map(
        lambda d: _dim_key(d) in LOGIT_AUDIT_DIMENSIONS
    )
    df["tier"] = df.apply(lambda r: classify_tier(r, iaa_map), axis=1)
    return df


def build_dimension_summary(tiered_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dim in sorted(tiered_df["dimension"].unique()):
        sub = tiered_df[tiered_df["dimension"] == dim]
        rows.append({
            "dimension": dim,
            "dimension_raw": sub["dimension_raw"].iloc[0],
            "n_tests": len(sub),
            "n_sig_bh": int(sub["sig_bh"].sum()),
            "n_primary": int((sub["tier"] == "primary").sum()),
            "n_exploratory": int((sub["tier"] == "exploratory").sum()),
            "mean_severity_avg": round(sub["mean_severity"].mean(), 4),
            "pct_nonzero_avg": round(sub["pct_nonzero"].mean(), 2),
            "iaa_alpha": sub["iaa_alpha"].iloc[0] if "iaa_alpha" in sub.columns else np.nan,
            "confidence": sub["confidence"].iloc[0] if "confidence" in sub.columns else "unknown",
            "logit_audit_validated": bool(sub["logit_audit_validated"].iloc[0]),
        })
    return pd.DataFrame(rows).sort_values("n_primary", ascending=False)


def export_practical_significance(
    results_df: pd.DataFrame,
    output_dir: str | Path,
    iaa_path: Path | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = add_cohens_d(results_df)
    iaa_map = load_iaa_alpha(iaa_path)

    sensitivity = build_sensitivity_grid(df)
    tiered = build_dimension_tiers(df, iaa_map)
    dim_summary = build_dimension_summary(tiered)

    primary = tiered[tiered["tier"] == "primary"].copy()
    exploratory = tiered[tiered["tier"] == "exploratory"].copy()

    sensitivity.to_csv(output_dir / "practical_significance_sensitivity.csv", index=False)
    tiered.to_csv(output_dir / "significance_results_with_tiers.csv", index=False)
    primary.to_csv(output_dir / "significance_primary_claims.csv", index=False)
    exploratory.to_csv(output_dir / "significance_exploratory_claims.csv", index=False)
    dim_summary.to_csv(output_dir / "dimension_tier_summary.csv", index=False)

    n_sig = int((df["sig_bh"] == True).sum())
    n_primary = len(primary)
    stats = {
        "n_total_tests": len(df),
        "n_bh_significant": n_sig,
        "n_primary_claims": n_primary,
        "n_exploratory_claims": len(exploratory),
        "primary_at_default_threshold": n_primary,
        "primary_mean_threshold": PRIMARY_MEAN,
        "primary_pct_threshold": PRIMARY_PCT,
        "primary_iaa_threshold": PRIMARY_IAA,
    }

    summary_lines = [
        "# Practical Significance Summary",
        "",
        f"- Total tests: {stats['n_total_tests']}",
        f"- BH-significant: {stats['n_bh_significant']} ({stats['n_bh_significant']/stats['n_total_tests']*100:.1f}%)",
        f"- Primary claims (BH-sig + mean≥{PRIMARY_MEAN} + pct≥{PRIMARY_PCT}% + IAA≥{PRIMARY_IAA} or logit audit): **{n_primary}**",
        f"- Exploratory (BH-sig but not primary): **{len(exploratory)}**",
        "",
        "## Sensitivity grid (BH-significant tests only)",
        "",
        sensitivity.to_markdown(index=False),
        "",
        "## Dimension tier summary",
        "",
        dim_summary.to_markdown(index=False),
    ]
    (output_dir / "PRACTICAL_SIGNIFICANCE_SUMMARY.md").write_text("\n".join(summary_lines))

    return stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python practical_significance.py [significance_results.csv] [output_dir]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(input_file)

    df = pd.read_csv(input_file)
    stats = export_practical_significance(df, output_dir)
    print(f"Exported practical significance analysis to {output_dir}")
    print(f"  Primary claims: {stats['n_primary_claims']}/{stats['n_bh_significant']} BH-significant")


if __name__ == "__main__":
    main()
