"""Summary table generation for thesis results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import PASS1_CATEGORIES
from .practical_significance import export_practical_significance
from .utils import is_flagged


def calculate_effect_sizes(df_pass2: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import rankdata

    print("Calculating effect sizes...")
    results = []
    datasets = df_pass2["dataset_name"].dropna().unique()

    for dataset in datasets:
        df_sub = df_pass2[df_pass2["dataset_name"] == dataset]
        effect_cols = [c for c in df_sub.columns if str(c).startswith("judge_effect_")]

        for col in effect_cols:
            dim = col.replace("judge_effect_", "")
            scores = pd.to_numeric(df_sub[col], errors="coerce").dropna()

            if len(scores) < 2 or (scores == 0).all():
                continue

            try:
                diffs = scores - 0
                diffs = diffs[diffs != 0]
                if len(diffs) == 0:
                    continue

                mean_score = scores.mean()
                std_score = scores.std(ddof=1)
                cohens_d = mean_score / std_score if std_score > 0 else np.nan

                results.append({
                    "dataset": dataset,
                    "dimension": dim.replace("_", " "),
                    "cohens_d": round(cohens_d, 4) if pd.notna(cohens_d) else np.nan,
                })
            except Exception:
                pass

    return pd.DataFrame(results)


def generate_tables(
    pass1_df: pd.DataFrame,
    pass2_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    output_dir: Path,
    iaa_agreement_path: Path | None = None,
    significance_csv: Path | None = None,
) -> pd.DataFrame:
    print("Generating tables...")
    out_dir = output_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_data = []
    for dataset in pass1_df["dataset_name"].unique():
        sub1 = pass1_df[pass1_df["dataset_name"] == dataset]
        n_total = len(sub1)

        flagged_count = 0
        for _, row in sub1.iterrows():
            pass1_cols = [
                c for c in row.index
                if str(c).startswith("judge_")
                and not str(c).endswith("_rationale")
                and c not in ["judge_schema", "judge_notes", "judge_raw_votes_json", "judge_steerability"]
            ]
            if any(is_flagged(row.get(c)) for c in pass1_cols):
                flagged_count += 1

        summary_data.append({
            "dataset": dataset,
            "n_total": n_total,
            "n_flagged_pass1": flagged_count,
            "pct_flagged": round(flagged_count / n_total * 100, 1) if n_total > 0 else 0,
        })
    pd.DataFrame(summary_data).to_csv(out_dir / "run_summary.csv", index=False)

    steer_counts = pass1_df["judge_steerability"].value_counts(dropna=False).reset_index()
    steer_counts.columns = ["steerability", "count"]
    steer_counts["pct"] = round(steer_counts["count"] / len(pass1_df) * 100, 1)
    steer_counts.to_csv(out_dir / "pass1_steerability_counts.csv", index=False)

    cat_rows = []
    for dataset in pass1_df["dataset_name"].dropna().unique():
        sub = pass1_df[pass1_df["dataset_name"] == dataset]
        n_total = len(sub)
        for cat in PASS1_CATEGORIES:
            col = f"judge_{cat}"
            if col not in sub.columns:
                continue
            n_flagged = sub[col].apply(is_flagged).sum()
            cat_rows.append({
                "dataset": dataset,
                "category": cat,
                "n_total": n_total,
                "n_flagged": int(n_flagged),
                "pct_flagged": round(n_flagged / n_total * 100, 1) if n_total > 0 else 0,
            })
    pd.DataFrame(cat_rows).to_csv(out_dir / "pass1_category_flags.csv", index=False)

    effect_sizes_df = calculate_effect_sizes(pass2_df)
    if not effect_sizes_df.empty:
        sig_df = sig_df.merge(effect_sizes_df, on=["dataset", "dimension"], how="left")
    else:
        sig_df = sig_df.copy()
        sig_df["cohens_d"] = np.nan

    sig_bh = sig_df[sig_df["sig_bh"] == True].copy()
    sig_bh.to_csv(out_dir / "significance_bh_significant.csv", index=False)

    sig_for_tiers = sig_df
    if significance_csv and significance_csv.exists():
        sig_for_tiers = pd.read_csv(significance_csv)
    export_practical_significance(sig_for_tiers, out_dir, iaa_path=iaa_agreement_path)

    dim_summary = sig_df.groupby("dimension").agg(
        n_tests=("dataset", "count"),
        n_sig_bh=("sig_bh", "sum"),
        mean_severity=("mean_severity", "mean"),
        mean_cohens_d=("cohens_d", "mean"),
    ).reset_index()
    dim_summary["pct_sig"] = round(dim_summary["n_sig_bh"] / dim_summary["n_tests"] * 100, 1)
    dim_summary["abs_mean_cohens_d"] = dim_summary["mean_cohens_d"].abs()
    max_d_by_dim = sig_df.groupby("dimension")["cohens_d"].apply(lambda x: x.abs().max())
    dim_summary["max_cohens_d"] = dim_summary["dimension"].map(max_d_by_dim)
    dim_summary.sort_values("mean_severity", ascending=False).to_csv(
        out_dir / "significance_by_dimension.csv", index=False
    )

    ds_summary = sig_df.groupby("dataset").agg(
        n_tests=("dimension", "count"),
        n_sig_bh=("sig_bh", "sum"),
        mean_severity=("mean_severity", "mean"),
        mean_cohens_d=("cohens_d", "mean"),
        max_cohens_d=("cohens_d", lambda x: x.abs().max()),
    ).reset_index()
    ds_summary["pct_sig"] = round(ds_summary["n_sig_bh"] / ds_summary["n_tests"] * 100, 1)
    ds_summary.sort_values("n_sig_bh", ascending=False).to_csv(
        out_dir / "significance_by_dataset.csv", index=False
    )

    sig_bh.sort_values("mean_severity", ascending=False).head(50).to_csv(
        out_dir / "top_effects_ranked.csv", index=False
    )

    asym_tests = sig_df[sig_df["dimension"] == "steering asymmetry"]
    pct_asym_datasets = round((asym_tests["sig_bh"].sum() / len(asym_tests)) * 100, 1) if len(asym_tests) > 0 else 0

    asym_scores = pd.to_numeric(pass2_df["judge_effect_steering_asymmetry"], errors="coerce")
    pct_asym_samples = round((asym_scores > 0).sum() / len(asym_scores.dropna()) * 100, 1) if len(asym_scores.dropna()) > 0 else 0

    pd.DataFrame([
        {
            "metric": "pct_datasets_with_significant_asymmetry",
            "value": pct_asym_datasets,
            "n_datasets": len(asym_tests),
        },
        {
            "metric": "pct_flagged_samples_with_asymmetry",
            "value": pct_asym_samples,
            "n_samples_evaluated": len(asym_scores.dropna()),
        },
    ]).to_csv(out_dir / "directional_asymmetry_stats.csv", index=False)

    deg_cols = [
        "judge_effect_label_content_contradiction",
        "judge_effect_hedging_escalation",
        "judge_effect_factual_reversal",
    ]
    deg_stats = []
    for col in deg_cols:
        scores = pd.to_numeric(pass2_df[col], errors="coerce").dropna()
        if len(scores) > 0:
            deg_stats.append({
                "effect": col.replace("judge_effect_", ""),
                "n_evaluated": len(scores),
                "n_non_zero": (scores > 0).sum(),
                "pct_non_zero": round((scores > 0).sum() / len(scores) * 100, 1),
                "mean_severity": round(scores.mean(), 3),
            })
    pd.DataFrame(deg_stats).to_csv(out_dir / "high_intensity_degradation_stats.csv", index=False)

    dec_scores = pd.to_numeric(pass2_df["judge_effect_logit_text_decoupling"], errors="coerce").dropna()
    pct_dec = round((dec_scores > 0).sum() / len(dec_scores) * 100, 1) if len(dec_scores) > 0 else 0
    pd.DataFrame([{
        "metric": "pct_flagged_samples_with_decoupling",
        "value": pct_dec,
        "n_samples_evaluated": len(dec_scores),
        "mean_severity": round(dec_scores.mean(), 3) if len(dec_scores) > 0 else 0,
    }]).to_csv(out_dir / "logit_text_decoupling_stats.csv", index=False)

    return sig_df


def export_iaa_metrics(iaa_path: Path, output_dir: Path) -> None:
    out_dir = output_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not iaa_path.exists():
        print(f"Warning: IAA metrics not found at {iaa_path}; skipping.")
        return

    iaa_df = pd.read_csv(iaa_path)
    for col in ("Krippendorff_Ordinal", "Krippendorff_Interval"):
        if col in iaa_df.columns:
            iaa_df[col] = pd.to_numeric(iaa_df[col], errors="coerce").round(4)
    iaa_df.to_csv(out_dir / "agreement_metrics.csv", index=False)
    print(f"Exported IAA metrics to {out_dir / 'agreement_metrics.csv'}")
