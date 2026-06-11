#!/usr/bin/env python3
"""Analyze human validation ratings: IAA, human--LLM concordance, prevalence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.metrics.logit_trends import compute_oracle_labels, logits_from_row  # noqa: E402
from human_eval_common import (  # noqa: E402
    DEFAULT_PASS2,
    HUMAN_EVAL_DIR,
    PRIMARY_DIMENSIONS,
    ref_llm_col,
)
from scripts.thesis_results.config import THESIS_ROOT  # noqa: E402

THESIS_TABLES = THESIS_ROOT / "results/tables"
THESIS_GENERATED = THESIS_ROOT / "thesis/sections/generated"

DIMENSION_TEX = {
    "steering_asymmetry": "Steering asymmetry",
    "token_inflation": "Token inflation",
    "factual_reversal": "Factual reversal",
    "inverse_logit_polarity": "Inverse logit polarity",
}

SUCCESS_ALPHA_ORD = 0.60
SUCCESS_KAPPA = 0.50
SUCCESS_RHO = 0.50
SUCCESS_PREVALENCE_PP = 10.0
SUCCESS_LOGIT_KAPPA = 0.60


def weighted_cohen_kappa(x: np.ndarray, y: np.ndarray, n_categories: int = 6) -> float:
    """Linear-weighted Cohen's kappa for two raters."""
    if len(x) == 0:
        return float("nan")
    conf = np.zeros((n_categories, n_categories))
    for a, b in zip(x, y):
        ai, bi = int(a), int(b)
        if 0 <= ai < n_categories and 0 <= bi < n_categories:
            conf[ai, bi] += 1
    n = conf.sum()
    if n == 0:
        return float("nan")
    weights = np.zeros((n_categories, n_categories))
    for i in range(n_categories):
        for j in range(n_categories):
            weights[i, j] = abs(i - j) / (n_categories - 1)
    po = 1.0 - (weights * conf / n).sum()
    row_m = conf.sum(axis=1) / n
    col_m = conf.sum(axis=0) / n
    pe = 1.0 - (weights * np.outer(row_m, col_m)).sum()
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return float((po - pe) / (1.0 - pe))


def krippendorff_alpha_two_raters(wide: pd.DataFrame) -> float:
    complete = wide.dropna(how="any")
    if complete.shape[0] < 2 or complete.shape[1] < 2:
        return float("nan")
    data = complete.T.to_numpy()
    return float(krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal"))


def human_human_metrics(merged: pd.DataFrame, split: str | None) -> pd.DataFrame:
    df = merged if split is None else merged[merged["split"] == split]
    rows = []
    for dim in PRIMARY_DIMENSIONS:
        wide = df.pivot_table(index="sample_id", columns="rater_id", values=dim, aggfunc="first")
        if wide.shape[1] < 2:
            continue
        complete = wide.dropna(how="any")
        n = len(complete)
        if n < 2:
            continue
        a = complete.iloc[:, 0].to_numpy()
        b = complete.iloc[:, 1].to_numpy()
        rows.append({
            "dimension": dim,
            "n_samples": n,
            "n_raters": 2,
            "alpha_ord": krippendorff_alpha_two_raters(wide),
            "weighted_kappa": weighted_cohen_kappa(a, b),
            "mae": float(np.mean(np.abs(a - b))),
        })
    return pd.DataFrame(rows)


def _rater_mean_scores(merged: pd.DataFrame) -> pd.DataFrame:
    return merged.groupby("sample_id", as_index=False).agg(
        {dim: "mean" for dim in PRIMARY_DIMENSIONS} | {"split": "first"}
    )


def _human_scores_for_llm(
    merged: pd.DataFrame,
    split: str | None,
    rater_id: str | None = None,
) -> pd.DataFrame:
    df = merged.copy()
    if rater_id is not None:
        df = df[df["rater_id"].astype(str).str.lower() == rater_id.lower()]
    human = _rater_mean_scores(df)
    if split:
        human = human[human["split"] == split]
    return human


def human_llm_metrics(
    merged: pd.DataFrame,
    subsample: pd.DataFrame,
    split: str | None,
    rater_id: str | None = None,
) -> pd.DataFrame:
    human = _human_scores_for_llm(merged, split, rater_id=rater_id)

    ref = subsample[["sample_id"] + [ref_llm_col(d) for d in PRIMARY_DIMENSIONS]].copy()
    joined = human.merge(ref, on="sample_id", how="inner")

    rows = []
    for dim in PRIMARY_DIMENSIONS:
        h = joined[dim].to_numpy()
        llm_col = ref_llm_col(dim)
        l = pd.to_numeric(joined[llm_col], errors="coerce").fillna(0).to_numpy()
        if len(h) < 3:
            continue
        rho, _ = spearmanr(h, l)
        rows.append({
            "dimension": dim,
            "n_samples": len(h),
            "spearman_rho": float(rho) if not np.isnan(rho) else float("nan"),
            "mae": float(np.mean(np.abs(h - l))),
            "human_mean": float(np.mean(h)),
            "llm_mean": float(np.mean(l)),
            "human_pct_nonzero": float(100 * np.mean(h > 0)),
            "llm_pct_nonzero": float(100 * np.mean(l > 0)),
            "prevalence_delta_pp": float(100 * np.mean(h > 0) - 100 * np.mean(l > 0)),
        })
    return pd.DataFrame(rows)


def single_rater_human_llm_metrics(
    merged: pd.DataFrame,
    subsample: pd.DataFrame,
    split: str,
    rater_id: str = "a",
) -> pd.DataFrame:
    """Human--LLM concordance when only one rater has scored each sample."""
    return human_llm_metrics(merged, subsample, split, rater_id=rater_id)


def logit_audit_metrics(
    merged: pd.DataFrame,
    subsample: pd.DataFrame,
    split: str | None,
) -> pd.DataFrame:
    human = _rater_mean_scores(merged)
    if split:
        human = human[human["split"] == split]
    human["human_ilp_nonzero"] = human["inverse_logit_polarity"] > 0

    ref = subsample.copy()
    audit_rows = []
    for _, row in ref.iterrows():
        logits_map = logits_from_row(row)
        oracle = compute_oracle_labels(logits_map)
        audit_rows.append({
            "sample_id": row["sample_id"],
            "audit_ilp_nonzero": bool(oracle.get("either_ep_inv", False)),
        })
    audit = pd.DataFrame(audit_rows)
    joined = human.merge(audit, on="sample_id", how="inner")
    if joined.empty:
        return pd.DataFrame()

    h = joined["human_ilp_nonzero"].to_numpy()
    a = joined["audit_ilp_nonzero"].to_numpy()
    # Cohen's kappa (unweighted binary)
    n = len(h)
    po = (h == a).mean()
    p_h = h.mean()
    p_a = a.mean()
    pe = p_h * p_a + (1 - p_h) * (1 - p_a)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0

    return pd.DataFrame([{
        "n_samples": n,
        "cohens_kappa": float(kappa),
        "human_pct_nonzero": float(100 * h.mean()),
        "audit_pct_nonzero": float(100 * a.mean()),
    }])


def prevalence_threshold_check(human_llm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in human_llm.iterrows():
        rows.append({
            "dimension": row["dimension"],
            "human_mean_ge_0_5": row["human_mean"] >= 0.5,
            "human_prevalence_ge_10pct": row["human_pct_nonzero"] >= 10.0,
            "prevalence_within_10pp": abs(row["prevalence_delta_pp"]) <= SUCCESS_PREVALENCE_PP,
        })
    return pd.DataFrame(rows)


def _fmt(x: float, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    return f"{x:.{nd}f}"


def write_calibration_iaa_tex(hh: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    iaa_lines = [
        r"\begin{table}[htbp]",
        r"  \centering\small",
        r"  \caption{Human--human agreement on primary dimensions (calibration set, two raters).}",
        r"  \label{tab:human_calibration_iaa}",
        r"  \begin{tabular}{@{}lrrr@{}}",
        r"    \toprule",
        r"    Dimension & $n$ & $\alpha_{\mathrm{ord}}$ & Weighted $\kappa$ \\",
        r"    \midrule",
    ]
    if hh.empty:
        iaa_lines.append(r"    \multicolumn{4}{c}{\textit{No paired calibration ratings}} \\")
    else:
        for _, row in hh.iterrows():
            dim = DIMENSION_TEX.get(row["dimension"], row["dimension"].replace("_", " "))
            iaa_lines.append(
                f"    {dim} & {int(row['n_samples'])} & {_fmt(row['alpha_ord'])} & {_fmt(row['weighted_kappa'])} \\\\"
            )
    iaa_lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
    (out_dir / "human_validation_calibration_iaa.tex").write_text("\n".join(iaa_lines) + "\n")


def write_exploratory_concordance_tex(hl: pd.DataFrame, out_dir: Path, rater_id: str = "a") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_samples = int(hl["n_samples"].max()) if not hl.empty else 0
    conc_lines = [
        r"\begin{table}[htbp]",
        r"  \centering\small",
        rf"  \caption{{Exploratory human--LLM concordance on partial main set ($n={n_samples}$, Rater {rater_id.upper()} only).}}",
        r"  \label{tab:human_exploratory_concordance}",
        r"  \begin{tabular}{@{}lrrrr@{}}",
        r"    \toprule",
        r"    Dimension & $\rho$ & MAE & Human \% & LLM \% \\",
        r"    \midrule",
    ]
    if hl.empty:
        conc_lines.append(r"    \multicolumn{5}{c}{\textit{No single-rater main ratings}} \\")
    else:
        for _, row in hl.iterrows():
            dim = DIMENSION_TEX.get(row["dimension"], row["dimension"].replace("_", " "))
            conc_lines.append(
                f"    {dim} & {_fmt(row['spearman_rho'])} & {_fmt(row['mae'])} & "
                f"{_fmt(row['human_pct_nonzero'], 1)} & {_fmt(row['llm_pct_nonzero'], 1)} \\\\"
            )
    conc_lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
    (out_dir / "human_validation_exploratory_concordance.tex").write_text("\n".join(conc_lines) + "\n")


def write_summary_md(
    path: Path,
    hh: pd.DataFrame,
    hl: pd.DataFrame,
    logit: pd.DataFrame,
    split: str,
) -> None:
    lines = [
        f"# Human validation analysis ({split})",
        "",
        "## Human--human agreement",
        "",
    ]
    if hh.empty:
        lines.append("_No paired ratings yet._")
    else:
        lines.append(hh.to_csv(index=False))
    lines.extend(["", "## Human--LLM concordance", ""])
    if hl.empty:
        lines.append("_No merged ratings with LLM reference._")
    else:
        lines.append(hl.to_csv(index=False))
    lines.extend(["", "## Inverse logit vs logit-derived audit", ""])
    if logit.empty:
        lines.append("_No logit audit comparison._")
    else:
        lines.append(logit.to_csv(index=False))
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze human validation study")
    parser.add_argument("--merged", type=Path, default=HUMAN_EVAL_DIR / "human_ratings_merged.csv")
    parser.add_argument("--subsample", type=Path, default=HUMAN_EVAL_DIR / "subsample_all.csv")
    parser.add_argument("--pass2", type=Path, default=DEFAULT_PASS2)
    parser.add_argument("--output-dir", type=Path, default=HUMAN_EVAL_DIR)
    parser.add_argument("--thesis-tables", type=Path, default=THESIS_TABLES)
    parser.add_argument(
        "--split",
        choices=["calibration", "main", "holdout", "all"],
        default="all",
    )
    parser.add_argument(
        "--thesis-generated",
        type=Path,
        default=THESIS_GENERATED,
        help="Directory for generated LaTeX table fragments",
    )
    parser.add_argument(
        "--write-tex",
        choices=["calibration-iaa", "exploratory-concordance"],
        default=None,
        help="Write a thesis table fragment for partial human validation reporting",
    )
    parser.add_argument(
        "--single-rater",
        default=None,
        help="Restrict human--LLM concordance to one rater (e.g. a)",
    )
    args = parser.parse_args()

    if not args.merged.exists():
        print(f"No merged ratings at {args.merged}; run collect_human_ratings.py after rating.")
        sys.exit(0)

    merged = pd.read_csv(args.merged)
    subsample_path = args.subsample
    if not subsample_path.exists():
        print(f"Missing subsample {subsample_path}")
        sys.exit(1)
    subsample = pd.read_csv(subsample_path, low_memory=False)

    split_filter = None if args.split == "all" else args.split
    tag = args.split

    hh = human_human_metrics(merged, split_filter)
    if args.single_rater:
        hl = single_rater_human_llm_metrics(
            merged, subsample, split_filter or args.split, rater_id=args.single_rater
        )
    else:
        hl = human_llm_metrics(merged, subsample, split_filter)
    logit = logit_audit_metrics(merged, subsample, split_filter)
    prev = prevalence_threshold_check(hl) if not hl.empty else pd.DataFrame()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.thesis_tables.mkdir(parents=True, exist_ok=True)

    suffix = "" if tag == "all" else f"_{tag}"
    hh.to_csv(args.thesis_tables / f"human_validation_iaa{suffix}.csv", index=False)
    hl.to_csv(args.thesis_tables / f"human_validation_concordance{suffix}.csv", index=False)
    if not logit.empty:
        logit.to_csv(args.thesis_tables / f"human_validation_logit_audit{suffix}.csv", index=False)
    if not prev.empty:
        prev.to_csv(args.thesis_tables / f"human_validation_prevalence{suffix}.csv", index=False)

    write_summary_md(args.output_dir / f"analysis_summary{suffix}.md", hh, hl, logit, tag)

    summary = {
        "split": tag,
        "success_criteria": {
            "alpha_ord": SUCCESS_ALPHA_ORD,
            "weighted_kappa": SUCCESS_KAPPA,
            "spearman_rho": SUCCESS_RHO,
            "prevalence_pp": SUCCESS_PREVALENCE_PP,
            "logit_kappa": SUCCESS_LOGIT_KAPPA,
        },
        "human_human": hh.to_dict(orient="records"),
        "human_llm": hl.to_dict(orient="records"),
    }
    (args.output_dir / f"analysis_summary{suffix}.json").write_text(
        json.dumps(summary, indent=2)
    )
    if args.write_tex == "calibration-iaa":
        write_calibration_iaa_tex(hh, args.thesis_generated)
        print(f"Wrote calibration IAA TeX -> {args.thesis_generated}")
    elif args.write_tex == "exploratory-concordance":
        write_exploratory_concordance_tex(
            hl, args.thesis_generated, rater_id=args.single_rater or "a"
        )
        print(f"Wrote exploratory concordance TeX -> {args.thesis_generated}")

    print(f"Wrote analysis for split={tag} -> {args.thesis_tables}")


if __name__ == "__main__":
    main()
