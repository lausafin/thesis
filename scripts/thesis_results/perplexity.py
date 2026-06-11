"""Perplexity vs Pass-2 side-effect correlation analysis."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .constants import (
    DIMENSION_LABELS,
    PASS2_EFFECT_KEYS,
    PRIMARY_DIMENSIONS,
    STEERING_MULTIPLIERS_FLOAT,
)

PPL_METRICS = ["max_abs_ppl_delta", "mean_steered_ppl", "baseline_ppl"]


def _format_multiplier(mult: float) -> str:
    mult_str = f"{float(mult):.1f}"
    if mult_str == "-0.0":
        mult_str = "0.0"
    return mult_str


def load_perplexity_from_directory(base_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    if not base_dir.is_dir():
        print(f"Warning: generations directory not found: {base_dir}")
        return pd.DataFrame()

    subdirs = sorted(d for d in os.listdir(base_dir) if (base_dir / d).is_dir())

    for dataset_name in subdirs:
        dataset_path = base_dir / dataset_name
        json_files = glob.glob(str(dataset_path / "*_generations.json"))
        if not json_files:
            continue

        json_file = sorted(json_files)[-1]
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: skipping {json_file}: {exc}")
            continue

        for item in data:
            record = {
                "dataset_name": dataset_name,
                "sample_idx": item.get("sample_idx"),
            }
            completions = item.get("completions", {})
            for val in completions.values():
                if not isinstance(val, dict):
                    continue
                mult = val.get("multiplier")
                if mult is None:
                    continue
                mult_str = _format_multiplier(mult)
                ppl = val.get("perplexity")
                if ppl is not None and ppl != float("inf") and not np.isnan(ppl):
                    record[f"perplexity_{mult_str}"] = float(ppl)
            records.append(record)

    df = pd.DataFrame(records)
    print(f"Loaded perplexity for {len(df)} samples from {len(subdirs)} dataset dirs.")
    return df


def compute_perplexity_features(row: pd.Series) -> dict[str, float]:
    ppls: dict[float, float] = {}
    for mult in STEERING_MULTIPLIERS_FLOAT:
        col = f"perplexity_{_format_multiplier(mult)}"
        val = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(val):
            ppls[mult] = float(val)

    out: dict[str, float] = {
        "max_abs_ppl_delta": np.nan,
        "mean_steered_ppl": np.nan,
        "baseline_ppl": np.nan,
    }
    if 0.0 not in ppls:
        return out

    baseline = ppls[0.0]
    out["baseline_ppl"] = baseline
    steered = [ppls[m] for m in STEERING_MULTIPLIERS_FLOAT if m != 0.0 and m in ppls]
    if steered:
        out["mean_steered_ppl"] = float(np.mean(steered))
        out["max_abs_ppl_delta"] = float(max(abs(p - baseline) for p in steered))
    return out


def has_parseable_pass2_scores(row: pd.Series) -> bool:
    for dim in PASS2_EFFECT_KEYS:
        val = pd.to_numeric(row.get(f"judge_effect_{dim}"), errors="coerce")
        if pd.notna(val):
            return True
    return False


def build_merged_df(
    generations_dir: Path,
    pass2_path: Path,
    max_rows: int | None = None,
) -> pd.DataFrame:
    ppl_df = load_perplexity_from_directory(generations_dir)
    print(f"Loading {pass2_path}...")
    pass2_df = pd.read_csv(pass2_path, low_memory=False)
    if max_rows:
        pass2_df = pass2_df.head(max_rows)

    merged = pass2_df.merge(ppl_df, on=["dataset_name", "sample_idx"], how="left")
    features = merged.apply(compute_perplexity_features, axis=1, result_type="expand")
    merged = pd.concat([merged, features], axis=1)
    merged = merged[merged["baseline_ppl"].notna()].copy()
    merged["pass2_scored"] = merged.apply(has_parseable_pass2_scores, axis=1)
    return merged


def aggregate_perplexity_by_alpha(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    rows = []
    for mult in STEERING_MULTIPLIERS_FLOAT:
        col = f"perplexity_{_format_multiplier(mult)}"
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        rows.append({
            "subset": subset,
            "alpha": float(mult),
            "mean_perplexity": vals.mean(),
            "sem_perplexity": vals.sem(),
            "n": len(vals),
        })
    return pd.DataFrame(rows)


def compute_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    scored = merged[merged["pass2_scored"]].copy()
    rows = []
    for dim in PASS2_EFFECT_KEYS:
        col = f"judge_effect_{dim}"
        severity = pd.to_numeric(scored[col], errors="coerce")
        for metric in PPL_METRICS:
            x = pd.to_numeric(scored[metric], errors="coerce")
            mask = x.notna() & severity.notna()
            n = int(mask.sum())
            if n < 3:
                rows.append({
                    "dimension": dim,
                    "perplexity_metric": metric,
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                    "spearman_r": np.nan,
                    "spearman_p": np.nan,
                    "n": n,
                })
                continue
            pr, pp = pearsonr(x[mask], severity[mask])
            sr, sp = spearmanr(x[mask], severity[mask])
            rows.append({
                "dimension": dim,
                "perplexity_metric": metric,
                "pearson_r": pr,
                "pearson_p": pp,
                "spearman_r": sr,
                "spearman_p": sp,
                "n": n,
            })
    return pd.DataFrame(rows)


def plot_perplexity_curve(
    by_alpha_corpus: pd.DataFrame,
    by_alpha_pass2: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for subset_df, label, color in [
        (by_alpha_corpus, "Full corpus", "steelblue"),
        (by_alpha_pass2, "Pass-2-evaluated subset", "coral"),
    ]:
        x = subset_df["alpha"].values
        y = subset_df["mean_perplexity"].values
        yerr = subset_df["sem_perplexity"].values
        ax.errorbar(
            x, y, yerr=yerr, marker="o", capsize=3, linewidth=2,
            label=label, color=color,
        )

    if 0.0 in by_alpha_corpus["alpha"].values:
        baseline = by_alpha_corpus.loc[
            by_alpha_corpus["alpha"] == 0.0, "mean_perplexity"
        ].iloc[0]
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.set_xlabel("Steering multiplier (α)")
    ax.set_ylabel("Mean generation perplexity")
    ax.set_title("Corpus-mean perplexity stays nearly flat across α")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_scatter_hexbin(merged: pd.DataFrame, out_path: Path) -> None:
    scored = merged[merged["pass2_scored"]].copy()
    n_scored = len(scored)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes_flat = axes.flatten()

    for ax, dim in zip(axes_flat, PRIMARY_DIMENSIONS):
        col = f"judge_effect_{dim}"
        x = pd.to_numeric(scored["max_abs_ppl_delta"], errors="coerce")
        y = pd.to_numeric(scored[col], errors="coerce")
        mask = x.notna() & y.notna()
        x_vals = x[mask].values
        y_vals = y[mask].values

        hb = ax.hexbin(
            x_vals, y_vals, gridsize=35, cmap="Blues", mincnt=1, linewidths=0.2,
        )
        pr, _ = pearsonr(x_vals, y_vals)
        label = DIMENSION_LABELS.get(dim, dim.replace("_", " ").title())
        ax.set_title(label)
        ax.set_xlabel("Max |Δperplexity| from baseline")
        ax.set_ylabel("Overall Pass-2 severity (0–5)")
        ax.set_ylim(-0.2, 5.2)
        ax.text(
            0.97, 0.97, f"ρ = {pr:.2f}\nn = {len(x_vals):,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )
        fig.colorbar(hb, ax=ax, label="Count")

    fig.suptitle(
        f"Perplexity change vs overall Pass-2 severity (Pass-2-evaluated, N={n_scored:,})",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_perplexity_analysis(
    generations_dir: Path,
    pass2_path: Path,
    output_dir: Path,
    max_rows: int | None = None,
) -> None:
    if not generations_dir.is_dir():
        print(f"Skipping perplexity analysis: {generations_dir} not found.")
        return

    merged = build_merged_df(generations_dir, pass2_path, max_rows)
    scored = merged[merged["pass2_scored"]]
    print(f"Pass-2 scored rows with perplexity: {len(scored)}")

    by_alpha_corpus = aggregate_perplexity_by_alpha(merged, "full_corpus")
    by_alpha_pass2 = aggregate_perplexity_by_alpha(scored, "pass2_scored")
    by_alpha = pd.concat([by_alpha_corpus, by_alpha_pass2], ignore_index=True)
    correlations = compute_correlations(merged)

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    correlations.to_csv(tables_dir / "perplexity_side_effect_correlations.csv", index=False)
    by_alpha.to_csv(tables_dir / "perplexity_by_alpha.csv", index=False)
    print(f"Saved perplexity tables to {tables_dir}")

    plot_perplexity_curve(
        by_alpha_corpus,
        by_alpha_pass2,
        figures_dir / "perplexity_vs_steering_strength.png",
    )
    plot_scatter_hexbin(merged, figures_dir / "perplexity_vs_side_effects_scatter.png")
    print(f"Saved perplexity figures to {figures_dir}")

    primary = correlations[
        (correlations["dimension"].isin(PRIMARY_DIMENSIONS))
        & (correlations["perplexity_metric"] == "max_abs_ppl_delta")
    ]
    print("\nPrimary dimensions (max |Δppl| vs overall severity):")
    for _, row in primary.iterrows():
        print(f"  {row['dimension']:25s} ρ = {row['pearson_r']:.3f}  n = {int(row['n'])}")
