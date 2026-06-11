"""Figure generation for thesis results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .constants import STEERING_MULTIPLIERS


def generate_figures(sig_df: pd.DataFrame, pass1_df: pd.DataFrame, pass2_df: pd.DataFrame, output_dir: Path) -> None:
    print("Generating figures...")
    out_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmap_data = sig_df.pivot(index="dataset", columns="dimension", values="mean_severity")
    plt.figure(figsize=(16, 12))
    sns.heatmap(heatmap_data, cmap="YlOrRd", annot=False, cbar_kws={"label": "Mean Severity"})
    plt.title("Mean Effect Severity by Dataset and Dimension")
    plt.tight_layout()
    plt.savefig(out_dir / "severity_heatmap_dataset_dimension.png", dpi=300)
    plt.close()

    dim_summary = sig_df.groupby("dimension")["mean_severity"].mean().sort_values(ascending=True)
    plt.figure(figsize=(10, 8))
    dim_summary.plot(kind="barh", color="steelblue")
    plt.title("Average Mean Severity Across All Datasets")
    plt.xlabel("Mean Severity")
    plt.tight_layout()
    plt.savefig(out_dir / "top_dimensions_bar.png", dpi=300)
    plt.close()

    run_summary_path = tables_dir / "run_summary.csv"
    if run_summary_path.exists():
        run_summary = pd.read_csv(run_summary_path)
        run_summary = run_summary.sort_values("pct_flagged", ascending=True)
        plt.figure(figsize=(10, 14))
        plt.barh(run_summary["dataset"], run_summary["pct_flagged"], color="coral")
        plt.xlabel("Pass-1 Flag Rate (%)")
        plt.title("Share of Samples Flagged in Pass 1 by Dataset")
        plt.tight_layout()
        plt.savefig(out_dir / "pass1_flag_rate_by_dataset.png", dpi=300)
        plt.close()

    if "logit_0.0" in pass2_df.columns:
        curves = []
        for alpha in STEERING_MULTIPLIERS:
            logit_col = f"logit_{alpha}"
            if logit_col not in pass2_df.columns:
                continue
            baseline = pd.to_numeric(pass2_df["logit_0.0"], errors="coerce")
            steered = pd.to_numeric(pass2_df[logit_col], errors="coerce")
            delta = (steered - baseline).abs()
            curves.append({
                "alpha": float(alpha),
                "mean_abs_logit_delta": delta.mean(),
            })
        if curves:
            curve_df = pd.DataFrame(curves)
            plt.figure(figsize=(8, 5))
            plt.plot(curve_df["alpha"], curve_df["mean_abs_logit_delta"], marker="o", linewidth=2)
            plt.xlabel("Steering Multiplier (α)")
            plt.ylabel("Mean |Δ logit| vs baseline (α=0)")
            plt.title("Logit Shift Magnitude vs Steering Strength (Pass-2 Subset)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "dimension_vs_steering_strength.png", dpi=300)
            plt.close()

    effect_cols = [c for c in pass2_df.columns if str(c).startswith("judge_effect_")]
    top_dims = (
        sig_df.groupby("dimension")["mean_severity"]
        .mean()
        .sort_values(ascending=False)
        .head(8)
        .index.tolist()
    )
    n_plots = len(top_dims)
    n_cols = 4
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
    axes = np.array(axes).reshape(-1)
    for i, dim in enumerate(top_dims):
        col = f"judge_effect_{dim.replace(' ', '_')}"
        if col not in pass2_df.columns:
            continue
        scores = pd.to_numeric(pass2_df[col], errors="coerce").dropna()
        axes[i].hist(scores, bins=6, range=(0, 5), color="steelblue", edgecolor="white")
        axes[i].set_title(dim[:28], fontsize=9)
        axes[i].set_xlabel("Severity")
    for j in range(n_plots, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Pass-2 Severity Distributions (Flagged Samples)", y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "score_distribution_by_dimension.png", dpi=300, bbox_inches="tight")
    plt.close()

    effect_means = []
    for col in effect_cols:
        dim = col.replace("judge_effect_", "").replace("_", " ")
        per_ds = pass2_df.groupby("dataset_name")[col].apply(
            lambda s: pd.to_numeric(s, errors="coerce").mean()
        )
        effect_means.append({
            "dimension": dim,
            "mean": per_ds.mean(),
            "sem": per_ds.sem(),
        })
    em_df = pd.DataFrame(effect_means).sort_values("mean", ascending=True)
    plt.figure(figsize=(10, 10))
    plt.barh(em_df["dimension"], em_df["mean"], xerr=em_df["sem"], color="teal", capsize=2)
    plt.xlabel("Mean Severity (Pass-2 Flagged Samples)")
    plt.title("Effect Severity by Dimension (± SEM across datasets)")
    plt.tight_layout()
    plt.savefig(out_dir / "dimension_effect_bar_with_se.png", dpi=300)
    plt.close()
