"""Objective per-steering-strength metrics and figures."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import STEERING_MULTIPLIERS


def text_similarity(a: str, b: str) -> float:
    if not a or not b or a == "N/A" or b == "N/A":
        return np.nan
    return SequenceMatcher(None, str(a).strip(), str(b).strip()).ratio()


def compute_row_metrics(row: pd.Series) -> dict:
    baseline = str(row.get("generation_0.0", "") or "")
    baseline_len = max(len(baseline), 1)
    baseline_logit = pd.to_numeric(row.get("logit_0.0"), errors="coerce")
    out = {}

    pos_devs, neg_devs = [], []
    for alpha in STEERING_MULTIPLIERS:
        if alpha == "0.0":
            continue
        gen = str(row.get(f"generation_{alpha}", "") or "")
        logit = pd.to_numeric(row.get(f"logit_{alpha}"), errors="coerce")
        prefix = alpha.replace(".", "_").replace("-", "neg")

        out[f"token_len_ratio_{prefix}"] = len(gen) / baseline_len if gen else np.nan
        out[f"text_similarity_{prefix}"] = text_similarity(baseline, gen)
        out[f"text_deviation_{prefix}"] = 1.0 - out[f"text_similarity_{prefix}"]
        if pd.notna(logit) and pd.notna(baseline_logit):
            out[f"abs_logit_delta_{prefix}"] = abs(logit - baseline_logit)
        else:
            out[f"abs_logit_delta_{prefix}"] = np.nan

        dev = out[f"text_deviation_{prefix}"]
        if pd.notna(dev):
            if float(alpha) > 0:
                pos_devs.append(dev)
            elif float(alpha) < 0:
                neg_devs.append(dev)

    out["asymmetry_pos_minus_neg_dev"] = (
        np.mean(pos_devs) - np.mean(neg_devs)
        if pos_devs and neg_devs
        else np.nan
    )
    return out


def aggregate_by_alpha(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    non_baseline = [a for a in STEERING_MULTIPLIERS if a != "0.0"]
    for alpha in non_baseline:
        prefix = alpha.replace(".", "_").replace("-", "neg")
        rows.append({
            "alpha": float(alpha),
            "mean_token_len_ratio": metrics_df[f"token_len_ratio_{prefix}"].mean(),
            "sem_token_len_ratio": metrics_df[f"token_len_ratio_{prefix}"].sem(),
            "mean_text_deviation": metrics_df[f"text_deviation_{prefix}"].mean(),
            "sem_text_deviation": metrics_df[f"text_deviation_{prefix}"].sem(),
            "mean_abs_logit_delta": metrics_df[f"abs_logit_delta_{prefix}"].mean(),
            "sem_abs_logit_delta": metrics_df[f"abs_logit_delta_{prefix}"].sem(),
            "n": len(metrics_df),
        })
    return pd.DataFrame(rows)


def plot_curves(agg: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    x = agg["alpha"].values

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].errorbar(
        x, agg["mean_token_len_ratio"], yerr=agg["sem_token_len_ratio"],
        marker="o", capsize=3, color="steelblue",
    )
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Steering multiplier (α)")
    axes[0].set_ylabel("Mean token length ratio vs baseline")
    axes[0].set_title("Token inflation proxy")
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(
        x, agg["mean_text_deviation"], yerr=agg["sem_text_deviation"],
        marker="o", capsize=3, color="coral",
    )
    axes[1].set_xlabel("Steering multiplier (α)")
    axes[1].set_ylabel("Mean text deviation (1 − similarity)")
    axes[1].set_title("Text change vs baseline")
    axes[1].grid(True, alpha=0.3)

    axes[2].errorbar(
        x, agg["mean_abs_logit_delta"], yerr=agg["sem_abs_logit_delta"],
        marker="o", capsize=3, color="teal",
    )
    axes[2].set_xlabel("Steering multiplier (α)")
    axes[2].set_ylabel("Mean |Δ logit| vs baseline")
    axes[2].set_title("Logit shift magnitude")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Automated per-α metrics (Pass-2 flagged subset)", y=1.02)
    plt.tight_layout()
    fig.savefig(figures_dir / "per_alpha_objective_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_per_alpha_metrics(
    pass2_path: Path,
    output_dir: Path,
    max_rows: int | None = None,
) -> None:
    print(f"Loading {pass2_path} for per-α metrics...")
    df = pd.read_csv(pass2_path, low_memory=False)
    if max_rows:
        df = df.head(max_rows)

    print(f"Computing per-row metrics for {len(df)} rows...")
    metrics = df.apply(compute_row_metrics, axis=1, result_type="expand")
    metrics_df = pd.concat([df[["dataset_name", "sample_idx"]], metrics], axis=1)

    agg = aggregate_by_alpha(metrics_df)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)

    agg.to_csv(tables_dir / "per_alpha_objective_metrics.csv", index=False)
    metrics_df.to_csv(tables_dir / "per_alpha_objective_metrics_by_sample.csv", index=False)
    print(f"Saved per-α tables to {tables_dir}")

    plot_curves(agg, figures_dir)
    print(f"Saved per-α figures to {figures_dir}")
