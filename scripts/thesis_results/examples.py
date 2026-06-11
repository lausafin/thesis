"""Qualitative example extraction for thesis appendix case studies."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def extract_qualitative_examples(pass2_df: pd.DataFrame, output_dir: Path) -> None:
    print("Extracting qualitative examples...")
    out_dir = output_dir / "examples"

    def save_examples(df_subset, sub_dir, sort_col, n=5):
        (out_dir / sub_dir).mkdir(parents=True, exist_ok=True)
        top_examples = df_subset.sort_values(sort_col, ascending=False).head(n)

        for i, (_, row) in enumerate(top_examples.iterrows()):
            with open(out_dir / sub_dir / f"example_{i+1}_{row['dataset_name']}.md", "w") as f:
                f.write(f"# Example: {sub_dir.replace('_', ' ').title()}\n")
                f.write(f"**Dataset:** {row['dataset_name']}\n")
                f.write(f"**Severity ({sort_col.replace('judge_effect_', '')}):** {row[sort_col]}\n\n")
                f.write(f"### Question\n```text\n{row['question']}\n```\n\n")
                f.write(f"### Baseline (0.0)\n```text\n{row.get('generation_0.0', 'N/A')}\n```\n\n")
                f.write(f"### Negative Steering (-2.0)\n```text\n{row.get('generation_-2.0', 'N/A')}\n```\n\n")
                f.write(f"### Positive Steering (+2.0)\n```text\n{row.get('generation_2.0', 'N/A')}\n```\n\n")
                f.write(f"### Judge Rationale\n```text\n{row.get('judge_raw_evidence', 'N/A')}\n```\n")

    df = pass2_df.copy()
    df["judge_effect_steering_asymmetry"] = pd.to_numeric(df["judge_effect_steering_asymmetry"], errors="coerce")
    save_examples(df[df["judge_effect_steering_asymmetry"] >= 4], "directional_asymmetry", "judge_effect_steering_asymmetry")

    df["judge_effect_logit_text_decoupling"] = pd.to_numeric(df["judge_effect_logit_text_decoupling"], errors="coerce")
    save_examples(df[df["judge_effect_logit_text_decoupling"] >= 3], "logit_text_decoupling", "judge_effect_logit_text_decoupling")

    df["judge_effect_label_content_contradiction"] = pd.to_numeric(df["judge_effect_label_content_contradiction"], errors="coerce")
    save_examples(df[df["judge_effect_label_content_contradiction"] >= 4], "high_intensity_degradation", "judge_effect_label_content_contradiction")
