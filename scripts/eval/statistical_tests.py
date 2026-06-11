#!/usr/bin/env python3
"""
Statistical significance testing for steering effects on discourse dimensions.

For each (dataset, dimension) pair:
- Runs one-sample Wilcoxon signed-rank test against 0 using Pratt's method
- Computes mean severity score
- Applies Bonferroni and Benjamini-Hochberg corrections across all tests

Usage:
    python statistical_tests.py [input_csv] [output_dir]
"""

import os
import sys

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon

from pipeline.constants import PASS2_CATEGORY_EFFECTS
from scripts.eval.practical_significance import add_cohens_d, export_practical_significance

ALPHA = 0.05

# =============================================================================
# Statistical Tests
# =============================================================================

def run_wilcoxon_test(scores):
    """
    Run one-sample Wilcoxon signed-rank test on severity scores against 0.
    Uses Pratt's method to appropriately handle zero-differences (ties at 0).
    Returns (statistic, p_value) or (NaN, 1.0) if the test cannot be run.
    """
    # Wilcoxon requires at least one non-zero difference
    if (scores == 0).all():
        return np.nan, 1.0
    try:
        # We test if the distribution of scores is significantly > 0
        stat, p_val = wilcoxon(scores, alternative='greater', zero_method='pratt')
        return stat, p_val
    except ValueError:
        return np.nan, np.nan


def run_all_tests(df):
    """
    Run Wilcoxon signed-rank test for each (dataset, dimension) pair.
    Returns a DataFrame with one row per test.
    """
    results = []

    datasets = sorted(df['dataset_name'].dropna().unique())

    for dataset in datasets:
        df_sub = df[df['dataset_name'] == dataset]

        for category, dimensions in PASS2_CATEGORY_EFFECTS.items():
            for dim in dimensions:
                col = f"judge_effect_{dim}"

                if col not in df_sub.columns:
                    continue

                scores = pd.to_numeric(df_sub[col], errors='coerce')

                # Drop rows with NaN scores
                scores = scores[scores.notna()]

                n_total = len(scores)
                if n_total < 2:
                    continue

                # Count non-zero severities
                n_nonzero = int((scores > 0).sum())

                # Compute stats
                mean_score = scores.mean()
                std_score = scores.std(ddof=1)

                w_stat, p_raw = run_wilcoxon_test(scores)

                results.append({
                    'category': category.replace('_', ' ').title(),
                    'dataset': dataset,
                    'dimension': dim.replace('_', ' '),
                    'dimension_raw': dim,
                    'n_total': n_total,
                    'n_nonzero': n_nonzero,
                    'pct_nonzero': round(n_nonzero / n_total * 100, 2),
                    'mean_severity': round(mean_score, 4),
                    'std_severity': round(std_score, 4) if pd.notna(std_score) else np.nan,
                    'wilcoxon_stat': w_stat,
                    'p_raw': p_raw,
                })

    return pd.DataFrame(results)


def apply_corrections(results_df):
    """Apply Bonferroni and Benjamini-Hochberg corrections."""
    valid = results_df['p_raw'].notna()
    p_values = results_df.loc[valid, 'p_raw'].values
    n_tests = len(p_values)

    print(f"\n  Total tests: {n_tests}")

    bonferroni_adjusted = np.minimum(p_values * n_tests, 1.0)
    bh_adjusted = stats.false_discovery_control(p_values, method='bh')

    results_df.loc[valid, 'p_bonferroni'] = bonferroni_adjusted
    results_df.loc[valid, 'p_bh'] = bh_adjusted
    results_df.loc[valid, 'sig_bonferroni'] = bonferroni_adjusted < ALPHA
    results_df.loc[valid, 'sig_bh'] = bh_adjusted < ALPHA

    # For invalid rows
    results_df.loc[~valid, 'p_bonferroni'] = np.nan
    results_df.loc[~valid, 'p_bh'] = np.nan
    results_df.loc[~valid, 'sig_bonferroni'] = False
    results_df.loc[~valid, 'sig_bh'] = False

    return results_df


# =============================================================================
# Summary Reports
# =============================================================================

def print_bh_significant_by_dataset(results_df):
    """
    For each dataset, print dimensions that are Benjamini–Hochberg significant,
    sorted by Mean Severity in decreasing order.
    """
    print(f"\n{'='*80}")
    print("BH-SIGNIFICANT DIMENSIONS BY DATASET (Mean Severity, decreasing)")
    print(f"{'='*80}")

    for ds in sorted(results_df['dataset'].unique(), key=lambda x: str(x)):
        sub = results_df[(results_df['dataset'] == ds) & (results_df['sig_bh'])].copy()
        print(f"\n{ds}")
        if sub.empty:
            print("  (none)")
            continue
        sub = sub.sort_values('mean_severity', ascending=False)
        print(f"  {'Dimension':<30} {'Severity':>9} {'p_bh':>11} {'n':>6}")
        print(f"  {'-'*58}")
        for _, row in sub.iterrows():
            print(
                f"  {row['dimension']:<30} {row['mean_severity']:>9.4f} "
                f"{row['p_bh']:>11.4e} {int(row['n_total']):>6}"
            )


def print_summary(results_df):
    """Print summary of significant results."""
    valid = results_df['p_raw'].notna()
    n_total = valid.sum()

    n_sig_raw = (results_df.loc[valid, 'p_raw'] < ALPHA).sum()
    n_sig_bonf = results_df['sig_bonferroni'].sum()
    n_sig_bh = results_df['sig_bh'].sum()

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests:                         {n_total}")
    print(f"Significant (raw p < {ALPHA}):          {n_sig_raw} ({n_sig_raw/n_total*100:.1f}%)")
    print(f"Significant (Bonferroni):             {n_sig_bonf} ({n_sig_bonf/n_total*100:.1f}%)")
    print(f"Significant (Benjamini-Hochberg):     {n_sig_bh} ({n_sig_bh/n_total*100:.1f}%)")

    # Per-dimension summary
    print(f"\n{'='*80}")
    print(f"PER-DIMENSION SUMMARY (Benjamini-Hochberg)")
    print(f"{'='*80}")
    print(f"{'Dimension':<30} {'N_tests':>8} {'N_sig':>6} {'%sig':>6} {'Mean Sev':>9} {'Max Sev':>9}")
    print("-" * 75)

    for dim in sorted(results_df['dimension'].unique()):
        subset = results_df[results_df['dimension'] == dim]
        n_tests = len(subset)
        n_sig = subset['sig_bh'].sum()
        mean_sev = subset['mean_severity'].mean()
        max_sev = subset['mean_severity'].max()
        print(f"{dim:<30} {n_tests:>8} {n_sig:>6} {n_sig/n_tests*100:>5.1f}% {mean_sev:>9.3f} {max_sev:>9.3f}")

    # Per-dataset summary
    print(f"\n{'='*80}")
    print(f"PER-DATASET SUMMARY (Benjamini-Hochberg)")
    print(f"{'='*80}")
    print(f"{'Dataset':<45} {'N_tests':>8} {'N_sig':>6} {'%sig':>6} {'Mean Sev':>9}")
    print("-" * 80)

    for ds in sorted(results_df['dataset'].unique()):
        subset = results_df[results_df['dataset'] == ds]
        n_tests = len(subset)
        n_sig = subset['sig_bh'].sum()
        mean_sev = subset['mean_severity'].mean()
        print(f"{ds:<45} {n_tests:>8} {n_sig:>6} {n_sig/n_tests*100:>5.1f}% {mean_sev:>9.3f}")

    # Top significant results by effect size
    print(f"\n{'='*80}")
    print(f"TOP 20 SIGNIFICANT RESULTS BY MEAN SEVERITY (Benjamini-Hochberg)")
    print(f"{'='*80}")
    sig = results_df[results_df['sig_bh'] == True].copy()
    top = sig.nlargest(20, 'mean_severity')
    print(f"{'Dataset':<35} {'Dimension':<25} {'Sev':>7} {'p_bh':>10} {'N':>5}")
    print("-" * 85)
    for _, row in top.iterrows():
        print(f"{row['dataset']:<35} {row['dimension']:<25} {row['mean_severity']:>7.3f} {row['p_bh']:>10.2e} {row['n_total']:>5}")

    print_bh_significant_by_dataset(results_df)


# =============================================================================
# Main
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python statistical_tests.py [input_csv] [output_dir]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'significance_tests/'

    os.makedirs(output_dir, exist_ok=True)
    print(f"Input:  {input_file}")
    print(f"Output: {output_dir}")
    print(f"Alpha:  {ALPHA}\n")

    # Load data
    print("Loading data...")
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows.")

    # Run tests
    print("\nRunning Wilcoxon signed-rank tests...")
    results_df = run_all_tests(df)
    if results_df.empty:
        print("No valid tests were run. Check that the input CSV contains columns like 'judge_effect_<dimension>'.")
        return
        
    print(f"  Computed {len(results_df)} tests")

    # Apply corrections
    print("\nApplying multiple testing corrections...")
    results_df = apply_corrections(results_df)
    results_df = add_cohens_d(results_df)

    # Save full results
    output_cols = [
        'category', 'dataset', 'dimension',
        'n_total', 'n_nonzero', 'pct_nonzero',
        'mean_severity', 'std_severity', 'cohens_d',
        'wilcoxon_stat', 'p_raw', 'p_bonferroni', 'p_bh',
        'sig_bonferroni', 'sig_bh'
    ]
    results_path = os.path.join(output_dir, 'significance_results.csv')
    results_df[output_cols].to_csv(results_path, index=False)
    print(f"\nSaved full results: {results_path}")

    # Save significant-only results
    sig_path = os.path.join(output_dir, 'significance_results_bh_significant.csv')
    sig_df = results_df[results_df['sig_bh'] == True][output_cols]
    sig_df.to_csv(sig_path, index=False)
    print(f"Saved BH-significant results: {sig_path} ({len(sig_df)} rows)")

    # Practical significance tiers and sensitivity tables
    print("\nExporting practical significance analysis...")
    tier_stats = export_practical_significance(results_df, output_dir)
    print(
        f"  Primary claims: {tier_stats['n_primary_claims']}/"
        f"{tier_stats['n_bh_significant']} BH-significant"
    )

    # Print summary
    print_summary(results_df)


if __name__ == '__main__':
    main()
