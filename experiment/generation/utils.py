"""
Utility functions for steering experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


def plot_perplexity_vs_steering(
    generation_results: List[Dict],
    multipliers: List[float],
    save_path: str = "perplexity_vs_steering.png"
):
    """
    Plot perplexity vs steering strength.
    
    Args:
        generation_results: List of generation results (from generate_texts_with_perplexity)
        multipliers: List of steering multipliers
        save_path: Path to save the plot
    """
    # Calculate mean and std perplexity for each multiplier
    mean_perplexities = []
    std_perplexities = []
    
    for mult in multipliers:
        key = f'multiplier_{mult}'
        perplexities = [
            r['completions'][key]['perplexity'] 
            for r in generation_results 
            if r['completions'][key]['perplexity'] != float('inf')
        ]
        if perplexities:
            mean_perplexities.append(np.mean(perplexities))
            std_perplexities.append(np.std(perplexities))
        else:
            mean_perplexities.append(np.nan)
            std_perplexities.append(np.nan)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot with error bars
    ax.errorbar(
        multipliers, mean_perplexities, yerr=std_perplexities,
        fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2,
        color='purple', ecolor='purple', alpha=0.8
    )
    
    # Mark baseline (multiplier = 0)
    if 0.0 in multipliers:
        baseline_idx = multipliers.index(0.0)
        ax.axhline(y=mean_perplexities[baseline_idx], color='gray', linestyle='--', 
                   alpha=0.5, label=f'Baseline (λ=0): {mean_perplexities[baseline_idx]:.2f}')
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Steering Multiplier (λ)', fontsize=12)
    ax.set_ylabel('Mean Perplexity', fontsize=12)
    ax.set_title('Perplexity vs Steering Strength\n(lower = more fluent/confident)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add annotations for negative/positive steering regions
    ax.annotate('← Counter-steering', xy=(min(multipliers), ax.get_ylim()[1]), 
                fontsize=10, color='blue', alpha=0.7)
    ax.annotate('Steering →', xy=(max(multipliers) - 0.5, ax.get_ylim()[1]), 
                fontsize=10, color='red', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Perplexity plot saved to {save_path}")


def plot_results(results: Dict, save_path: str = "steering_results.png"):
    """Plot the steering results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Propensity curve (mean logit diff vs multiplier)
    ax1 = axes[0]
    ax1.plot(results['multipliers'], results['mean_logit_diffs'], 'b-o', linewidth=2, markersize=8)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Steering Multiplier (λ)', fontsize=12)
    ax1.set_ylabel('Mean Logit Difference', fontsize=12)
    ax1.set_title(f'Propensity Curve\n(Steerability: {results["aggregate_steerability"]:.3f})', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Per-sample steerability distribution
    ax2 = axes[1]
    steerabilities = results['per_sample_steerability']
    ax2.hist(steerabilities, bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero steerability')
    ax2.axvline(x=np.median(steerabilities), color='green', linestyle='-', linewidth=2, label=f'Median: {np.median(steerabilities):.3f}')
    ax2.set_xlabel('Per-sample Steerability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Per-sample Steerability', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Fraction of anti-steerable samples
    ax3 = axes[2]
    anti_steerable = sum(1 for s in steerabilities if s < 0)
    total = len(steerabilities)
    ax3.bar(['Anti-steerable\n(< 0)', 'Steerable\n(≥ 0)'], 
            [anti_steerable, total - anti_steerable],
            color=['red', 'green'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title(f'Anti-steerable Samples: {anti_steerable}/{total} ({100*anti_steerable/total:.1f}%)', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")
