"""
Steering Vectors Experiment
===========================
Replicating: "Analyzing the Generalization and Reliability of Steering Vectors" (arXiv:2407.12404)

This script:
1. Extracts steering vectors using Contrastive Activation Addition (CAA)
2. Generates text with and without steering across different multipliers
3. Calculates perplexity for each completion
4. Evaluates and stores steerability results
"""

import os
import torch
import json
import random
import argparse
import numpy as np
from typing import List, Dict
from tqdm import tqdm

from data import (
    ContrastiveSample,
    load_myopic_reward_dataset,
    format_as_contrastive_prompt,
)
from extractor import SteeringVectorExtractor
from evaluator import SteeringEvaluator
from utils import plot_results, plot_perplexity_vs_steering


# ============================================================================
# Data Loading
# ============================================================================

def load_and_split_dataset(
    dataset_path: str,
    num_train: int,
    num_val: int,
    num_test: int,
) -> tuple[List[ContrastiveSample], List[ContrastiveSample], List[ContrastiveSample]]:
    """Load dataset and split into train/val/test sets."""
    print(f"Loading dataset from {dataset_path}...")
    raw_samples = load_myopic_reward_dataset(dataset_path)
    print(f"Loaded {len(raw_samples)} samples")
    
    random.shuffle(raw_samples)
    samples = [format_as_contrastive_prompt(s) for s in raw_samples]
    
    train_samples = samples[:num_train]
    val_samples = samples[num_train:num_train + num_val]
    test_samples = samples[num_train + num_val:num_train + num_val + num_test]
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    return train_samples, val_samples, test_samples


# ============================================================================
# Steering Vector Extraction
# ============================================================================

def find_optimal_layer(
    extractor: SteeringVectorExtractor,
    train_samples: List[ContrastiveSample],
    val_samples: List[ContrastiveSample],
    multipliers: List[float],
    specified_layer: int = None,
    layers_to_test: List[int] = None,
) -> int:
    """Find the optimal layer for steering via sweep or use specified layer."""
    if specified_layer is not None:
        print(f"Using specified layer: {specified_layer}")
        return specified_layer
    
    print("\n" + "="*60)
    print("Running layer sweep to find optimal layer...")
    print("="*60)
    
    # Default layers to test (middle layers typically work best)
    if layers_to_test is None:
        layers_to_test = [10, 11, 12, 13, 14, 15, 16, 18, 20]
    
    print(f"Testing layers: {layers_to_test}")
    print(f"Using multipliers: {multipliers}")
    
    evaluator = SteeringEvaluator(extractor)
    layer_steerabilities = {}
    
    for layer in layers_to_test:
        print(f"\n--- Testing Layer {layer} ---")
        sv = extractor.extract_steering_vector(train_samples, layer)
        
        # Evaluate each validation sample across multipliers
        generation_results = []
        for sample in tqdm(val_samples, desc=f"Layer {layer} eval"):
            result = evaluator.evaluate_sample_across_multipliers(
                sample, sv, layer, multipliers, max_new_tokens=5
            )
            generation_results.append(result)
    
        # Compute steerability from results
        steerability = evaluator.compute_steerability_from_results(
            generation_results, multipliers
        )
        layer_steerabilities[layer] = steerability['aggregate_steerability']
        print(f"Layer {layer} steerability: {steerability['aggregate_steerability']:.4f}")
    
    best_layer = max(layer_steerabilities, key=layer_steerabilities.get)
    print(f"\nBest layer: {best_layer} (steerability: {layer_steerabilities[best_layer]:.4f})")
    return best_layer


def extract_steering_vector(
    extractor: SteeringVectorExtractor,
    train_samples: List[ContrastiveSample],
    layer: int,
    save_path: str,
    model_name: str,
    batch_size: int = 1,
    normalize: bool = False,
) -> torch.Tensor:
    """Extract and save steering vector."""
    print("\n" + "="*60)
    print(f"Extracting steering vector at layer {layer}...")
    if batch_size > 1:
        print(f"Using batched extraction (batch_size={batch_size})")
    if normalize:
        print("Normalization: ENABLED (unit norm)")
    else:
        print("Normalization: DISABLED (natural magnitude)")
    print("="*60)
    
    # Use batched or sequential extraction based on batch_size
    if batch_size > 1:
        steering_vector = extractor.extract_steering_vector_batched(
            train_samples, layer, batch_size, normalize=normalize
        )
    else:
        steering_vector = extractor.extract_steering_vector(
            train_samples, layer, normalize=normalize
        )
    
    torch.save({
        'steering_vector': steering_vector,
        'layer': layer,
        'model_name': model_name,
        'num_train_samples': len(train_samples),
        'normalized': normalize,
    }, save_path)
    print(f"Steering vector saved to {save_path}")
    
    return steering_vector


# ============================================================================
# Text Generation with Perplexity
# ============================================================================

def generate_texts_with_perplexity(
    extractor: SteeringVectorExtractor,
    test_samples: List[ContrastiveSample],
    steering_vector: torch.Tensor,
    layer: int,
    multipliers: List[float],
    max_new_tokens: int,
    batch_size: int = 1,
) -> List[Dict]:
    """
    Generate text for each test sample across all multipliers.
    Each completion includes perplexity score.
    
    Args:
        extractor: The model extractor
        test_samples: Test samples to evaluate
        steering_vector: The steering vector
        layer: Layer to apply steering
        multipliers: List of multipliers (0.0 = no steering)
        max_new_tokens: Max tokens to generate
        batch_size: Number of samples to process at once (>1 enables batching)
        
    Returns:
        List of results with completions and perplexity for each sample
    """
    print("\n" + "="*60)
    print("Generating text with and without steering...")
    print(f"Multipliers: {multipliers}")
    if batch_size > 1:
        print(f"Using batched generation (batch_size={batch_size})")
    print("="*60)
    
    evaluator = SteeringEvaluator(extractor)
    
    # Use batched or sequential generation based on batch_size
    if batch_size > 1:
        results = evaluator.evaluate_samples_batched(
            test_samples, steering_vector, layer, multipliers, max_new_tokens, batch_size
        )
    else:
        results = []
        for i, sample in enumerate(tqdm(test_samples, desc="Generating texts")):
            sample_result = evaluator.evaluate_sample_across_multipliers(
                sample, steering_vector, layer, multipliers, max_new_tokens
            )
            sample_result['sample_idx'] = i
            results.append(sample_result)
    
    return results


# ============================================================================
# Steerability Evaluation
# ============================================================================

def compute_steerability_from_generation(
    extractor: SteeringVectorExtractor,
    generation_results: List[Dict],
    multipliers: List[float],
) -> Dict:
    """
    Compute steerability from already-generated results.
    
    This is efficient because it uses the logit_diffs computed during generation,
    avoiding extra forward passes.
    """
    print("\n" + "="*60)
    print("Computing steerability from generation results...")
    print("="*60)
    
    evaluator = SteeringEvaluator(extractor)
    return evaluator.compute_steerability_from_results(generation_results, multipliers)


# ============================================================================
# Results Saving and Printing
# ============================================================================

def save_generation_results(results: List[Dict], save_path: str):
    """Save generated texts and perplexity scores."""
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Generation results saved to {save_path}")


def save_steerability_results(
    results: Dict,
    model_name: str,
    layer: int,
    num_test: int,
    save_path: str,
):
    """Save steerability evaluation results."""
    anti_steerable = sum(1 for s in results['per_sample_steerability'] if s < 0)
    
    data = {
        'model': model_name,
        'layer': layer,
        'multipliers': results['multipliers'],
        'mean_logit_diffs': results['mean_logit_diffs'],
        'aggregate_steerability': results['aggregate_steerability'],
        'per_sample_steerability': results['per_sample_steerability'],
        'anti_steerable_count': anti_steerable,
        'total_test_samples': num_test,
    }
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Steerability results saved to {save_path}")


def print_perplexity_summary(generation_results: List[Dict], multipliers: List[float]):
    """Print perplexity summary across multipliers."""
    print("\n" + "="*60)
    print("PERPLEXITY SUMMARY")
    print("="*60)
    
    for mult in multipliers:
        key = f'multiplier_{mult}'
        perplexities = [
            r['completions'][key]['perplexity'] 
            for r in generation_results 
            if r['completions'][key]['perplexity'] != float('inf')
        ]
        if perplexities:
            label = "No steering" if mult == 0.0 else f"λ={mult}"
            print(f"\n{label}:")
            print(f"  Mean perplexity: {np.mean(perplexities):.4f}")
            print(f"  Std perplexity:  {np.std(perplexities):.4f}")


def print_steerability_summary(
    results: Dict,
    model_name: str,
    layer: int,
    num_test: int,
):
    """Print steerability summary."""
    print("\n" + "="*60)
    print("STEERABILITY SUMMARY")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Steering Layer: {layer}")
    print(f"Aggregate Steerability: {results['aggregate_steerability']:.4f}")
    print(f"Mean per-sample steerability: {np.mean(results['per_sample_steerability']):.4f}")
    print(f"Std per-sample steerability: {np.std(results['per_sample_steerability']):.4f}")
    
    anti_steerable = sum(1 for s in results['per_sample_steerability'] if s < 0)
    print(f"Anti-steerable samples: {anti_steerable}/{num_test} ({100*anti_steerable/num_test:.1f}%)")
    
    print("\nPropensity curve (multiplier -> mean logit diff):")
    for mult, ld in zip(results['multipliers'], results['mean_logit_diffs']):
        print(f"  λ={mult:+.1f}: {ld:+.4f}")


# ============================================================================
# Main
# ============================================================================

def main(
    dataset: str,
    model: str,
    layer: int,
    num_train: int,
    num_val: int,
    num_test: int,
    max_new_tokens: int,
    multipliers: List[float],
    output_dir: str,
    batch_size: int = 1,
    layers_to_test: List[int] = None,
    normalize: bool = False,
    steerability_multipliers: List[float] = None,
):
    """
    Main experiment function.
    
    Args:
        dataset: Path to dataset file
        model: Model name or path
        layer: Layer to steer at (None for auto-selection via layer sweep)
        num_train: Number of training samples
        num_val: Number of validation samples
        num_test: Number of test samples
        max_new_tokens: Max tokens to generate
        multipliers: Multipliers for text generation (full range including extremes)
        output_dir: Output directory prefix
        batch_size: Batch size for processing (1=sequential, >1=batched for efficiency)
        layers_to_test: Layers to test in layer sweep (if layer is None)
        normalize: If True, normalize steering vector to unit norm
        steerability_multipliers: Multipliers for steerability computation (excludes extremes).
                                  If None, uses multipliers excluding min and max values.
    """
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Compute steerability multipliers (exclude extreme values) if not provided
    if steerability_multipliers is None:
        sorted_mults = sorted(multipliers)
        # Exclude min and max values for steerability computation
        steerability_multipliers = sorted_mults[1:-1] if len(sorted_mults) > 2 else sorted_mults
    
    print(f"Generation multipliers: {multipliers}")
    print(f"Steerability multipliers: {steerability_multipliers}")
    
    # Create output directory if it doesn't exist
    output_parent = os.path.dirname(output_dir)
    if output_parent and not os.path.exists(output_parent):
        os.makedirs(output_parent, exist_ok=True)
        print(f"Created output directory: {output_parent}")
    
    # Load data
    train_samples, val_samples, test_samples = load_and_split_dataset(
        dataset, num_train, num_val, num_test
    )
    
    # Initialize model
    extractor = SteeringVectorExtractor(model_name=model)
    
    # Find optimal layer (or use specified layer)
    # Uses steerability_multipliers (excludes extremes) to avoid degenerate outputs
    best_layer = find_optimal_layer(
        extractor, train_samples, val_samples, steerability_multipliers, layer, layers_to_test
    )
    
    # Extract steering vector
    sv_path = f"{output_dir}_steering_vector.pt"
    steering_vector = extract_steering_vector(
        extractor, train_samples, best_layer, sv_path, model, batch_size, normalize
    )
    
    # Combine all samples for generation (train + val + test)
    all_samples = train_samples + val_samples + test_samples
    print(f"\nTotal samples for generation: {len(all_samples)} "
          f"(train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)})")
    
    # Generate texts with perplexity and logit_diff for ALL samples
    generation_results = generate_texts_with_perplexity(
        extractor, all_samples, steering_vector, best_layer,
        multipliers, max_new_tokens, batch_size
    )
    
    # Add split information to each result
    for i, result in enumerate(generation_results):
        if i < len(train_samples):
            result['split'] = 'train'
        elif i < len(train_samples) + len(val_samples):
            result['split'] = 'val'
        else:
            result['split'] = 'test'
    
    # Save generation results
    gen_path = f"{output_dir}_generations.json"
    save_generation_results(generation_results, gen_path)
    
    # Print perplexity summary
    print_perplexity_summary(generation_results, multipliers)
    
    # Compute steerability from generation results (efficient - no extra forward passes)
    # Uses steerability_multipliers (excludes extremes like -2.0 and 2.0)
    steerability_results = compute_steerability_from_generation(
        extractor, generation_results, steerability_multipliers
    )
    
    # Save steerability results
    steer_path = f"{output_dir}_steerability.json"
    save_steerability_results(
        steerability_results, model, best_layer, len(all_samples), steer_path
    )
    
    # Print steerability summary
    print_steerability_summary(
        steerability_results, model, best_layer, len(all_samples)
    )
    
    # Plot results
    plot_path = f"{output_dir}_plots.png"
    plot_results(steerability_results, plot_path)
    
    # Plot perplexity vs steering strength
    perplexity_plot_path = f"{output_dir}_perplexity.png"
    plot_perplexity_vs_steering(generation_results, multipliers, perplexity_plot_path)
    
    return {
        'generation_results': generation_results,
        'steerability_results': steerability_results,
    }


if __name__ == "__main__":
    # Default multipliers for unnormalized vs normalized steering vectors
    # Generation uses full range; steerability excludes extremes by default
    MULTIPLIERS_UNNORMALIZED = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    MULTIPLIERS_NORMALIZED = [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    
    parser = argparse.ArgumentParser(description='Steering Vectors Experiment')
    parser.add_argument('--dataset', type=str, default='myopic-reward.jsonl',
                        help='Path to the dataset file')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                        help='Model name or path')
    parser.add_argument('--layer', type=int, default=None,
                        help='Layer to steer at (if not specified, will do layer sweep)')
    parser.add_argument('--num_train', type=int, default=600,
                        help='Number of training samples')
    parser.add_argument('--num_val', type=int, default=100,
                        help='Number of validation samples')
    parser.add_argument('--num_test', type=int, default=300,
                        help='Number of test samples')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='Max tokens to generate')
    parser.add_argument('--multipliers', type=float, nargs='+', default=None,
                        help='Multipliers for text generation (full range). If not specified, uses defaults based on --normalize: '
                             f'unnormalized={MULTIPLIERS_UNNORMALIZED}, normalized={MULTIPLIERS_NORMALIZED}')
    parser.add_argument('--steerability_multipliers', type=float, nargs='+', default=None,
                        help='Multipliers for steerability/layer sweep (excludes extremes). '
                             'If not specified, uses --multipliers with min/max values removed.')
    parser.add_argument('--output', type=str, default='results',
                        help='Output prefix for result files')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing (1=sequential, >1=batched for efficiency)')
    parser.add_argument('--layers_to_test', type=int, nargs='+', default=None,
                        help='Layers to test in layer sweep (e.g., --layers_to_test 8 10 12 14 16). '
                             'Only used if --layer is not specified.')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize steering vector to unit norm. Changes default multipliers to [-10, 10] range.')
    parser.add_argument('--test-run', action='store_true',
                        help='Quick test run with 100 samples total (60 train, 10 val, 30 test) and layer 13')
    
    args = parser.parse_args()
    
    # Set default multipliers based on normalization if not explicitly provided
    if args.multipliers is None:
        if args.normalize:
            args.multipliers = MULTIPLIERS_NORMALIZED
            print(f"Using normalized multipliers: {args.multipliers}")
        else:
            args.multipliers = MULTIPLIERS_UNNORMALIZED
            print(f"Using unnormalized multipliers: {args.multipliers}")
    
    # Override settings for test run
    if args.test_run:
        print("\n" + "="*60)
        print("TEST RUN MODE: Using reduced sample sizes")
        print("="*60)
        args.num_train = 60
        args.num_val = 10
        args.num_test = 30
        if args.layer is None:
            args.layer = 13  # Skip layer sweep, use middle layer
            print(f"Using fixed layer {args.layer} (skipping layer sweep)")
        print(f"Samples: {args.num_train} train, {args.num_val} val, {args.num_test} test")
        print("="*60 + "\n")
    
    main(
        dataset=args.dataset,
        model=args.model,
        layer=args.layer,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        max_new_tokens=args.max_new_tokens,
        multipliers=args.multipliers,
        output_dir=args.output,
        batch_size=args.batch_size,
        layers_to_test=args.layers_to_test,
        normalize=args.normalize,
        steerability_multipliers=args.steerability_multipliers,
    )
