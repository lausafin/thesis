"""
run_annotations.py

This script evaluates the side effects of activation steering on Large Language Models.
It takes a CSV containing baseline and steered generations (9 steps from -2.0 to +2.0) plus logits.
To ensure robustness, it calls an LLM-as-a-Judge (DeepSeek) 3 times per prompt, parses the 
JSON responses, and aggregates the results (majority voting for categories/steerability).
It supports resuming interrupted runs and outputs a final CSV with aggregated metrics.
"""

import os
import sys
import json
import argparse
import pathlib
from pathlib import Path
from typing import Any, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

THESIS_ROOT = Path(__file__).resolve().parents[2]
if str(THESIS_ROOT) not in sys.path:
    sys.path.insert(0, str(THESIS_ROOT))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from core.utils import (
    construct_few_shot_prompt,
    call_deepseek,
    extract_json_from_text,
    get_majority,
    load_data_from_directory,
    is_flagged_value
)
from core.data import filter_dataset_slice
from pipeline.constants import PASS2_EFFECT_KEYS
from pipeline.judge_parser import build_flagged_categories_text, should_process_row, aggregate_evaluations

load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")

def evaluate_single(prompt, temp):
    """Executes a single evaluation pass."""
    res_text = call_deepseek(prompt, temp=temp)
    if res_text:
        return extract_json_from_text(res_text)
    return None

def process_row_multi(idx, row, n_evals=3, template_path=None, compress_text=False):
    """
    Executes n sequential evaluation calls for a single dataset row.
    """
    dataset = row["dataset_name"]
    sample_idx = row["sample_idx"]
    pos_target = row.get("positive_steering", "N/A")
    neg_target = row.get("negative_steering", "N/A")
    question = row.get("question", row.get("prompt", "N/A"))

    # Dynamically extract all available multipliers (alpha values) from the column names
    alpha_keys = sorted([
        k.replace("generation_", "") 
        for k in row.keys() 
        if str(k).startswith("generation_") and pd.notna(row.get(k))
    ], key=float)
    
    gens = {}
    logits = {}
    
    for alpha_key in alpha_keys:
        f_val = float(alpha_key)
        # Format explicitly to force + on positive numbers (except 0.0) 
        s_key = f"+{f_val:.1f}" if f_val > 0 else f"{f_val:.1f}"
        
        gens[s_key] = str(row[f"generation_{alpha_key}"])
        
        logit_key = f"logit_{alpha_key}"
        if logit_key in row and pd.notna(row[logit_key]):
            logits[s_key] = str(row[logit_key])

    prompt = construct_few_shot_prompt(
        dataset,
        sample_idx,
        question,
        gens,
        logits,
        pos_target,
        neg_target,
        flagged_categories=build_flagged_categories_text(row),
        steerability_result=row.get("judge_steerability", "N/A"),
        steerability_rationale=row.get("judge_steerability_rationale", "N/A"),
        template_path=template_path,
        compress_text=compress_text
    )
    
    results = []
    temps = [0.6] * n_evals
    
    for t in temps:
        res = evaluate_single(prompt, t)
        if res: 
            results.append(res)
            
    return idx, results

def evaluate_dataset(input_csv, output_csv, num_evals=1, max_workers=None, samples_per_dataset=None, max_datasets=None, dataset_filter=None, random_sampling=False, save_all=False, template_path=None, compress_text=False, force_rerun=False):
    template_name = os.path.basename(template_path) if template_path else ""
    using_pass2_template = "pass2" in template_name.lower()

    if max_workers is None:
        # Increase max_workers since threaded pool is now flat (sequential per row)
        max_workers = 20 if num_evals == 1 else 15
    else:
        max_workers = max_workers

    input_file = input_csv
    output_file = output_csv
    
    # 1. Always load the full dataset first to ensure we have all rows
    if os.path.isdir(input_file):
        print(f"Input is a directory: {input_file}. Scanning for generation JSONs...")
        df = load_data_from_directory(input_file)
    else:
        df = pd.read_csv(input_file)

    if df.empty:
        print("No data loaded. Exiting.")
        return

    new_cols = [
        "judge_steerability", "judge_steerability_rationale",
        "judge_factual_integrity", "judge_factual_integrity_rationale",
        "judge_logical_coherence", "judge_logical_coherence_rationale",
        "judge_behavioral_resistance", "judge_behavioral_resistance_rationale",
        "judge_output_shape", "judge_output_shape_rationale",
        "judge_tonal_change", "judge_tonal_change_rationale",
        "judge_schema", "judge_notes", "judge_raw_votes_json", "judge_raw_evidence",
    ]
    new_cols.extend([f"judge_effect_{k}" for k in PASS2_EFFECT_KEYS])
    
    for c in new_cols:
        if c not in df.columns:
            df[c] = pd.Series(dtype='object')
        else:
            df[c] = df[c].astype("object")

    # 2. Check for resume capability: If output file already exists, load it
    # and update the full dataframe with the previously processed rows.
    if os.path.exists(output_file) and not force_rerun:
        print(f"Output file {output_file} exists. Loading to resume progress...")
        existing_df = pd.read_csv(output_file)
        
        # Merge the existing progress into the full dataset
        if not existing_df.empty:
            # Set indices for safe updating based on dataset_name and sample_idx
            if 'dataset_name' in df.columns and 'sample_idx' in df.columns:
                already_evaluated = (existing_df['judge_raw_votes_json'].notna()).sum()
                print(f"Found {already_evaluated} previously evaluated samples to resume from.")
                
                df.set_index(['dataset_name', 'sample_idx'], inplace=True)
                existing_df.set_index(['dataset_name', 'sample_idx'], inplace=True)
                
                # Update the original df with values from existing_df
                df.update(existing_df)
                
                # Reset indices back to standard
                df.reset_index(inplace=True)
            else:
                print("Warning: Missing dataset_name or sample_idx columns. Cannot safely resume.")
    
    # Filter the underlying dataframe so we don't save out-of-scope rows
    df = filter_dataset_slice(df, samples_per_dataset, max_datasets, dataset_filter, random_sampling)
    
    # Identify rows to process
    rows_to_process = []
    
    valid_indices = set(df.index)
            
    skipped_no_flag = 0
    skipped_no_pass1 = 0

    for idx, row in df.iterrows():
        if idx not in valid_indices:
            continue
            
        process, reason = should_process_row(row, force_rerun, using_pass2_template)
        if process:
            rows_to_process.append((idx, row))
        else:
            if reason == "no_flag":
                skipped_no_flag += 1
            elif reason == "no_pass1":
                skipped_no_pass1 += 1

    if using_pass2_template:
        print(f"  Pass 2 filter: {skipped_no_flag} rows skipped (no pass 1 flags), "
              f"{skipped_no_pass1} rows skipped (pass 1 not run).")

    print(f"Submitting {len(rows_to_process)} rows for {num_evals}x evaluation (max_workers={max_workers})...")

    # We collect updates in a dictionary to avoid multi-thread mutation of the underlying DataFrame
    updates = {}

    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_row_multi, idx, row, num_evals, template_path, compress_text): idx
            for idx, row in rows_to_process
        }
        
        with tqdm(total=len(rows_to_process), desc=f"Evaluating via DeepSeek ({num_evals}x)") as pbar:
            for future in as_completed(futures):
                idx_original = futures[future]
                try:
                    idx_ret, results = future.result()
                    if results and len(results) > 0:
                        row_update = aggregate_evaluations(results, df.loc[idx_ret])
                        
                        if row_update:
                            updates[idx_ret] = row_update
                        
                        # Apply immediately to the DataFrame and save atomically
                        if len(updates) > 0:
                            for u_idx, u_dict in updates.items():
                                for k, v in u_dict.items():
                                    df.at[u_idx, k] = v
                            updates.clear()
                            
                            temp_file = f"{output_file}.tmp"
                            df.to_csv(temp_file, index=False)
                            os.replace(temp_file, output_file) # Atomic save

                except Exception as exc:
                    print(f"Row {idx_original} exception: {exc}")
                pbar.update(1)
                
    # Final flush of remaining updates
    if len(updates) > 0:
        for u_idx, u_dict in updates.items():
            for k, v in u_dict.items():
                df.at[u_idx, k] = v
        updates.clear()
    
    if not save_all:
        df = df

    temp_file = f"{output_file}.tmp"
    df.to_csv(temp_file, index=False)
    os.replace(temp_file, output_file)            
    print("Done!")


def main():
    import datetime
    
    parser = argparse.ArgumentParser(description="Evaluate steering generations multiple times via DeepSeek API.")
    parser.add_argument("--input", type=str, default="data/generations", help="Path to input CSV file or directory containing generations. Defaults to data/generations")
    
    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument("--output", type=str, default=None, help="Path to save the output CSV with evaluated metrics. Defaults to results/full-run-by-shivam/<timestamp>/eval_pass*.csv")
    out_group.add_argument("--resume_dir", type=str, default=None, help="Path to an existing results directory to resume an interrupted run.")
    out_group.add_argument("--resume_latest", action="store_true", help="Automatically find and resume from the most recently created timestamped directory in results/full-run-by-shivam/.")

    parser.add_argument("--num_evals", type=int, default=3, help="Number of evaluation passes per prompt (default: 3 for NeurIPS-qualified robustness).")
    parser.add_argument("--max_workers", type=int, default=None, help="Number of concurrent rows to process.")
    parser.add_argument("--samples_per_dataset", type=int, default=None, help="Number of samples to evaluate per dataset. Leave empty for all.")
    parser.add_argument("--max_datasets", type=int, default=None, help="Optionally limit to N datasets randomly.")
    parser.add_argument("--dataset_filter", type=str, default=None, help="Optionally filter to exactly one dataset name.")
    parser.add_argument("--random_sampling", action="store_true", help="Randomly sample rows rather than taking the first n.")
    parser.add_argument("--save_all", action="store_true", help="Save all rows, including unevaluated ones. Default is to save only evaluated rows.")
    parser.add_argument("--template_path", type=str, default=None, help="Optional prompt template path (e.g., prompts/prompt_pass1.txt).")
    parser.add_argument("--compress_text", action="store_true", help="Use Semantic Diff Mask to calculate string deltas against baseline to save input tokens.")
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun on all rows instead of just skipping non-flagged.")
    parser.add_argument("--force_new_run", action="store_true", help="Ignore recent incomplete runs and start a fresh run.")
    parser.add_argument("--run_both_passes", action=argparse.BooleanOptionalAction, default=True, help="Run Pass 1 followed automatically by Pass 2. Default is True. Use --no-run_both_passes to disable.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Error: Input path '{args.input}' does not exist. Please ensure you have the dataset available.")

    out_dir = "."
    pass1_out = None
    if not args.output:
        # Check if we are resuming from an existing directory
        target_resume_dir = None
        base_results_dir = "results/full-run-by-shivam"
        
        if args.resume_latest:
            if os.path.exists(base_results_dir):
                # Find all timestamped directories and sort by name (which is timestamp format)
                dirs = [os.path.join(base_results_dir, d) for d in os.listdir(base_results_dir) 
                        if os.path.isdir(os.path.join(base_results_dir, d))]
                if dirs:
                    target_resume_dir = max(dirs, key=os.path.getmtime)
                    print(f"Auto-detected latest run directory: {target_resume_dir}")
                else:
                    print(f"Warning: --resume_latest used but no directories found in {base_results_dir}. Starting fresh.")
            else:
                print(f"Warning: --resume_latest used but {base_results_dir} does not exist. Starting fresh.")
        elif hasattr(args, 'resume_dir') and args.resume_dir:
            target_resume_dir = args.resume_dir

        if target_resume_dir:
            out_dir = target_resume_dir
            if args.run_both_passes:
                pass1_out = f"{out_dir}/eval_pass1.csv"
                args.output = f"{out_dir}/eval_pass2.csv"
            else:
                if os.path.exists(f"{out_dir}/eval_pass1.csv") and not os.path.exists(f"{out_dir}/eval_pass2.csv"):
                    args.output = f"{out_dir}/eval_pass1.csv"
                elif os.path.exists(f"{out_dir}/eval_pass2.csv") and not os.path.exists(f"{out_dir}/eval_pass1.csv"):
                    args.output = f"{out_dir}/eval_pass2.csv"
                else:
                    args.output = f"{out_dir}/eval_pass1.csv" if (args.template_path and "pass1" in args.template_path) else f"{out_dir}/eval_pass2.csv"
        else:
            if not args.force_new_run and os.path.exists(base_results_dir):
                import time
                current_time = time.time()
                incomplete_2h = []
                incomplete_24h = []
                
                dirs = [os.path.join(base_results_dir, d) for d in os.listdir(base_results_dir) 
                        if os.path.isdir(os.path.join(base_results_dir, d))]
                
                for d in dirs:
                    if not os.path.exists(os.path.join(d, ".run_complete")):
                        age_hours = (current_time - os.path.getmtime(d)) / 3600
                        if age_hours < 2.0:
                            incomplete_2h.append((d, age_hours))
                        elif age_hours < 24.0:
                            incomplete_24h.append((d, age_hours))
                
                if incomplete_2h:
                    latest, age = min(incomplete_2h, key=lambda x: x[1])
                    raise FileExistsError(
                        f"Found an incomplete run from {age:.1f} hours ago ({latest}).\n"
                        f"Please pass --resume_latest to continue it, or pass --force_new_run to ignore."
                    )
                elif incomplete_24h:
                    print(f"Notice: Found {len(incomplete_24h)} incomplete run(s) from the last 24 hours:")
                    for d, age in sorted(incomplete_24h, key=lambda x: x[1]):
                        print(f"  - {d} ({age:.1f} hours ago)")
                    print("Proceeding with a fresh run. You can use --resume_latest to finish the latest incomplete run next time.\n")

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = f"results/full-run-by-shivam/{timestamp}"
            os.makedirs(out_dir, exist_ok=True)
            
            if args.run_both_passes:
                pass1_out = f"{out_dir}/eval_pass1.csv"
                args.output = f"{out_dir}/eval_pass2.csv"
            else:
                if args.template_path and "pass1" in args.template_path:
                    args.output = f"{out_dir}/eval_pass1.csv"
                elif args.template_path and "pass2" in args.template_path:
                    args.output = f"{out_dir}/eval_pass2.csv"
                else:
                    args.output = f"{out_dir}/eval.csv"
    else:
        out_dir = os.path.dirname(args.output) or "."
        os.makedirs(out_dir, exist_ok=True)
        if args.run_both_passes:
            if "pass2" in args.output:
                pass1_out = args.output.replace("pass2", "pass1")
            else:
                pass1_out = args.output.replace(".csv", "_pass1.csv") if args.output.endswith(".csv") else args.output + "_pass1.csv"

    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()
            
    sys.stdout = Logger(os.path.join(out_dir, "run_terminal.log"))

    print(f"=== Experiment Run Started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Arguments: {vars(args)}")

    if args.run_both_passes:
        print(f"\n{'='*50}\n=== Starting Pass 1 (Output: {pass1_out}) ===\n{'='*50}")
        evaluate_dataset(
            input_csv=args.input,
            output_csv=pass1_out,
            num_evals=args.num_evals,
            max_workers=args.max_workers,
            samples_per_dataset=args.samples_per_dataset,
            max_datasets=args.max_datasets,
            dataset_filter=args.dataset_filter,
            random_sampling=args.random_sampling,
            save_all=args.save_all,
            template_path="prompts/prompt_pass1.txt",
            compress_text=args.compress_text,
            force_rerun=args.force_rerun
        )
        print(f"\n{'='*50}\n=== Starting Pass 2 (Output: {args.output}) ===\n{'='*50}")
        evaluate_dataset(
            input_csv=pass1_out,
            output_csv=args.output,
            num_evals=args.num_evals,
            max_workers=args.max_workers,
            samples_per_dataset=args.samples_per_dataset,
            max_datasets=args.max_datasets,
            dataset_filter=args.dataset_filter,
            random_sampling=args.random_sampling,
            save_all=args.save_all,
            template_path="prompts/prompt_pass2_new_format.txt",
            compress_text=args.compress_text,
            force_rerun=args.force_rerun
        )
    else:
        evaluate_dataset(
            input_csv=args.input,
            output_csv=args.output,
            num_evals=args.num_evals,
            max_workers=args.max_workers,
            samples_per_dataset=args.samples_per_dataset,
            max_datasets=args.max_datasets,
            dataset_filter=args.dataset_filter,
            random_sampling=args.random_sampling,
            save_all=args.save_all,
            template_path=args.template_path,
            compress_text=args.compress_text,
            force_rerun=args.force_rerun
        )

    print(f"\n{'='*50}\n=== Calculating Inter-Rater Reliability ===\n{'='*50}")
    skip_agreement = args.template_path and "per_alpha" in os.path.basename(args.template_path).lower()
    if skip_agreement:
        print("Skipping standard IAA (per-α template uses separate comparison script).")
    else:
        try:
            from pathlib import Path
            scripts_eval_dir = Path(__file__).parent
            sys.path.insert(0, str(scripts_eval_dir))
            from calculate_agreement import calculate_agreement

            keys_to_check = [
                "steerability", "factual_integrity", "logical_coherence",
                "behavioral_resistance", "output_shape", "tonal_change",
            ]

            if args.run_both_passes or (args.template_path and "pass2" in args.template_path):
                from pipeline.constants import PASS2_EFFECT_KEYS
                keys_to_check.extend(PASS2_EFFECT_KEYS)

            keys_to_check = list(dict.fromkeys(keys_to_check))

            agreement_df = calculate_agreement(args.output, keys_to_check)
            if not agreement_df.empty:
                agreement_csv = os.path.join(out_dir, "agreement_metrics.csv")
                agreement_df.to_csv(agreement_csv, index=False)

                print(f"\n{'='*50}\n=== Final Agreement Metrics ===\n{'='*50}")

                pass1_df = agreement_df[agreement_df['Pass'] == '1'].drop('Pass', axis=1)
                pass2_df = agreement_df[agreement_df['Pass'] == '2'].drop('Pass', axis=1)

                if not pass1_df.empty:
                    print(f"\n--- Pass 1: Core Categories ---")
                    print(pass1_df.to_markdown(index=False))

                if not pass2_df.empty:
                    print(f"\n--- Pass 2: Effect Dimensions ---")
                    print(pass2_df.to_markdown(index=False))

                print(f"\nSaved agreement metrics to {agreement_csv}")
        except Exception as e:
            print(f"Error calculating agreement: {e}")

    # Mark run as complete
    with open(os.path.join(out_dir, ".run_complete"), "w") as f:
        f.write("")

if __name__ == "__main__":
    main()
