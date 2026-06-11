"""
Utility functions for steering experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import pandas as pd
import json
import os
import requests
import time
import re
import glob
from collections import Counter
from dotenv import load_dotenv

from core.metrics.logit_trends import compute_logit_trend_hint

load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")

_PROMPT_TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "prompt_pass1.txt")

def load_prompt_template(path):
    try:
        with open(path, "r") as _f:
            return _f.read()
    except FileNotFoundError:
        print(f"Warning: {path} not found.")
        return ""

PROMPT_TEMPLATE = load_prompt_template(_PROMPT_TEMPLATE_PATH)

import difflib

def compute_semantic_diff(baseline, steered):
    if baseline == "N/A" or steered == "N/A" or not baseline or not steered:
        return str(steered)[:3000]
    
    if baseline == steered:
        return "[NO CHANGE FROM BASELINE]"
        
    b_words = str(baseline).split()
    s_words = str(steered).split()
        
    s = difflib.SequenceMatcher(None, b_words, s_words)
    if s.ratio() < 0.2:
        return str(steered)[:3000]
        
    b_out = []
    has_b_change = False
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            words = b_words[i1:i2]
            if len(words) > 30:
                words = words[:10] + ["..."] + words[-10:]
            b_out.append(" ".join(words))
        elif tag in ('replace', 'delete'):
            has_b_change = True
            b_out.append(f"<HIGHLIGHT> {' '.join(b_words[i1:i2])} </HIGHLIGHT>")

    s_out = []
    has_s_change = False
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            words = s_words[j1:j2]
            if len(words) > 30:
                words = words[:10] + ["..."] + words[-10:]
            s_out.append(" ".join(words))
        elif tag in ('replace', 'insert'):
            has_s_change = True
            s_out.append(f"<HIGHLIGHT> {' '.join(s_words[j1:j2])} </HIGHLIGHT>")
            
    if not has_b_change and not has_s_change:
        return "[NO CHANGE FROM BASELINE]"
        
    b_final = " ".join(b_out).replace(" <HIGHLIGHT>  </HIGHLIGHT> ", " ").replace("<HIGHLIGHT>", "").replace("</HIGHLIGHT>", "").strip() if not has_b_change else " ".join(b_out).strip()
    s_final = " ".join(s_out).replace(" <HIGHLIGHT>  </HIGHLIGHT> ", " ").replace("<HIGHLIGHT>", "").replace("</HIGHLIGHT>", "").strip() if not has_s_change else " ".join(s_out).strip()
    
    # Clean up double spaces from joins
    b_final = re.sub(r'\s+', ' ', b_final)
    s_final = re.sub(r'\s+', ' ', s_final)
    
    return f'\n<BASELINE>\n{b_final}\n</BASELINE>\n<STEERED>\n{s_final}\n</STEERED>'

def construct_few_shot_prompt(dataset_name, sample_idx, question, text_map, logits_map, pos_target, neg_target, 
                            flagged_categories=None, steerability_result=None, steerability_rationale=None,
                            template_path=None, compress_text=False):
    """
    Constructs the prompt for the DeepSeek judge with new instructions and input format.
    If template_path is provided, loads that template. Otherwise uses default PROMPT_TEMPLATE.
    """
    if template_path:
        template = load_prompt_template(template_path)
    else:
        template = PROMPT_TEMPLATE

    sorted_keys = ["-2.0", "-1.5", "-1.0", "-0.5", "0.0", "+0.5", "+1.0", "+1.5", "+2.0"]
    
    gens_formatted = ""
    logits_formatted = ""
    
    baseline_text = text_map.get("0.0", "N/A")

    logit_trend_hint = compute_logit_trend_hint(logits_map)

    for k in sorted_keys:
        text = text_map.get(k, "N/A")
        if text != "N/A":
            # Enforce the mathematically validated 250-word threshold to bound API tokens
            # without discarding structural divergence signals.
            words = str(text).split()
            if len(words) > 250:
                text = " ".join(words[:250]) + "...[TRUNCATED_AT_250_WORDS]"
                
            if compress_text and k != "0.0":
                diff = compute_semantic_diff(baseline_text, str(text))
                gens_formatted += f"{k}: {diff}\n"
            else:
                gens_formatted += f"{k}: {str(text)}\n"
        
        logit = logits_map.get(k, "N/A")
        if logit != "N/A":
            logits_formatted += f"{k}: {logit}\n"

    kwargs = {
        "dataset_name": dataset_name,
        "sample_idx": sample_idx,
        "question": question,
        "pos_target": pos_target,
        "neg_target": neg_target,
        "gens_formatted": gens_formatted,
        "logits_formatted": logits_formatted,
        "logit_trend_hint": logit_trend_hint,
    }

    if flagged_categories is not None:
        kwargs["flagged_categories"] = flagged_categories
    if steerability_result is not None:
        kwargs["steerability_result"] = steerability_result
    if steerability_rationale is not None:
        kwargs["steerability_rationale"] = steerability_rationale

    # Use regex substitution instead of str.format() so that literal JSON braces
    # in the template body (e.g. worked examples) are left untouched.
    def _replace(m):
        key = m.group(1)
        return str(kwargs[key]) if key in kwargs else m.group(0)

    return re.sub(r'\{([a-zA-Z_][a-zA-Z_0-9]*)\}', _replace, template)


import difflib
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

def call_deepseek(prompt, temp=0.6, system_prompt=None):
    """
    Calls the DeepSeek API with robust exponential backoff for rate limit handling.
    """
    if system_prompt is None:
        system_prompt = (
            "You are a precise AI Judge. Your task is to output a VALID JSON object evaluating the provided text based on the taxonomy provided."
            " Do NOT output 'Understood', conversational text, or markdown code blocks (```)."
            " Your response must be ONLY the raw JSON object starting with '{' and ending with '}'."
        )
    
    @retry(
        wait=wait_exponential(multiplier=2, min=5, max=120), # Waits 5, 10, 20, 40, 80, 120 secs
        stop=stop_after_attempt(8),
        retry=retry_if_exception_type(Exception)
    )
    def _do_call():
        # print(f"Sending prompt to DeepSeek ({len(prompt)} chars)...")  # Removed to avoid console spam in MT
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temp,
                "max_tokens": 4000
            },
            timeout=300
        )
        if response.status_code == 429:
            print(f"RATE LIMITED. Backing off... ({response.text})")
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    try:
        return _do_call()
    except Exception as e:
        print(f"DeepSeek connection error after all retries: {e}")
    return None


def extract_json_from_text(text):
    """
    Extracts JSON cleanly from text.
    """
    if not text:
        return None
    try:
        text = text.replace("```json", "").replace("```", "").strip()
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            print(f"DEBUG: No braces found in text: {text[:200]}...")
            return None
        json_str = text[start:end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            import re
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            return json.loads(json_str, strict=False)
    except Exception as e:
        print(f"DEBUG: JSON decode error: {e}")
        print(f"DEBUG: Problematic text snippet: {text[:200]}...")
    return None

import string

def is_flagged_value(val) -> bool:
    """
    Determines if a qualitative LLM judge evaluation output evaluates to a 'flagged' or 'true' state.
    Uses robust string normalization to handle natural language artifacts like "Yes", "1", "Flagged".
    """
    if val is True or val == 1:
        return True
    if not isinstance(val, str):
        val = str(val)
        
    val_str = val.strip().upper()
    clean_val_str = val_str.translate(str.maketrans('', '', string.punctuation))
    
    truthy_words = {"FLAGGED", "YES", "TRUE", "1"}
    negation_words = {"NOT", "NO", "FALSE", "NEVER", "0"}
    
    is_truthy = (clean_val_str in truthy_words or any(x in clean_val_str.split() for x in truthy_words))
    is_negated = any(x in clean_val_str.split() for x in negation_words)
    
    return is_truthy and not is_negated


def get_majority(values):
    """Returns the most common string value, or None if empty."""
    if not values: return None
    c = Counter(values)
    return c.most_common(1)[0][0]


def load_data_from_directory(base_dir):
    """
    Crawls a directory structure for `*_generations.json` files and aggregates them into a DataFrame.
    Expects subdirectories: base_dir/[dataset_name]/[timestamp]_generations.json
    """
    records = []
    
    # Iterate over all subdirectories in base_dir
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist.")
        return pd.DataFrame()

    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"  Found {len(subdirs)} subdirectories in {base_dir}")
    
    for dataset_name in subdirs:
        dataset_path = os.path.join(base_dir, dataset_name)
        # Find the generations json file
        json_files = glob.glob(os.path.join(dataset_path, "*_generations.json"))
        
        if not json_files:
            continue
            
        # If multiple files exist, take the most recent one (sorted by name usually works for timestamped files)
        json_file = sorted(json_files)[-1]
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue
            
        for item in data:
            record = {
                "dataset_name": dataset_name,
                "sample_idx": item.get("sample_idx"),
                "question": item.get("question"),
                "positive_steering": item.get("answer_matching_behavior"),
                "negative_steering": item.get("answer_not_matching_behavior"),
            }
            
            # Extract completions
            completions = item.get("completions", {})
            for key, val in completions.items():
                if isinstance(val, dict):
                    mult = val.get("multiplier")
                    if mult is not None:
                        # Format multiplier to string like "-2.0"
                        mult_str = f"{float(mult):.1f}"
                        if mult_str == "-0.0": mult_str = "0.0"
                        
                        record[f"generation_{mult_str}"] = val.get("text")
                        record[f"logit_{mult_str}"] = val.get("logit_diff")
            
            records.append(record)

    df = pd.DataFrame(records)
    print(f"  Loaded {len(df)} samples from directory scan.")
    return df


def ensure_dir(file_path):
    """
    Ensures that the directory for a given file path exists.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)



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
