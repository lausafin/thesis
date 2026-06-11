"""
Data loading and formatting utilities for steering experiments.
"""

import json
from typing import List, Dict
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class ContrastiveSample:
    """A single contrastive sample with question and positive/negative options."""
    question: str
    answer_matching_behavior: str  # The "positive" answer (e.g., "(A)" or "(B)")
    answer_not_matching_behavior: str  # The "negative" answer
    full_prompt_positive: str = ""
    full_prompt_negative: str = ""


def load_myopic_reward_dataset(filepath: str) -> List[Dict]:
    """Load the myopic-reward dataset from a JSONL file."""
    samples = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def format_as_contrastive_prompt(sample: Dict) -> ContrastiveSample:
    """
    Format a raw sample into a contrastive prompt.
    
    The dataset has:
    - question: The question text with choices
    - answer_matching_behavior: The answer that matches the target behavior (e.g., "(B)")
    - answer_not_matching_behavior: The answer that doesn't match (e.g., "(A)")
    """
    question = sample['question']
    pos_answer = sample['answer_matching_behavior'].strip()
    neg_answer = sample['answer_not_matching_behavior'].strip()
    
    return ContrastiveSample(
        question=question,
        answer_matching_behavior=pos_answer,
        answer_not_matching_behavior=neg_answer,
    )


def create_chat_prompt(question: str, system_prompt: str = "You are a helpful, honest and concise assistant.") -> str:
    """
    Create a chat-formatted prompt for instruction-tuned models.
    For Llama-3.1, we use the chat template format.
    """
    # Llama 3.1 chat format
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt

def filter_dataset_slice(df, samples_per_dataset=None, max_datasets=None, dataset_filter=None, random_sampling=False):
    """Filters a DataFrame safely for subset evaluation."""
    if samples_per_dataset is None and dataset_filter is None and max_datasets is None:
        return df

    valid_indices = set()
    filtered_df = df
    
    if dataset_filter is not None:
        filtered_df = df[df["dataset_name"] == dataset_filter]
        
    unique_datasets = filtered_df["dataset_name"].unique()
    if max_datasets is not None and len(unique_datasets) > max_datasets:
        import numpy as np
        if random_sampling:
            unique_datasets = np.random.choice(unique_datasets, max_datasets, replace=False)
        else:
            unique_datasets = unique_datasets[:max_datasets]
            
    filtered_df = filtered_df[filtered_df["dataset_name"].isin(unique_datasets)]
        
    for _, group in filtered_df.groupby("dataset_name", sort=False):
        if samples_per_dataset is not None:
            n = min(samples_per_dataset, len(group))
            if random_sampling:
                valid_indices.update(group.sample(n=n).index)
            else:
                valid_indices.update(group.head(n).index)
        else:
            valid_indices.update(group.index)
            
    return df.loc[sorted(list(valid_indices))].copy()
