"""
Data loading and formatting utilities for steering experiments.
"""

import json
from typing import List, Dict
from dataclasses import dataclass


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
