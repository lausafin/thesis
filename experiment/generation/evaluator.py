"""
Steering Vector Evaluator for measuring steerability and generating text.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from collections import defaultdict

from data import ContrastiveSample, create_chat_prompt


class SteeringEvaluator:
    """
    Evaluates the effectiveness of steering vectors.
    """
    
    def __init__(self, extractor):
        self.extractor = extractor
        self.model = extractor.model
        self.tokenizer = extractor.tokenizer
        self.device = extractor.device
    
    def _create_steering_hook(self, steering_vector: torch.Tensor, multiplier: float):
        """Create a steering hook for a given multiplier."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                modified = hidden_states.clone()
                # Steer only the last token position (steers last prompt token + each generated token)
                modified[:, -1, :] += multiplier * steering_vector.to(hidden_states.device).to(hidden_states.dtype)
                # Alternative: steer all positions (steers entire prompt + all generated tokens)
                # modified[0, :, :] += multiplier * steering_vector.to(hidden_states.device).to(hidden_states.dtype)
                return (modified,) + output[1:]
            else:
                modified = output.clone()
                # Steer only the last token position (steers last prompt token + each generated token)
                modified[:, -1, :] += multiplier * steering_vector.to(output.device).to(output.dtype)
                # Alternative: steer all positions (steers entire prompt + all generated tokens)
                # modified[0, :, :] += multiplier * steering_vector.to(output.device).to(output.dtype)
                return modified
        return hook
    
    def _calculate_perplexity_from_scores(
        self,
        scores: Tuple[torch.Tensor, ...],
        generated_token_ids: torch.Tensor,
    ) -> float:
        """
        Calculate perplexity from generation scores.
        
        Args:
            scores: Tuple of logits tensors, one per generated token.
                    Each tensor has shape [batch_size, vocab_size].
            generated_token_ids: The token IDs that were generated.
            
        Returns:
            Perplexity score (lower = more confident/fluent)
        """
        if len(scores) == 0:
            return float('inf')
        
        total_nll = 0.0
        num_tokens = 0
        
        for score, token_id in zip(scores, generated_token_ids):
            # score shape: [batch_size, vocab_size]
            log_probs = torch.nn.functional.log_softmax(score, dim=-1)
            token_nll = -log_probs[0, token_id].item()
            total_nll += token_nll
            num_tokens += 1
        
        if num_tokens == 0:
            return float('inf')
        
        avg_nll = total_nll / num_tokens
        perplexity = np.exp(avg_nll)
        return perplexity
    
    def _calculate_logit_diff(
        self,
        first_token_logits: torch.Tensor,
        sample: ContrastiveSample,
    ) -> float:
        """
        Calculate logit difference following paper methodology.
        
        Compares logit("A") vs logit("B") at the first generation position,
        where A/B are extracted from answers like " (A)" or " (B)".
        
        Args:
            first_token_logits: Logits tensor for first generated token [batch_size, vocab_size]
            sample: The contrastive sample with answer options
            
        Returns:
            Logit difference: logit(positive_letter) - logit(negative_letter)
        """
        # Extract just the letter (A or B) from answers like " (A)" or " (B)"
        pos_letter = sample.answer_matching_behavior.strip().strip("()")  # "A" or "B"
        neg_letter = sample.answer_not_matching_behavior.strip().strip("()")  # "B" or "A"
        
        # Get single token IDs for "A" and "B"
        pos_id = self.tokenizer.encode(pos_letter, add_special_tokens=False)[0]
        neg_id = self.tokenizer.encode(neg_letter, add_special_tokens=False)[0]
        
        logit_diff = first_token_logits[0, pos_id].item() - first_token_logits[0, neg_id].item()
        return logit_diff
    
    def generate_text(
        self,
        sample: ContrastiveSample,
        steering_vector: torch.Tensor = None,
        layer: int = None,
        multiplier: float = 0.0,
        max_new_tokens: int = 100,
        system_prompt: str = "You are a helpful, honest and concise assistant.",
    ) -> Dict:
        """
        Generate text, optionally with steering, and calculate perplexity and logit diff.
        
        Perplexity and logit difference are calculated during generation, ensuring 
        steering is properly applied at each token generation step.
        
        Args:
            sample: The sample to generate text for
            steering_vector: Optional steering vector to apply
            layer: Layer to apply steering at
            multiplier: Strength of steering (0.0 = no steering)
            max_new_tokens: Maximum number of tokens to generate
            system_prompt: System prompt
            
        Returns:
            Dict with:
                - text: Generated text
                - perplexity: Perplexity score
                - logit_diff: Logit difference (positive - negative answer)
                - first_token_logits: Raw logits for first generated token (optional use)
        """
        prompt = create_chat_prompt(sample.question, system_prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Register steering hook if multiplier is non-zero
        handle = None
        if steering_vector is not None and multiplier != 0.0:
            handle = self.model.model.layers[layer].register_forward_hook(
                self._create_steering_hook(steering_vector, multiplier)
            )
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            
            generated_ids = outputs.sequences
            scores = outputs.scores  # Tuple of logits, one per generated token
            
            # Get the token IDs that were actually generated
            generated_token_ids = generated_ids[0][inputs.input_ids.shape[1]:]
            
            # Decode generated text
            generated_text = self.tokenizer.decode(
                generated_token_ids, 
                skip_special_tokens=True
            )
            
            # Calculate perplexity from generation scores
            perplexity = self._calculate_perplexity_from_scores(scores, generated_token_ids)
            
            # Calculate logit difference from first token scores
            first_token_logits = scores[0] if len(scores) > 0 else None
            logit_diff = self._calculate_logit_diff(first_token_logits, sample) if first_token_logits is not None else 0.0
                
        finally:
            if handle is not None:
                handle.remove()
        
        return {
            'text': generated_text,
            'perplexity': perplexity,
            'logit_diff': logit_diff,
            'first_token_logits': first_token_logits,
        }
    
    
    def _check_answer_choice(
        self,
        generated_text: str,
        answer_matching: str,
        answer_not_matching: str,
    ) -> str:
        """
        Check which answer the generated text chose.
        
        Returns:
            'matching', 'not_matching', or 'unclear'
        """
        text_start = generated_text.strip()[:20].upper()
        matching_clean = answer_matching.strip().upper()
        not_matching_clean = answer_not_matching.strip().upper()
        
        if text_start.startswith(matching_clean) or matching_clean in text_start:
            return 'matching'
        elif text_start.startswith(not_matching_clean) or not_matching_clean in text_start:
            return 'not_matching'
        else:
            return 'unclear'
    
    def generate_and_evaluate(
        self,
        sample: ContrastiveSample,
        steering_vector: torch.Tensor,
        layer: int,
        multiplier: float,
        max_new_tokens: int = 100,
        system_prompt: str = "You are a helpful, honest and concise assistant.",
    ) -> Dict:
        """
        Generate text and calculate perplexity for a single sample at a single multiplier.
        
        Perplexity and logit difference are calculated during generation, ensuring 
        steering is properly applied at each token generation step.
        
        Returns:
            Dict with generated text, perplexity, logit_diff, and answer choice
        """
        gen_result = self.generate_text(
            sample, steering_vector, layer, multiplier, max_new_tokens, system_prompt
        )
        
        answer_choice = self._check_answer_choice(
            gen_result['text'],
            sample.answer_matching_behavior,
            sample.answer_not_matching_behavior,
        )
        
        return {
            'text': gen_result['text'],
            'perplexity': gen_result['perplexity'],
            'logit_diff': gen_result['logit_diff'],
            'chose_matching_behavior': answer_choice == 'matching',
            'chose_not_matching_behavior': answer_choice == 'not_matching',
            'answer_choice': answer_choice,
        }
    
    def evaluate_sample_across_multipliers(
        self,
        sample: ContrastiveSample,
        steering_vector: torch.Tensor,
        layer: int,
        multipliers: List[float],
        max_new_tokens: int = 100,
        system_prompt: str = "You are a helpful, honest and concise assistant.",
    ) -> Dict:
        """
        Generate text and calculate perplexity for a sample across multiple multipliers.
        
        Args:
            sample: The sample to evaluate
            steering_vector: The steering vector
            layer: Layer to apply steering
            multipliers: List of multipliers (should include 0.0 for baseline)
            max_new_tokens: Max tokens to generate
            system_prompt: System prompt
            
        Returns:
            Dict with sample info and results for each multiplier
        """
        result = {
            'question': sample.question,
            'answer_matching_behavior': sample.answer_matching_behavior,
            'answer_not_matching_behavior': sample.answer_not_matching_behavior,
            'completions': {},
        }
        
        for mult in multipliers:
            gen_result = self.generate_and_evaluate(
                sample, steering_vector, layer, mult, max_new_tokens, system_prompt
            )
            result['completions'][f'multiplier_{mult}'] = {
                'multiplier': mult,
                'text': gen_result['text'],
                'perplexity': gen_result['perplexity'],
                'logit_diff': gen_result['logit_diff'],
                'chose_matching_behavior': gen_result['chose_matching_behavior'],
                'chose_not_matching_behavior': gen_result['chose_not_matching_behavior'],
                'answer_choice': gen_result['answer_choice'],
            }
        
        return result
    
    def compute_steerability_from_results(
        self,
        generation_results: List[Dict],
        multipliers: List[float],
    ) -> Dict:
        """
        Compute steerability metrics from already-generated results.
        
        This is more efficient than compute_steerability() as it uses the logit_diffs
        that were already computed during generation, avoiding extra forward passes.
        
        Args:
            generation_results: List of results from generate_texts (with logit_diff in completions)
            multipliers: List of multipliers used during generation
            
        Returns:
            Dict with steerability metrics
        """
        results = {
            'multipliers': multipliers,
            'mean_logit_diffs': [],
            'per_sample_logit_diffs': defaultdict(list),
            'per_sample_steerability': [],
        }
        
        # Extract logit_diffs from generation results
        for mult in multipliers:
            key = f'multiplier_{mult}'
            logit_diffs = []
            for i, sample_result in enumerate(generation_results):
                ld = sample_result['completions'][key]['logit_diff']
                logit_diffs.append(ld)
                results['per_sample_logit_diffs'][i].append(ld)
            results['mean_logit_diffs'].append(np.mean(logit_diffs))
        
        # Compute per-sample steerability (slope of logit_diff vs multiplier)
        for i in range(len(generation_results)):
            lds = results['per_sample_logit_diffs'][i]
            slope, _ = np.polyfit(multipliers, lds, 1)
            results['per_sample_steerability'].append(slope)
        
        # Compute aggregate steerability
        slope, _ = np.polyfit(multipliers, results['mean_logit_diffs'], 1)
        results['aggregate_steerability'] = slope
        
        return results
    
    # =========================================================================
    # Batched Methods (for efficiency with batch_size > 1)
    # =========================================================================
    
    def generate_text_batched(
        self,
        samples: List[ContrastiveSample],
        steering_vector: torch.Tensor,
        layer: int,
        multiplier: float,
        max_new_tokens: int = 100,
        batch_size: int = 4,
        system_prompt: str = "You are a helpful, honest and concise assistant.",
    ) -> List[Dict]:
        """
        Generate text for multiple samples at once with the same multiplier.
        
        Batched generation is faster but requires more GPU memory.
        
        Args:
            samples: List of samples to generate text for
            steering_vector: The steering vector
            layer: Layer to apply steering
            multiplier: Steering multiplier
            max_new_tokens: Max tokens to generate
            batch_size: Number of samples to process at once
            system_prompt: System prompt
            
        Returns:
            List of dicts with text, perplexity, logit_diff for each sample
        """
        results = []
        
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i + batch_size]
            prompts = [create_chat_prompt(s.question, system_prompt) for s in batch_samples]
            
            inputs = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
            ).to(self.device)
            
            # Register steering hook
            handle = None
            if steering_vector is not None and multiplier != 0.0:
                handle = self.model.model.layers[layer].register_forward_hook(
                    self._create_steering_hook(steering_vector, multiplier)
                )
            
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=0.0,
                        pad_token_id=self.tokenizer.eos_token_id,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                
                # Process each sample in the batch
                for j, sample in enumerate(batch_samples):
                    # Get generated tokens for this sample
                    gen_ids = outputs.sequences[j][inputs.input_ids.shape[1]:]
                    
                    # Find actual end of generation (first EOS/PAD token)
                    # This is important because batched generation pads shorter sequences
                    eos_positions = (gen_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                    if len(eos_positions) > 0:
                        # Include the EOS token in perplexity calculation
                        end_pos = eos_positions[0].item() + 1
                        gen_ids_for_ppl = gen_ids[:end_pos]
                    else:
                        gen_ids_for_ppl = gen_ids
                    
                    text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                    
                    # Calculate perplexity from scores (only up to actual end of generation)
                    # Scores shape: (num_generated_tokens, batch_size, vocab_size)
                    num_tokens_for_ppl = len(gen_ids_for_ppl)
                    sample_scores = tuple(s[j:j+1] for s in outputs.scores[:num_tokens_for_ppl])
                    perplexity = self._calculate_perplexity_from_scores(sample_scores, gen_ids_for_ppl)
                    
                    # Calculate logit diff from first token
                    first_token_logits = outputs.scores[0][j:j+1] if len(outputs.scores) > 0 else None
                    logit_diff = self._calculate_logit_diff(first_token_logits, sample) if first_token_logits is not None else 0.0
                    
                    # Check answer choice
                    answer_choice = self._check_answer_choice(
                        text, sample.answer_matching_behavior, sample.answer_not_matching_behavior
                    )
                    
                    results.append({
                        'text': text,
                        'perplexity': perplexity,
                        'logit_diff': logit_diff,
                        'chose_matching_behavior': answer_choice == 'matching',
                        'chose_not_matching_behavior': answer_choice == 'not_matching',
                        'answer_choice': answer_choice,
                    })
                    
            finally:
                if handle:
                    handle.remove()
        
        return results
    
    def evaluate_samples_batched(
        self,
        samples: List[ContrastiveSample],
        steering_vector: torch.Tensor,
        layer: int,
        multipliers: List[float],
        max_new_tokens: int = 100,
        batch_size: int = 4,
        system_prompt: str = "You are a helpful, honest and concise assistant.",
    ) -> List[Dict]:
        """
        Evaluate multiple samples across all multipliers using batched processing.
        
        For each multiplier, processes samples in batches. This is more efficient
        than processing sample-by-sample.
        
        Args:
            samples: List of samples to evaluate
            steering_vector: The steering vector
            layer: Layer to apply steering
            multipliers: List of multipliers to evaluate
            max_new_tokens: Max tokens to generate
            batch_size: Number of samples to process at once
            system_prompt: System prompt
            
        Returns:
            List of dicts (one per sample) with completions for each multiplier
        """
        # Initialize results structure
        results = []
        for i, sample in enumerate(samples):
            results.append({
                'sample_idx': i,
                'question': sample.question,
                'answer_matching_behavior': sample.answer_matching_behavior,
                'answer_not_matching_behavior': sample.answer_not_matching_behavior,
                'completions': {},
            })
        
        # Process each multiplier
        for mult in tqdm(multipliers, desc="Multipliers"):
            # Generate for all samples with this multiplier
            batch_results = self.generate_text_batched(
                samples, steering_vector, layer, mult, max_new_tokens, batch_size, system_prompt
            )
            
            # Add to results
            for i, gen_result in enumerate(batch_results):
                results[i]['completions'][f'multiplier_{mult}'] = {
                    'multiplier': mult,
                    'text': gen_result['text'],
                    'perplexity': gen_result['perplexity'],
                    'logit_diff': gen_result['logit_diff'],
                    'chose_matching_behavior': gen_result['chose_matching_behavior'],
                    'chose_not_matching_behavior': gen_result['chose_not_matching_behavior'],
                    'answer_choice': gen_result['answer_choice'],
                }
        
        return results
