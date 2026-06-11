"""
Steering Vector Extractor using Contrastive Activation Addition (CAA).
"""

import torch
from typing import List, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from data import ContrastiveSample, create_chat_prompt


class SteeringVectorExtractor:
    """
    Extracts steering vectors using Contrastive Activation Addition (CAA).
    
    The method:
    1. For each sample, get activations when the model sees the positive answer
    2. Get activations when the model sees the negative answer
    3. Compute the mean difference across all samples
    
    Supports optional batch processing with left padding for efficiency.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.float16,
    ):
        print(f"Loading model: {model_name}")
        print(f"Device: {device}")
        
        self.device = device
        self.model_name = model_name
        
        # Load tokenizer with left padding for batch processing
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding is required for decoder-only models in batch mode
        # This ensures the last token position is always the actual last token
        self.tokenizer.padding_side = "left"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        
        # Get number of layers
        self.num_layers = self.model.config.num_hidden_layers
        print(f"Model has {self.num_layers} layers")
        
        # Storage for activations
        self.activations = {}
        
    def _get_activation_hook(self, layer_idx: int):
        """Create a hook to capture activations at a specific layer."""
        def hook(module, input, output):
            # output is a tuple, first element is the hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.activations[layer_idx] = hidden_states.detach()
        return hook
    
    def get_activations_at_position(
        self,
        prompt: str,
        target_token: str,
        layer: int,
    ) -> torch.Tensor:
        """
        Get activations at the position of the target token.
        
        Args:
            prompt: The full prompt including the answer token
            target_token: The token we want activations for (e.g., "(A)" or "(B)")
            layer: Which layer to extract from
            
        Returns:
            Activation tensor at the target position
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Register hook for the target layer
        # For Llama models, the layers are in model.model.layers
        handle = self.model.model.layers[layer].register_forward_hook(
            self._get_activation_hook(layer)
        )
        
        try:
            # Forward pass
            with torch.no_grad():
                _ = self.model(**inputs)
            
            # Get the activations
            activations = self.activations[layer]
            
            # Get activation at the last token position (the answer token)
            # Shape: [batch, seq_len, hidden_dim] -> [hidden_dim]
            last_token_activation = activations[0, -1, :].cpu()
            
        finally:
            handle.remove()
            self.activations.clear()
        
        return last_token_activation
    
    def extract_steering_vector(
        self,
        samples: List[ContrastiveSample],
        layer: int,
        system_prompt: str = "You are a helpful, honest and concise assistant.",
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Extract a steering vector using mean difference of activations.
        
        Args:
            samples: List of contrastive samples
            layer: Layer to extract from
            system_prompt: System prompt to use
            normalize: If True, normalize steering vector to unit norm
            
        Returns:
            Steering vector (mean difference of positive - negative activations)
        """
        positive_activations = []
        negative_activations = []
        
        print(f"Extracting steering vector from layer {layer}...")
        
        for sample in tqdm(samples, desc="Processing samples"):
            # Create prompts with positive and negative answers
            prompt_pos = create_chat_prompt(sample.question, system_prompt) + sample.answer_matching_behavior
            prompt_neg = create_chat_prompt(sample.question, system_prompt) + sample.answer_not_matching_behavior
            
            # Get activations
            act_pos = self.get_activations_at_position(prompt_pos, sample.answer_matching_behavior, layer)
            act_neg = self.get_activations_at_position(prompt_neg, sample.answer_not_matching_behavior, layer)
            
            positive_activations.append(act_pos)
            negative_activations.append(act_neg)
        
        # Stack and compute mean difference
        pos_stack = torch.stack(positive_activations)
        neg_stack = torch.stack(negative_activations)
        
        steering_vector = (pos_stack - neg_stack).mean(dim=0)
        
        original_norm = steering_vector.norm().item()
        print(f"Steering vector shape: {steering_vector.shape}")
        print(f"Steering vector norm (before normalization): {original_norm:.4f}")
        
        if normalize:
            steering_vector = steering_vector / steering_vector.norm()
            print(f"Steering vector normalized to unit norm (1.0)")
        
        return steering_vector
    
    # =========================================================================
    # Batched Methods (for efficiency with batch_size > 1)
    # =========================================================================
    
    def _get_batched_activations(
        self,
        prompts: List[str],
        layer: int,
    ) -> torch.Tensor:
        """
        Get last-token activations for a batch of prompts.
        
        With left padding, the rightmost token is always the actual last token,
        so we can simply take position -1 for all sequences.
        
        Args:
            prompts: List of prompts to process
            layer: Layer to extract activations from
            
        Returns:
            Tensor of shape [batch_size, hidden_dim] with last-token activations
        """
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(self.device)
        
        handle = self.model.model.layers[layer].register_forward_hook(
            self._get_activation_hook(layer)
        )
        
        try:
            with torch.no_grad():
                _ = self.model(**inputs)
            
            # With left padding, position -1 is always the last real token
            # Shape: [batch, seq_len, hidden_dim] -> [batch, hidden_dim]
            last_token_acts = self.activations[layer][:, -1, :].cpu()
        finally:
            handle.remove()
            self.activations.clear()
        
        return last_token_acts
    
    def extract_steering_vector_batched(
        self,
        samples: List[ContrastiveSample],
        layer: int,
        batch_size: int = 8,
        system_prompt: str = "You are a helpful, honest and concise assistant.",
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Extract a steering vector using batched processing.
        
        Significantly faster than sequential processing for large datasets.
        
        Args:
            samples: List of contrastive samples
            layer: Layer to extract from
            batch_size: Number of samples to process at once
            system_prompt: System prompt to use
            normalize: If True, normalize steering vector to unit norm
            
        Returns:
            Steering vector (mean difference of positive - negative activations)
        """
        print(f"Extracting steering vector from layer {layer} (batch_size={batch_size})...")
        
        # Prepare all prompts
        pos_prompts = [
            create_chat_prompt(s.question, system_prompt) + s.answer_matching_behavior 
            for s in samples
        ]
        neg_prompts = [
            create_chat_prompt(s.question, system_prompt) + s.answer_not_matching_behavior 
            for s in samples
        ]
        
        positive_activations = []
        negative_activations = []
        
        # Process in batches
        num_batches = (len(samples) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(samples), batch_size), desc="Extracting (batched)", total=num_batches):
            batch_pos = pos_prompts[i:i + batch_size]
            batch_neg = neg_prompts[i:i + batch_size]
            
            pos_acts = self._get_batched_activations(batch_pos, layer)
            neg_acts = self._get_batched_activations(batch_neg, layer)
            
            positive_activations.append(pos_acts)
            negative_activations.append(neg_acts)
        
        # Concatenate all batches
        pos_stack = torch.cat(positive_activations, dim=0)
        neg_stack = torch.cat(negative_activations, dim=0)
        
        steering_vector = (pos_stack - neg_stack).mean(dim=0)
        
        original_norm = steering_vector.norm().item()
        print(f"Steering vector shape: {steering_vector.shape}")
        print(f"Steering vector norm (before normalization): {original_norm:.4f}")
        
        if normalize:
            steering_vector = steering_vector / steering_vector.norm()
            print(f"Steering vector normalized to unit norm (1.0)")
        
        return steering_vector