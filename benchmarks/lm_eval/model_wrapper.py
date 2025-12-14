"""
Custom model wrapper to integrate Infini-LLM with LM Evaluation Harness
"""

import torch
from typing import List, Optional, Union
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from transformers import AutoTokenizer


@register_model("infinillm")
class InfiniLLMWrapper(LM):
    """
    Wrapper class to make Infini-LLM compatible with lm-evaluation-harness
    """

    def __init__(
        self,
        model,
        tokenizer_path: str,
        batch_size: int = 1,
        max_length: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Infini-LLM wrapper

        Args:
            model: Infini-LLM model instance
            tokenizer_path: Path to the tokenizer
            batch_size: Batch size for evaluation
            max_length: Maximum sequence length
            device: Device to run on ('cuda' or 'cpu')
        """
        super().__init__()
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._batch_size = batch_size
        self._max_length = max_length
        self._device = device

        # Move model to device
        self.model = self.model.to(self._device)
        self.model.eval()

        # Log model configuration
        print(f"\nModel Configuration:")
        print(f"  Attention method: {self.model.layers[0].attn_method if hasattr(self.model.layers[0], 'attn_method') else 'unknown'}")
        print(f"  Window size: {self.model.layers[0].window_size if hasattr(self.model.layers[0], 'window_size') else 'N/A'}")
        print(f"  Use cache: {self.model.layers[0].attention.use_cache if hasattr(self.model.layers[0].attention, 'use_cache') else 'unknown'}")
        print()

    @property
    def eot_token_id(self):
        """End of text token ID"""
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        """Maximum sequence length the model can handle"""
        return self._max_length if self._max_length else 2048

    @property
    def max_gen_toks(self):
        """Maximum number of tokens to generate"""
        return 256

    @property
    def batch_size(self):
        """Batch size for evaluation"""
        return self._batch_size

    @property
    def device(self):
        """Device the model is on"""
        return self._device

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        """
        Encode a string into tokens

        Args:
            string: Input string to encode

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: List[int], **kwargs) -> str:
        """
        Decode tokens into a string

        Args:
            tokens: List of token IDs to decode

        Returns:
            Decoded string
        """
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _reset_cache(self):
        """Reset the model's KV cache state between independent evaluations"""
        for layer in self.model.layers:
            if hasattr(layer, 'attention'):
                attention = layer.attention
                # Reset all cache-related state
                if hasattr(attention, 'prompt_len'):
                    attention.prompt_len = 0
                if hasattr(attention, '_prompt_positions'):
                    attention._prompt_positions = None
                if hasattr(attention, '_window_positions'):
                    attention._window_positions = None
                # Recreate cache tensors with correct shape instead of just zeroing
                if hasattr(attention, 'cache_k') and attention.cache_k is not None:
                    cache_shape = attention.cache_k.shape
                    attention.cache_k = torch.zeros(cache_shape, dtype=attention.cache_k.dtype, device=attention.cache_k.device)
                if hasattr(attention, 'cache_v') and attention.cache_v is not None:
                    cache_shape = attention.cache_v.shape
                    attention.cache_v = torch.zeros(cache_shape, dtype=attention.cache_v.dtype, device=attention.cache_v.device)
                # Reset any cache state variables
                if hasattr(attention, '_cache_positions'):
                    attention._cache_positions = None
                if hasattr(attention, 'last_attention_weights'):
                    attention.last_attention_weights = None

    def _model_call(self, inps: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Run the model on the input tokens

        Args:
            inps: Input token tensor of shape (batch, seq_len)

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size)
        """
        with torch.no_grad():
            # Reset cache before each independent call
            self._reset_cache()
            inps = inps.to(self._device)
            outputs = self.model(inps, 0)
            return outputs

    def _model_generate(
        self,
        context: Union[str, torch.Tensor],
        max_length: int,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generate text from the model

        Args:
            context: Input context (string or token tensor)
            max_length: Maximum number of tokens to generate
            stop: List of stop strings

        Returns:
            Generated text
        """
        if isinstance(context, str):
            input_ids = self.tokenizer.encode(context, return_tensors="pt")
        else:
            input_ids = context

        input_ids = input_ids.to(self._device)
        generated = input_ids.clone()

        with torch.no_grad():
            # Process initial context
            outputs = self.model(generated, 0)

            # Generate tokens one by one
            for i in range(max_length):
                # Get logits for next token
                next_token_logits = outputs[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Get next output
                current_pos = generated.shape[1] - 1
                outputs = self.model(next_token, current_pos)

                # Check stop conditions
                if stop:
                    generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                    if any(stop_str in generated_text for stop_str in stop):
                        break

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def loglikelihood(self, requests) -> List[tuple]:
        """
        Compute log-likelihood for the given requests

        Args:
            requests: List of Instance objects with args = (context, continuation)

        Returns:
            List of (log_likelihood, is_greedy) tuples
        """
        results = []

        # Track sequence length statistics
        seq_lengths = []

        for idx, request in enumerate(requests):
            # Extract context and continuation from Instance object
            context, continuation = request.args
            # Encode context and continuation
            context_tokens = self.tok_encode(context)
            continuation_tokens = self.tok_encode(continuation)

            # Create full sequence
            full_tokens = torch.tensor(
                [context_tokens + continuation_tokens],
                dtype=torch.long,
                device=self._device
            )

            seq_len = len(context_tokens) + len(continuation_tokens)
            seq_lengths.append(seq_len)

            # Get model outputs
            with torch.no_grad():
                logits = self._model_call(full_tokens)

            # Compute log probabilities for continuation tokens
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Get log probability of continuation
            continuation_start = len(context_tokens)
            continuation_end = len(context_tokens) + len(continuation_tokens)

            total_log_prob = 0.0
            is_greedy = True

            for i, token_id in enumerate(continuation_tokens):
                pos = continuation_start + i
                if pos > 0:  # Skip if no previous context
                    token_log_prob = log_probs[0, pos - 1, token_id].item()
                    total_log_prob += token_log_prob

                    # Check if this is the greedy choice
                    greedy_token = torch.argmax(log_probs[0, pos - 1]).item()
                    if greedy_token != token_id:
                        is_greedy = False

            results.append((total_log_prob, is_greedy))

        # Print sequence length statistics
        if seq_lengths:
            import numpy as np
            print(f"\nSequence Length Statistics:")
            print(f"  Min: {np.min(seq_lengths)}")
            print(f"  Max: {np.max(seq_lengths)}")
            print(f"  Mean: {np.mean(seq_lengths):.1f}")
            print(f"  Median: {np.median(seq_lengths):.1f}")
            print(f"  >512 tokens: {sum(1 for x in seq_lengths if x > 512)} ({100*sum(1 for x in seq_lengths if x > 512)/len(seq_lengths):.1f}%)")
            print(f"  >1024 tokens: {sum(1 for x in seq_lengths if x > 1024)} ({100*sum(1 for x in seq_lengths if x > 1024)/len(seq_lengths):.1f}%)")
            print()

        return results

    def loglikelihood_rolling(self, requests) -> List[float]:
        """
        Compute rolling log-likelihood for the given requests

        Args:
            requests: List of Instance objects with args = (string,)

        Returns:
            List of log-likelihoods
        """
        results = []

        for request in requests:
            # Extract string from Instance object
            (string,) = request.args
            tokens = self.tok_encode(string)
            token_tensor = torch.tensor([tokens], dtype=torch.long, device=self._device)

            with torch.no_grad():
                logits = self._model_call(token_tensor)

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Sum log probabilities
            total_log_prob = 0.0
            for i in range(1, len(tokens)):
                token_log_prob = log_probs[0, i - 1, tokens[i]].item()
                total_log_prob += token_log_prob

            results.append(total_log_prob)

        return results

    def generate_until(self, requests) -> List[str]:
        """
        Generate text until stop conditions are met

        Args:
            requests: List of Instance objects with args = (context, generation_kwargs)

        Returns:
            List of generated strings
        """
        results = []

        for request in requests:
            # Extract context and generation kwargs from Instance object
            context, gen_kwargs = request.args
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            stop = gen_kwargs.get("until", [])

            generated_text = self._model_generate(
                context=context,
                max_length=max_gen_toks,
                stop=stop
            )

            # Remove context from generated text
            if generated_text.startswith(context):
                generated_text = generated_text[len(context):]

            results.append(generated_text)

        return results
