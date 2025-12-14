"""
RULER task definitions and configurations

RULER expands on the needle-in-a-haystack (NIAH) test with:
1. Retrieval tasks (single/multi-needle, multi-key)
2. Multi-hop tracing
3. Aggregation
4. Question answering
"""

import random
import string
from enum import Enum
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


class RULERTaskType(Enum):
    """RULER task categories"""
    NIAH_SINGLE = "niah_single_needle"  # Single needle in haystack
    NIAH_MULTI = "niah_multi_needle"  # Multiple needles
    NIAH_MULTIKEY = "niah_multikey"  # Multiple key-value pairs
    MULTIHOP_TRACING = "multihop_tracing"  # Multi-hop reasoning
    AGGREGATION = "aggregation"  # Aggregation tasks
    QA = "question_answering"  # Question answering


@dataclass
class RULERTaskConfig:
    """Configuration for RULER task"""
    task_type: RULERTaskType
    context_length: int  # Target context length in tokens
    num_needles: int = 1  # Number of needles to insert
    num_hops: int = 1  # For multi-hop tasks
    haystack_type: str = "repeat"  # "repeat", "essay", "code"
    insert_positions: str = "random"  # "random", "begin", "end", "uniform"


class RULERSyntheticDataGenerator:
    """
    Generate synthetic data for RULER benchmark tasks
    """

    def __init__(self, tokenizer):
        """
        Initialize generator

        Args:
            tokenizer: Tokenizer to use for token counting
        """
        self.tokenizer = tokenizer

        # Sample haystack texts
        self.haystack_templates = {
            "repeat": "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ",
            "essay": "In the realm of natural language processing, transformers have revolutionized the field. "
                     "These models leverage self-attention mechanisms to process sequences of arbitrary length. "
                     "The ability to handle long contexts is crucial for many downstream tasks. ",
            "code": "def process_data(input_data):\n    result = []\n    for item in input_data:\n        "
                    "if item is not None:\n            result.append(item)\n    return result\n\n",
        }

    def generate_niah_single(
        self,
        context_length: int,
        haystack_type: str = "repeat",
        insert_position: str = "random",
    ) -> Tuple[str, str, str]:
        """
        Generate single needle-in-a-haystack task

        Returns:
            Tuple of (context, needle, answer)
        """
        # Generate unique needle
        needle_id = ''.join(random.choices(string.digits, k=6))
        needle = f"The special magic number is: {needle_id}"
        answer = needle_id

        # Generate haystack
        haystack = self._generate_haystack(context_length, haystack_type)

        # Insert needle at specified position
        context = self._insert_needle(haystack, needle, insert_position, context_length)

        return context, needle, answer

    def generate_niah_multi(
        self,
        context_length: int,
        num_needles: int = 5,
        haystack_type: str = "repeat",
    ) -> Tuple[str, List[str], List[str]]:
        """
        Generate multi-needle task

        Returns:
            Tuple of (context, needles, answers)
        """
        needles = []
        answers = []

        for i in range(num_needles):
            needle_id = ''.join(random.choices(string.digits, k=6))
            needle = f"Special number {i+1} is: {needle_id}"
            needles.append(needle)
            answers.append(needle_id)

        # Generate haystack
        haystack = self._generate_haystack(context_length, haystack_type)

        # Insert needles uniformly
        context = haystack
        positions = [int(i * len(haystack) / (num_needles + 1)) for i in range(1, num_needles + 1)]

        for needle, pos in zip(needles, positions):
            context = context[:pos] + f" {needle}. " + context[pos:]

        return context, needles, answers

    def generate_niah_multikey(
        self,
        context_length: int,
        num_keys: int = 10,
    ) -> Tuple[str, Dict[str, str], str]:
        """
        Generate multi-key-value retrieval task

        Returns:
            Tuple of (context, key_value_dict, target_key)
        """
        # Generate key-value pairs
        kv_pairs = {}
        for i in range(num_keys):
            key = f"key_{i:03d}"
            value = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            kv_pairs[key] = value

        # Generate context with key-value pairs
        context_parts = []
        haystack_base = self.haystack_templates["repeat"]

        for key, value in kv_pairs.items():
            context_parts.append(f"The value for {key} is {value}.")
            # Add haystack between pairs
            context_parts.append(haystack_base * 5)

        context = " ".join(context_parts)

        # Pad to target length if needed
        while len(self.tokenizer.encode(context)) < context_length:
            context += haystack_base

        # Select random key to query
        target_key = random.choice(list(kv_pairs.keys()))

        return context, kv_pairs, target_key

    def generate_multihop_tracing(
        self,
        context_length: int,
        num_hops: int = 3,
    ) -> Tuple[str, List[str], str]:
        """
        Generate multi-hop tracing task

        Example: A->B, B->C, C->D, find path from A to D

        Returns:
            Tuple of (context, chain, answer)
        """
        # Generate chain
        chain = []
        nodes = [f"node_{i}" for i in range(num_hops + 1)]

        for i in range(num_hops):
            relation = f"{nodes[i]} connects to {nodes[i+1]}"
            chain.append(relation)

        # Shuffle chain and embed in haystack
        random.shuffle(chain)

        haystack = self._generate_haystack(context_length, "essay")
        context_parts = [haystack[:len(haystack)//4]]

        for relation in chain:
            context_parts.append(f"{relation}.")
            context_parts.append(haystack[:100])

        context_parts.append(haystack[len(haystack)//4:])
        context = " ".join(context_parts)

        # Answer is the final node
        answer = nodes[-1]

        return context, chain, answer

    def generate_aggregation(
        self,
        context_length: int,
        num_items: int = 20,
    ) -> Tuple[str, List[int], int]:
        """
        Generate aggregation task (e.g., count, sum)

        Returns:
            Tuple of (context, numbers, sum)
        """
        # Generate random numbers
        numbers = [random.randint(1, 100) for _ in range(num_items)]

        # Embed in context
        haystack = self._generate_haystack(context_length, "repeat")
        context_parts = []

        for i, num in enumerate(numbers):
            context_parts.append(f"Item {i+1} has value {num}.")
            context_parts.append(haystack[:50])

        context = " ".join(context_parts)

        # Pad to target length
        while len(self.tokenizer.encode(context)) < context_length:
            context += haystack

        answer = sum(numbers)

        return context, numbers, answer

    def _generate_haystack(self, target_tokens: int, haystack_type: str) -> str:
        """Generate haystack text of approximately target_tokens length"""
        template = self.haystack_templates.get(haystack_type, self.haystack_templates["repeat"])

        haystack = ""
        while len(self.tokenizer.encode(haystack)) < target_tokens:
            haystack += template

        return haystack

    def _insert_needle(
        self,
        haystack: str,
        needle: str,
        position: str,
        target_tokens: int,
    ) -> str:
        """Insert needle into haystack at specified position"""
        if position == "begin":
            context = f"{needle}. {haystack}"
        elif position == "end":
            context = f"{haystack} {needle}."
        else:  # random or uniform
            # Insert at random position
            insert_pos = random.randint(0, len(haystack))
            context = haystack[:insert_pos] + f" {needle}. " + haystack[insert_pos:]

        # Trim to approximately target length
        encoded = self.tokenizer.encode(context)
        if len(encoded) > target_tokens:
            trimmed = self.tokenizer.decode(encoded[:target_tokens])
            # Ensure needle is still present
            if needle not in trimmed:
                # Re-insert needle if trimmed out
                insert_pos = random.randint(0, len(trimmed))
                context = trimmed[:insert_pos] + f" {needle}. " + trimmed[insert_pos:]
            else:
                context = trimmed

        return context


class RULERTaskFactory:
    """Factory for creating RULER task instances"""

    def __init__(self, tokenizer):
        """
        Initialize factory

        Args:
            tokenizer: Tokenizer for token counting
        """
        self.generator = RULERSyntheticDataGenerator(tokenizer)

    def create_task(self, config: RULERTaskConfig) -> Dict[str, Any]:
        """
        Create a RULER task based on configuration

        Args:
            config: Task configuration

        Returns:
            Dictionary with task data
        """
        if config.task_type == RULERTaskType.NIAH_SINGLE:
            context, needle, answer = self.generator.generate_niah_single(
                config.context_length,
                config.haystack_type,
                config.insert_positions,
            )
            return {
                "task_type": config.task_type.value,
                "context": context,
                "needle": needle,
                "answer": answer,
                "query": "What is the special magic number mentioned in the text?",
            }

        elif config.task_type == RULERTaskType.NIAH_MULTI:
            context, needles, answers = self.generator.generate_niah_multi(
                config.context_length,
                config.num_needles,
                config.haystack_type,
            )
            return {
                "task_type": config.task_type.value,
                "context": context,
                "needles": needles,
                "answers": answers,
                "query": "List all the special numbers mentioned in the text.",
            }

        elif config.task_type == RULERTaskType.NIAH_MULTIKEY:
            context, kv_pairs, target_key = self.generator.generate_niah_multikey(
                config.context_length,
                config.num_needles,  # Use as num_keys
            )
            return {
                "task_type": config.task_type.value,
                "context": context,
                "kv_pairs": kv_pairs,
                "target_key": target_key,
                "answer": kv_pairs[target_key],
                "query": f"What is the value for {target_key}?",
            }

        elif config.task_type == RULERTaskType.MULTIHOP_TRACING:
            context, chain, answer = self.generator.generate_multihop_tracing(
                config.context_length,
                config.num_hops,
            )
            return {
                "task_type": config.task_type.value,
                "context": context,
                "chain": chain,
                "answer": answer,
                "query": f"Starting from {chain[0].split()[0]}, what is the final node after {config.num_hops} hops?",
            }

        elif config.task_type == RULERTaskType.AGGREGATION:
            context, numbers, answer = self.generator.generate_aggregation(
                config.context_length,
                config.num_needles,  # Use as num_items
            )
            return {
                "task_type": config.task_type.value,
                "context": context,
                "numbers": numbers,
                "answer": answer,
                "query": "What is the sum of all item values mentioned?",
            }

        else:
            raise ValueError(f"Unknown task type: {config.task_type}")
