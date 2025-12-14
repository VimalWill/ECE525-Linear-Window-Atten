"""
LM Evaluation Harness integration for Infini-LLM
Provides wrapper around EleutherAI's lm-evaluation-harness
"""

from .runner import LMEvalHarnessRunner
from .model_wrapper import InfiniLLMWrapper

__all__ = ['LMEvalHarnessRunner', 'InfiniLLMWrapper']
