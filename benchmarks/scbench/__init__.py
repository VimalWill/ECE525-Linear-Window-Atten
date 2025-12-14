"""
SCBench (SharedContextBench) integration for Infini-LLM
KV cache-centric analysis of long-context methods
"""

from .runner import SCBenchRunner
from .tasks import SCBenchTaskRegistry

__all__ = ['SCBenchRunner', 'SCBenchTaskRegistry']
