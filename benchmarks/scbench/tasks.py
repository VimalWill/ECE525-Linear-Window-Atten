"""
SCBench task definitions and registry
Based on: https://arxiv.org/abs/2412.10319

SCBench includes 12 tasks across 4 categories:
1. String Retrieval
2. Semantic Retrieval
3. Global Information Processing
4. Multi-tasking

Across 2 modes:
- Multi-turn: Sequential queries in same session
- Multi-request: Parallel requests with shared context
"""

from enum import Enum
from typing import Dict, List, Any
from dataclasses import dataclass


class SCBenchTaskType(Enum):
    """SCBench task categories"""
    STRING_RETRIEVAL = "string_retrieval"
    SEMANTIC_RETRIEVAL = "semantic_retrieval"
    GLOBAL_INFO = "global_info"
    MULTITASKING = "multitasking"


class SCBenchMode(Enum):
    """SCBench evaluation modes"""
    MULTI_TURN = "multi_turn"
    MULTI_REQUEST = "multi_request"


@dataclass
class SCBenchTask:
    """SCBench task configuration"""
    name: str
    task_type: SCBenchTaskType
    mode: SCBenchMode
    description: str
    metrics: List[str]
    domains: List[str]


class SCBenchTaskRegistry:
    """
    Registry of SCBench tasks

    SCBench consists of 931 multi-turn sessions with 4,853 queries
    """

    TASKS = {
        # String Retrieval Tasks
        "string_needle": SCBenchTask(
            name="string_needle",
            task_type=SCBenchTaskType.STRING_RETRIEVAL,
            mode=SCBenchMode.MULTI_TURN,
            description="Retrieve specific strings from long context",
            metrics=["accuracy", "recall"],
            domains=["retrieval"]
        ),
        "string_code": SCBenchTask(
            name="string_code",
            task_type=SCBenchTaskType.STRING_RETRIEVAL,
            mode=SCBenchMode.MULTI_REQUEST,
            description="Retrieve code snippets from codebase context",
            metrics=["accuracy", "precision"],
            domains=["code"]
        ),

        # Semantic Retrieval Tasks
        "semantic_kv": SCBenchTask(
            name="semantic_kv",
            task_type=SCBenchTaskType.SEMANTIC_RETRIEVAL,
            mode=SCBenchMode.MULTI_TURN,
            description="Semantic key-value retrieval from context",
            metrics=["accuracy", "f1"],
            domains=["retrieval", "qa"]
        ),
        "semantic_qa": SCBenchTask(
            name="semantic_qa",
            task_type=SCBenchTaskType.SEMANTIC_RETRIEVAL,
            mode=SCBenchMode.MULTI_TURN,
            description="Question answering requiring semantic understanding",
            metrics=["accuracy", "rouge", "bleu"],
            domains=["qa"]
        ),

        # Global Information Processing Tasks
        "global_summary": SCBenchTask(
            name="global_summary",
            task_type=SCBenchTaskType.GLOBAL_INFO,
            mode=SCBenchMode.MULTI_TURN,
            description="Summarize information across entire context",
            metrics=["rouge", "coherence"],
            domains=["summarization"]
        ),
        "global_counting": SCBenchTask(
            name="global_counting",
            task_type=SCBenchTaskType.GLOBAL_INFO,
            mode=SCBenchMode.MULTI_REQUEST,
            description="Count occurrences across full context",
            metrics=["accuracy"],
            domains=["counting"]
        ),

        # Multi-tasking Tasks
        "multitask_icl": SCBenchTask(
            name="multitask_icl",
            task_type=SCBenchTaskType.MULTITASKING,
            mode=SCBenchMode.MULTI_TURN,
            description="In-context learning across multiple tasks",
            metrics=["accuracy", "task_success_rate"],
            domains=["icl"]
        ),
        "multitask_tracing": SCBenchTask(
            name="multitask_tracing",
            task_type=SCBenchTaskType.MULTITASKING,
            mode=SCBenchMode.MULTI_TURN,
            description="Multi-hop tracing across context",
            metrics=["accuracy", "hop_accuracy"],
            domains=["reasoning", "tracing"]
        ),

        # Additional domain-specific tasks
        "code_completion": SCBenchTask(
            name="code_completion",
            task_type=SCBenchTaskType.SEMANTIC_RETRIEVAL,
            mode=SCBenchMode.MULTI_REQUEST,
            description="Code completion with long file context",
            metrics=["exact_match", "pass@k"],
            domains=["code"]
        ),
        "doc_qa": SCBenchTask(
            name="doc_qa",
            task_type=SCBenchTaskType.SEMANTIC_RETRIEVAL,
            mode=SCBenchMode.MULTI_TURN,
            description="Question answering over long documents",
            metrics=["accuracy", "f1", "em"],
            domains=["qa", "retrieval"]
        ),
        "kv_retrieval": SCBenchTask(
            name="kv_retrieval",
            task_type=SCBenchTaskType.STRING_RETRIEVAL,
            mode=SCBenchMode.MULTI_TURN,
            description="Key-value pair retrieval with KV cache reuse",
            metrics=["accuracy", "cache_hit_rate"],
            domains=["retrieval"]
        ),
        "multihop_qa": SCBenchTask(
            name="multihop_qa",
            task_type=SCBenchTaskType.MULTITASKING,
            mode=SCBenchMode.MULTI_TURN,
            description="Multi-hop reasoning across documents",
            metrics=["accuracy", "reasoning_steps"],
            domains=["qa", "reasoning"]
        ),
    }

    @classmethod
    def get_task(cls, task_name: str) -> SCBenchTask:
        """Get task by name"""
        if task_name not in cls.TASKS:
            raise ValueError(f"Task {task_name} not found. Available: {list(cls.TASKS.keys())}")
        return cls.TASKS[task_name]

    @classmethod
    def list_tasks(cls, task_type: SCBenchTaskType = None, mode: SCBenchMode = None) -> List[str]:
        """List tasks, optionally filtered by type or mode"""
        tasks = []
        for name, task in cls.TASKS.items():
            if task_type and task.task_type != task_type:
                continue
            if mode and task.mode != mode:
                continue
            tasks.append(name)
        return tasks

    @classmethod
    def get_all_tasks(cls) -> Dict[str, SCBenchTask]:
        """Get all tasks"""
        return cls.TASKS.copy()
