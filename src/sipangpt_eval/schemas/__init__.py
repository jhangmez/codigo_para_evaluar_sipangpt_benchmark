"""
Módulo de esquemas de datos tipados estrictamente con Pydantic v2.
"""

from sipangpt_eval.schemas.benchmark import (
    TurnType,
    ModuleCategory,
    ConversationMessage,
    BenchmarkCase,
)
from sipangpt_eval.schemas.inference import (
    CitationSource,
    InferenceOutput,
)
from sipangpt_eval.schemas.evaluation import (
    ScoreCategory,
    EvaluatedPair,
    BenchmarkSummaryMetrics,
)

__all__ = [
    "TurnType",
    "ModuleCategory",
    "ConversationMessage",
    "BenchmarkCase",
    "CitationSource",
    "InferenceOutput",
    "ScoreCategory",
    "EvaluatedPair",
    "BenchmarkSummaryMetrics",
]
