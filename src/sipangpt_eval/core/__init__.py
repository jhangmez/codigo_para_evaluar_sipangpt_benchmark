"""
Módulo core de ejecución de inferencia y evaluación objetiva del benchmark.
"""

from sipangpt_eval.core.runner import BenchmarkRunner
from sipangpt_eval.core.scorer import BenchmarkScorer

__all__ = [
    "BenchmarkRunner",
    "BenchmarkScorer",
]
