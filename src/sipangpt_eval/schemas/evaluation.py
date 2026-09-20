from enum import Enum
from pydantic import BaseModel, Field
from sipangpt_eval.schemas.benchmark import BenchmarkCase
from sipangpt_eval.schemas.inference import InferenceOutput


class ScoreCategory(str, Enum):
    CORRECTA = "C"      # 1.0 punto
    PARCIAL = "P"       # 0.5 puntos
    INCORRECTA = "I"    # 0.0 puntos (alucinación o error fáctico)


class EvaluatedPair(BaseModel):
    caso: BenchmarkCase
    inferencia_finetuned: InferenceOutput
    inferencia_rag: InferenceOutput
    score_finetuned: ScoreCategory
    score_rag: ScoreCategory
    observaciones: str = ""


class BenchmarkSummaryMetrics(BaseModel):
    total_preguntas: int = 50
    exactitud_finetuned_pct: float
    exactitud_rag_pct: float
    alucinaciones_finetuned_count: int
    alucinaciones_rag_count: int
    latencia_media_finetuned_ms: float
    latencia_media_rag_ms: float
