from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel, Field
from sipangpt_eval.schemas.benchmark import BenchmarkCase
from sipangpt_eval.schemas.inference import InferenceOutput


class ScoreCategory(str, Enum):
    CORRECTA = "C"      # 1.0 punto
    PARCIAL = "P"       # 0.5 puntos
    INCORRECTA = "I"    # 0.0 puntos (alucinación o error fáctico)


class JudgeEvaluationOutput(BaseModel):
    analisis_comparativo: str = Field(
        ...,
        description="Razonamiento paso a paso comparando la respuesta del asistente contra la respuesta oficial de referencia."
    )
    omisiones_o_errores: str = Field(
        ...,
        description="Detalle de pasos faltantes, contradicciones o datos inventados (si los hubiera)."
    )
    alucinacion_detectada: bool = Field(
        ...,
        description="True si el asistente inventó datos, enlaces o normativas inexistentes."
    )
    veredicto: ScoreCategory = Field(
        ...,
        description="Etiqueta final estricta: 'C' (Correcta), 'P' (Parcial) o 'I' (Incorrecta)."
    )


class JevEvaluationOutput(BaseModel):
    """Resultado estructurado de la evaluación efectuada por TypeSafe AI Jev (System One)."""
    veredicto: ScoreCategory = Field(
        ...,
        description="Veredicto calibrado C (Correcta), P (Parcial) o I (Incorrecta)"
    )
    veredicto_choice: str = Field(
        ...,
        description="Opción ganadora seleccionada por Jev ('C', 'P', 'I')"
    )
    veredicto_confianza: float = Field(
        ...,
        description="Nivel de confianza de la decisión de veredicto (0.0 a 1.0)"
    )
    veredicto_probabilidades: Dict[str, float] = Field(
        default_factory=dict,
        description="Distribución de probabilidades calibradas para cada opción (C, P, I)"
    )
    alucinacion_detectada: bool = Field(
        ...,
        description="Indicador binario de alucinación según umbral probabilístico"
    )
    alucinacion_probabilidad: float = Field(
        ...,
        description="Probabilidad de alucinación o datos inventados asignada por Jev (0.0 a 1.0)"
    )
    calidad_score: float = Field(
        ...,
        description="Puntuación en la escala ordenada de calidad técnica y apego normativo"
    )
    calidad_confianza: float = Field(
        ...,
        description="Nivel de confianza en la graduación de calidad"
    )
    latencia_ms: float = Field(
        default=0.0,
        description="Tiempo de respuesta del Gateway / Jev en milisegundos"
    )


class EvaluatedPair(BaseModel):
    caso: BenchmarkCase
    inferencia_finetuned: InferenceOutput
    inferencia_rag: InferenceOutput
    score_finetuned: ScoreCategory
    score_rag: ScoreCategory
    observaciones: str = ""
    judge_eval_finetuned: Optional[JudgeEvaluationOutput] = None
    judge_eval_rag: Optional[JudgeEvaluationOutput] = None
    jev_eval_finetuned: Optional[JevEvaluationOutput] = None
    jev_eval_rag: Optional[JevEvaluationOutput] = None


class BenchmarkSummaryMetrics(BaseModel):
    total_preguntas: int = 50
    exactitud_finetuned_pct: float
    exactitud_rag_pct: float
    alucinaciones_finetuned_count: int
    alucinaciones_rag_count: int
    latencia_media_finetuned_ms: float
    latencia_media_rag_ms: float


class JevSummaryMetrics(BaseModel):
    total_preguntas: int = 50
    exactitud_finetuned_pct: float
    exactitud_rag_pct: float
    alucinaciones_finetuned_count: int
    alucinaciones_rag_count: int
    promedio_calidad_finetuned: float
    promedio_calidad_rag: float
    latencia_media_ms: float
