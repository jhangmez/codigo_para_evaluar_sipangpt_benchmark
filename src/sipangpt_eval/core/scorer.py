from typing import List, Sequence, Tuple
from sipangpt_eval.schemas.benchmark import BenchmarkCase
from sipangpt_eval.schemas.evaluation import (
    BenchmarkSummaryMetrics,
    EvaluatedPair,
    ScoreCategory,
)
from sipangpt_eval.schemas.inference import InferenceOutput


import unicodedata


def _normalize_text(text: str) -> str:
    """Normaliza texto eliminando diacríticos (tildes) y convirtiendo a minúsculas."""
    nfd = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


class BenchmarkScorer:
    """Rúbrica de evaluación objetiva y métricas de desempeño."""

    @staticmethod
    def evaluate_response(
        case: BenchmarkCase, output: InferenceOutput, is_rag: bool = False
    ) -> ScoreCategory:
        """Evalúa una respuesta frente a la Verdad Terreno (Ground Truth).

        C (Correcta = 1.0): Coincidencia con procedimiento oficial. Si es RAG, requiere citas.
        P (Parcial = 0.5): Idea general adecuada con omisión menor.
        I (Incorrecta = 0.0): Alucinación o error fáctico.
        """
        resp: str = _normalize_text(output.respuesta_generada)
        expected: str = _normalize_text(case.respuesta_esperada)

        # Extraer palabras clave de la respuesta esperada
        keywords: List[str] = [
            w.strip(".,;:()") for w in expected.split() if len(w.strip(".,;:()")) > 3
        ]
        matches: int = sum(1 for kw in keywords if kw in resp)
        ratio: float = matches / max(1, len(keywords))

        if is_rag:
            has_citations: bool = len(output.citas) > 0
            if ratio >= 0.5 and has_citations:
                return ScoreCategory.CORRECTA
            elif ratio >= 0.25:
                return ScoreCategory.PARCIAL
            else:
                return ScoreCategory.INCORRECTA
        else:
            if ratio >= 0.5:
                return ScoreCategory.CORRECTA
            elif ratio >= 0.25:
                return ScoreCategory.PARCIAL
            else:
                return ScoreCategory.INCORRECTA

    def evaluate_all(
        self,
        cases: Sequence[BenchmarkCase],
        gemma_outputs: Sequence[InferenceOutput],
        rag_outputs: Sequence[InferenceOutput],
    ) -> Tuple[List[EvaluatedPair], BenchmarkSummaryMetrics]:
        """Evalúa el conjunto de 50 pares de inferencia y genera la matriz y resumen de métricas."""
        pairs: List[EvaluatedPair] = []

        sum_points_ft: float = 0.0
        sum_points_rag: float = 0.0

        aluc_ft: int = 0
        aluc_rag: int = 0

        latencies_ft: List[float] = []
        latencies_rag: List[float] = []

        for case, out_ft, out_rag in zip(cases, gemma_outputs, rag_outputs):
            score_ft: ScoreCategory = self.evaluate_response(case, out_ft, is_rag=False)
            score_rag: ScoreCategory = self.evaluate_response(case, out_rag, is_rag=True)

            # Puntos por categoría
            pts_map: dict[ScoreCategory, float] = {
                ScoreCategory.CORRECTA: 1.0,
                ScoreCategory.PARCIAL: 0.5,
                ScoreCategory.INCORRECTA: 0.0,
            }

            pts_ft: float = pts_map[score_ft]
            pts_rag: float = pts_map[score_rag]

            sum_points_ft += pts_ft
            sum_points_rag += pts_rag

            if score_ft == ScoreCategory.INCORRECTA:
                aluc_ft += 1
            if score_rag == ScoreCategory.INCORRECTA:
                aluc_rag += 1

            latencies_ft.append(out_ft.tiempo_total_ms)
            latencies_rag.append(out_rag.tiempo_total_ms)

            obs: str = (
                f"FineTuned score: {score_ft.value}, RAG score: {score_rag.value}. "
                f"Citas RAG: {len(out_rag.citas)}."
            )

            pairs.append(
                EvaluatedPair(
                    caso=case,
                    inferencia_finetuned=out_ft,
                    inferencia_rag=out_rag,
                    score_finetuned=score_ft,
                    score_rag=score_rag,
                    observaciones=obs,
                )
            )

        total: int = len(cases)
        exactitud_ft_pct: float = (sum_points_ft / max(1, total)) * 100.0
        exactitud_rag_pct: float = (sum_points_rag / max(1, total)) * 100.0

        mean_lat_ft: float = sum(latencies_ft) / max(1, len(latencies_ft))
        mean_lat_rag: float = sum(latencies_rag) / max(1, len(latencies_rag))

        summary = BenchmarkSummaryMetrics(
            total_preguntas=total,
            exactitud_finetuned_pct=round(exactitud_ft_pct, 2),
            exactitud_rag_pct=round(exactitud_rag_pct, 2),
            alucinaciones_finetuned_count=aluc_ft,
            alucinaciones_rag_count=aluc_rag,
            latencia_media_finetuned_ms=round(mean_lat_ft, 2),
            latencia_media_rag_ms=round(mean_lat_rag, 2),
        )

        return pairs, summary
