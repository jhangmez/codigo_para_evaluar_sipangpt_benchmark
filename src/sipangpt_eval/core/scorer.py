import json
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from google import genai
from google.genai import types

from sipangpt_eval.config import settings
from sipangpt_eval.schemas.benchmark import BenchmarkCase
from sipangpt_eval.schemas.evaluation import (
    BenchmarkSummaryMetrics,
    EvaluatedPair,
    JevEvaluationOutput,
    JudgeEvaluationOutput,
    ScoreCategory,
)
from sipangpt_eval.schemas.inference import InferenceOutput

JUDGE_SYSTEM_PROMPT = """Eres un Juez Evaluador Imparcial y Riguroso de sistemas de Inteligencia Artificial para soporte de TI universitario en la Universidad Señor de Sipán (USS).
Tu objetivo es contrastar la respuesta emitida por un asistente conversacional frente a la respuesta oficial de referencia (Ground Truth) extraída de los manuales y reglamentos oficiales de la universidad.

Reglas de Calificación Estricta (Zheng et al., NeurIPS 2023):
1. Califica como 'C' (Correcta) si la respuesta contiene los pasos indispensables y los datos fácticos exactos según la referencia (fechas, requisitos, flujos). Pequeñas variaciones de redacción que no alteren el procedimiento son válidas. Para sistemas RAG, valora la inclusión de citas o referencias normativas.
2. Califica como 'P' (Parcial) si la orientación general es correcta pero omite un paso esencial, un requisito clave o es imprecisa en fechas o plazos secundarios.
3. Califica como 'I' (Incorrecta) si contiene alucinaciones (inventa plataformas, enlaces, costos o normas), contradice la respuesta de referencia o no resuelve la duda del estudiante.

Debes ser completamente objetivo. No permitas que la elocuencia o longitud de la respuesta influyan en tu juicio fáctico.
"""


class BenchmarkScorer:
    """Rúbrica de evaluación objetiva basada en Reference-Guided LLM-as-a-Judge (Zheng et al., NeurIPS 2023)."""

    def __init__(self, api_key: str = "", model_name: str = "") -> None:
        self.api_key: str = (
            api_key
            or settings.gemini_api_key
            or os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY", "")
        )
        self.model_name: str = model_name or settings.judge_model_name or "gemini-3.5-flash"
        self._client: genai.Client | None = None
        if self.api_key:
            self._client = genai.Client(api_key=self.api_key)

    def evaluate_with_llm_judge(
        self,
        pregunta: str,
        respuesta_referencia: str,
        respuesta_candidata: str,
        is_rag: bool = False,
    ) -> JudgeEvaluationOutput:
        """Evalúa una respuesta usando Reference-Guided LLM-as-a-Judge con Gemini 3.5 Flash."""
        if not self._client:
            raise ValueError("GEMINI_API_KEY no configurada para el Juez LLM.")

        prompt: str = (
            f"[Pregunta del Usuario]\n{pregunta}\n\n"
            f"[Respuesta Oficial de Referencia (Ground Truth)]\n{respuesta_referencia}\n\n"
            f"[Respuesta Generada por el Asistente a Evaluar ('{'Sipán-STAIR RAG' if is_rag else 'Gemma-4 Fine-Tuned'}')]\n{respuesta_candidata}"
        )

        config = types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=JudgeEvaluationOutput,
        )

        models_to_try: List[str] = [self.model_name, "gemini-3.5-flash-lite"]
        last_error: Exception | None = None

        for model in models_to_try:
            for attempt in range(5):
                try:
                    res = self._client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    )
                    if res.text:
                        return JudgeEvaluationOutput.model_validate_json(res.text)
                except Exception as exc:
                    last_error = exc
                    err_msg: str = str(exc)
                    if "404" in err_msg or "limit: 20" in err_msg:
                        break
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "503" in err_msg:
                        wait_secs: float = 6.0 * (attempt + 1)
                        print(f"  [429/503 en {model}] Reintentando en {wait_secs:.0f}s...")
                        time.sleep(wait_secs)
                    else:
                        time.sleep(2.0 * (attempt + 1))

        if last_error:
            raise last_error
        raise RuntimeError("No se pudo obtener veredicto del Juez Gemini LLM.")

    def evaluate_all(
        self,
        cases: Sequence[BenchmarkCase],
        gemma_outputs: Sequence[InferenceOutput],
        rag_outputs: Sequence[InferenceOutput],
        cache_file_path: Path | None = None,
    ) -> Tuple[List[EvaluatedPair], BenchmarkSummaryMetrics]:
        """Evalúa el conjunto de pares de inferencia con soporte de checkpointing / caché persistente."""
        cache_path: Path = cache_file_path or (settings.results_dir / "judge_evaluations_cache.json")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        cached_evals: Dict[str, Dict[str, str | bool]] = {}
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    if isinstance(raw_data, dict):
                        cached_evals = raw_data
            except Exception:
                cached_evals = {}

        pairs: List[EvaluatedPair] = []
        sum_points_ft: float = 0.0
        sum_points_rag: float = 0.0
        aluc_ft: int = 0
        aluc_rag: int = 0
        latencies_ft: List[float] = []
        latencies_rag: List[float] = []

        pts_map: Dict[ScoreCategory, float] = {
            ScoreCategory.CORRECTA: 1.0,
            ScoreCategory.PARCIAL: 0.5,
            ScoreCategory.INCORRECTA: 0.0,
        }

        total_cases: int = len(cases)

        # Cargar resultados cacheados de Jev si existen
        jev_cache_path: Path = settings.results_dir / "jev_evaluations_cache.json"
        cached_jev: Dict[str, JevEvaluationOutput] = {}
        if jev_cache_path.exists():
            try:
                with open(jev_cache_path, "r", encoding="utf-8") as f:
                    raw_jev = json.load(f)
                    if isinstance(raw_jev, dict):
                        for jk, jv in raw_jev.items():
                            if isinstance(jk, str) and isinstance(jv, dict):
                                cached_jev[jk] = JevEvaluationOutput.model_validate(jv)
            except Exception:
                pass

        for idx, (case, out_ft, out_rag) in enumerate(zip(cases, gemma_outputs, rag_outputs), 1):
            key_ft: str = f"case_{case.id}_ft"
            key_rag: str = f"case_{case.id}_rag"

            # Evaluar FineTuned
            if key_ft in cached_evals:
                judge_ft = JudgeEvaluationOutput.model_validate(cached_evals[key_ft])
            else:
                judge_ft = self.evaluate_with_llm_judge(
                    pregunta=case.pregunta_usuario,
                    respuesta_referencia=case.respuesta_esperada,
                    respuesta_candidata=out_ft.respuesta_generada,
                    is_rag=False,
                )
                cached_evals[key_ft] = judge_ft.model_dump()
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cached_evals, f, ensure_ascii=False, indent=2)
                time.sleep(1.0)

            # Evaluar RAG
            if key_rag in cached_evals:
                judge_rag = JudgeEvaluationOutput.model_validate(cached_evals[key_rag])
            else:
                judge_rag = self.evaluate_with_llm_judge(
                    pregunta=case.pregunta_usuario,
                    respuesta_referencia=case.respuesta_esperada,
                    respuesta_candidata=out_rag.respuesta_generada,
                    is_rag=True,
                )
                cached_evals[key_rag] = judge_rag.model_dump()
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cached_evals, f, ensure_ascii=False, indent=2)
                time.sleep(1.0)

            score_ft: ScoreCategory = judge_ft.veredicto
            score_rag: ScoreCategory = judge_rag.veredicto

            sum_points_ft += pts_map[score_ft]
            sum_points_rag += pts_map[score_rag]

            if judge_ft.alucinacion_detectada or score_ft == ScoreCategory.INCORRECTA:
                aluc_ft += 1
            if judge_rag.alucinacion_detectada or score_rag == ScoreCategory.INCORRECTA:
                aluc_rag += 1

            latencies_ft.append(out_ft.tiempo_total_ms)
            latencies_rag.append(out_rag.tiempo_total_ms)

            obs: str = (
                f"FT: [{score_ft.value}] {judge_ft.omisiones_o_errores} | "
                f"RAG: [{score_rag.value}] {judge_rag.omisiones_o_errores}"
            )

            pairs.append(
                EvaluatedPair(
                    caso=case,
                    inferencia_finetuned=out_ft,
                    inferencia_rag=out_rag,
                    score_finetuned=score_ft,
                    score_rag=score_rag,
                    observaciones=obs,
                    judge_eval_finetuned=judge_ft,
                    judge_eval_rag=judge_rag,
                    jev_eval_finetuned=cached_jev.get(key_ft),
                    jev_eval_rag=cached_jev.get(key_rag),
                )
            )

            print(f"[JUEZ LLM {idx}/{total_cases}] Caso #{case.id} | FT: {score_ft.value} | RAG: {score_rag.value}")

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
