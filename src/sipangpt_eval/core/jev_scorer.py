import json
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import httpx

from sipangpt_eval.config import settings
from sipangpt_eval.schemas.benchmark import BenchmarkCase
from sipangpt_eval.schemas.evaluation import (
    EvaluatedPair,
    JevEvaluationOutput,
    JevSummaryMetrics,
    ScoreCategory,
)
from sipangpt_eval.schemas.inference import InferenceOutput


class JevScorer:
    """Evaluador de decisiones tipadas basado en el modelo System One TypeSafe AI Jev vía Vercel AI Gateway."""

    def __init__(
        self,
        api_key: str = "",
        gateway_url: str = "",
    ) -> None:
        self.api_key: str = (
            api_key
            or settings.vercel_ai_gateway_key
            or os.environ.get("VERCEL_AI_GATEWAY_KEY", "")
            or os.environ.get("AI_GATEWAY_API_KEY", "")
        )
        self.gateway_url: str = (
            gateway_url
            or settings.vercel_ai_gateway_url
            or "https://ai-gateway.vercel.sh/v1/evaluate"
        )

    def evaluate_with_jev(
        self,
        pregunta: str,
        respuesta_referencia: str,
        respuesta_candidata: str,
        is_rag: bool = False,
    ) -> JevEvaluationOutput:
        """Evalúa una respuesta usando el modelo de decisión tipada Jev (System One)."""
        if not self.api_key:
            raise ValueError("VERCEL_AI_GATEWAY_KEY no configurada para el evaluador Jev.")

        state_context: str = (
            f"[Pregunta del Estudiante USS]\n{pregunta}\n\n"
            f"[Respuesta Oficial de Referencia (Ground Truth USS)]\n{respuesta_referencia}\n\n"
            f"[Respuesta Generada por el Asistente a Evaluar ('{'Sipán-STAIR RAG' if is_rag else 'Gemma-4 Fine-Tuned'}')]\n{respuesta_candidata}"
        )

        payload: Dict[str, object] = {
            "model": "typesafe-ai/jev",
            "state": state_context,
            "questions": {
                "veredicto": {
                    "type": "choice",
                    "instructions": (
                        "Evalúa estrictamente si la respuesta generada es fáctica y conforme a la referencia oficial:\n"
                        "- 'C' (Correcta): Contiene los pasos indispensables y datos fácticos exactos según la referencia USS.\n"
                        "- 'P' (Parcial): Orientación general útil pero omite pasos esenciales, requisitos clave o datos fácticos indispensables.\n"
                        "- 'I' (Incorrecta): Contiene alucinaciones, inventa plataformas o procedimientos, o contradice la normativa oficial USS."
                    ),
                    "criteria": {
                        "C": "Correcta: pasos indispensables y datos fácticos exactos según la referencia oficial USS.",
                        "P": "Parcial: orientación útil pero omite pasos clave, requisitos fácticos indispensables o plazos.",
                        "I": "Incorrecta: alucinaciones, procedimientos inventados, enlaces falsos o contradicciones normativas.",
                    },
                },
                "alucinacion": {
                    "type": "boolean",
                    "instructions": "¿La respuesta generada contiene alucinaciones, afirmaciones falsas o inventa plataformas/trámites inexistentes en la USS?",
                },
                "calidad_tecnica": {
                    "type": "score",
                    "instructions": "Nivel de calidad técnica, completitud y apego normativo oficial:",
                    "criteria": [
                        "deficiente con errores o alucinaciones",
                        "regular con omisiones notorias",
                        "buena y precisa",
                        "excelente y totalmente exhaustiva",
                    ],
                },
            },
        }

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start_time: float = time.perf_counter()
        last_err: Exception | None = None
        data: Dict[str, object] = {}

        for attempt in range(6):
            try:
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(self.gateway_url, headers=headers, json=payload)
                if response.status_code == 200:
                    raw_data = response.json()
                    if isinstance(raw_data, dict):
                        data = raw_data
                    break
                elif response.status_code in (429, 503, 504):
                    wait_time: float = 4.0 * (attempt + 1)
                    print(f"  [429/503 Rate Limit en Jev] Esperando {wait_time:.0f}s (intento {attempt + 1}/6)...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"Error en Vercel AI Gateway (Jev): {response.status_code} - {response.text}"
                    )
            except httpx.RequestError as exc:
                last_err = exc
                time.sleep(3.0 * (attempt + 1))

        if not data:
            if last_err:
                raise last_err
            raise RuntimeError("No se pudo obtener respuesta válida de Vercel AI Gateway (Jev).")

        latencia_ms: float = (time.perf_counter() - start_time) * 1000.0
        raw_answers = data.get("answers", {})
        answers: Dict[str, Dict[str, object]] = {}
        if isinstance(raw_answers, dict):
            for k_ans, v_ans in raw_answers.items():
                if isinstance(k_ans, str) and isinstance(v_ans, dict):
                    sub_d: Dict[str, object] = {}
                    for sub_k, sub_v in v_ans.items():
                        if isinstance(sub_k, str):
                            sub_d[sub_k] = sub_v
                    answers[k_ans] = sub_d

        veredicto_data = answers.get("veredicto", {})
        raw_choice = veredicto_data.get("choice", "I")
        choice_val: str = str(raw_choice) if raw_choice is not None else "I"
        veredicto_cat: ScoreCategory
        if choice_val == "C":
            veredicto_cat = ScoreCategory.CORRECTA
        elif choice_val == "P":
            veredicto_cat = ScoreCategory.PARCIAL
        else:
            veredicto_cat = ScoreCategory.INCORRECTA

        raw_conf = veredicto_data.get("confidence", 0.0)
        confianza_val: float = float(raw_conf) if isinstance(raw_conf, (int, float)) else 0.0

        raw_probs = veredicto_data.get("probabilities", {})
        probs_map: Dict[str, float] = {}
        if isinstance(raw_probs, dict):
            for k_pr, v_pr in raw_probs.items():
                if isinstance(v_pr, (int, float)):
                    probs_map[str(k_pr)] = float(v_pr)

        aluc_data = answers.get("alucinacion", {})
        raw_aluc_p = aluc_data.get("probability", 0.0)
        aluc_prob: float = float(raw_aluc_p) if isinstance(raw_aluc_p, (int, float)) else 0.0
        aluc_detected: bool = aluc_prob >= 0.50 or veredicto_cat == ScoreCategory.INCORRECTA

        calidad_data = answers.get("calidad_tecnica", {})
        raw_cal_s = calidad_data.get("score", 0.0)
        calidad_score_val: float = float(raw_cal_s) if isinstance(raw_cal_s, (int, float)) else 0.0
        raw_cal_c = calidad_data.get("confidence", 0.0)
        calidad_conf_val: float = float(raw_cal_c) if isinstance(raw_cal_c, (int, float)) else 0.0

        return JevEvaluationOutput(
            veredicto=veredicto_cat,
            veredicto_choice=choice_val,
            veredicto_confianza=confianza_val,
            veredicto_probabilidades=probs_map,
            alucinacion_detectada=aluc_detected,
            alucinacion_probabilidad=aluc_prob,
            calidad_score=calidad_score_val,
            calidad_confianza=calidad_conf_val,
            latencia_ms=round(latencia_ms, 2),
        )

    def evaluate_all(
        self,
        cases: Sequence[BenchmarkCase],
        gemma_outputs: Sequence[InferenceOutput],
        rag_outputs: Sequence[InferenceOutput],
        cache_file_path: Path | None = None,
    ) -> Tuple[List[EvaluatedPair], JevSummaryMetrics]:
        """Evalúa todos los casos con Jev System One utilizando persistencia progresiva en caché."""
        cache_path: Path = cache_file_path or (settings.results_dir / "jev_evaluations_cache.json")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        cached_evals: Dict[str, Dict[str, object]] = {}
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    if isinstance(raw_data, dict):
                        for k, v in raw_data.items():
                            if isinstance(k, str) and isinstance(v, dict):
                                sub_dict: Dict[str, object] = {str(sk): sv for sk, sv in v.items()}
                                cached_evals[k] = sub_dict
            except Exception:
                cached_evals = {}

        pairs: List[EvaluatedPair] = []
        pts_map: Dict[ScoreCategory, float] = {
            ScoreCategory.CORRECTA: 1.0,
            ScoreCategory.PARCIAL: 0.5,
            ScoreCategory.INCORRECTA: 0.0,
        }

        sum_points_ft: float = 0.0
        sum_points_rag: float = 0.0
        aluc_ft: int = 0
        aluc_rag: int = 0
        calidad_ft_list: List[float] = []
        calidad_rag_list: List[float] = []
        latencies_jev: List[float] = []

        total_cases: int = len(cases)
        print(f"\n[JEV SYSTEM ONE] Evaluando {total_cases} casos con TypeSafe AI Jev...")

        for idx, (case, out_ft, out_rag) in enumerate(zip(cases, gemma_outputs, rag_outputs), 1):
            key_ft: str = f"case_{case.id}_ft"
            key_rag: str = f"case_{case.id}_rag"

            # Evaluar Gemma-4 Fine-Tuned
            if key_ft in cached_evals:
                jev_ft = JevEvaluationOutput.model_validate(cached_evals[key_ft])
            else:
                jev_ft = self.evaluate_with_jev(
                    pregunta=case.pregunta_usuario,
                    respuesta_referencia=case.respuesta_esperada,
                    respuesta_candidata=out_ft.respuesta_generada,
                    is_rag=False,
                )
                cached_evals[key_ft] = jev_ft.model_dump()
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cached_evals, f, ensure_ascii=False, indent=2)
                time.sleep(1.0)

            # Evaluar Sipán-STAIR RAG
            if key_rag in cached_evals:
                jev_rag = JevEvaluationOutput.model_validate(cached_evals[key_rag])
            else:
                jev_rag = self.evaluate_with_jev(
                    pregunta=case.pregunta_usuario,
                    respuesta_referencia=case.respuesta_esperada,
                    respuesta_candidata=out_rag.respuesta_generada,
                    is_rag=True,
                )
                cached_evals[key_rag] = jev_rag.model_dump()
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cached_evals, f, ensure_ascii=False, indent=2)
                time.sleep(1.0)

            score_ft = jev_ft.veredicto
            score_rag = jev_rag.veredicto

            sum_points_ft += pts_map[score_ft]
            sum_points_rag += pts_map[score_rag]

            if jev_ft.alucinacion_detectada:
                aluc_ft += 1
            if jev_rag.alucinacion_detectada:
                aluc_rag += 1

            calidad_ft_list.append(jev_ft.calidad_score)
            calidad_rag_list.append(jev_rag.calidad_score)
            latencies_jev.extend([jev_ft.latencia_ms, jev_rag.latencia_ms])

            obs: str = (
                f"JEV FT: [{score_ft.value}] (Conf: {jev_ft.veredicto_confianza:.2f}, Aluc: {jev_ft.alucinacion_probabilidad:.2f}) | "
                f"JEV RAG: [{score_rag.value}] (Conf: {jev_rag.veredicto_confianza:.2f}, Aluc: {jev_rag.alucinacion_probabilidad:.2f})"
            )

            pairs.append(
                EvaluatedPair(
                    caso=case,
                    inferencia_finetuned=out_ft,
                    inferencia_rag=out_rag,
                    score_finetuned=score_ft,
                    score_rag=score_rag,
                    observaciones=obs,
                    jev_eval_finetuned=jev_ft,
                    jev_eval_rag=jev_rag,
                )
            )

            print(
                f"[JEV {idx}/{total_cases}] Caso #{case.id} [{case.modulo.value}] | "
                f"FT: {score_ft.value} (conf: {jev_ft.veredicto_confianza:.2f}) | "
                f"RAG: {score_rag.value} (conf: {jev_rag.veredicto_confianza:.2f}) | "
                f"Lat: {jev_ft.latencia_ms:.0f}ms / {jev_rag.latencia_ms:.0f}ms"
            )

        exactitud_ft_pct: float = (sum_points_ft / max(1, total_cases)) * 100.0
        exactitud_rag_pct: float = (sum_points_rag / max(1, total_cases)) * 100.0
        mean_calidad_ft: float = sum(calidad_ft_list) / max(1, len(calidad_ft_list))
        mean_calidad_rag: float = sum(calidad_rag_list) / max(1, len(calidad_rag_list))
        mean_lat_jev: float = sum(latencies_jev) / max(1, len(latencies_jev))

        metrics = JevSummaryMetrics(
            total_preguntas=total_cases,
            exactitud_finetuned_pct=round(exactitud_ft_pct, 2),
            exactitud_rag_pct=round(exactitud_rag_pct, 2),
            alucinaciones_finetuned_count=aluc_ft,
            alucinaciones_rag_count=aluc_rag,
            promedio_calidad_finetuned=round(mean_calidad_ft, 2),
            promedio_calidad_rag=round(mean_calidad_rag, 2),
            latencia_media_ms=round(mean_lat_jev, 2),
        )

        return pairs, metrics
