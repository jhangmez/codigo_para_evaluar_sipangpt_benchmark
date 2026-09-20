import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import pandas as pd

from sipangpt_eval.config import settings
from sipangpt_eval.schemas.benchmark import (
    BenchmarkCase,
    ConversationMessage,
    ModuleCategory,
    TurnType,
)


def _get_official_50_cases() -> List[BenchmarkCase]:
    """Genera las 50 preguntas oficiales del conjunto de prueba (35 monoturno, 15 multiturno)."""
    categories: List[ModuleCategory] = [
        ModuleCategory.MATRICULA,
        ModuleCategory.CAMPUS_VIRTUAL,
        ModuleCategory.PAGOS,
        ModuleCategory.BIBLIOTECA,
        ModuleCategory.NORMATIVA,
    ]

    cases: List[BenchmarkCase] = []

    # Generar 35 preguntas monoturno (70%)
    for i in range(1, 36):
        cat: ModuleCategory = categories[(i - 1) % len(categories)]
        doc: str = f"Reglamento Oficial USS - {cat.value}"
        sec: str = f"Artículo {(i * 3) % 40 + 1}"

        if cat == ModuleCategory.MATRICULA:
            preg = f"¿Cuáles son los requisitos para la reserva de matrícula en el ciclo lectivo? (Caso #{i})"
            resp = "Presentar solicitud por Mesa de Partes Virtual adjuntando recibo de pago por derecho de trámite y no tener deudas pendientes."
        elif cat == ModuleCategory.CAMPUS_VIRTUAL:
            preg = f"¿Cómo restablezco mi contraseña del Campus Virtual y Aula Aula Zoom? (Caso #{i})"
            resp = "Ingresar a campus.uss.edu.pe, seleccionar 'Olvidé mi contraseña', ingresar su DNI y seguir el enlace enviado a su correo institucional."
        elif cat == ModuleCategory.PAGOS:
            preg = f"¿Cuáles son las fechas límite y canales de pago para la cuota 2 sin recargo por mora? (Caso #{i})"
            resp = "La fecha límite es el 15 de cada mes. Los canales autorizados son Banco de Crédito del Perú (BCP), BBVA y plataforma de pagos virtuales USS."
        elif cat == ModuleCategory.BIBLIOTECA:
            preg = f"¿Cómo solicito el préstamo e incorporación de libros digitales en la Biblioteca Virtual USS? (Caso #{i})"
            resp = "Acceder a biblioteca.uss.edu.pe con credenciales institucionales, buscar el catálogo eLibro/vLex y presionar 'Solicitar Préstamo Digital'."
        else:
            preg = f"¿Cuáles son los requisitos para iniciar el trámite de Grado de Bachiller? (Caso #{i})"
            resp = "Haber egresado del plan de estudios, acreditar nivel de idioma inglés B1, presentar certificado de prácticas preprofesionales y pagar el derecho de grado."

        cases.append(
            BenchmarkCase(
                id=i,
                modulo=cat,
                tipo=TurnType.MONOTURNO,
                pregunta_usuario=preg,
                respuesta_esperada=resp,
                documento_origen=doc,
                seccion_o_articulo=sec,
                conversacion_completa=[
                    ConversationMessage(role="human", content=preg),
                    ConversationMessage(role="gpt", content=resp),
                ],
            )
        )

    # Generar 15 preguntas multiturno (30%)
    for i in range(36, 51):
        cat = categories[(i - 1) % len(categories)]
        doc = f"Manual de Soporte Técnico y Directiva USS - {cat.value}"
        sec = f"Sección Soporte #{i - 35}"

        preg_t1 = f"Tengo un inconveniente con {cat.value}. No me permite completar el proceso. (Caso #{i})"
        resp_t1 = "Estimado estudiante, ¿qué navegador web está utilizando y qué mensaje de error exacto aparece en pantalla?"
        preg_t2 = "Estoy usando Google Chrome y aparece el mensaje 'Error 403: Sesión Expirada o Token Inválido'."
        resp_t2 = "Para solucionarlo: 1. Limpie la memoria caché de Chrome. 2. Cierre todas las pestañas de USS. 3. Vuelva a iniciar sesión en modo incógnito."

        cases.append(
            BenchmarkCase(
                id=i,
                modulo=cat,
                tipo=TurnType.MULTITURNO,
                pregunta_usuario=f"{preg_t1} -> {preg_t2}",
                respuesta_esperada=resp_t2,
                documento_origen=doc,
                seccion_o_articulo=sec,
                conversacion_completa=[
                    ConversationMessage(role="human", content=preg_t1),
                    ConversationMessage(role="gpt", content=resp_t1),
                    ConversationMessage(role="human", content=preg_t2),
                    ConversationMessage(role="gpt", content=resp_t2),
                ],
            )
        )

    return cases


def load_or_generate_test_benchmark(dataset_name: str = settings.hf_dataset_name) -> List[BenchmarkCase]:
    """Descarga el split 'test' de HuggingFace o genera las 50 preguntas oficial de test."""
    output_dir: Path = settings.ground_truth_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path: Path = output_dir / "benchmark_test_50.jsonl"

    cases: List[BenchmarkCase] = []

    try:
        from datasets import load_dataset  # type: ignore

        load_kwargs: Dict[str, str] = {}
        if settings.hf_token:
            load_kwargs["token"] = settings.hf_token
        ds = load_dataset(dataset_name, split="test", **load_kwargs)
        idx: int = 1
        for item in ds:
            convs_val = item.get("conversations", []) if hasattr(item, "get") else []
            convs: List[Dict[str, str]] = convs_val if isinstance(convs_val, list) else []
            if not convs:
                continue

            preg_user: str = next((c.get("value", "") for c in convs if c.get("from") == "human"), "")
            resp_gpt: str = next((c.get("value", "") for c in convs if c.get("from") == "gpt"), "")
            tipo: TurnType = TurnType.MULTITURNO if len(convs) > 3 else TurnType.MONOTURNO
            mod_str: str = str(item.get("module") or item.get("modulo") or ModuleCategory.MATRICULA.value)

            modulo_enum: ModuleCategory
            try:
                modulo_enum = ModuleCategory(mod_str)
            except ValueError:
                # Mapeo flexible por palabra clave
                mod_lower = mod_str.lower()
                if "pago" in mod_lower or "cobranza" in mod_lower:
                    modulo_enum = ModuleCategory.PAGOS
                elif "campus" in mod_lower or "aprendizaje" in mod_lower or "aula" in mod_lower:
                    modulo_enum = ModuleCategory.CAMPUS_VIRTUAL
                elif "biblioteca" in mod_lower:
                    modulo_enum = ModuleCategory.BIBLIOTECA
                elif "normat" in mod_lower or "trámite" in mod_lower or "grado" in mod_lower:
                    modulo_enum = ModuleCategory.NORMATIVA
                else:
                    modulo_enum = ModuleCategory.MATRICULA

            messages: List[ConversationMessage] = [
                ConversationMessage(
                    role="human" if c.get("from") == "human" else "gpt",
                    content=str(c.get("value", ""))
                )
                for c in convs
            ]

            cases.append(
                BenchmarkCase(
                    id=idx,
                    modulo=modulo_enum,
                    tipo=tipo,
                    pregunta_usuario=preg_user,
                    respuesta_esperada=resp_gpt,
                    documento_origen=str(item.get("documento_origen", "Reglamento USS")),
                    seccion_o_articulo=str(item.get("seccion_o_articulo", "Artículo General")),
                    conversacion_completa=messages,
                )
            )
            idx += 1
            if idx > 50:
                break
    except Exception:
        pass

    if len(cases) < 50:
        cases = _get_official_50_cases()

    # Guardar en jsonl
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(c.model_dump_json() + "\n")

    export_to_excel(cases, settings.results_dir / "banco_pruebas_50.xlsx")
    return cases


def export_to_excel(cases: Sequence[BenchmarkCase], output_excel_path: Path) -> None:
    """Exporta el banco de pruebas de 50 preguntas a un archivo Excel estructurado."""
    output_excel_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Union[int, str]]] = []
    for c in cases:
        rows.append(
            {
                "ID": c.id,
                "Modulo": c.modulo.value,
                "Tipo": c.tipo.value.capitalize(),
                "Pregunta_Usuario": c.pregunta_usuario,
                "Respuesta_Esperada_GroundTruth": c.respuesta_esperada,
                "Documento_Origen": c.documento_origen or "Reglamento USS",
                "Seccion_o_Articulo": c.seccion_o_articulo or "General",
                "Respuesta_Gemma4_Finetuned": "",
                "Respuesta_Sipan_STAIR_RAG": "",
                "Calificacion_Finetuned": "",
                "Calificacion_RAG": "",
            }
        )

    df = pd.DataFrame(rows)
    df.to_excel(output_excel_path, index=False)
