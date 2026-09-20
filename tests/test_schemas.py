from sipangpt_eval.schemas.benchmark import (
    BenchmarkCase,
    ConversationMessage,
    ModuleCategory,
    TurnType,
)
from sipangpt_eval.schemas.evaluation import (
    BenchmarkSummaryMetrics,
    EvaluatedPair,
    ScoreCategory,
)
from sipangpt_eval.schemas.inference import CitationSource, InferenceOutput


def test_benchmark_case_schema() -> None:
    case = BenchmarkCase(
        id=1,
        modulo=ModuleCategory.MATRICULA,
        tipo=TurnType.MONOTURNO,
        pregunta_usuario="¿Cómo realizo el trámite de reserva de matrícula?",
        respuesta_esperada="Presentar solicitud en Mesa de Partes Virtual.",
        documento_origen="Reglamento USS",
        seccion_o_articulo="Artículo 12",
        conversacion_completa=[
            ConversationMessage(role="human", content="¿Cómo realizo el trámite?"),
            ConversationMessage(role="gpt", content="Presentar solicitud."),
        ],
    )
    assert case.id == 1
    assert case.modulo == ModuleCategory.MATRICULA
    assert case.tipo == TurnType.MONOTURNO
    assert case.conversacion_completa is not None
    assert len(case.conversacion_completa) == 2


def test_inference_output_schema() -> None:
    citation = CitationSource(
        documento="Reglamento Académico 2024",
        articulo_o_seccion="Art. 45",
        url_publica="https://www.uss.edu.pe/reglamentos",
        similitud=0.95,
    )
    output = InferenceOutput(
        modelo_nombre="SipánGPT-RAG-STAIR",
        respuesta_generada="De acuerdo con el Reglamento Académico...",
        tiempo_total_ms=450.0,
        tiempo_busqueda_ms=120.0,
        tiempo_generacion_ms=330.0,
        tokens_totales=45,
        citas=[citation],
    )
    assert output.modelo_nombre == "SipánGPT-RAG-STAIR"
    assert len(output.citas) == 1
    assert output.citas[0].similitud == 0.95


def test_evaluation_schemas() -> None:
    metrics = BenchmarkSummaryMetrics(
        total_preguntas=50,
        exactitud_finetuned_pct=65.0,
        exactitud_rag_pct=92.0,
        alucinaciones_finetuned_count=12,
        alucinaciones_rag_count=2,
        latencia_media_finetuned_ms=480.0,
        latencia_media_rag_ms=850.0,
    )
    assert metrics.total_preguntas == 50
    assert metrics.exactitud_rag_pct > metrics.exactitud_finetuned_pct
