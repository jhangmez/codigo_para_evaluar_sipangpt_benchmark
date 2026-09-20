from pathlib import Path
from sipangpt_eval.clients.local_gemma import LocalGemmaClient
from sipangpt_eval.clients.sipan_rag import SipanRAGClient
from sipangpt_eval.config import settings
from sipangpt_eval.core.runner import BenchmarkRunner
from sipangpt_eval.core.scorer import BenchmarkScorer
from sipangpt_eval.extraction.hf_loader import load_or_generate_test_benchmark
from sipangpt_eval.reporting.charts import generate_all_reports_and_charts
from sipangpt_eval.reporting.excel_generator import generate_comparative_excel


def test_full_evaluation_pipeline(tmp_path: Path) -> None:
    # 1. Extraer casos (50 casos)
    cases = load_or_generate_test_benchmark()
    assert len(cases) == 50

    # Contar monoturno vs multiturno
    monoturno_count = sum(1 for c in cases if c.tipo.value == "monoturno")
    multiturno_count = sum(1 for c in cases if c.tipo.value == "multiturno")
    assert monoturno_count + multiturno_count == 50
    assert monoturno_count >= 30
    assert multiturno_count >= 10

    # 2. Correr inferencia con clientes de prueba/mock fallback
    gemma_client = LocalGemmaClient(api_url="")
    rag_client = SipanRAGClient(api_url="")
    runner = BenchmarkRunner(gemma_client, rag_client, results_dir=tmp_path / "results")

    outs_ft, outs_rag = runner.run_benchmark(cases[:3])
    assert len(outs_ft) == 3
    assert len(outs_rag) == 3

    # 3. Evaluar y calificar usando mock para pruebas unitarias rápidas
    from unittest.mock import patch
    from sipangpt_eval.schemas.evaluation import JudgeEvaluationOutput, ScoreCategory

    mock_judge = JudgeEvaluationOutput(
        analisis_comparativo="Prueba unitaria correcta",
        omisiones_o_errores="Sin omisiones",
        alucinacion_detectada=False,
        veredicto=ScoreCategory.CORRECTA,
    )

    with patch.object(BenchmarkScorer, "evaluate_with_llm_judge", return_value=mock_judge):
        scorer = BenchmarkScorer()
        pairs, metrics = scorer.evaluate_all(cases[:3], outs_ft[:3], outs_rag[:3])
        assert len(pairs) == 3
        assert metrics.total_preguntas == 3

    # 4. Generar Excel
    excel_file = tmp_path / "results" / "matriz_comparativa_final.xlsx"
    generate_comparative_excel(pairs, metrics, output_path=excel_file)
    assert excel_file.exists()

    # 5. Generar reportes y gráficos
    fig_dir = tmp_path / "reports" / "figures"
    tab_dir = tmp_path / "reports" / "tables"

    # Redireccionar temporales en settings para la prueba
    orig_fig_dir = settings.figures_dir
    orig_tab_dir = settings.tables_dir
    try:
        generate_all_reports_and_charts(metrics, pairs)
        assert (settings.figures_dir / "curva_exactitud_comparada.png").exists()
        assert (settings.figures_dir / "latencia_boxplots.png").exists()
        assert (settings.tables_dir / "tabla_exactitud_50_preguntas.md").exists()
    finally:
        pass


def test_jev_evaluation_pipeline(tmp_path: Path) -> None:
    from unittest.mock import MagicMock, patch
    from sipangpt_eval.core.jev_scorer import JevScorer
    from sipangpt_eval.reporting.excel_generator import generate_jev_comparative_excel
    from sipangpt_eval.schemas.evaluation import JevEvaluationOutput, ScoreCategory

    cases = load_or_generate_test_benchmark()[:2]
    gemma_client = LocalGemmaClient(api_url="")
    rag_client = SipanRAGClient(api_url="")
    runner = BenchmarkRunner(gemma_client, rag_client, results_dir=tmp_path / "results")
    outs_ft, outs_rag = runner.run_benchmark(cases)

    mock_jev = JevEvaluationOutput(
        veredicto=ScoreCategory.CORRECTA,
        veredicto_choice="C",
        veredicto_confianza=0.95,
        veredicto_probabilidades={"C": 0.95, "P": 0.04, "I": 0.01},
        alucinacion_detectada=False,
        alucinacion_probabilidad=0.02,
        calidad_score=2.85,
        calidad_confianza=0.90,
        latencia_ms=450.0,
    )

    with patch.object(JevScorer, "evaluate_with_jev", return_value=mock_jev):
        scorer = JevScorer(api_key="mock_key")
        pairs, j_metrics = scorer.evaluate_all(
            cases, outs_ft, outs_rag, cache_file_path=tmp_path / "results" / "jev_cache.json"
        )
        assert len(pairs) == 2
        assert j_metrics.total_preguntas == 2
        assert j_metrics.exactitud_finetuned_pct == 100.0
        assert j_metrics.exactitud_rag_pct == 100.0

    excel_file = tmp_path / "results" / "matriz_evaluacion_jev.xlsx"
    generate_jev_comparative_excel(pairs, j_metrics, output_path=excel_file)
    assert excel_file.exists()
