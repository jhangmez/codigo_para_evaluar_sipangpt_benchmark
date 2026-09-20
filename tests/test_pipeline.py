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
    gemma_client = LocalGemmaClient()
    rag_client = SipanRAGClient()
    runner = BenchmarkRunner(gemma_client, rag_client, results_dir=tmp_path / "results")

    outs_ft, outs_rag = runner.run_benchmark(cases)
    assert len(outs_ft) == 50
    assert len(outs_rag) == 50

    # 3. Evaluar y calificar
    scorer = BenchmarkScorer()
    pairs, metrics = scorer.evaluate_all(cases, outs_ft, outs_rag)
    assert len(pairs) == 50
    assert metrics.total_preguntas == 50

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
