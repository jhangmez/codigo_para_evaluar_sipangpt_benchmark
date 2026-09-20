from pathlib import Path
from typing import List, Tuple
from rich.console import Console
from rich.table import Table
import typer

from sipangpt_eval.clients.local_gemma import LocalGemmaClient
from sipangpt_eval.clients.sipan_rag import SipanRAGClient
from sipangpt_eval.config import settings
from sipangpt_eval.core.runner import BenchmarkRunner
from sipangpt_eval.core.scorer import BenchmarkScorer
from sipangpt_eval.extraction.hf_loader import load_or_generate_test_benchmark
from sipangpt_eval.reporting.charts import generate_all_reports_and_charts
from sipangpt_eval.reporting.excel_generator import generate_comparative_excel
from sipangpt_eval.schemas.benchmark import BenchmarkCase
from sipangpt_eval.schemas.evaluation import BenchmarkSummaryMetrics, EvaluatedPair
from sipangpt_eval.schemas.inference import InferenceOutput

app = typer.Typer(
    name="codigo-para-evaluar-sipangpt-benchmark",
    help="Suite de Evaluación Comparativa de Rigor Científico: Gemma-4 Fine-Tuned vs Sipán-STAIR RAG",
)
console = Console()


@app.command()
def extract() -> None:
    """Extrae y genera el conjunto de prueba oficial de 50 preguntas (Ground Truth)."""
    console.print("[bold blue]Descargando / Extrayendo 50 preguntas de prueba...[/bold blue]")
    cases: List[BenchmarkCase] = load_or_generate_test_benchmark()
    console.print(
        f"[bold green]✓ Exitosamente extraídas {len(cases)} preguntas de prueba.[/bold green]"
    )
    console.print(f"-> Guardado en: {settings.ground_truth_dir / 'benchmark_test_50.jsonl'}")
    console.print(f"-> Excel generado: {settings.results_dir / 'banco_pruebas_50.xlsx'}")


@app.command()
def run() -> Tuple[List[InferenceOutput], List[InferenceOutput]]:
    """Ejecuta inferencia de los 50 casos en Gemma-4 Fine-Tuned y Sipán-STAIR RAG."""
    cases: List[BenchmarkCase] = load_or_generate_test_benchmark()

    gemma_client = LocalGemmaClient(
        api_url=settings.gemma_api_url, model_name=settings.gemma_model_name
    )
    rag_client = SipanRAGClient(
        api_url=settings.sipan_rag_api_url, api_token=settings.sipan_rag_api_token
    )

    runner = BenchmarkRunner(
        gemma_client=gemma_client, rag_client=rag_client, results_dir=settings.results_dir
    )

    console.print("[bold yellow]Iniciando ejecución de inferencia en ambos modelos...[/bold yellow]")
    outs_ft, outs_rag = runner.run_benchmark(cases)
    console.print("[bold green]✓ Inferencia completada con éxito.[/bold green]")
    console.print(f"-> Outputs Gemma-4: {settings.results_dir / 'run_gemma4_finetuned.json'}")
    console.print(f"-> Outputs RAG: {settings.results_dir / 'run_sipan_stair_rag.json'}")
    return outs_ft, outs_rag


@app.command()
def evaluate() -> Tuple[List[EvaluatedPair], BenchmarkSummaryMetrics]:
    """Evalúa los resultados de inferencia, califica (C/P/I) y genera la matriz Excel."""
    cases: List[BenchmarkCase] = load_or_generate_test_benchmark()

    gemma_client = LocalGemmaClient(
        api_url=settings.gemma_api_url, model_name=settings.gemma_model_name
    )
    rag_client = SipanRAGClient(
        api_url=settings.sipan_rag_api_url, api_token=settings.sipan_rag_api_token
    )
    runner = BenchmarkRunner(
        gemma_client=gemma_client, rag_client=rag_client, results_dir=settings.results_dir
    )

    outs_ft, outs_rag = runner.run_benchmark(cases)

    scorer = BenchmarkScorer()
    pairs, metrics = scorer.evaluate_all(cases, outs_ft, outs_rag)

    excel_path: Path = settings.results_dir / "matriz_comparativa_final.xlsx"
    generate_comparative_excel(pairs, metrics, excel_path)

    # Imprimir tabla resumida en consola
    table = Table(title="Resumen General de Métricas del Benchmark")
    table.add_column("Métrica", style="cyan", no_wrap=True)
    table.add_column("Gemma-4 Fine-Tuned", style="magenta")
    table.add_column("Sipán-STAIR (RAG)", style="green")

    table.add_row("Total Preguntas Evaluadas", str(metrics.total_preguntas), str(metrics.total_preguntas))
    table.add_row("Exactitud Global (%)", f"{metrics.exactitud_finetuned_pct:.2f}%", f"{metrics.exactitud_rag_pct:.2f}%")
    table.add_row("Total Alucinaciones (Conteo I)", str(metrics.alucinaciones_finetuned_count), str(metrics.alucinaciones_rag_count))
    table.add_row("Latencia Media (ms)", f"{metrics.latencia_media_finetuned_ms:.2f} ms", f"{metrics.latencia_media_rag_ms:.2f} ms")

    console.print(table)
    console.print(f"[bold green]✓ Matriz Excel generada:[/bold green] {excel_path}")
    return pairs, metrics


@app.command()
def report() -> None:
    """Genera gráficos a 300 DPI y tablas Markdown para la Tesis."""
    pairs, metrics = evaluate()
    console.print("[bold cyan]Generando gráficos comparativos (300 DPI) y tablas para la Tesis...[/bold cyan]")
    generate_all_reports_and_charts(metrics, pairs)
    console.print(f"-> Gráficos guardados en: {settings.figures_dir}")
    console.print(f"-> Tabla Markdown para Tesis: {settings.tables_dir / 'tabla_exactitud_50_preguntas.md'}")


@app.command(name="all")
def full_pipeline() -> None:
    """Ejecuta el flujo de trabajo completo: Extracción -> Inferencia -> Evaluación -> Reportes."""
    console.print("[bold green]=== INICIANDO PIPELINE DE EVALUACIÓN BENCHMARK SIPANGPT ===[/bold green]")
    extract()
    report()
    console.print("[bold green]=== PIPELINE COMPLETADO EXITOSAMENTE ===[/bold green]")


if __name__ == "__main__":
    app()
