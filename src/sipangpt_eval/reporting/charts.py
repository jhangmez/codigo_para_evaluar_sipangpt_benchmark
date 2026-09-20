from pathlib import Path
from typing import Dict, List, Sequence
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sipangpt_eval.config import settings
from sipangpt_eval.schemas.evaluation import (
    BenchmarkSummaryMetrics,
    EvaluatedPair,
    ScoreCategory,
)


def generate_accuracy_chart(
    metrics: BenchmarkSummaryMetrics,
    pairs: Sequence[EvaluatedPair],
    output_path: Path = settings.figures_dir / "curva_exactitud_comparada.png",
) -> None:
    """Genera gráfico comparativo de exactitud global y por módulo a 300 DPI."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calcular exactitud por modulo
    mod_scores_ft: Dict[str, List[float]] = {}
    mod_scores_rag: Dict[str, List[float]] = {}

    pts_map: Dict[ScoreCategory, float] = {
        ScoreCategory.CORRECTA: 1.0,
        ScoreCategory.PARCIAL: 0.5,
        ScoreCategory.INCORRECTA: 0.0,
    }

    for pair in pairs:
        mod_name: str = pair.caso.modulo.value
        if mod_name not in mod_scores_ft:
            mod_scores_ft[mod_name] = []
            mod_scores_rag[mod_name] = []

        mod_scores_ft[mod_name].append(pts_map[pair.score_finetuned])
        mod_scores_rag[mod_name].append(pts_map[pair.score_rag])

    mod_data: List[Dict[str, str | float]] = []
    for mod in mod_scores_ft.keys():
        pct_ft: float = (sum(mod_scores_ft[mod]) / len(mod_scores_ft[mod])) * 100.0
        pct_rag: float = (sum(mod_scores_rag[mod]) / len(mod_scores_rag[mod])) * 100.0

        mod_data.append({"Módulo": mod, "Modelo": "Gemma-4 Fine-Tuned", "Exactitud (%)": round(pct_ft, 1)})
        mod_data.append({"Módulo": mod, "Modelo": "Sipán-STAIR (RAG)", "Exactitud (%)": round(pct_rag, 1)})

    df_plot = pd.DataFrame(mod_data)

    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    palette: Dict[str, str] = {
        "Gemma-4 Fine-Tuned": "#D9534F",
        "Sipán-STAIR (RAG)": "#0275D8",
    }

    ax = sns.barplot(
        data=df_plot,
        x="Exactitud (%)",
        y="Módulo",
        hue="Modelo",
        palette=palette,
    )

    plt.title(
        "Exactitud Comparada por Módulo Académico (Fine-Tuning vs RAG)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Exactitud (%)", fontsize=11, fontweight="bold")
    plt.ylabel("Módulo Evaluado", fontsize=11, fontweight="bold")
    plt.xlim(0, 100)

    from matplotlib.container import BarContainer

    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_latency_boxplot(
    pairs: Sequence[EvaluatedPair],
    output_path: Path = settings.figures_dir / "latencia_boxplots.png",
) -> None:
    """Genera gráfico de caja (boxplot) comparativo de latencias a 300 DPI."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    latency_records: List[Dict[str, str | float]] = []
    for pair in pairs:
        latency_records.append(
            {
                "Modelo": "Gemma-4 Fine-Tuned",
                "Latencia (ms)": pair.inferencia_finetuned.tiempo_total_ms,
            }
        )
        latency_records.append(
            {
                "Modelo": "Sipán-STAIR (RAG)",
                "Latencia (ms)": pair.inferencia_rag.tiempo_total_ms,
            }
        )

    df_lat = pd.DataFrame(latency_records)

    plt.figure(figsize=(8, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    palette: Dict[str, str] = {
        "Gemma-4 Fine-Tuned": "#5CB85C",
        "Sipán-STAIR (RAG)": "#F0AD4E",
    }

    sns.boxplot(
        data=df_lat,
        x="Modelo",
        y="Latencia (ms)",
        hue="Modelo",
        palette=palette,
        legend=False,
        width=0.4,
    )

    plt.title(
        "Distribución de Latencia de Inferencia (ms)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Modelo Evaluado", fontsize=11, fontweight="bold")
    plt.ylabel("Tiempo de Respuesta (ms)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_markdown_table(
    metrics: BenchmarkSummaryMetrics,
    pairs: Sequence[EvaluatedPair],
    output_path: Path = settings.tables_dir / "tabla_exactitud_50_preguntas.md",
) -> None:
    """Genera la tabla de resultados del Capítulo V formateada en Markdown para el informe de Tesis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Conteo por categorías
    c_ft: int = sum(1 for p in pairs if p.score_finetuned == ScoreCategory.CORRECTA)
    p_ft: int = sum(1 for p in pairs if p.score_finetuned == ScoreCategory.PARCIAL)
    i_ft: int = sum(1 for p in pairs if p.score_finetuned == ScoreCategory.INCORRECTA)

    c_rag: int = sum(1 for p in pairs if p.score_rag == ScoreCategory.CORRECTA)
    p_rag: int = sum(1 for p in pairs if p.score_rag == ScoreCategory.PARCIAL)
    i_rag: int = sum(1 for p in pairs if p.score_rag == ScoreCategory.INCORRECTA)

    md_content: str = f"""# Capítulo V: Resultados y Evaluación del Benchmark

## Tabla V.1: Matriz Comparativa de Desempeño (50 Preguntas de Prueba)

| Métrica / Dimensión de Evaluación | Gemma-4 Fine-Tuned (LoRA) | Sipán-STAIR (RAG Architecture) | Diferencia (%) |
| :--- | :---: | :---: | :---: |
| **Total de Casos Evaluados** | {metrics.total_preguntas} | {metrics.total_preguntas} | - |
| **Respuestas Correctas (C - 1.0 pt)** | {c_ft} ({c_ft / metrics.total_preguntas * 100:.1f}%) | {c_rag} ({c_rag / metrics.total_preguntas * 100:.1f}%) | +{(c_rag - c_ft) / metrics.total_preguntas * 100:.1f}% |
| **Respuestas Parciales (P - 0.5 pt)** | {p_ft} ({p_ft / metrics.total_preguntas * 100:.1f}%) | {p_rag} ({p_rag / metrics.total_preguntas * 100:.1f}%) | {(p_rag - p_ft) / metrics.total_preguntas * 100:.1f}% |
| **Respuestas Incorrectas / Alucinaciones (I - 0.0 pt)** | {i_ft} ({i_ft / metrics.total_preguntas * 100:.1f}%) | {i_rag} ({i_rag / metrics.total_preguntas * 100:.1f}%) | -{(i_ft - i_rag) / metrics.total_preguntas * 100:.1f}% |
| **Porcentaje Global de Exactitud** | **{metrics.exactitud_finetuned_pct:.2f}%** | **{metrics.exactitud_rag_pct:.2f}%** | **+{(metrics.exactitud_rag_pct - metrics.exactitud_finetuned_pct):.2f}%** |
| **Latencia Media por Consulta** | {metrics.latencia_media_finetuned_ms:.2f} ms | {metrics.latencia_media_rag_ms:.2f} ms | +{metrics.latencia_media_rag_ms - metrics.latencia_media_finetuned_ms:.2f} ms |

---

### Resumen Técnico de los Resultados

1. **Exactitud y Reducción de Alucinaciones:**
   La arquitectura **Sipán-STAIR (RAG)** alcanza una exactitud del **{metrics.exactitud_rag_pct:.2f}%** en comparación con el **{metrics.exactitud_finetuned_pct:.2f}%** obtenido por **Gemma-4 Fine-Tuned**.
   Esto representa un incremento neto de **{(metrics.exactitud_rag_pct - metrics.exactitud_finetuned_pct):.2f}%** en la fidelidad fáctica gracias al mecanismo de recuperación de contexto y citación directa de normativas institucionales.

2. **Evaluación de Latencia:**
   El modelo fine-tuned presenta una latencia media de **{metrics.latencia_media_finetuned_ms:.2f} ms**, mientras que el pipeline RAG requiere **{metrics.latencia_media_rag_ms:.2f} ms** debido a la etapa adicional de búsqueda vectorial y reranking.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)


def generate_all_reports_and_charts(
    metrics: BenchmarkSummaryMetrics, pairs: Sequence[EvaluatedPair]
) -> None:
    """Genera todos los reportes, figuras a 300 DPI y tablas Markdown."""
    generate_accuracy_chart(metrics, pairs)
    generate_latency_boxplot(pairs)
    generate_markdown_table(metrics, pairs)
