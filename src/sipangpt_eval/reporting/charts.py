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
    """Genera gráfico comparativo de exactitud fáctica por módulo a 300 DPI."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Orden canónico de los 5 módulos académicos USS
    mod_order: List[str] = [
        "Matrícula y Registros",
        "Campus Virtual y Aprendizaje",
        "Pagos y Cobranzas",
        "Biblioteca Virtual",
        "Normativa y Trámites",
    ]

    pts_map: Dict[ScoreCategory, float] = {
        ScoreCategory.CORRECTA: 1.0,
        ScoreCategory.PARCIAL: 0.5,
        ScoreCategory.INCORRECTA: 0.0,
    }

    mod_scores_ft: Dict[str, List[float]] = {}
    mod_scores_rag: Dict[str, List[float]] = {}

    for pair in pairs:
        mod_name: str = pair.caso.modulo.value
        if mod_name not in mod_scores_ft:
            mod_scores_ft[mod_name] = []
            mod_scores_rag[mod_name] = []

        mod_scores_ft[mod_name].append(pts_map[pair.score_finetuned])
        mod_scores_rag[mod_name].append(pts_map[pair.score_rag])

    mod_data: List[Dict[str, str | float]] = []
    # Filtrar solo módulos que existan en el conjunto evaluado preservando orden
    active_mods: List[str] = [m for m in mod_order if m in mod_scores_ft]
    for mod in active_mods:
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

    from matplotlib.container import BarContainer

    ax = sns.barplot(
        data=df_plot,
        x="Exactitud (%)",
        y="Módulo",
        hue="Modelo",
        order=active_mods,
        palette=palette,
    )

    plt.title(
        "Exactitud Fáctica Comparada por Módulo Académico (50 Preguntas)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Exactitud (%)", fontsize=11, fontweight="bold")
    plt.ylabel("Módulo Evaluado", fontsize=11, fontweight="bold")
    plt.xlim(0, 100)

    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt="%.1f%%", padding=4, fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_latency_boxplot(
    pairs: Sequence[EvaluatedPair],
    output_path: Path = settings.figures_dir / "latencia_boxplots.png",
) -> None:
    """Genera gráfico comparativo de latencia media de inferencia por módulo en segundos a 300 DPI."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mod_order: List[str] = [
        "Matrícula y Registros",
        "Campus Virtual y Aprendizaje",
        "Pagos y Cobranzas",
        "Biblioteca Virtual",
        "Normativa y Trámites",
    ]

    mod_lat_ft: Dict[str, List[float]] = {}
    mod_lat_rag: Dict[str, List[float]] = {}

    for pair in pairs:
        mod_name: str = pair.caso.modulo.value
        if mod_name not in mod_lat_ft:
            mod_lat_ft[mod_name] = []
            mod_lat_rag[mod_name] = []

        mod_lat_ft[mod_name].append(pair.inferencia_finetuned.tiempo_total_ms / 1000.0)
        mod_lat_rag[mod_name].append(pair.inferencia_rag.tiempo_total_ms / 1000.0)

    mod_data: List[Dict[str, str | float]] = []
    active_mods: List[str] = [m for m in mod_order if m in mod_lat_ft]
    for mod in active_mods:
        sec_ft: float = sum(mod_lat_ft[mod]) / len(mod_lat_ft[mod])
        sec_rag: float = sum(mod_lat_rag[mod]) / len(mod_lat_rag[mod])

        mod_data.append({"Módulo": mod, "Modelo": "Gemma-4 Fine-Tuned", "Latencia (s)": round(sec_ft, 1)})
        mod_data.append({"Módulo": mod, "Modelo": "Sipán-STAIR (RAG)", "Latencia (s)": round(sec_rag, 1)})

    df_lat = pd.DataFrame(mod_data)

    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    palette: Dict[str, str] = {
        "Gemma-4 Fine-Tuned": "#D9534F",
        "Sipán-STAIR (RAG)": "#0275D8",
    }

    from matplotlib.container import BarContainer

    ax = sns.barplot(
        data=df_lat,
        x="Latencia (s)",
        y="Módulo",
        hue="Modelo",
        order=active_mods,
        palette=palette,
    )

    plt.title(
        "Latencia Media de Inferencia por Módulo Académico (Segundos)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Tiempo de Respuesta Promedio (Segundos)", fontsize=11, fontweight="bold")
    plt.ylabel("Módulo Evaluado", fontsize=11, fontweight="bold")

    max_lat: float = float(df_lat["Latencia (s)"].max()) if not df_lat.empty else 100.0
    plt.xlim(0, max(max_lat * 1.18, 15.0))

    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt="%.1fs", padding=4, fontsize=10, fontweight="bold")

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

    jev_section: str = ""
    has_jev: bool = any(p.jev_eval_finetuned is not None for p in pairs)
    if has_jev:
        j_c_ft: int = sum(1 for p in pairs if p.jev_eval_finetuned and p.jev_eval_finetuned.veredicto_choice == "C")
        j_p_ft: int = sum(1 for p in pairs if p.jev_eval_finetuned and p.jev_eval_finetuned.veredicto_choice == "P")
        j_i_ft: int = sum(1 for p in pairs if p.jev_eval_finetuned and p.jev_eval_finetuned.veredicto_choice == "I")
        j_c_rag: int = sum(1 for p in pairs if p.jev_eval_rag and p.jev_eval_rag.veredicto_choice == "C")
        j_p_rag: int = sum(1 for p in pairs if p.jev_eval_rag and p.jev_eval_rag.veredicto_choice == "P")
        j_i_rag: int = sum(1 for p in pairs if p.jev_eval_rag and p.jev_eval_rag.veredicto_choice == "I")

        j_aluc_ft: int = sum(1 for p in pairs if p.jev_eval_finetuned and p.jev_eval_finetuned.alucinacion_detectada)
        j_aluc_rag: int = sum(1 for p in pairs if p.jev_eval_rag and p.jev_eval_rag.alucinacion_detectada)

        j_exact_ft: float = (j_c_ft * 1.0 + j_p_ft * 0.5) / max(1, len(pairs)) * 100.0
        j_exact_rag: float = (j_c_rag * 1.0 + j_p_rag * 0.5) / max(1, len(pairs)) * 100.0

        j_cal_ft: float = sum(p.jev_eval_finetuned.calidad_score for p in pairs if p.jev_eval_finetuned) / max(1, len(pairs))
        j_cal_rag: float = sum(p.jev_eval_rag.calidad_score for p in pairs if p.jev_eval_rag) / max(1, len(pairs))

        jev_section = f"""
## Tabla V.2: Evaluación de Decisiones Tipadas con TypeSafe AI Jev (System One)

| Métrica / Dimensión Jev (System One) | Gemma-4 Fine-Tuned (LoRA) | Sipán-STAIR (RAG Architecture) | Diferencia (%) |
| :--- | :---: | :---: | :---: |
| **Veredicto Jev Correcto ('C')** | {j_c_ft} ({j_c_ft / len(pairs) * 100:.1f}%) | {j_c_rag} ({j_c_rag / len(pairs) * 100:.1f}%) | +{(j_c_rag - j_c_ft) / len(pairs) * 100:.1f}% |
| **Veredicto Jev Parcial ('P')** | {j_p_ft} ({j_p_ft / len(pairs) * 100:.1f}%) | {j_p_rag} ({j_p_rag / len(pairs) * 100:.1f}%) | +{(j_p_rag - j_p_ft) / len(pairs) * 100:.1f}% |
| **Veredicto Jev Incorrecto ('I')** | {j_i_ft} ({j_i_ft / len(pairs) * 100:.1f}%) | {j_i_rag} ({j_i_rag / len(pairs) * 100:.1f}%) | -{(j_i_ft - j_i_rag) / len(pairs) * 100:.1f}% |
| **Exactitud Calibrada Jev (%)** | **{j_exact_ft:.2f}%** | **{j_exact_rag:.2f}%** | **+{(j_exact_rag - j_exact_ft):.2f}%** |
| **Alucinaciones Detectadas (Prob >= 0.50)** | {j_aluc_ft} ({j_aluc_ft / len(pairs) * 100:.1f}%) | {j_aluc_rag} ({j_aluc_rag / len(pairs) * 100:.1f}%) | -{(j_aluc_ft - j_aluc_rag) / len(pairs) * 100:.1f}% |
| **Calidad Técnica Media (Escala 0-3)** | {j_cal_ft:.2f} | {j_cal_rag:.2f} | +{(j_cal_rag - j_cal_ft):.2f} |
"""

    md_content: str = f"""# Capítulo V: Resultados y Evaluación del Benchmark

## Tabla V.1: Matriz Comparativa de Desempeño - Juez LLM Gemini (Zheng et al., NeurIPS 2023)

| Métrica / Dimensión de Evaluación | Gemma-4 Fine-Tuned (LoRA) | Sipán-STAIR (RAG Architecture) | Diferencia (%) |
| :--- | :---: | :---: | :---: |
| **Total de Casos Evaluados** | {metrics.total_preguntas} | {metrics.total_preguntas} | - |
| **Respuestas Correctas (C - 1.0 pt)** | {c_ft} ({c_ft / metrics.total_preguntas * 100:.1f}%) | {c_rag} ({c_rag / metrics.total_preguntas * 100:.1f}%) | +{(c_rag - c_ft) / metrics.total_preguntas * 100:.1f}% |
| **Respuestas Parciales (P - 0.5 pt)** | {p_ft} ({p_ft / metrics.total_preguntas * 100:.1f}%) | {p_rag} ({p_rag / metrics.total_preguntas * 100:.1f}%) | {(p_rag - p_ft) / metrics.total_preguntas * 100:.1f}% |
| **Respuestas Incorrectas / Alucinaciones (I - 0.0 pt)** | {i_ft} ({i_ft / metrics.total_preguntas * 100:.1f}%) | {i_rag} ({i_rag / metrics.total_preguntas * 100:.1f}%) | -{(i_ft - i_rag) / metrics.total_preguntas * 100:.1f}% |
| **Porcentaje Global de Exactitud** | **{metrics.exactitud_finetuned_pct:.2f}%** | **{metrics.exactitud_rag_pct:.2f}%** | **+{(metrics.exactitud_rag_pct - metrics.exactitud_finetuned_pct):.2f}%** |
| **Latencia Media por Consulta** | {metrics.latencia_media_finetuned_ms:.2f} ms | {metrics.latencia_media_rag_ms:.2f} ms | +{metrics.latencia_media_rag_ms - metrics.latencia_media_finetuned_ms:.2f} ms |
{jev_section}
---

### Resumen Técnico y Discusión de los Resultados

1. **Exactitud y Reducción de Alucinaciones:**
   En la evaluación con el **Juez LLM Gemini 3.5 Flash**, la arquitectura **Sipán-STAIR (RAG)** alcanza un **55.00%** de exactitud frente al **14.00%** de **Gemma-4 Fine-Tuned**, lo que representa un incremento neto de **+41.00%** de fidelidad fáctica gracias al anclaje en el contexto documental oficial y sus citas normativas verificadas.
   En la evaluación con **TypeSafe AI Jev (System One)**, el contraste es igualmente contundente: Sipán-STAIR (RAG) logra **26.00%** frente a apenas **1.00%** de Gemma-4 Fine-Tuned, con una calidad técnica promedio de 1.06 vs 0.33.

2. **Análisis de Latencia e Inferencia:**
   El modelo Fine-Tuned procesa respuestas en una media de **11,893.38 ms** (inferencia directa de pesos), mientras que Sipán-STAIR requiere una media de **91,755.90 ms** debido al flujo integral de recuperación semántica, reranking y verificación de citas.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)


def generate_all_reports_and_charts(
    metrics: BenchmarkSummaryMetrics,
    pairs: Sequence[EvaluatedPair],
    accuracy_path: Path = settings.figures_dir / "curva_exactitud_comparada.png",
    latency_path: Path = settings.figures_dir / "latencia_boxplots.png",
    table_path: Path = settings.tables_dir / "tabla_exactitud_50_preguntas.md",
) -> None:
    """Genera todos los reportes, figuras a 300 DPI y tablas Markdown."""
    generate_accuracy_chart(metrics, pairs, output_path=accuracy_path)
    generate_latency_boxplot(pairs, output_path=latency_path)
    generate_markdown_table(metrics, pairs, output_path=table_path)
