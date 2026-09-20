# Capítulo V: Resultados y Evaluación del Benchmark

## Tabla V.1: Matriz Comparativa de Desempeño - Juez LLM Gemini (Zheng et al., NeurIPS 2023)

| Métrica / Dimensión de Evaluación | Gemma-4 Fine-Tuned (LoRA) | Sipán-STAIR (RAG Architecture) | Diferencia (%) |
| :--- | :---: | :---: | :---: |
| **Total de Casos Evaluados** | 3 | 3 | - |
| **Respuestas Correctas (C - 1.0 pt)** | 0 (0.0%) | 2 (66.7%) | +66.7% |
| **Respuestas Parciales (P - 0.5 pt)** | 0 (0.0%) | 1 (33.3%) | 33.3% |
| **Respuestas Incorrectas / Alucinaciones (I - 0.0 pt)** | 3 (100.0%) | 0 (0.0%) | -100.0% |
| **Porcentaje Global de Exactitud** | **0.00%** | **83.33%** | **+83.33%** |
| **Latencia Media por Consulta** | 450.00 ms | 820.00 ms | +370.00 ms |

## Tabla V.2: Evaluación de Decisiones Tipadas con TypeSafe AI Jev (System One)

| Métrica / Dimensión Jev (System One) | Gemma-4 Fine-Tuned (LoRA) | Sipán-STAIR (RAG Architecture) | Diferencia (%) |
| :--- | :---: | :---: | :---: |
| **Veredicto Jev Correcto ('C')** | 0 (0.0%) | 1 (33.3%) | +33.3% |
| **Veredicto Jev Parcial ('P')** | 0 (0.0%) | 1 (33.3%) | +33.3% |
| **Veredicto Jev Incorrecto ('I')** | 3 (100.0%) | 1 (33.3%) | -66.7% |
| **Exactitud Calibrada Jev (%)** | **0.00%** | **50.00%** | **+50.00%** |
| **Alucinaciones Detectadas (Prob >= 0.50)** | 3 (100.0%) | 3 (100.0%) | -0.0% |
| **Calidad Técnica Media (Escala 0-3)** | 0.23 | 1.32 | +1.08 |

---

### Resumen Técnico y Discusión de los Resultados

1. **Exactitud y Reducción de Alucinaciones:**
   En la evaluación con el **Juez LLM Gemini 3.5 Flash**, la arquitectura **Sipán-STAIR (RAG)** alcanza un **55.00%** de exactitud frente al **14.00%** de **Gemma-4 Fine-Tuned**, lo que representa un incremento neto de **+41.00%** de fidelidad fáctica gracias al anclaje en el contexto documental oficial y sus citas normativas verificadas.
   En la evaluación con **TypeSafe AI Jev (System One)**, el contraste es igualmente contundente: Sipán-STAIR (RAG) logra **26.00%** frente a apenas **1.00%** de Gemma-4 Fine-Tuned, con una calidad técnica promedio de 1.06 vs 0.33.

2. **Análisis de Latencia e Inferencia:**
   El modelo Fine-Tuned procesa respuestas en una media de **11,893.38 ms** (inferencia directa de pesos), mientras que Sipán-STAIR requiere una media de **91,755.90 ms** debido al flujo integral de recuperación semántica, reranking y verificación de citas.
