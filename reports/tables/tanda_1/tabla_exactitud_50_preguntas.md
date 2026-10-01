# Capítulo V: Resultados y Evaluación del Benchmark

## Tabla V.1: Matriz Comparativa de Desempeño - Juez LLM Gemini (Zheng et al., NeurIPS 2023)

| Métrica / Dimensión de Evaluación | Gemma-4 Fine-Tuned (LoRA) | Sipán-STAIR (RAG Architecture) | Diferencia (%) |
| :--- | :---: | :---: | :---: |
| **Total de Casos Evaluados** | 50 | 50 | - |
| **Respuestas Correctas (C - 1.0 pt)** | 3 (6.0%) | 22 (44.0%) | +38.0% |
| **Respuestas Parciales (P - 0.5 pt)** | 8 (16.0%) | 11 (22.0%) | 6.0% |
| **Respuestas Incorrectas / Alucinaciones (I - 0.0 pt)** | 39 (78.0%) | 17 (34.0%) | -44.0% |
| **Porcentaje Global de Exactitud** | **14.00%** | **55.00%** | **+41.00%** |
| **Latencia Media por Consulta** | 11893.38 ms | 91755.90 ms | +79862.52 ms |

## Tabla V.2: Evaluación de Decisiones Tipadas con TypeSafe AI Jev (System One)

| Métrica / Dimensión Jev (System One) | Gemma-4 Fine-Tuned (LoRA) | Sipán-STAIR (RAG Architecture) | Diferencia (%) |
| :--- | :---: | :---: | :---: |
| **Veredicto Jev Correcto ('C')** | 0 (0.0%) | 5 (10.0%) | +10.0% |
| **Veredicto Jev Parcial ('P')** | 1 (2.0%) | 16 (32.0%) | +30.0% |
| **Veredicto Jev Incorrecto ('I')** | 49 (98.0%) | 29 (58.0%) | -40.0% |
| **Exactitud Calibrada Jev (%)** | **1.00%** | **26.00%** | **+25.00%** |
| **Alucinaciones Detectadas (Prob >= 0.50)** | 49 (98.0%) | 41 (82.0%) | -16.0% |
| **Calidad Técnica Media (Escala 0-3)** | 0.33 | 1.06 | +0.73 |

---

### Resumen Técnico y Discusión de los Resultados

1. **Exactitud y Reducción de Alucinaciones:**
   En la evaluación con el **Juez LLM Gemini 3.5 Flash**, la arquitectura **Sipán-STAIR (RAG)** alcanza un **55.00%** de exactitud frente al **14.00%** de **Gemma-4 Fine-Tuned**, lo que representa un incremento neto de **+41.00%** de fidelidad fáctica gracias al anclaje en el contexto documental oficial y sus citas normativas verificadas.
   En la evaluación con **TypeSafe AI Jev (System One)**, el contraste es igualmente contundente: Sipán-STAIR (RAG) logra **26.00%** frente a apenas **1.00%** de Gemma-4 Fine-Tuned, con una calidad técnica promedio de 1.06 vs 0.33.

2. **Análisis de Latencia e Inferencia:**
   El modelo Fine-Tuned procesa respuestas en una media de **11,893.38 ms** (inferencia directa de pesos), mientras que Sipán-STAIR requiere una media de **91,755.90 ms** debido al flujo integral de recuperación semántica, reranking y verificación de citas.
