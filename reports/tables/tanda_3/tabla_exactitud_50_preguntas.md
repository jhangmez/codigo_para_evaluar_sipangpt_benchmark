# Capítulo V: Resultados y Evaluación del Benchmark

## Tabla V.1: Matriz Comparativa de Desempeño - Juez LLM Gemini 3.8 Flash (Zheng et al., NeurIPS 2023)

| Métrica / Dimensión de Evaluación | Gemma-4 Fine-Tuned (LoRA) | Sipán-STAIR (RAG Architecture) | Diferencia (%) |
| :--- | :---: | :---: | :---: |
| **Total de Casos Evaluados** | 50 | 50 | - |
| **Respuestas Correctas (C - 1.0 pt)** | 5 (10.0%) | 7 (14.0%) | +4.0% |
| **Respuestas Parciales (P - 0.5 pt)** | 3 (6.0%) | 6 (12.0%) | +6.0% |
| **Respuestas Incorrectas / Alucinaciones (I - 0.0 pt)** | 42 (84.0%) | 37 (74.0%) | -10.0% |
| **Porcentaje Global de Exactitud** | **13.00%** | **20.00%** | **+7.00%** |
| **Latencia Media por Consulta** | 11465.35 ms | 8434.52 ms | -3030.83 ms |

---

### Resumen Técnico y Discusión de los Resultados

1. **Exactitud y Reducción de Alucinaciones:**
   En la evaluación con el **Juez LLM Gemini 3.8 Flash (Reference-Guided)**, la arquitectura **Sipán-STAIR (RAG con gemini-3.8-flash)** alcanza un **20.00%** de exactitud frente al **13.00%** de **Gemma-4 Fine-Tuned**, manteniendo una ventaja neta de **+7.00%** en fidelidad procedimental e institucional y reduciendo las alucinaciones frente al modelo ajustado por pesos.

2. **Análisis de Latencia e Inferencia:**
   El modelo Fine-Tuned procesa respuestas en una media de **11,465.35 ms** (~11.47 s), mientras que Sipán-STAIR con `gemini-3.8-flash` registra **8,434.52 ms** (~8.43 s), confirmando que el pipeline RAG optimizado opera un **26.4% más rápido** que la inferencia local completa.
