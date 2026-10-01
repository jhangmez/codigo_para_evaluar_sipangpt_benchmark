# Capítulo V: Resultados y Evaluación del Benchmark

## Tabla V.1: Matriz Comparativa de Desempeño - Juez LLM Gemini (Zheng et al., NeurIPS 2023)

| Métrica / Dimensión de Evaluación | Gemma-4 Fine-Tuned (LoRA) | Sipán-STAIR (RAG Architecture) | Diferencia (%) |
| :--- | :---: | :---: | :---: |
| **Total de Casos Evaluados** | 50 | 50 | - |
| **Respuestas Correctas (C - 1.0 pt)** | 1 (2.0%) | 19 (38.0%) | +36.0% |
| **Respuestas Parciales (P - 0.5 pt)** | 4 (8.0%) | 7 (14.0%) | +6.0% |
| **Respuestas Incorrectas / Alucinaciones (I - 0.0 pt)** | 45 (90.0%) | 24 (48.0%) | -42.0% |
| **Porcentaje Global de Exactitud** | **6.00%** | **45.00%** | **+39.00%** |
| **Latencia Media por Consulta** | 12145.39 ms | 7472.28 ms | -4673.11 ms |

---

### Resumen Técnico y Discusión de los Resultados

1. **Exactitud y Reducción de Alucinaciones:**
   En la evaluación con el **Juez LLM Gemini (Reference-Guided)**, la arquitectura **Sipán-STAIR (RAG)** alcanza un **45.00%** de exactitud frente al **6.00%** de **Gemma-4 Fine-Tuned**, lo que representa una ventaja neta de **+39.00%** de fidelidad fáctica gracias al anclaje en el contexto documental institucional verificado.

2. **Análisis de Latencia e Inferencia:**
   El modelo Fine-Tuned procesa respuestas en una media de **12145.39 ms**, mientras que Sipán-STAIR registra **7472.28 ms**, demostrando la alta eficiencia del motor RAG con reranking semántico y streaming de respuestas bajo condiciones óptimas de red.
