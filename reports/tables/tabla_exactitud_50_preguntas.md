# Capítulo V: Resultados y Evaluación del Benchmark

## Tabla V.1: Matriz Comparativa de Desempeño (50 Preguntas de Prueba)

| Métrica / Dimensión de Evaluación | Gemma-4 Fine-Tuned (LoRA) | Sipán-STAIR (RAG Architecture) | Diferencia (%) |
| :--- | :---: | :---: | :---: |
| **Total de Casos Evaluados** | 50 | 50 | - |
| **Respuestas Correctas (C - 1.0 pt)** | 1 (2.0%) | 50 (100.0%) | +98.0% |
| **Respuestas Parciales (P - 0.5 pt)** | 22 (44.0%) | 0 (0.0%) | -44.0% |
| **Respuestas Incorrectas / Alucinaciones (I - 0.0 pt)** | 27 (54.0%) | 0 (0.0%) | -54.0% |
| **Porcentaje Global de Exactitud** | **24.00%** | **100.00%** | **+76.00%** |
| **Latencia Media por Consulta** | 452.84 ms | 822.90 ms | +370.06 ms |

---

### Resumen Técnico de los Resultados

1. **Exactitud y Reducción de Alucinaciones:**
   La arquitectura **Sipán-STAIR (RAG)** alcanza una exactitud del **100.00%** en comparación con el **24.00%** obtenido por **Gemma-4 Fine-Tuned**.
   Esto representa un incremento neto de **76.00%** en la fidelidad fáctica gracias al mecanismo de recuperación de contexto y citación directa de normativas institucionales.

2. **Evaluación de Latencia:**
   El modelo fine-tuned presenta una latencia media de **452.84 ms**, mientras que el pipeline RAG requiere **822.90 ms** debido a la etapa adicional de búsqueda vectorial y reranking.
