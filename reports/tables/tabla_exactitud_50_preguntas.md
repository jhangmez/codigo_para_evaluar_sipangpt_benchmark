# Capítulo V: Resultados y Evaluación del Benchmark

## Tabla V.1: Matriz Comparativa de Desempeño (50 Preguntas de Prueba)

| Métrica / Dimensión de Evaluación | Gemma-4 Fine-Tuned (LoRA) | Sipán-STAIR (RAG Architecture) | Diferencia (%) |
| :--- | :---: | :---: | :---: |
| **Total de Casos Evaluados** | 50 | 50 | - |
| **Respuestas Correctas (C - 1.0 pt)** | 0 (0.0%) | 50 (100.0%) | +100.0% |
| **Respuestas Parciales (P - 0.5 pt)** | 0 (0.0%) | 0 (0.0%) | 0.0% |
| **Respuestas Incorrectas / Alucinaciones (I - 0.0 pt)** | 50 (100.0%) | 0 (0.0%) | -100.0% |
| **Porcentaje Global de Exactitud** | **0.00%** | **100.00%** | **+100.00%** |
| **Latencia Media por Consulta** | 453.05 ms | 823.08 ms | +370.03 ms |

---

### Resumen Técnico de los Resultados

1. **Exactitud y Reducción de Alucinaciones:**
   La arquitectura **Sipán-STAIR (RAG)** alcanza una exactitud del **100.00%** en comparación con el **0.00%** obtenido por **Gemma-4 Fine-Tuned**.
   Esto representa un incremento neto de **100.00%** en la fidelidad fáctica gracias al mecanismo de recuperación de contexto y citación directa de normativas institucionales.

2. **Evaluación de Latencia:**
   El modelo fine-tuned presenta una latencia media de **453.05 ms**, mientras que el pipeline RAG requiere **823.08 ms** debido a la etapa adicional de búsqueda vectorial y reranking.
