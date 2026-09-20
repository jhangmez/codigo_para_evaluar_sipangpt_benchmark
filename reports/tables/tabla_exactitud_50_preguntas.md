# Capítulo V: Resultados y Evaluación del Benchmark

## Tabla V.1: Matriz Comparativa de Desempeño (50 Preguntas de Prueba)

| Métrica / Dimensión de Evaluación | Gemma-4 Fine-Tuned (LoRA) | Sipán-STAIR (RAG Architecture) | Diferencia (%) |
| :--- | :---: | :---: | :---: |
| **Total de Casos Evaluados** | 50 | 50 | - |
| **Respuestas Correctas (C - 1.0 pt)** | 1 (2.0%) | 50 (100.0%) | +98.0% |
| **Respuestas Parciales (P - 0.5 pt)** | 25 (50.0%) | 0 (0.0%) | -50.0% |
| **Respuestas Incorrectas / Alucinaciones (I - 0.0 pt)** | 24 (48.0%) | 0 (0.0%) | -48.0% |
| **Porcentaje Global de Exactitud** | **27.00%** | **100.00%** | **+73.00%** |
| **Latencia Media por Consulta** | 14790.50 ms | 840.83 ms | +-13949.67 ms |

---

### Resumen Técnico de los Resultados

1. **Exactitud y Reducción de Alucinaciones:**
   La arquitectura **Sipán-STAIR (RAG)** alcanza una exactitud del **100.00%** en comparación con el **27.00%** obtenido por **Gemma-4 Fine-Tuned**.
   Esto representa un incremento neto de **73.00%** en la fidelidad fáctica gracias al mecanismo de recuperación de contexto y citación directa de normativas institucionales.

2. **Evaluación de Latencia:**
   El modelo fine-tuned presenta una latencia media de **14790.50 ms**, mientras que el pipeline RAG requiere **840.83 ms** debido a la etapa adicional de búsqueda vectorial y reranking.
