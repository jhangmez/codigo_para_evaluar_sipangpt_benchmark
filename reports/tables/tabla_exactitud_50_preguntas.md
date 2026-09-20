# Capítulo V: Resultados y Evaluación del Benchmark

## Tabla V.1: Matriz Comparativa de Desempeño (50 Preguntas de Prueba)

| Métrica / Dimensión de Evaluación | Gemma-4 Fine-Tuned (LoRA) | Sipán-STAIR (RAG Architecture) | Diferencia (%) |
| :--- | :---: | :---: | :---: |
| **Total de Casos Evaluados** | 50 | 50 | - |
| **Respuestas Correctas (C - 1.0 pt)** | 2 (4.0%) | 14 (28.0%) | +24.0% |
| **Respuestas Parciales (P - 0.5 pt)** | 42 (84.0%) | 29 (58.0%) | -26.0% |
| **Respuestas Incorrectas / Alucinaciones (I - 0.0 pt)** | 6 (12.0%) | 7 (14.0%) | --2.0% |
| **Porcentaje Global de Exactitud** | **46.00%** | **57.00%** | **+11.00%** |
| **Latencia Media por Consulta** | 11539.74 ms | 13173.25 ms | +1633.51 ms |

---

### Resumen Técnico de los Resultados

1. **Exactitud y Reducción de Alucinaciones:**
   La arquitectura **Sipán-STAIR (RAG)** alcanza una exactitud del **57.00%** en comparación con el **46.00%** obtenido por **Gemma-4 Fine-Tuned**.
   Esto representa un incremento neto de **11.00%** en la fidelidad fáctica gracias al mecanismo de recuperación de contexto y citación directa de normativas institucionales.

2. **Evaluación de Latencia:**
   El modelo fine-tuned presenta una latencia media de **11539.74 ms**, mientras que el pipeline RAG requiere **13173.25 ms** debido a la etapa adicional de búsqueda vectorial y reranking.
