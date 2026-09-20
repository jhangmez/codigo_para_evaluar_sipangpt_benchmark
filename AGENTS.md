# AGENTS.md — Guía Maestra para Agentes de Inteligencia Artificial y Desarrolladores

> **Propósito de este archivo**: Proveer el contexto arquitectónico, metodológico, técnico y operativo completo de este repositorio para cualquier agente autónomo de IA (o desarrollador) que continúe, extienda o audite el proyecto.

---

## 1. Identidad y Contexto del Proyecto

* **Repositorio**: `codigo_para_evaluar_sipangpt_benchmark`
* **Autor / Investigador**: **Gomez, J.**
* **Contexto de Investigación**: Proyecto de Tesis de Ingeniería de Sistemas para la evaluación científica y cuantitativa de asistentes virtuales universitarios en la **Universidad Señor de Sipán (USS)**.
* **Problema Científico Central**: Comparar rigurosamente el desempeño de dos arquitecturas de Inteligencia Artificial Generativa para soporte y trámites académicos universitarios:
  1. **Modelo Generativo Fine-Tuned (Pesos Paramétricos)**: `Gemma-4` ajustado con LoRA/QLoRA (4-bit, unsloth) sobre diálogos institucionales.
  2. **Arquitectura RAG (Retrieval-Augmented Generation / Anclaje Fáctico Dinámico)**: `Sipán-STAIR` montado sobre Next.js 14, vector store institucional y reranker semántico.
* **Repositorios Relacionados del Ecosistema**:
  * Suite de Benchmark (este repo): `https://github.com/jhangmez/codigo_para_evaluar_sipangpt_benchmark/tree/main`
  * Síntesis de Dataset ShareGPT: `https://github.com/jhangmez/codigo_para_sintesis_dataset_sharegpt`
  * Generador de Dataset Fine-Tuning: `https://github.com/jhangmez/codigo_para_generar_dataset_finetuning`
  * Asistente Oficial SipánGPT (Next.js RAG): `https://github.com/jhangmez/sipangpt/tree/feature/V2`
  * Dataset Oficial en Hugging Face: `https://huggingface.co/datasets/ussipan/sipangpt-V2`

---

## 2. Principios de Diseño Obligatorios (Reglas Inquebrantables)

Si como agente vas a modificar o agregar código a este repositorio, DEBES cumplir estrictamente las siguientes directivas:

1. **Cero Tolerancia a tipos `Any` (`Strict Type Annotations`)**:
   * **Ninguna función, método, parámetro, variable o retorno puede utilizar `typing.Any`.**
   * Todo el código debe pasar `mypy src/` con 0 errores de tipado estricto.
   * Utiliza uniones tipadas (`str | float | None`), genéricos (`Sequence[T]`, `Dict[str, int]`) o modelos Pydantic específicos.
2. **Pydantic v2 Estricto**:
   * Todos los data contracts heredan de `pydantic.BaseModel` con validación v2 (`model_validate`, `model_dump`, `Field(...)`).
   * Configuración de entorno centralizada con `pydantic-settings` (`BaseSettings`, `SettingsConfigDict`).
3. **Estructura src-layout**:
   * Todo el código fuente del paquete reside exclusivamente en `src/sipangpt_eval/`.
4. **Idempotencia y Checkpointing Continuo**:
   * Cada caso procesado (tanto en inferencia como en calificación de jueces) debe ser persistido de inmediato en disco (`data/results/`).
   * Si el pipeline se interrumpe (por cortes de red, rate-limits 429 de APIs o fallos locales), la reanudación debe saltarse los casos ya procesados sin re-ejecutar llamadas redundantes ni duplicar costos.
5. **Aislamiento Absoluto de Pruebas Unitarias (`tmp_path`)**:
   * Los tests en `tests/` **NUNCA** deben sobreescribir las figuras en `reports/figures/` ni los datos en `data/results/`.
   * En `test_pipeline.py`, todos los artefactos generados deben dirigirse a `tmp_path`.
6. **Formato de Citación y Nombres de Autor**:
   * En citas APA 7 y referencias del proyecto, el autor debe citarse estrictamente como: **`Gomez, J.`** (sin tilde en la 'o' y sin el segundo apellido 'Padilla').

---

## 3. Estructura del Repositorio (`src-layout`)

```text
codigo_para_evaluar_sipangpt_benchmark/
│
├── .env.example                     # Plantilla de variables de entorno requeridas
├── pyproject.toml                   # Configuración del paquete y dependencias (mypy, pytest)
├── README.md                        # Informe metodológico, tablas y visualizaciones oficiales
├── AGENTS.md                        # Esta guía maestra para agentes y desarrolladores
├── main.py                          # CLI principal de orquestación con Typer
│
├── data/
│   ├── ground_truth/                # Banco de pruebas oficial (50 preguntas, 5 módulos)
│   │   └── benchmark_test_50.jsonl
│   └── results/                     # Resultados persistidos y cachés
│       ├── run_gemma4_finetuned.json          # 50 respuestas de Gemma-4 Fine-Tuned
│       ├── run_sipan_stair_rag.json           # 50 respuestas de Sipán-STAIR RAG
│       ├── judge_evaluations_cache.json       # Evaluaciones persistentes del Juez Gemini LLM
│       ├── jev_evaluations_cache.json         # Evaluaciones persistentes de TypeSafe AI Jev
│       ├── matriz_comparativa_final.xlsx      # Matriz Excel oficial de Tesis
│       ├── matriz_evaluacion_jev.xlsx         # Matriz Excel de apoyo experimental Jev
│       └── banco_pruebas_50.xlsx              # Banco de preguntas exportado a Excel
│
├── reports/
│   ├── figures/                     # Gráficos estadísticos en alta resolución (300 DPI)
│   │   ├── curva_exactitud_comparada.png      # Exactitud por módulo (Juez LLM Gemini)
│   │   ├── latencia_boxplots.png              # Latencia media por módulo (Segundos)
│   │   ├── jev_exactitud_comparada.png        # Exactitud calibrada por módulo (Jev)
│   │   └── jev_calidad_tecnica.png            # Calidad técnica media por módulo (Jev)
│   └── tables/                      # Tablas Markdown para el informe de Tesis
│       └── tabla_exactitud_50_preguntas.md
│
├── src/
│   └── sipangpt_eval/
│       ├── config.py                # Pydantic Settings con lectura automática de .env
│       ├── schemas/                 # Data contracts con tipado estricto (0 Any)
│       │   ├── benchmark.py         # BenchmarkCase, ModuleCategory, TurnType
│       │   ├── inference.py         # InferenceOutput, DocumentCitation
│       │   ├── evaluation.py        # ScoreCategory, JudgeEvaluationOutput, EvaluatedPair, BenchmarkSummaryMetrics
│       │   └── jev.py               # JevChoiceResponse, JevScoreResponse, JevEvaluationOutput, JevSummaryMetrics
│       ├── clients/                 # Conectores desacoplados para inferencia
│       │   ├── base.py              # BaseLLMClient (clase abstracta con query())
│       │   ├── local_gemma.py       # Conector HTTP/Ollama/LM Studio para Gemma-4 con System Prompt oficial
│       │   └── sipan_rag.py         # Conector HTTP Next.js (/api/chat) con bypass de testing o sesión NextAuth
│       ├── extraction/              # Carga y partición del conjunto de prueba
│       │   └── hf_loader.py         # load_or_generate_test_benchmark() desde ussipan/sipangpt-V2
│       ├── core/                    # Lógica de inferencia y evaluación
│       │   ├── runner.py            # BenchmarkRunner con persistencia continua por caso
│       │   ├── scorer.py            # BenchmarkScorer: Juez LLM Gemini 3.5 Flash con rubric C/P/I
│       │   └── jev_scorer.py        # JevScorer: Evaluador experimental TypeSafe AI Jev vía Vercel AI Gateway
│       └── reporting/               # Exportadores y graficadores
│           ├── excel_generator.py   # Generación de hojas Excel multi-pestaña con openpyxl
│           └── charts.py            # Generación de gráficos 300 DPI y tablas Markdown
│
└── tests/                           # Suite de pruebas automatizadas con pytest
    ├── test_schemas.py              # Validación de modelos y data contracts
    └── test_pipeline.py             # Pruebas integrales de flujo con mocking y aislamiento de artefactos
```

---

## 4. Metodología de Evaluación Dual

El benchmark implementa un enfoque de evaluación dual riguroso:

### 4.1. Juez Principal: Reference-Guided LLM Judge (`Gemini 3.5 Flash`)
* **Fundamento**: Basado en el estándar de oro de NeurIPS 2023 (*Judging LLM-as-a-Judge*, Zheng et al., UC Berkeley / LMSYS).
* **Rúbrica Cualitativa y Cuantitativa**:
  * **`CORRECTA` (`C` - 1.0 punto)**: La respuesta es precisa fáctica y documentalmente, cubre los puntos clave requeridos y no contiene contradicciones ni inventa procedimientos.
  * **`PARCIAL` (`P` - 0.5 puntos)**: La respuesta contiene información correcta pero omite detalles normativos esenciales, pasos clave o condicionantes institucionales.
  * **`INCORRECTA` (`I` - 0.0 puntos)**: La respuesta alucina, inventa plazos o requisitos no existentes, se desvía del tema o proporciona información errónea perjudicial para el estudiante.
* **Control de Tasa y Resiliencia**: El cliente en `scorer.py` maneja reintentos con backoff exponencial para errores `429` (Rate Limit) y `503` (Service Unavailable) de Google Gemini API, asegurando que la evaluación de 50 casos termine sin caídas.

### 4.2. Apoyo Experimental: Decisiones Tipadas Calibradas (`TypeSafe AI Jev`)
* **Fundamento**: Arquitectura *System One* (Kahneman, 2011) que evalúa mediante decisiones tipadas y probabilísticas estructuradas sin texto libre de chatbot.
* **Endpoint**: Conexión a `https://ai-gateway.vercel.sh/v1/evaluate` vía Vercel AI Gateway.
* **Salidas Calibradas**:
  1. `choice`: Clasificación categórica `['C', 'P', 'I']` con vector de probabilidades y confianza calibrada.
  2. `boolean`: Probabilidad directa de alucinación fáctica (`alucinacion_detectada` con umbral $\ge 0.50$).
  3. `score`: Calidad técnica y exhaustividad en escala continua `[0.0, 3.0]`.

---

## 5. Módulos Académicos y Cobertura del Dataset

El banco de pruebas oficial (`data/ground_truth/benchmark_test_50.jsonl`) contiene exactamente **50 casos**, balanceados en los **5 módulos académicos** oficiales de la USS:

| Módulo Académico | N° de Preguntas | Descripción Temática |
| :--- | :---: | :--- |
| **Matrícula y Registros** | 10 | Rectificaciones de matrícula, reservas, convalidaciones, actas y certificados. |
| **Campus Virtual y Aprendizaje** | 14 | Aula Virtual Moodle/Canvas, acceso a Zoom, entrega de tareas, exámenes y soporte técnico. |
| **Pagos y Cobranzas** | 12 | Pensiones, cuotas, fraccionamientos, canales de pago autorizados y conceptos de caja. |
| **Biblioteca Virtual** | 8 | Catálogo en línea, repositorios institucionales, bases de datos científicas (Scopus, EBSCO). |
| **Normativa y Trámites** | 6 | Grados y títulos, carné universitario Sunedu, reglamentos y mesa de partes virtual. |
| **Total General** | **50** | **35 preguntas monoturno y 15 diálogos multiturno.** |

> [!IMPORTANT]
> Todos los gráficos comparativos (`curva_exactitud_comparada.png`, `latencia_boxplots.png`, `jev_exactitud_comparada.png`, `jev_calidad_tecnica.png`) DEBEN mostrar sin excepción los **5 módulos académicos**.
> En el caso de Jev, si Gemma-4 obtiene `0.0%` en alguna categoría, la barra debe figurar con su etiqueta numérica `0.0%` para constatar que el módulo fue evaluado.

---

## 6. Variables de Entorno Requeridas (`.env`)

```env
# Endpoints de Inferencia
GEMMA_API_URL=http://localhost:11434/api/generate
GEMMA_MODEL_NAME=unsloth_gemma-4-E2B-it_1789791679-GGUF

SIPAN_RAG_API_URL=http://localhost:3000/api/chat
SIPAN_RAG_API_KEY="sipangpt-local-perf-key"

# Hugging Face Hub
HF_DATASET_NAME=ussipan/sipangpt-V2

# Juez Principal: Gemini 3.5 Flash
GEMINI_API_KEY="AIzaSy..."
JUDGE_MODEL_NAME=gemini-3.5-flash

# Evaluador Experimental: TypeSafe AI Jev
VERCEL_AI_GATEWAY_KEY="vck_..."
VERCEL_AI_GATEWAY_URL="https://ai-gateway.vercel.sh/v1/evaluate"
```

---

## 7. Comandos CLI (`main.py`) y Flujo de Trabajo

Para ejecutar cualquiera de las etapas del benchmark, activar el entorno virtual y usar `main.py`:

```bash
source .venv/bin/activate

# 1. Extraer el banco de 50 preguntas oficial
python main.py extract

# 2. Ejecutar inferencia en ambos modelos (aprovecha checkpoints en disco)
python main.py run

# 3. Evaluar con Gemini 3.5 Flash (Juez LLM)
python main.py evaluate

# 4. Generar reportes Markdown, Excel y gráficos 300 DPI
python main.py report

# 5. [Opcional] Ejecutar evaluación con TypeSafe AI Jev
python main.py evaluate-jev

# 6. Pipeline completo
python main.py all
```

---

## 8. Guía para Pruebas y Tipado Estricto

Antes de enviar cualquier commit o push a la rama `main`:

```bash
# 1. Verificación estricta de tipos (0 Any)
source .venv/bin/activate
mypy src/
# Salida requerida: "Success: no issues found in 20 source files"

# 2. Suite de pruebas unitarias e integración
pytest -v
# Salida requerida: "5 passed in ..."
```

> [!CAUTION]
> Si agregas nuevas pruebas que llamen a `generate_all_reports_and_charts` o funciones de `charts.py`, asegúrate de pasar rutas de salida basadas en `tmp_path` (por ejemplo `accuracy_path=tmp_path / "acc.png"`). Si dejas los valores por defecto, sobreescribirás los gráficos oficiales de 300 DPI con datos incompletos del test.

---

## 9. Fuentes de Información y Referencias Bibliográficas Oficiales (Normas APA 7ma Edición)

Cualquier agente o investigador que cite, extienda o documente este proyecto DEBE conocer y respetar la siguiente taxonomía de fuentes oficiales:

### 9.1. Literatura Científica y Metodología de Evaluación
* **Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023)**. *QLoRA: Efficient finetuning of quantized LLMs*. Advances in Neural Information Processing Systems (NeurIPS 2023), 36, 10088–10115. https://doi.org/10.48550/arXiv.2305.14314
  > *Rol en el proyecto*: Base metodológica para la cuantización y ajuste eficiente en 4-bit de Gemma-4 mediante adaptadores LoRA.
* **Gemma Team, Mesnard, T., Hardin, C., Dadashi, R., Bhupatiraju, S., Pathak, S., Sifre, L., Rivière, M., Kale, M. S., Love, J., Tafti, P., Léonard, L., & Google DeepMind. (2024)**. *Gemma: Open models based on Gemini research and technology*. arXiv preprint arXiv:2403.08295. https://doi.org/10.48550/arXiv.2403.08295
  > *Rol en el proyecto*: Arquitectura base del modelo open-weights seleccionado para el fine-tuning institucional.
* **Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021)**. *LoRA: Low-rank adaptation of large language models*. arXiv preprint arXiv:2106.09685. https://doi.org/10.48550/arXiv.2106.09685
  > *Rol en el proyecto*: Fundamento matemático de adaptación de bajo rango sobre las matrices de atención del transformer.
* **Kahneman, D. (2011)**. *Thinking, fast and slow*. Farrar, Straus and Giroux.
  > *Rol en el proyecto*: Fundamento cognitivo para justificar la integración experimental de TypeSafe AI Jev (System One: decisiones calibradas rápidas y tipadas) como complemento al Juez LLM generativo (System Two: razonamiento paso a paso).
* **Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., Riedel, S., & Kiela, D. (2020)**. *Retrieval-augmented generation for knowledge-intensive NLP tasks*. Advances in Neural Information Processing Systems (NeurIPS 2020), 33, 9459–9474. https://doi.org/10.48550/arXiv.2005.11401
  > *Rol en el proyecto*: Principio fundacional de la arquitectura Sipán-STAIR (RAG) para eliminar alucinaciones mediante grounding fáctico.
* **Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023)**. *Judging LLM-as-a-judge with MT-Bench and Chatbot Arena*. Advances in Neural Information Processing Systems (NeurIPS 2023), 36, 46595–46623. https://doi.org/10.48550/arXiv.2306.05685
  > *Rol en el proyecto*: Estándar de oro de evaluación guiada por referencia (*Reference-Guided LLM-as-a-Judge*) adoptado para el juez automatizado con rúbrica C/P/I.

### 9.2. Normativas Institucionales de la Universidad Señor de Sipán (USS)
Estas normativas constituyen el **Ground Truth** indiscutible del benchmark para la calificación de respuestas fácticas:
* **Universidad Señor de Sipán. (2022)**. *Reglamento de Grados y Títulos de la Universidad Señor de Sipán*. Vicerrectorado de Investigación, Dirección de Grados y Títulos. Chiclayo, Perú.
* **Universidad Señor de Sipán. (2023)**. *Manual de Usuario de la Plataforma Aula Virtual y Campus Virtual USS*. Dirección de Tecnologías de Información y Comunicaciones (DTI). Chiclayo, Perú.
* **Universidad Señor de Sipán. (2023)**. *Reglamento de Cobranzas y Derechos Académicos*. Dirección General de Administración y Finanzas. Chiclayo, Perú.
* **Universidad Señor de Sipán. (2023)**. *Reglamento General de Matrícula y Registros Académicos*. Vicerrectorado Académico, Dirección de Registros Académicos. Chiclayo, Perú.
* **Universidad Señor de Sipán. (2023)**. *Tarifario Oficial de Tasas y Servicios Administrativos de Pregrado y Posgrado*. Chiclayo, Perú.
* **Universidad Señor de Sipán. (2024)**. *Guía de Acceso al Sistema de Catálogo en Línea y Recursos Bibliográficos Digitales (Scopus, EBSCO, e-Library)*. Dirección del Centro de Información y Biblioteca Central. Chiclayo, Perú.

### 9.3. Datasets y Repositorios Oficiales del Ecosistema SipánGPT
Todos los artefactos de software y conjuntos de datos del proyecto están autorizados por **Gomez, J.**:
* **Gomez, J. (2026)**. *SipánGPT Benchmark Test Dataset (ussipan/sipangpt-V2)* [Conjunto de datos]. Hugging Face Hub. https://huggingface.co/datasets/ussipan/sipangpt-V2
* **Gomez, J. (2026)**. *Suite de Evaluación Comparativa de Rigor Científico: Gemma-4 Fine-Tuned vs. Sipán-STAIR (RAG)* [Código fuente]. GitHub (rama main). https://github.com/jhangmez/codigo_para_evaluar_sipangpt_benchmark/tree/main
* **Gomez, J. (2026)**. *Pipeline de Síntesis y Transformación de Dataset Multiturno en Formato ShareGPT* [Código fuente]. GitHub. https://github.com/jhangmez/codigo_para_sintesis_dataset_sharegpt
* **Gomez, J. (2026)**. *Generador y Validador de Dataset para Fine-Tuning de Modelos Gemma* [Código fuente]. GitHub. https://github.com/jhangmez/codigo_para_generar_dataset_finetuning
* **Gomez, J. (2026)**. *SipánGPT: Asistente Virtual Oficial Inteligente de la Universidad Señor de Sipán basado en RAG y Next.js* [Código fuente]. GitHub (rama feature/V2). https://github.com/jhangmez/sipangpt/tree/feature/V2

### 9.4. Tecnologías, APIs y Frameworks
* **Google AI. (2024)**. *Gemini API: Structured outputs and developer guide*. Google DeepMind. https://ai.google.dev/
* **TypeSafe AI. (2024)**. *Jev: The first System One decision model for typed software evaluations*. TypeSafe AI Inc. https://jevtypesafeai.com/
* **Unsloth AI. (2024)**. *Unsloth: Fast and memory-efficient LLM fine-tuning with 4-bit LoRA and GGUF export*. https://github.com/unslothai/unsloth
* **Vercel. (2024)**. *Vercel AI SDK & AI Gateway: Unified specification for model inference and evaluation routing*. Vercel Inc. https://ai-gateway.vercel.sh/

---

## 10. Directivas Cruciales para Agentes Autónomos

1. **Autoría**: Citar siempre al autor del proyecto como **`Gomez, J.`** (sin tildes en 'o' y sin segundo apellido).
2. **Ramas oficiales**: Enlazar siempre `codigo_para_evaluar_sipangpt_benchmark` a `tree/main` y `sipangpt` a `tree/feature/V2`.
3. **Integridad de datos persistidos**: Nunca borrar archivos en `data/results/` a menos que el usuario solicite explícitamente recalcular todo desde cero.
4. **Gráficos y Visualizaciones**: Conservar siempre los 5 módulos en los gráficos estadísticos y tablas.
5. **No Alucinaciones en el Código**: Toda dependencia debe estar registrada en `pyproject.toml`.

---

## 11. Pie de Página y Firma Oficial del Proyecto

En el archivo `README.md`, el documento finaliza obligatoriamente con la firma institucional:

```html
---

## SipánGPT

<div style="display: flex; align-items: center; height: fit-content;">
  <img src="https://avatars.githubusercontent.com/u/60937214?v=4" width="40" style="margin-right: 10px;"/>
  <span>Hecho con ❤️ por Jhan Gómez P.</span>
</div>
```
