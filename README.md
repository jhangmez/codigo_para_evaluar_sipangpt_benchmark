# Suite de Evaluación Comparativa de Rigor Científico: Gemma-4 Fine-Tuned vs. Sipán-STAIR (RAG)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2.8%2B-green.svg)](https://docs.pydantic.dev/)
[![Code Style: Strict Typing](https://img.shields.io/badge/mypy-strict-brightgreen.svg)](https://mypy.readthedocs.io/)

Este repositorio contiene la suite de evaluación automatizada y reproducible desarrollada para el informe de tesis de la **Universidad Nacional Pedro Ruiz Gallo (UNPRG)**. El objetivo principal es evaluar cuantitativa y cualitativamente el desempeño de dos aproximaciones tecnológicas para asistentes virtuales académicos:
1. **Gemma-4 Fine-Tuned (LoRA):** Modelo de lenguaje adaptado localmente.
2. **Sipán-STAIR (RAG Architecture):** Sistema de Generación Aumentada por Recuperación con citación contextual y verificación fáctica.

---

## 📁 Estructura del Proyecto (`src-layout`)

Seguimos las mejores prácticas de la industria en Python con arquitectura modular desacoplada, tipado estricto (Pydantic v2) y configuración vía `pyproject.toml`:

```text
codigo_para_evaluar_sipangpt_benchmark/
├── .env.example                     # Plantilla de variables de entorno (URLs, Tokens)
├── .gitignore                       # Configuración de exclusión para Git
├── pyproject.toml                   # Dependencias, linters y metadata del paquete
├── README.md                        # Guía técnica de ejecución y fundamentación
├── AGENTS.md                        # Documentación de arquitectura para agentes e IA
│
├── data/                            # Almacenamiento local de datos
│   ├── ground_truth/                # Conjunto de prueba oficial (las 50 preguntas)
│   │   └── benchmark_test_50.jsonl
│   └── results/                     # Resultados generados por las corridas de prueba
│       ├── run_gemma4_finetuned.json
│       ├── run_sipan_stair_rag.json
│       ├── banco_pruebas_50.xlsx
│       └── matriz_comparativa_final.xlsx
│
├── reports/                         # Evidencias para el Capítulo V de la Tesis
│   ├── figures/                     # Gráficos en alta resolución (300 DPI)
│   │   ├── curva_exactitud_comparada.png
│   │   └── latencia_boxplots.png
│   └── tables/                      # Tablas formateadas en Markdown / LaTeX para tesis
│       └── tabla_exactitud_50_preguntas.md
│
├── src/
│   └── sipangpt_eval/               # Paquete principal con tipado estricto
│       ├── __init__.py
│       ├── config.py                # Configuración global (Pydantic Settings)
│       │
│       ├── schemas/                 # Interfaces estrictas sin Any (Pydantic v2)
│       │   ├── __init__.py
│       │   ├── benchmark.py         # Modelos de pregunta, Ground Truth y Turnos
│       │   ├── inference.py         # Modelos de respuesta de IA, latencias y tokens
│       │   └── evaluation.py        # Rúbrica de calificación (C / P / I) y métricas
│       │
│       ├── clients/                 # Conectores desacoplados a modelos
│       │   ├── __init__.py
│       │   ├── base.py              # Clase base abstracta (ABC)
│       │   ├── local_gemma.py       # Cliente para Gemma-4 (Ollama / REST)
│       │   └── sipan_rag.py         # Cliente HTTP para SipánGPT RAG (STAIR)
│       │
│       ├── extraction/              # Carga y particionado de datos
│       │   ├── __init__.py
│       │   └── hf_loader.py         # Descarga y parseo del split 'test' de Hugging Face
│       │
│       ├── core/                    # Lógica de orquestación y rúbrica
│       │   ├── __init__.py
│       │   ├── runner.py            # Orquestador de inferencia sobre los 50 casos
│       │   └── scorer.py            # Rúbrica objetiva (C, P, I, alucinación, citas)
│       │
│       └── reporting/               # Generadores de evidencias visuales
│           ├── __init__.py
│           ├── excel_generator.py   # Matriz comparativa en Excel (.xlsx)
│           └── charts.py            # Generación de figuras PNG 300 DPI y tablas MD
│
├── tests/                           # Pruebas unitarias e integración (pytest + mypy)
│   ├── test_schemas.py
│   └── test_pipeline.py
│
└── main.py                          # CLI interactivo (Typer + Rich)
```

---

## 🧪 Metodología de Evaluación

El conjunto de prueba consta de **50 preguntas de prueba (Ground Truth)** distribuidas científicamente:
* **70% Monoturno (35 preguntas directas):** Consultas cerradas que exigen datos normativos específicos (fechas, costos, requisitos de matrícula).
* **30% Multiturno (15 diálogos de diagnóstico):** Interacciones continuas donde se evalúa el mantenimiento del contexto y la solución progresiva de problemas.

### Rúbrica de Calificación (3 Valores):
* **C (Correcta - 1.0 pto):** Procedimiento oficial exacto sin invención fáctica. En RAG exige cita válida al reglamento.
* **P (Parcial - 0.5 ptos):** Idea general correcta con omisión menor de detalle no crítico.
* **I (Incorrecta - 0.0 ptos):** Alucinación fáctica, enlace roto o procedimiento inexistente.

Fórmula de Exactitud Global:
$$\text{Exactitud Global (\%)} = \frac{\sum \text{Puntos Obtenidos}}{50} \times 100$$

---

## ⚙️ Conexión e Inferencia con Modelos Reales

### A. Modelo Gemma-4 Fine-Tuned (Unsloth GGUF)
Para inferir sobre la versión fine-tuned (`unsloth_gemma-4-E2B-it_1789791679-GGUF`):
* Se puede utilizar **LM Studio** o **Ollama / llama.cpp** como servidor de inferencia API REST local.
* En **LM Studio**, inicie el servidor local en el puerto `1234` (o `11434` en Ollama).
* **System Prompt Oficial:** La suite envía automáticamente en cada consulta el prompt del sistema oficial de SipánGPT:
  > *"Eres SipánGPT, el Asistente Virtual Oficial de la Universidad Señor de Sipán (USS) basado en inteligencia artificial generativa, experto en soporte técnico informático, plataformas digitales (Campus Virtual, Aula Virtual, Sistema de Registros Académicos, Biblioteca Virtual) y normativas institucionales..."*

### B. Modelo Sipán-STAIR (RAG en Next.js)
El sistema RAG se ejecuta desde la aplicación Next.js (`/Users/jhan/Documents/Proyectos/Nextjs/sipangpt`):
* Levantar el servidor dev de Next.js en `http://localhost:3000`.
* **Manejo de Autenticación / Sesión:** La ruta `/api/chat` de Next.js requiere sesión activa mediante NextAuth. En la variable de entorno `SIPAN_RAG_COOKIE` de la suite, configure la cookie `next-auth.session-token` obtenida desde el navegador al iniciar sesión en la aplicación.

---

## 🛠️ Instalación y Configuración

```bash
# 1. Crear y activar entorno virtual Python 3.11+
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar dependencias del proyecto en modo editable
pip install -e .
```

---

## 🚀 Uso del CLI (`main.py`)

Puedes ejecutar los comandos individualmente o correr el flujo completo:

```bash
# Extraer las 50 preguntas del conjunto de prueba
python main.py extract

# Ejecutar inferencia en ambos modelos (Gemma-4 vs RAG)
python main.py run

# Calificar y generar la matriz comparativa en Excel
python main.py evaluate

# Generar gráficos a 300 DPI y tablas Markdown para la Tesis
python main.py report

# Ejecutar PIPELINE COMPLETO de inicio a fin
python main.py all
```

---

## 🔍 Verificación de Tipos y Pruebas Unitarias

Para garantizar consistencia sin tipos implícitos (`Any`), el proyecto implementa análisis estático estricto:

```bash
# Análisis estático de tipos con Mypy (modo estricto, 0 Any)
mypy src/

# Ejecutar pruebas unitarias e integración
pytest tests/
```

---

## 📊 Salidas para el Informe de Tesis (Capítulo V)

Al finalizar la ejecución, se habrán generado las siguientes evidencias:
1. `data/results/matriz_comparativa_final.xlsx`: Matriz con detalle caso por caso y hoja de métricas resumidas.
2. `reports/figures/curva_exactitud_comparada.png`: Gráfico de exactitud comparada por módulo (300 DPI).
3. `reports/figures/latencia_boxplots.png`: Gráfico de cajas de distribución de latencia (300 DPI).
4. `reports/tables/tabla_exactitud_50_preguntas.md`: Tabla formateada lista para insertar en el documento Word/LaTeX del informe de tesis.
