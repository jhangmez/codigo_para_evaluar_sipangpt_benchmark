# AGENTS.md — Agent & Developer Guide

## Overview

This repository (`codigo-para-evaluar-sipangpt-benchmark`) implements a scientifically rigorous LLM evaluation benchmark for comparing **Gemma-4 Fine-Tuned (LoRA)** against **Sipán-STAIR (RAG Architecture)** for academic support at Universidad Señor de Sipán.

## Key Design Principles & Rules

1. **Strict Type Annotations**:
   - Every function, method, parameter, return value, and class attribute MUST be strictly typed.
   - **NO `Any` types allowed anywhere in the codebase.**
   - All Pydantic models use Pydantic v2 conventions.

2. **Project Architecture (`src-layout`)**:
   - `src/sipangpt_eval/schemas/`: Strongly-typed data contracts (`BenchmarkCase`, `InferenceOutput`, `EvaluatedPair`, `BenchmarkSummaryMetrics`, etc.).
   - `src/sipangpt_eval/clients/`: Model connectors implementing `BaseLLMClient`.
   - `src/sipangpt_eval/extraction/`: Ground truth extractor and dataset loader (`load_or_generate_test_benchmark`).
   - `src/sipangpt_eval/core/`: Inference runner (`BenchmarkRunner`) and scoring rubric (`BenchmarkScorer`).
   - `src/sipangpt_eval/reporting/`: Excel generator (`excel_generator.py`) and high-res chart generator (`charts.py`).

3. **Running the Pipeline**:
   - CLI entrypoint: `python main.py [extract|run|evaluate|report|all]`
   - Package setup: `pip install -e .`

4. **Testing and Verification**:
   - Type-checking: `mypy src/`
   - Unit & Integration Tests: `pytest tests/`

5. **Key Outputs**:
   - `data/ground_truth/benchmark_test_50.jsonl`
   - `data/results/matriz_comparativa_final.xlsx`
   - `reports/figures/curva_exactitud_comparada.png` (300 DPI)
   - `reports/figures/latencia_boxplots.png` (300 DPI)
   - `reports/tables/tabla_exactitud_50_preguntas.md`
