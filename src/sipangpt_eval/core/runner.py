import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from sipangpt_eval.clients.base import BaseLLMClient
from sipangpt_eval.clients.local_gemma import LocalGemmaClient
from sipangpt_eval.clients.sipan_rag import SipanRAGClient
from sipangpt_eval.config import settings
from sipangpt_eval.schemas.benchmark import BenchmarkCase
from sipangpt_eval.schemas.inference import InferenceOutput


class BenchmarkRunner:
    """Orquestador de inferencia sobre el banco de casos de prueba."""

    def __init__(
        self,
        gemma_client: BaseLLMClient,
        rag_client: BaseLLMClient,
        results_dir: Path = settings.results_dir
    ) -> None:
        self.gemma_client: BaseLLMClient = gemma_client
        self.rag_client: BaseLLMClient = rag_client
        self.results_dir: Path = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_benchmark(
        self, cases: Sequence[BenchmarkCase]
    ) -> Tuple[List[InferenceOutput], List[InferenceOutput]]:
        """Ejecuta la inferencia de los 50 casos en ambos modelos y guarda los JSONs."""
        gemma_outputs: List[InferenceOutput] = []
        rag_outputs: List[InferenceOutput] = []

        for case in cases:
            out_gemma: InferenceOutput = self.gemma_client.query(case)
            out_rag: InferenceOutput = self.rag_client.query(case)

            gemma_outputs.append(out_gemma)
            rag_outputs.append(out_rag)

        # Guardar outputs en archivos JSON
        path_gemma: Path = self.results_dir / "run_gemma4_finetuned.json"
        path_rag: Path = self.results_dir / "run_sipan_stair_rag.json"

        with open(path_gemma, "w", encoding="utf-8") as f:
            json.dump([out.model_dump() for out in gemma_outputs], f, indent=2, ensure_ascii=False)

        with open(path_rag, "w", encoding="utf-8") as f:
            json.dump([out.model_dump() for out in rag_outputs], f, indent=2, ensure_ascii=False)

        return gemma_outputs, rag_outputs
