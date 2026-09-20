import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from rich.console import Console

from sipangpt_eval.clients.base import BaseLLMClient
from sipangpt_eval.config import settings
from sipangpt_eval.schemas.benchmark import BenchmarkCase
from sipangpt_eval.schemas.inference import InferenceOutput

console = Console()


class BenchmarkRunner:
    """Orquestador de inferencia sobre el banco de casos de prueba con soporte de checkpoints."""

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

    def _save_checkpoint(
        self,
        path_gemma: Path,
        path_rag: Path,
        gemma_outputs: List[InferenceOutput],
        rag_outputs: List[InferenceOutput]
    ) -> None:
        """Persiste los resultados de inferencia tras cada caso procesado."""
        with open(path_gemma, "w", encoding="utf-8") as f:
            json.dump([out.model_dump() for out in gemma_outputs], f, indent=2, ensure_ascii=False)

        with open(path_rag, "w", encoding="utf-8") as f:
            json.dump([out.model_dump() for out in rag_outputs], f, indent=2, ensure_ascii=False)

    def run_benchmark(
        self, cases: Sequence[BenchmarkCase], resume: bool = True
    ) -> Tuple[List[InferenceOutput], List[InferenceOutput]]:
        """Ejecuta inferencia con persistencia continua y reutilización de checkpoints para evitar repeticiones."""
        path_gemma: Path = self.results_dir / "run_gemma4_finetuned.json"
        path_rag: Path = self.results_dir / "run_sipan_stair_rag.json"

        cached_gemma: Dict[int, InferenceOutput] = {}
        cached_rag: Dict[int, InferenceOutput] = {}

        if resume:
            if path_gemma.exists():
                try:
                    with open(path_gemma, "r", encoding="utf-8") as f:
                        data_gemma = json.load(f)
                        if isinstance(data_gemma, list):
                            for idx_c, item in enumerate(data_gemma):
                                out = InferenceOutput.model_validate(item)
                                case_id = cases[idx_c].id if idx_c < len(cases) else idx_c + 1
                                cached_gemma[case_id] = out
                except Exception:
                    pass

            if path_rag.exists():
                try:
                    with open(path_rag, "r", encoding="utf-8") as f:
                        data_rag = json.load(f)
                        if isinstance(data_rag, list):
                            for idx_c, item in enumerate(data_rag):
                                out = InferenceOutput.model_validate(item)
                                case_id = cases[idx_c].id if idx_c < len(cases) else idx_c + 1
                                cached_rag[case_id] = out
                except Exception:
                    pass

        gemma_outputs: List[InferenceOutput] = []
        rag_outputs: List[InferenceOutput] = []
        total_cases: int = len(cases)

        console.print(f"[bold yellow]Iniciando inferencia (Total: {total_cases} preguntas de prueba)...[/bold yellow]")

        for idx, case in enumerate(cases, 1):
            out_gemma: Optional[InferenceOutput] = cached_gemma.get(case.id)
            out_rag: Optional[InferenceOutput] = cached_rag.get(case.id)

            is_from_cache: bool = False
            if out_gemma is not None and out_rag is not None:
                is_from_cache = True
            else:
                out_gemma = self.gemma_client.query(case)
                out_rag = self.rag_client.query(case)

            gemma_outputs.append(out_gemma)
            rag_outputs.append(out_rag)

            # Guardar avances tras cada caso
            self._save_checkpoint(path_gemma, path_rag, gemma_outputs, rag_outputs)

            status_tag = "[blue][REUTILIZADO][/blue]" if is_from_cache else "[bold green][PROCESADO][/bold green]"
            console.print(
                f"{status_tag} [bold green][{idx}/{total_cases}][/bold green] Caso #{case.id} ({case.tipo.value}) "
                f"[{case.modulo.value}] | "
                f"Gemma-4: [magenta]{out_gemma.tiempo_total_ms:.0f}ms[/magenta] | "
                f"RAG: [cyan]{out_rag.tiempo_total_ms:.0f}ms[/cyan] (Citas: {len(out_rag.citas)})"
            )

        return gemma_outputs, rag_outputs
