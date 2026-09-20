from abc import ABC, abstractmethod
from sipangpt_eval.schemas.benchmark import BenchmarkCase
from sipangpt_eval.schemas.inference import InferenceOutput


class BaseLLMClient(ABC):
    """Clase base abstracta para clientes de inferencia de modelos."""

    @abstractmethod
    def query(self, case: BenchmarkCase) -> InferenceOutput:
        """Envia un caso de benchmark al modelo y retorna la respuesta generada con métricas."""
        ...
