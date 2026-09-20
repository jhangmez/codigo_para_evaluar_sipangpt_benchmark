from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Configuración global del sistema de evaluación utilizando Pydantic Settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    gemma_api_url: str = Field(
        default="http://localhost:11434/api/generate",
        description="URL del endpoint local para Ollama / llama.cpp / Gemma-4"
    )
    gemma_model_name: str = Field(
        default="gemma4:finetuned",
        description="Nombre del modelo fine-tuned"
    )
    sipan_rag_api_url: str = Field(
        default="http://localhost:8000/api/v1/query",
        description="URL del endpoint API de SipánGPT RAG (STAIR)"
    )
    sipan_rag_api_token: str = Field(
        default="",
        description="Token opcional de autorización para la API de SipánGPT"
    )
    hf_dataset_name: str = Field(
        default="ussipan/sipangpt-V2",
        description="Identificador del dataset oficial en HuggingFace Hub"
    )
    data_dir: str = Field(
        default="data",
        description="Directorio raíz para almacenamiento de datos"
    )
    reports_dir: str = Field(
        default="reports",
        description="Directorio raíz para salidas de informes y gráficos"
    )

    @property
    def ground_truth_dir(self) -> Path:
        return Path(self.data_dir) / "ground_truth"

    @property
    def results_dir(self) -> Path:
        return Path(self.data_dir) / "results"

    @property
    def figures_dir(self) -> Path:
        return Path(self.reports_dir) / "figures"

    @property
    def tables_dir(self) -> Path:
        return Path(self.reports_dir) / "tables"


settings = Settings()
