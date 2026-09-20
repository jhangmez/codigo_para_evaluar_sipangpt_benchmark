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
        description="URL del endpoint local para Ollama / LM Studio / llama.cpp / Gemma-4"
    )
    gemma_model_name: str = Field(
        default="unsloth_gemma-4-E2B-it_1789791679-GGUF",
        description="Nombre del modelo fine-tuned en LM Studio / Ollama"
    )
    gemma_system_prompt: str = Field(
        default=(
            "Eres SipánGPT, el Asistente Virtual Oficial de la Universidad Señor de Sipán (USS) "
            "basado en inteligencia artificial generativa, experto en soporte técnico informático, "
            "plataformas digitales (Campus Virtual, Aula Virtual, Sistema de Registros Académicos, "
            "Biblioteca Virtual) y normativas institucionales. Tu deber es brindar respuestas precisas, "
            "empáticas, estructuradas y estrictamente apegadas a los reglamentos y manuales oficiales de la USS."
        ),
        description="System prompt oficial para la inferencia del modelo fine-tuned"
    )
    sipan_rag_api_url: str = Field(
        default="http://localhost:3000/api/chat",
        description="URL del endpoint de la app Next.js SipánGPT (RAG STAIR)"
    )
    sipan_rag_api_token: str = Field(
        default="",
        description="Token opcional de autorización para la API de SipánGPT"
    )
    sipan_rag_cookie: str = Field(
        default="",
        description="Cookie de sesión de NextAuth para autenticación en /api/chat"
    )
    hf_dataset_name: str = Field(
        default="ussipan/sipangpt-V2",
        description="Identificador del dataset oficial en HuggingFace Hub"
    )
    hf_token: str = Field(
        default="",
        description="Token de autenticación opcional de HuggingFace Hub"
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
