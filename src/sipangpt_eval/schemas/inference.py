from typing import List, Optional
from pydantic import BaseModel, Field


class CitationSource(BaseModel):
    documento: str
    articulo_o_seccion: Optional[str] = None
    url_publica: Optional[str] = None
    similitud: Optional[float] = None


class InferenceOutput(BaseModel):
    modelo_nombre: str
    respuesta_generada: str
    tiempo_total_ms: float
    tiempo_busqueda_ms: Optional[float] = 0.0
    tiempo_generacion_ms: Optional[float] = 0.0
    tokens_totales: Optional[int] = None
    citas: List[CitationSource] = Field(default_factory=list)
