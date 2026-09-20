from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class TurnType(str, Enum):
    MONOTURNO = "monoturno"
    MULTITURNO = "multiturno"


class ModuleCategory(str, Enum):
    MATRICULA = "Matrícula y Registros Académicos"
    CAMPUS_VIRTUAL = "Campus Virtual, Aula y Zoom"
    PAGOS = "Pagos Virtuales y Cobranzas"
    BIBLIOTECA = "Biblioteca Virtual"
    TRAMITES_GRADOS = "Grados, Títulos y Régimen Estudiantil"


class ConversationMessage(BaseModel):
    role: str = Field(..., description="'human' o 'gpt'")
    content: str


class BenchmarkCase(BaseModel):
    id: int
    modulo: ModuleCategory
    tipo: TurnType
    pregunta_usuario: str
    respuesta_esperada: str
    documento_origen: Optional[str] = None
    seccion_o_articulo: Optional[str] = None
    conversacion_completa: Optional[List[ConversationMessage]] = None
