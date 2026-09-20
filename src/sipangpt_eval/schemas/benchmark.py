from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class TurnType(str, Enum):
    MONOTURNO = "monoturno"
    MULTITURNO = "multiturno"


class ModuleCategory(str, Enum):
    PAGOS = "Pagos y Cobranzas"
    CAMPUS_VIRTUAL = "Campus Virtual y Aprendizaje"
    MATRICULA = "Matrícula y Registros"
    NORMATIVA = "Normativa y Trámites"
    BIBLIOTECA = "Biblioteca Virtual"


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
