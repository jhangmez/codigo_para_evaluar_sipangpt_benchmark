from enum import Enum
from typing import Dict, List, Literal
from pydantic import BaseModel, Field

from sipangpt_eval.schemas.evaluation import JevEvaluationOutput, ScoreCategory


class JevQuestionType(str, Enum):
    CHOICE = "choice"
    SCORE = "score"
    BOOLEAN = "boolean"


class JevChoiceQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str
    criteria: Dict[str, str]


class JevBooleanQuestion(BaseModel):
    type: Literal["boolean"] = "boolean"
    instructions: str


class JevScoreQuestion(BaseModel):
    type: Literal["score"] = "score"
    instructions: str
    criteria: List[str]


class JevChoiceAnswer(BaseModel):
    type: str = "choice"
    choice: str
    probabilities: Dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0


class JevBooleanAnswer(BaseModel):
    type: str = "boolean"
    probability: float = 0.0


class JevScoreAnswer(BaseModel):
    type: str = "score"
    score: float = 0.0
    probabilities: Dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0


__all__ = [
    "JevQuestionType",
    "JevChoiceQuestion",
    "JevBooleanQuestion",
    "JevScoreQuestion",
    "JevChoiceAnswer",
    "JevBooleanAnswer",
    "JevScoreAnswer",
    "JevEvaluationOutput",
    "ScoreCategory",
]
