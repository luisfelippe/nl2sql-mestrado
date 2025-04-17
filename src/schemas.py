"""Esquemas de validação de dados."""

from typing import Literal
from pydantic import BaseModel


class PredictionModel(BaseModel):
    """Modelo de predição."""

    id: str
    db_id: str
    difficulty: Literal["easy", "medium", "hard", "extra hard"]
    instruction: str
    nl: str
    sql_expected: str
    sql_predicted: str
