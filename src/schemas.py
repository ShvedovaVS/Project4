"""Pydantic-схемы для API-запросов и ответов."""
from pydantic import BaseModel


class ReviewRequest(BaseModel):
    """Запрос на анализ тональности."""

    text: str


class PredictionResponse(BaseModel):
    """Ответ с результатом анализа тональности."""

    text: str
    sentiment: str
    confidence: float
