"""Загрузка моделей и предсказание тональности."""
from pathlib import Path
from typing import Optional

import joblib

from src.preprocessing import clean_text

LABELS = {0: "negative", 1: "neutral", 2: "positive"}


def load_model(path: str):
    """Загружает обученную модель (joblib)."""
    return joblib.load(Path(path))


def predict(text: str, model) -> dict:
    """Возвращает {'text': ..., 'sentiment': ..., 'confidence': ...}."""
    cleaned = clean_text(text)
    proba = model.predict_proba([cleaned])[0]
    idx = int(proba.argmax())
    return {
        "text": text,
        "sentiment": LABELS[idx],
        "confidence": round(float(proba[idx]), 3),
    }


def predict_with_all(
    text: str,
    models: dict[str, object],
) -> list[dict]:
    """Анализирует текст всеми моделями и возвращает список результатов."""
    results = []
    for name, model in models.items():
        result = predict(text, model)
        result["model"] = name
        results.append(result)
    return results


def ensemble_predict(
    text: str,
    models: dict[str, object],
) -> dict:
    """Совместное предсказание: голосование моделей с учётом уверенности."""
    results = predict_with_all(text, models)

    # Считаем взвешенные голоса для каждого класса
    class_scores = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    for r in results:
        class_scores[r["sentiment"]] += r["confidence"]

    # Выбираем класс с максимальной суммарной уверенностью
    best_sentiment = max(class_scores, key=class_scores.get)
    total_score = sum(class_scores.values())
    best_confidence = round(class_scores[best_sentiment] / total_score, 3)

    return {
        "text": text,
        "sentiment": best_sentiment,
        "confidence": best_confidence,
        "models_results": results,
        "class_scores": {k: round(v / total_score, 3) for k, v in class_scores.items()},
    }