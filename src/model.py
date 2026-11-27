"""Загрузка моделей и предсказание тональности."""

from pathlib import Path

import joblib

from src.preprocessing import clean_text

LABELS = {0: "negative", 1: "neutral", 2: "positive"}


def load_model(path: str):
    """Загружает обученную модель (joblib)."""
    return joblib.load(Path(path))


def predict(text: str, model) -> dict:
    """Возвращает {'text': ..., 'sentiment': ..., 'confidence': ..., 'proba': ...}."""
    cleaned = clean_text(text)
    proba = model.predict_proba([cleaned])[0]
    idx = int(proba.argmax())
    class_labels = [LABELS[int(class_id)] for class_id in model.classes_]

    # Полное распределение вероятностей по всем классам
    proba_dict = {
        label: round(float(probability), 4)
        for label, probability in zip(class_labels, proba)
    }
    sentiment = class_labels[idx]

    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": round(float(proba[idx]), 3),
        "proba": proba_dict,
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
    """Совместное предсказание: усреднение полных распределений вероятностей."""
    if not models:
        return {
            "text": text,
            "sentiment": "unknown",
            "confidence": 0.0,
            "models_results": [],
            "class_scores": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
        }

    results = predict_with_all(text, models)
    num_models = len(models)

    # Усредняем полные распределения вероятностей по всем моделям
    averaged = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    for r in results:
        for label, prob in r["proba"].items():
            averaged[label] += prob

    for label in averaged:
        averaged[label] /= num_models

    # Выбираем класс с максимальной средней вероятностью
    best_sentiment = max(averaged, key=averaged.get)
    best_confidence = round(averaged[best_sentiment], 3)

    return {
        "text": text,
        "sentiment": best_sentiment,
        "confidence": best_confidence,
        "models_results": results,
        "class_scores": {k: round(v, 3) for k, v in averaged.items()},
    }