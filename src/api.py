"""FastAPI: эндпоинты /health и /predict."""
import logging

from fastapi import FastAPI, HTTPException

from src.config import settings
from src.model import ensemble_predict, load_model, predict
from src.preprocessing import validate_text
from src.schemas import PredictionResponse, ReviewRequest

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment API")

# Загружаем все доступные модели
models: dict[str, object] = {}
try:
    models["LogisticRegression"] = load_model(settings.model_path)
except FileNotFoundError:
    logger.warning(f"Модель не найдена: {settings.model_path}")

try:
    models["NaiveBayes"] = load_model(settings.model_path_nb)
except FileNotFoundError:
    logger.warning(f"Модель не найдена: {settings.model_path_nb}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest) -> PredictionResponse:
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Текст не может быть пустым",
        )

    if validate_text(text) is None:
        raise HTTPException(
            status_code=400,
            detail="Текст слишком короткий или не содержит осмысленных слов для анализа",
        )

    try:
        if len(models) >= 2:
            result = ensemble_predict(text, models)
        elif models:
            result = predict(text, next(iter(models.values())))
        else:
            raise HTTPException(
                status_code=503,
                detail="Модели не загружены. Обратитесь к администратору.",
            )

        return PredictionResponse(
            text=text,
            sentiment=result["sentiment"],
            confidence=result["confidence"],
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Ошибка при анализе")
        raise HTTPException(
            status_code=500,
            detail="Произошла ошибка при анализе текста",
        )


@app.get("/")
def home():
    return {"message": "Sentiment API"}