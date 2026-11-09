"""FastAPI: эндпоинты /health и /predict."""
from fastapi import FastAPI

from src.config import settings
from src.model import load_model, predict
from src.schemas import ReviewRequest, PredictionResponse

app = FastAPI(title="Sentiment API")
model = load_model(settings.model_path)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest) -> PredictionResponse:
    return PredictionResponse(**predict(request.text, model))

@app.get("/")
def home():
    return {"message": "Sentiment API"}