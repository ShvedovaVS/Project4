"""Обучение моделей и сохранение в models/."""
from pathlib import Path

from datasets import load_dataset
import joblib
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_text


def prepare_data() -> tuple:
    """Загружает и подготавливает датасет ai-forever/ru-reviews-classification."""
    dataset = load_dataset("ai-forever/ru-reviews-classification")
    df = pd.DataFrame(dataset["train"])
    df = df.rename(columns={"label_text": "sentiment"})
    df = df[["text", "sentiment"]]

    label_map = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }
    df["sentiment"] = df["sentiment"].map(label_map)
    df["sentiment"] = df["sentiment"].astype(int)

    df["clean_text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"],
    )
    return X_train, X_test, y_train, y_test


def train_logreg(X_train, X_test, y_train, y_test) -> None:
    """Обучает LogisticRegression + TfidfVectorizer."""
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
            ("model", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"LogisticRegression точность на тесте: {score:.4f}")

    model_path = Path("models/logreg.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Модель сохранена: {model_path}")


def train_naive_bayes(X_train, X_test, y_train, y_test) -> None:
    """Обучает MultinomialNB + CountVectorizer."""
    pipeline = Pipeline(
        [
            ("count", CountVectorizer(max_features=50000, ngram_range=(1, 2))),
            ("model", MultinomialNB(alpha=1.0)),
        ]
    )
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"MultinomialNB точность на тесте: {score:.4f}")

    model_path = Path("models/naivebayes.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Модель сохранена: {model_path}")


def train() -> None:
    """Обучает все модели на датасете ai-forever/ru-reviews-classification."""
    print("Загрузка и подготовка данных...")
    X_train, X_test, y_train, y_test = prepare_data()

    print("\nОбучение LogisticRegression...")
    train_logreg(X_train, X_test, y_train, y_test)

    print("\nОбучение MultinomialNB...")
    train_naive_bayes(X_train, X_test, y_train, y_test)

    print("\nВсе модели обучены!")


if __name__ == "__main__":
    train()