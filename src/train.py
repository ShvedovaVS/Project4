"""Обучение моделей и сохранение в models/."""

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from src.preprocessing import clean_text

LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}
CLASS_NAMES = tuple(
    label for label, _ in sorted(LABEL_MAP.items(), key=lambda item: item[1])
)
CLASS_IDS = tuple(range(len(CLASS_NAMES)))
RANDOM_STATE = 42
FEATURE_CONFIGS = (
    {"features": "word", "word_min_df": 1, "word_max_features": 50_000},
    {"features": "word", "word_min_df": 2, "word_max_features": 100_000},
    {
        "features": "word+char",
        "word_min_df": 1,
        "word_max_features": 50_000,
        "char_min_df": 2,
        "char_max_features": 100_000,
    },
    {
        "features": "word+char",
        "word_min_df": 2,
        "word_max_features": 100_000,
        "char_min_df": 3,
        "char_max_features": 150_000,
    },
)
MODEL_PARAMETER_VALUES = {
    "LogisticRegression": (1.0, 0.5, 2.0),
    "MultinomialNB": (1.0, 0.5, 2.0),
}


def _validate_and_prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Проверяет метки и тексты, выводит баланс и сведения о дубликатах."""
    required_columns = {"text", "sentiment"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"В датасете отсутствуют колонки: {sorted(missing_columns)}")

    df = df[["text", "sentiment"]].copy()
    labels = df["sentiment"].astype("string").str.strip().str.lower()
    invalid_labels = labels.isna() | ~labels.isin(LABEL_MAP)
    if invalid_labels.any():
        values = df.loc[invalid_labels, "sentiment"].drop_duplicates().tolist()
        raise ValueError(f"Неизвестные или пустые метки тональности: {values}")
    df["sentiment"] = labels

    invalid_text_types = df["text"].notna() & ~df["text"].map(
        lambda value: isinstance(value, str)
    )
    if invalid_text_types.any():
        raise ValueError(
            "В колонке text обнаружены значения, которые не являются строками."
        )

    df["clean_text"] = df["text"].map(clean_text)
    empty_texts = df["clean_text"].eq("")
    if empty_texts.any():
        print(f"Удалены строки с пустым или отсутствующим текстом: {empty_texts.sum()}")
        df = df.loc[~empty_texts].copy()
    if df.empty:
        raise ValueError(
            "После удаления пустых текстов в датасете не осталось примеров."
        )

    duplicate_rows = df["text"].duplicated(keep=False)
    duplicate_groups = int(df.loc[duplicate_rows, "text"].nunique())
    duplicate_row_count = int(duplicate_rows.sum())
    print(
        "Точные дубликаты текстов — групп: "
        f"{duplicate_groups}; строк в этих группах: {duplicate_row_count}."
    )
    if duplicate_groups:
        conflicting_groups = (
            df.loc[duplicate_rows].groupby("text")["sentiment"].nunique().gt(1).sum()
        )
        if conflicting_groups:
            print(f"Группы одинаковых текстов с разными метками: {conflicting_groups}.")

    class_counts = df["sentiment"].value_counts().reindex(LABEL_MAP, fill_value=0)
    print(
        "Распределение классов: "
        + ", ".join(f"{label}={class_counts[label]}" for label in LABEL_MAP)
    )
    df["sentiment"] = df["sentiment"].map(LABEL_MAP).astype(int)
    return df


def _deduplicate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Убирает повторы model-input и исключает противоречивую разметку."""
    label_counts = df.groupby("clean_text")["sentiment"].nunique()
    conflicting_groups = label_counts[label_counts > 1]
    if not conflicting_groups.empty:
        conflicting_rows = df["clean_text"].isin(conflicting_groups.index)
        print(
            "Группы с конфликтующими метками удалены: "
            f"{len(conflicting_groups)}; строк исключено: {int(conflicting_rows.sum())}."
        )
        df = df.loc[~conflicting_rows].copy()

    duplicates = df["clean_text"].duplicated(keep="first")
    duplicate_count = int(duplicates.sum())
    if duplicate_count:
        print(f"Удалены повторные очищенные тексты до разделения: {duplicate_count}.")
        df = df.loc[~duplicates].copy()
    return df


def _split_dataset(df: pd.DataFrame) -> tuple:
    """Дедуплицирует и делит данные на train/validation/test в пропорции 70/15/15."""
    df = _deduplicate_dataset(df)
    class_counts = df["sentiment"].value_counts().reindex(CLASS_IDS, fill_value=0)
    print(
        "Баланс классов после дедупликации: "
        + ", ".join(
            f"{CLASS_NAMES[class_id]}={class_counts[class_id]}"
            for class_id in CLASS_IDS
        )
    )
    if (class_counts < 6).any():
        insufficient_classes = {
            CLASS_NAMES[class_id]: int(count)
            for class_id, count in class_counts.items()
            if count < 6
        }
        raise ValueError(
            "Для stratified train/validation/test нужно не менее 6 уникальных "
            f"примеров каждого класса; найдено: {insufficient_classes}."
        )

    X_train, X_temporary, y_train, y_temporary = train_test_split(
        df["clean_text"],
        df["sentiment"],
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=df["sentiment"],
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temporary,
        y_temporary,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temporary,
    )

    feature_splits = (X_train, X_validation, X_test)
    all_features = [text for split in feature_splits for text in split]
    if len(all_features) != len(set(all_features)):
        raise RuntimeError("Обнаружено пересечение текстов между частями датасета.")

    expected_classes = set(CLASS_IDS)
    label_splits = (y_train, y_validation, y_test)
    for split_name, labels in zip(("train", "validation", "test"), label_splits):
        if set(labels.unique()) != expected_classes:
            raise RuntimeError(f"В части {split_name} отсутствует один из классов.")

    print(
        "Размеры частей: "
        f"train={len(X_train)}, validation={len(X_validation)}, test={len(X_test)}."
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def prepare_data() -> tuple:
    """Загружает, проверяет и делит датасет на train/validation/test."""
    from datasets import load_dataset

    dataset = load_dataset("ai-forever/ru-reviews-classification")
    df = pd.DataFrame(dataset["train"])
    df = df.rename(columns={"label_text": "sentiment"})
    df = _validate_and_prepare_dataset(df)
    return _split_dataset(df)


def _calculate_metrics(y_true, y_pred) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in CLASS_NAMES
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_IDS).tolist(),
    }


def _evaluate_model(name: str, split_name: str, pipeline, X, y) -> dict:
    metrics = _calculate_metrics(y, pipeline.predict(X))
    _print_metrics(name, split_name, metrics)
    return metrics


def _print_metrics(name: str, split_name: str, metrics: dict) -> None:
    print(
        f"{name} / {split_name}: accuracy={metrics['accuracy']:.4f}, "
        f"macro-F1={metrics['macro_f1']:.4f}"
    )
    for label, scores in metrics["per_class"].items():
        print(
            f"  {label}: precision={scores['precision']:.4f}, "
            f"recall={scores['recall']:.4f}, F1={scores['f1']:.4f}"
        )
    print("  Матрица ошибок (строки — истинный, столбцы — предсказанный класс):")
    print(metrics["confusion_matrix"])


def _build_vectorizer(model_name: str, config: dict):
    vectorizer_type = (
        TfidfVectorizer if model_name == "LogisticRegression" else CountVectorizer
    )
    word_vectorizer = vectorizer_type(
        max_features=config["word_max_features"],
        min_df=config["word_min_df"],
        ngram_range=(1, 2),
    )
    if config["features"] == "word":
        return word_vectorizer

    char_vectorizer = vectorizer_type(
        analyzer="char_wb",
        binary=model_name == "MultinomialNB",
        max_features=config["char_max_features"],
        min_df=config["char_min_df"],
        ngram_range=(3, 5),
    )
    char_weight = 0.2 if model_name == "MultinomialNB" else 0.5
    return FeatureUnion(
        [("word", word_vectorizer), ("char", char_vectorizer)],
        transformer_weights={"word": 1.0, "char": char_weight},
    )


def _build_estimator(model_name: str, config: dict):
    if model_name == "LogisticRegression":
        return LogisticRegression(
            C=config["C"], max_iter=500, random_state=RANDOM_STATE
        )
    if model_name == "MultinomialNB":
        return MultinomialNB(alpha=config["alpha"])
    raise ValueError(f"Неизвестная модель для подбора: {model_name}")


def _build_pipeline(model_name: str, config: dict) -> Pipeline:
    return Pipeline(
        [
            ("features", _build_vectorizer(model_name, config)),
            ("model", _build_estimator(model_name, config)),
        ]
    )


def _tune_model(model_name: str, X_train, y_train, X_validation, y_validation):
    """Выбирает признаки и параметры только по validation macro-F1."""
    parameter_name = {"LogisticRegression": "C", "MultinomialNB": "alpha"}.get(
        model_name
    )
    if parameter_name is None:
        raise ValueError(f"Неизвестная модель для подбора: {model_name}")

    best_config = None
    best_metrics = None
    candidate_results = []
    candidate_number = 0
    total_candidates = len(FEATURE_CONFIGS) * len(MODEL_PARAMETER_VALUES[model_name])

    for feature_config in FEATURE_CONFIGS:
        vectorizer = _build_vectorizer(model_name, feature_config)
        train_features = vectorizer.fit_transform(X_train)
        validation_features = vectorizer.transform(X_validation)

        for parameter_value in MODEL_PARAMETER_VALUES[model_name]:
            config = {**feature_config, parameter_name: parameter_value}
            estimator = _build_estimator(model_name, config)
            estimator.fit(train_features, y_train)
            metrics = _calculate_metrics(
                y_validation, estimator.predict(validation_features)
            )
            candidate_results.append({"config": config, "validation": metrics})
            candidate_number += 1
            print(
                f"{model_name} validation {candidate_number}/{total_candidates}: "
                f"{config}, macro-F1={metrics['macro_f1']:.4f}"
            )
            if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
                best_config = config
                best_metrics = metrics

    _print_metrics(f"{model_name} selected", "validation", best_metrics)
    return best_config, best_metrics, candidate_results


def _fit_test_comparison(
    model_name: str,
    variant_name: str,
    config: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    model_path: Path | None = None,
) -> dict:
    pipeline = _build_pipeline(model_name, config)
    pipeline.fit(X_train, y_train)
    metrics = _evaluate_model(
        f"{model_name} {variant_name}", "test", pipeline, X_test, y_test
    )
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)
        print(f"Модель сохранена: {model_path}")
    return metrics


def _test_set_fingerprint(X_test, y_test) -> str:
    rows = sorted(
        (str(text), int(label)) for text, label in zip(X_test.tolist(), y_test.tolist())
    )
    return hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _save_baseline_report(
    model_metrics: dict,
    X_test,
    y_test,
    report_path: Path = Path("models/baseline_metrics.json"),
) -> Path:
    test_fingerprint = _test_set_fingerprint(X_test, y_test)
    report = {
        "dataset": "ai-forever/ru-reviews-classification",
        "split": {
            "train": 0.7,
            "validation": 0.15,
            "test": 0.15,
            "random_state": RANDOM_STATE,
            "deduplicated_by": "clean_text",
        },
        "test_fingerprint_sha256": test_fingerprint,
        "models": model_metrics,
    }
    if report_path.exists():
        existing_report = json.loads(report_path.read_text(encoding="utf-8"))
        if existing_report.get("test_fingerprint_sha256") != test_fingerprint:
            raise ValueError(
                f"Существующий baseline {report_path} относится к другому test split; "
                "он сохранён без перезаписи."
            )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Baseline-отчёт сохранён: {report_path}")
    return report_path


def _save_final_report(
    selections: dict,
    test_comparison: dict,
    X_test,
    y_test,
    report_path: Path = Path("models/final_metrics.json"),
) -> Path:
    test_fingerprint = _test_set_fingerprint(X_test, y_test)
    report = {
        "dataset": "ai-forever/ru-reviews-classification",
        "split": {
            "train": 0.7,
            "validation": 0.15,
            "test": 0.15,
            "random_state": RANDOM_STATE,
            "deduplicated_by": "clean_text",
            "final_fit_uses": "train+validation",
        },
        "test_fingerprint_sha256": test_fingerprint,
        "validation_selection": selections,
        "test_comparison": test_comparison,
    }
    if report_path.exists():
        existing_report = json.loads(report_path.read_text(encoding="utf-8"))
        if existing_report.get("test_fingerprint_sha256") != test_fingerprint:
            report_path = report_path.with_name(
                f"{report_path.stem}_{test_fingerprint[:8]}{report_path.suffix}"
            )

    if report_path.exists():
        existing_report = json.loads(report_path.read_text(encoding="utf-8"))
        if existing_report.get("test_fingerprint_sha256") != test_fingerprint:
            raise ValueError(
                f"Существующий отчёт {report_path} относится к другому test split; "
                "он сохранён без перезаписи."
            )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Итоговый отчёт сохранён: {report_path}")
    return report_path


def train() -> None:
    """Подбирает настройки по validation и сравнивает модели на test."""
    print("Загрузка и подготовка данных...")
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_data()

    selections = {}
    selected_configs = {}
    for model_name in ("LogisticRegression", "MultinomialNB"):
        print(f"\nПодбор настроек {model_name} по validation...")
        config, validation_metrics, candidates = _tune_model(
            model_name, X_train, y_train, X_validation, y_validation
        )
        selected_configs[model_name] = config
        selections[model_name] = {
            "selected_config": config,
            "selected_validation_metrics": validation_metrics,
            "candidates": candidates,
        }

    X_train_validation = pd.concat([X_train, X_validation])
    y_train_validation = pd.concat([y_train, y_validation])
    test_comparison = {}
    for model_name, model_path in (
        ("LogisticRegression", Path("models/logreg.pkl")),
        ("MultinomialNB", Path("models/naivebayes.pkl")),
    ):
        parameter_name = "C" if model_name == "LogisticRegression" else "alpha"
        baseline_config = {**FEATURE_CONFIGS[0], parameter_name: 1.0}
        baseline_metrics = _fit_test_comparison(
            model_name,
            "baseline refit",
            baseline_config,
            X_train_validation,
            y_train_validation,
            X_test,
            y_test,
        )
        selected_metrics = _fit_test_comparison(
            model_name,
            "selected",
            selected_configs[model_name],
            X_train_validation,
            y_train_validation,
            X_test,
            y_test,
            model_path,
        )
        test_comparison[model_name] = {
            "baseline_refit": baseline_metrics,
            "selected": selected_metrics,
        }

    _save_final_report(selections, test_comparison, X_test, y_test)

    print("\nВсе модели обучены!")


if __name__ == "__main__":
    train()