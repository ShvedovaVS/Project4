"""Telegram-бот для анализа тональности текста."""
import asyncio
import logging

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command

from src.config import settings
from src.model import ensemble_predict, load_model, predict

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

bot = Bot(token=settings.bot_token)
dp = Dispatcher()

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

if not models:
    logger.error("Ни одна модель не загружена! Запустите: python -m src.train")

# Словарь для красивого отображения тональности
SENTIMENT_LABELS = {
    "positive": "😊 Положительная",
    "neutral": "😐 Нейтральная",
    "negative": "😠 Отрицательная",
}

MODEL_NAMES = {
    "LogisticRegression": "📐 Логистическая регрессия",
    "NaiveBayes": "📊 Наивный Байес",
}


def _format_single(text: str, sentiment: str, confidence: float) -> str:
    """Форматирует ответ бота с эмодзи и пояснением (одна модель)."""
    label = SENTIMENT_LABELS.get(sentiment, sentiment)
    confidence_pct = int(confidence * 100)

    if confidence_pct >= 80:
        comment = "Бот очень уверен в этом результате!"
    elif confidence_pct >= 50:
        comment = "Бот довольно уверен."
    else:
        comment = "Бот не совсем уверен, результат может быть другим."

    # Обрезаем текст только если он слишком длинный для Telegram (4096 символов)
    display_text = text[:300]
    if len(text) > 300:
        display_text += "..."

    return (
        f"📊 *Анализ тональности*\n\n"
        f"📝 *Текст:* «{display_text}»\n\n"
        f"🎯 *Тональность:* {label}\n"
        f"📈 *Уверенность:* {confidence_pct}%\n\n"
        f"💬 {comment}"
    )


def _format_comparison(text: str, result: dict) -> str:
    """Форматирует ответ с результатами всех моделей и ensemble."""
    label = SENTIMENT_LABELS.get(result["sentiment"], result["sentiment"])
    confidence_pct = int(result["confidence"] * 100)

    display_text = text[:300]
    if len(text) > 300:
        display_text += "..."

    lines = [f"📊 *Анализ тональности*\n", f"📝 *Текст:* «{display_text}»\n", "🔬 *Результаты моделей:*"]

    # Результаты каждой модели
    for r in result["models_results"]:
        model_name = MODEL_NAMES.get(r["model"], r["model"])
        s_label = SENTIMENT_LABELS.get(r["sentiment"], r["sentiment"])
        s_pct = int(r["confidence"] * 100)
        lines.append(f"  • {model_name}: {s_label} ({s_pct}%)")
    lines.append("")

    # Итоговый результат
    lines.append(f"🏆 *Итог (ensemble):* {label}")
    lines.append(f"📈 *Уверенность:* {confidence_pct}%")
    lines.append("")

    # Распределение по классам
    lines.append("📋 *Распределение:*")
    for cls, score in result["class_scores"].items():
        cls_label = SENTIMENT_LABELS.get(cls, cls)
        bar_len = int(score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(f"  {cls_label}: {bar} {int(score*100)}%")
    lines.append("")

    # Комментарий
    if confidence_pct >= 80:
        comment = "Все модели согласны — результат надёжный!"
    elif confidence_pct >= 50:
        comment = "Модели в основном согласны."
    else:
        comment = "Модели не могут определиться — текст неоднозначный."
    lines.append(f"💬 {comment}")

    return "\n".join(lines)


@dp.message(Command("start"))
async def cmd_start(message: types.Message) -> None:
    """Команда /start — приветствие и инструкция."""
    model_count = len(models)
    model_list = "\n".join(
        f"  • {name}" for name in MODEL_NAMES.values()
    )
    welcome = (
        "👋 *Привет! Я — бот для анализа тональности текста.*\n\n"
        "Я помогу определить, положительное ли сообщение, нейтральное или отрицательное.\n\n"
        "🤖 *Мои модели ({count}):*\n{list}\n\n"
        "📌 *Как пользоваться:*\n"
        "• Просто *напишите мне текст* — я проанализирую его всеми моделями.\n\n"
        "⚠️ Обратите внимание: я анализирую русскоязычные тексты.\n\n"
        "🚀 Попробуйте — отправьте любой текст!"
    ).format(count=model_count, list=model_list)

    await message.answer(welcome, parse_mode="Markdown")


@dp.message(lambda message: message.text and not message.text.startswith("/"))
async def handle_message(message: types.Message) -> None:
    """Обработка обычных текстовых сообщений."""
    text = message.text.strip()
    if not text:
        await message.answer("✏️ Пожалуйста, отправьте текст для анализа.")
        return

    if not models:
        await message.answer(
            "⚠️ Модели не загружены. Обратитесь к администратору бота."
        )
        return

    try:
        if len(models) >= 2:
            # Если моделей ≥ 2 — используем ensemble
            result = ensemble_predict(text, models)
            response = _format_comparison(text, result)
        else:
            # Одна модель — простой ответ
            result = predict(text, list(models.values())[0])
            response = _format_single(
                text, result["sentiment"], result["confidence"]
            )
        await message.answer(response, parse_mode="Markdown")
    except Exception as exc:
        logger.exception("Ошибка при анализе сообщения")
        await message.answer(
            "❌ Произошла ошибка при анализе. Попробуйте позже."
        )


async def main() -> None:
    """Запуск бота в режиме polling."""
    logger.info("Запуск бота...")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())