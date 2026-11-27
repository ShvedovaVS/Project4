import re
import unicodedata


def clean_text(text: str) -> str:
    """Очищает текст, сохраняя ZWJ в составных emoji."""
    if not text or not isinstance(text, str):
        return ""

    # Нормализация Unicode
    text = unicodedata.normalize("NFKC", text)

    # Удаляет невидимые символы, кроме ZWJ, который соединяет составные emoji
    text = re.sub(r"[\u200b\u200c\ufeff\u2060]", "", text)

    # Удаляет URL
    text = re.sub(r"http\S+", " ", text)

    # Заменяет множественные пробелы, табуляции, переносы строк на один пробел
    text = re.sub(r"\s+", " ", text).strip()

    return text


def validate_text(text: str, min_words: int = 1) -> str | None:
    """Проверяет, что очищенный текст содержит слова или emoji-сигнал.

    Возвращает очищенный текст, если он валиден, или None, если слишком короткий.
    """
    stripped = text.strip()
    if not stripped:
        return None

    cleaned = clean_text(stripped)
    if not cleaned or len(cleaned.split()) < min_words:
        return None
    has_word_signal = any(
        sum(character.isalnum() for character in token) >= 2
        for token in cleaned.split()
    )
    has_symbol_signal = any(
        unicodedata.category(character).startswith("S") for character in cleaned
    )
    if not has_word_signal and not has_symbol_signal:
        return None

    return cleaned