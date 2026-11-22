import re
import unicodedata


def clean_text(text: str) -> str:
    """Очищает текст: Unicode, zero-width символы, URL, множественные пробелы."""
    if not text or not isinstance(text, str):
        return ""

    # Нормализация Unicode
    text = unicodedata.normalize("NFKC", text)

    # Удаляет zero-width и другие невидимые Unicode-символы
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u2060]", "", text)

    # Удаляет URL
    text = re.sub(r"http\S+", " ", text)

    # Заменяет множественные пробелы, табуляции, переносы строк на один пробел
    text = re.sub(r"\s+", " ", text).strip()

    return text


def validate_text(text: str, min_words: int = 2) -> str | None:
    """Проверяет, что текст не пустой и содержит достаточно слов после очистки.

    Возвращает очищенный текст, если он валиден, или None, если слишком короткий.
    """
    stripped = text.strip()
    if not stripped:
        return None

    cleaned = clean_text(stripped)
    if not cleaned or len(cleaned.split()) < min_words:
        return None

    return cleaned