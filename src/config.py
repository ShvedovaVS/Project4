from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Путь к .env файлу в корне проекта (относительно этого файла в src/)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    bot_token: str
    model_path: str = "models/logreg.pkl"
    model_path_nb: str = "models/naivebayes.pkl"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=str(_ENV_PATH) if _ENV_PATH.exists() else "",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()

# Проверка: токен должен быть задан
if not settings.bot_token:
    raise ValueError(
        "bot_token is not configured. "
        "Set BOT_TOKEN environment variable or create .env file with bot_token=..."
    )