from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration.

    Values are loaded from environment variables and .env.
    """

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------

    APP_NAME: str = "Municipal AI Service"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "local"
    APP_DEBUG: bool = False

    # -------------------------------------------------------------------------
    # API
    # -------------------------------------------------------------------------

    API_PREFIX: str = "/api/v1"

    # -------------------------------------------------------------------------
    # Server
    # -------------------------------------------------------------------------

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # -------------------------------------------------------------------------
    # AI Models
    # -------------------------------------------------------------------------

    # Supported values:
    # auto, cuda, mps, cpu
    MODEL_DEVICE: str = "auto"

    # Root directory containing model artifacts.
    MODEL_DIR: str = "./models"

    # -------------------------------------------------------------------------
    # Language Detection Model
    # -------------------------------------------------------------------------

    LANGUAGE_MODEL_PATH: str = "models/language/v1"
    LANGUAGE_MODEL_VERSION: str = "1.0.0"

    # -------------------------------------------------------------------------
    # Sentiment Classification Model
    # -------------------------------------------------------------------------

    SENTIMENT_MODEL_PATH: str = "models/sentiment/v1"
    SENTIMENT_MODEL_VERSION: str = "1.0.0"

    SENTIMENT_NUM_LABELS: int = 3

    # -------------------------------------------------------------------------
    # Topic Classification Model
    # -------------------------------------------------------------------------

    TOPIC_MODEL_PATH: str = "models/topic/v1"
    TOPIC_MODEL_VERSION: str = "1.0.0"

    # Final municipal topic taxonomy.
    TOPIC_NUM_LABELS: int = 25

    # -------------------------------------------------------------------------
    # Priority Classification Model
    # -------------------------------------------------------------------------

    PRIORITY_MODEL_PATH: str = "models/priority/v1"
    PRIORITY_MODEL_VERSION: str = "1.0.0"

    # -------------------------------------------------------------------------
    # Laravel → Python security
    # -------------------------------------------------------------------------

    INTERNAL_API_KEY: str = Field(
        ...,
        min_length=32,
    )

    # -------------------------------------------------------------------------
    # Request / inference
    # -------------------------------------------------------------------------

    MAX_TEXT_LENGTH: int = 5000

    MAX_BATCH_SIZE: int = 500

    INFERENCE_TIMEOUT_SECONDS: float = 30.0

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    LOG_LEVEL: str = "INFO"

    # -------------------------------------------------------------------------
    # Pydantic settings
    # -------------------------------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Return the cached application settings.

    Configuration is loaded once and reused throughout
    the lifetime of the application.
    """
    return Settings()
