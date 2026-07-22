from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 50051
    health_port: int = 8080
    data_path: Path = Path("data/training_examples.jsonl")
    models_dir: Path = Path("models")
    min_confidence: float = 0.8
    min_examples_per_chat: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="EXPENSE_ML_",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
