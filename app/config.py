"""Server-level settings from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_key: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    log_level: str = "info"
    request_timeout: int = 300
    max_connections: int = 500
    models_path: str = "models.json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()