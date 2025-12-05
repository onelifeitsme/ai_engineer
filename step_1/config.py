from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    DATABASE_URL: str
    MISTRAL_API_KEY: str

    @property
    def DATABASE_URL_asyncpg(self):
        return self.DATABASE_URL

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent / ".env"),
        env_file_encoding="utf-8",
    )


settings = Settings()
