"""Application settings loaded from environment variables or a local .env file."""

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the verification application."""

    gemini_api_key: SecretStr | None = None
    tesseract_cmd: str = "tesseract"
    tesseract_psm: int = 3

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def blank_key_is_unset(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
