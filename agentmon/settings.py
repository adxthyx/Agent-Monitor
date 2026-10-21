from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    database_url: str = "postgresql+psycopg://agentmon:agentmon@localhost:5433/agentmon"
    slots_path: Path = ROOT / "config" / "slots.yaml"
    alert_webhook_url: str | None = None

    # Email alerts (SMTP). Disabled unless smtp_host and alert_email_to are set.
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_security: Literal["starttls", "ssl", "none"] = "starttls"
    alert_email_from: str = "agentmon@localhost"
    alert_email_to: str | None = None  # comma-separated

    # Receiver that webhook-style targets post their replies to.
    callback_host: str = "127.0.0.1"
    callback_port: int = 8200
    callback_public_url: str | None = None  # how the target reaches the receiver; default http://host:port


@lru_cache
def get_settings() -> Settings:
    return Settings()
