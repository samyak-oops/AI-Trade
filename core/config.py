"""
core/config.py – Loads config.yaml + .env into a single Settings object.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Always load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env", override=False)


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    llm_model: str = Field(
        default="meta-llama/llama-4-scout-17b-16e-instruct", env="LLM_MODEL"
    )

    # ── Angel One ─────────────────────────────────────────────────────────────
    angel_api_key: str = Field(default="", env="ANGEL_API_KEY")
    angel_client_id: str = Field(default="", env="ANGEL_CLIENT_ID")
    angel_password: str = Field(default="", env="ANGEL_PASSWORD")
    angel_totp_secret: str = Field(default="", env="ANGEL_TOTP_SECRET")

    # ── News ──────────────────────────────────────────────────────────────────
    serper_api_key: str = Field(default="", env="SERPER_API_KEY")

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: str = Field(default="development", env="APP_ENV")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    db_path: str = Field(default="./data/trading.db", env="DB_PATH")
    cache_dir: str = Field(default="./data/cache", env="CACHE_DIR")
    paper_trading: bool = Field(default=True, env="PAPER_TRADING")
    default_capital: float = Field(default=100_000.0, env="DEFAULT_CAPITAL")
    risk_per_trade_pct: float = Field(default=1.0, env="RISK_PER_TRADE_PCT")

    # ── YAML overlay (loaded separately) ─────────────────────────────────────
    _yaml: Dict[str, Any] = {}

    model_config = {"env_file": ".env", "extra": "ignore"}

    def yaml_section(self, key: str) -> Dict[str, Any]:
        """Return a top-level section from config.yaml."""
        return self._yaml.get(key, {})


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    yaml_path = Path(__file__).parent.parent / "config.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as fh:
            settings._yaml = yaml.safe_load(fh) or {}
    # Ensure cache dir exists
    Path(settings.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    return settings
