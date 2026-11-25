"""Process settings. Real credentials stay in the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    llm_mode: str = "local"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    db_path: Path = Path("data/agentops.sqlite")
    data_dir: Path = Path("data")
    max_steps: int = 8
    tool_timeout_seconds: float = 15.0
    tool_retries: int = 2
    retry_backoff_seconds: float = 0.2
    spreadsheet_fail_first: int = 0
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            llm_mode=os.environ.get("AGENTOPS_LLM_MODE", "local").strip().lower(),
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            db_path=Path(os.environ.get("AGENTOPS_DB_PATH", "data/agentops.sqlite")),
            data_dir=Path(os.environ.get("AGENTOPS_DATA_DIR", "data")),
            max_steps=_env_int("AGENTOPS_MAX_STEPS", 8),
            tool_timeout_seconds=_env_float("AGENTOPS_TOOL_TIMEOUT_SECONDS", 15.0),
            tool_retries=_env_int("AGENTOPS_TOOL_RETRIES", 2),
            retry_backoff_seconds=_env_float("AGENTOPS_RETRY_BACKOFF_SECONDS", 0.2),
            spreadsheet_fail_first=_env_int("AGENTOPS_SPREADSHEET_FAIL_FIRST", 0),
            host=os.environ.get("AGENTOPS_HOST", "0.0.0.0"),
            port=_env_int("AGENTOPS_PORT", 8000),
        )
