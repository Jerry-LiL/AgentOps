"""JSON files shared by the spreadsheet and email-draft servers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def data_dir() -> Path:
    return Path(os.environ.get("AGENTOPS_DATA_DIR", "data"))


def load_json(name: str, default: Any) -> Any:
    path = data_dir() / name
    if not path.exists():
        save_json(name, default)
        return json.loads(json.dumps(default))
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(name: str, payload: Any) -> None:
    path = data_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
