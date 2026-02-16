"""Kaggle configuration settings for voicegenhub."""

import json
from pathlib import Path
from typing import Dict, Any

DEFAULT_CONFIG = {
    "deployment_timeout_minutes": 30,
    "polling_interval_seconds": 60,
    "retry_interval_seconds": 60,
    "kernel_id": "leventecsibi/voicegenhub-gpu",
}


def load_kaggle_config() -> Dict[str, Any]:
    """Load Kaggle configuration from kaggle_settings.json or return defaults."""
    config_path = Path(__file__).parent / "config" / "kaggle_settings.json"

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return {**DEFAULT_CONFIG, **config}
        except (json.JSONDecodeError, Exception):
            pass

    return DEFAULT_CONFIG.copy()
