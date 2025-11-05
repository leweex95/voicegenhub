"""Simple configuration for VoiceGenHub."""

import os
from typing import Optional


class Settings:
    """Simple application settings."""

    def __init__(self):
        # General settings
        self.debug = os.getenv("VOICEGENHUB_DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("VOICEGENHUB_LOG_LEVEL", "INFO").upper()

        # Audio settings
        self.default_format = "mp3"
        self.max_text_length = 10000


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
