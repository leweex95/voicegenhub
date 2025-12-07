"""Unit tests for configuration settings."""
import os
from unittest.mock import patch

from voicegenhub.config.settings import Settings, get_settings


class TestSettings:
    """Unit tests for Settings class."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()

        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.default_format == "mp3"
        assert settings.max_text_length == 10000

    @patch.dict(os.environ, {"VOICEGENHUB_DEBUG": "true"})
    def test_debug_from_env(self):
        """Test debug setting from environment."""
        settings = Settings()
        assert settings.debug is True

    @patch.dict(os.environ, {"VOICEGENHUB_DEBUG": "false"})
    def test_debug_false_from_env(self):
        """Test debug setting false from environment."""
        settings = Settings()
        assert settings.debug is False

    @patch.dict(os.environ, {"VOICEGENHUB_LOG_LEVEL": "DEBUG"})
    def test_log_level_from_env(self):
        """Test log level from environment."""
        settings = Settings()
        assert settings.log_level == "DEBUG"

    def test_get_settings_singleton(self):
        """Test get_settings returns singleton instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    @patch.dict(os.environ, {"VOICEGENHUB_DEBUG": "true", "VOICEGENHUB_LOG_LEVEL": "ERROR"})
    def test_settings_with_env_vars(self):
        """Test settings with multiple environment variables."""
        # Clear singleton
        import voicegenhub.config.settings as settings_module
        settings_module._settings = None

        settings = get_settings()
        assert settings.debug is True
        assert settings.log_level == "ERROR"
