"""Unit tests for logging utilities."""
from unittest.mock import patch

from voicegenhub.utils.logger import configure_logging, get_logger


class TestLogger:
    """Unit tests for logging utilities."""

    @patch("voicegenhub.utils.logger.logging.basicConfig")
    @patch("voicegenhub.utils.logger.structlog.configure")
    def test_configure_logging_default(self, mock_structlog, mock_basic_config):
        """Test default logging configuration."""
        configure_logging()

        mock_basic_config.assert_called_once()
        mock_structlog.assert_called_once()

        # Check basic config call
        call_args = mock_basic_config.call_args
        assert call_args[1]["level"] == 20  # INFO level

    @patch("voicegenhub.utils.logger.logging.basicConfig")
    @patch("voicegenhub.utils.logger.structlog.configure")
    def test_configure_logging_debug(self, mock_structlog, mock_basic_config):
        """Test debug logging configuration."""
        configure_logging(level="DEBUG")

        call_args = mock_basic_config.call_args
        assert call_args[1]["level"] == 10  # DEBUG level

    @patch("voicegenhub.utils.logger.logging.basicConfig")
    @patch("voicegenhub.utils.logger.structlog.configure")
    def test_configure_logging_rich_format(self, mock_structlog, mock_basic_config):
        """Test rich format logging configuration."""
        configure_logging(format_type="rich")

        # Should use RichHandler
        mock_basic_config.assert_called_once()

    @patch("voicegenhub.utils.logger.logging.basicConfig")
    @patch("voicegenhub.utils.logger.structlog.configure")
    @patch("voicegenhub.utils.logger.warnings.filterwarnings")
    def test_configure_logging_warnings_filtered(self, mock_warnings, mock_structlog, mock_basic_config):
        """Test that warnings are filtered."""
        configure_logging()

        # Check that warnings.filterwarnings was called multiple times
        assert mock_warnings.call_count >= 3

    def test_get_logger(self):
        """Test getting a logger instance."""
        logger = get_logger("test")
        assert logger is not None

    def test_get_logger_no_name(self):
        """Test getting logger without name."""
        logger = get_logger()
        assert logger is not None
