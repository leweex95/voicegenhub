"""
Logging utilities for VoiceGenHub.

Provides structured logging configuration with support for different
output formats and log levels.
"""

import logging
import warnings
from typing import Optional

import structlog
from rich.console import Console
from rich.logging import RichHandler


def configure_logging(
    level: str = "INFO", format_type: str = "rich", show_locals: bool = False
) -> None:
    """
    Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Logging format ('rich', 'json', 'standard')
        show_locals: Whether to show local variables in tracebacks
    """
    # Suppress third-party library warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    warnings.filterwarnings("ignore", message=".*Defaulting repo_id.*")
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
    warnings.filterwarnings("ignore", message=".*LoRACompatibleLinear.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*LlamaModel is using LlamaSdpaAttention.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*past_key_values.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*scaled_dot_product_attention.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*torch.backends.cuda.sdp_kernel.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=r".*Reference mel length is not equal to 2 \* reference token length.*")
    warnings.filterwarnings("ignore", message=".*LlamaModel is using LlamaSdpaAttention.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*past_key_values.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*scaled_dot_product_attention.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*torch.backends.cuda.sdp_kernel.*", category=FutureWarning)

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Suppress third-party library loggers
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers.utils.hub").setLevel(logging.ERROR)
    logging.getLogger("chatterbox").setLevel(logging.ERROR)
    logging.getLogger("kokoro").setLevel(logging.ERROR)
    logging.getLogger("misaki").setLevel(logging.ERROR)
    logging.getLogger("edge_tts").setLevel(logging.ERROR)
    logging.getLogger("elevenlabs").setLevel(logging.ERROR)
    logging.getLogger("s3gen").setLevel(logging.ERROR)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Configure stdlib logging
    if format_type == "rich":
        import sys
        rich_handler = RichHandler(console=Console(file=sys.stderr, force_terminal=False))
        rich_handler.show_locals = show_locals
        handlers = [rich_handler]
    else:
        handlers = []

    logging.basicConfig(
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        level=log_level,
    )

    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if format_type == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.stdlib.ProcessorFormatter.wrap_for_formatter)

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (defaults to caller's module name)

    Returns:
        Configured logger instance
    """
    return structlog.get_logger(name)


# Configure default logging
configure_logging()
