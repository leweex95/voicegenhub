"""
Logging utilities for VoiceGenHub.

Provides structured logging configuration with support for different
output formats and log levels.
"""

import logging
import sys
from typing import Optional
import structlog
from rich.logging import RichHandler
from rich.console import Console


def configure_logging(
    level: str = "INFO",
    format_type: str = "rich",
    show_locals: bool = False
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Logging format ('rich', 'json', 'standard')
        show_locals: Whether to show local variables in tracebacks
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure stdlib logging
    if format_type == "rich":
        rich_handler = RichHandler(console=Console(stderr=True))
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