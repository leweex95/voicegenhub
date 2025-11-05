"""Simple TTS Providers Package."""

from .base import (
    AudioFormat,
    AuthenticationError,
    ProviderCapabilities,
    ProviderNotAvailableError,
    RateLimitError,
    TextTooLongError,
    TTSError,
    TTSProvider,
    TTSRequest,
    TTSResponse,
    Voice,
    VoiceGender,
    VoiceNotFoundError,
    VoiceType,
)
from .factory import ProviderFactory, provider_factory

__all__ = [
    # Base classes and types
    "TTSProvider",
    "Voice",
    "VoiceGender",
    "VoiceType",
    "AudioFormat",
    "TTSRequest",
    "TTSResponse",
    "ProviderCapabilities",
    # Exceptions
    "TTSError",
    "ProviderNotAvailableError",
    "VoiceNotFoundError",
    "TextTooLongError",
    "RateLimitError",
    "AuthenticationError",
    # Factory
    "ProviderFactory",
    "provider_factory",
]
