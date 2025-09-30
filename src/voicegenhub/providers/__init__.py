"""Simple TTS Providers Package."""

from .base import (
    TTSProvider,
    Voice,
    VoiceGender,
    VoiceType,
    AudioFormat,
    TTSRequest,
    TTSResponse,
    ProviderCapabilities,
    TTSError,
    ProviderNotAvailableError,
    VoiceNotFoundError,
    TextTooLongError,
    RateLimitError,
    AuthenticationError,
)

from .factory import (
    ProviderFactory,
    provider_factory,
)

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