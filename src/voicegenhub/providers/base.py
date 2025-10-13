"""
Base TTS Provider Interface

This module defines the abstract base class and interfaces that all TTS providers
must implement to ensure consistent behavior across different TTS services.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import asyncio

from pydantic import BaseModel


class AudioFormat(Enum):
    """Supported audio formats."""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    FLAC = "flac"
    AAC = "aac"


class VoiceGender(Enum):
    """Voice gender categories."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class VoiceType(Enum):
    """Voice type categories."""
    STANDARD = "standard"
    NEURAL = "neural" 
    PREMIUM = "premium"
    WAVENET = "wavenet"


@dataclass
class Voice:
    """Voice metadata and configuration."""
    id: str
    name: str
    language: str
    locale: str
    gender: VoiceGender
    voice_type: VoiceType
    provider: str
    sample_rate: int = 22050
    description: Optional[str] = None
    emotions: Optional[List[str]] = None
    styles: Optional[List[str]] = None
    age_group: Optional[str] = None
    quality_score: Optional[float] = None


class TTSRequest(BaseModel):
    """TTS generation request parameters."""
    text: str
    voice_id: str
    language: Optional[str] = None
    audio_format: AudioFormat = AudioFormat.MP3
    sample_rate: int = 22050
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    emotion: Optional[str] = None
    style: Optional[str] = None
    ssml: bool = False
    
    class Config:
        use_enum_values = True


class TTSResponse(BaseModel):
    """TTS generation response."""
    audio_data: bytes
    format: AudioFormat
    sample_rate: int
    duration: float
    voice_used: str
    metadata: Dict[str, Any] = {}


class ProviderCapabilities(BaseModel):
    """Provider capability information."""
    supports_ssml: bool = False
    supports_emotions: bool = False
    supports_styles: bool = False
    supports_speed_control: bool = False
    supports_pitch_control: bool = False
    supports_volume_control: bool = False
    supports_streaming: bool = False
    max_text_length: int = 5000
    rate_limit_per_minute: int = 60
    supported_formats: List[AudioFormat] = [AudioFormat.MP3]
    supported_sample_rates: List[int] = [22050]


class TTSError(Exception):
    """Base TTS error class."""
    def __init__(self, message: str, error_code: str = "UNKNOWN", provider: str = None):
        super().__init__(message)
        self.error_code = error_code
        self.provider = provider


class ProviderNotAvailableError(TTSError):
    """Raised when a provider is not available or configured."""
    pass


class VoiceNotFoundError(TTSError):
    """Raised when a requested voice is not found."""
    pass


class TextTooLongError(TTSError):
    """Raised when text exceeds provider limits."""
    pass


class RateLimitError(TTSError):
    """Raised when rate limits are exceeded."""
    pass


class AuthenticationError(TTSError):
    """Raised when authentication fails."""
    pass


class TTSProvider(ABC):
    """
    Abstract base class for all TTS providers.
    
    This class defines the interface that all TTS providers must implement
    to ensure consistent behavior and integration with the VoiceGenHub system.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self._voices_cache: Optional[List[Voice]] = None
        self._capabilities: Optional[ProviderCapabilities] = None
    
    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique identifier for this provider."""
        pass
    
    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable display name for this provider."""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the provider.
        
        This method should handle any setup required for the provider,
        such as authentication, connection setup, etc.
        """
        pass
    
    @abstractmethod
    async def get_voices(self, language: Optional[str] = None) -> List[Voice]:
        """
        Get available voices from the provider.
        
        Args:
            language: Optional language filter (e.g., 'en', 'en-US')
        
        Returns:
            List of available voices
        """
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> ProviderCapabilities:
        """
        Get provider capabilities.
        
        Returns:
            Provider capabilities information
        """
        pass
    
    @abstractmethod
    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """
        Synthesize speech from text.
        
        Args:
            request: TTS request parameters
        
        Returns:
            TTS response with audio data
        
        Raises:
            TTSError: If synthesis fails
        """
        pass
    
    async def synthesize_streaming(
        self, 
        request: TTSRequest
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize speech with streaming response.
        
        Default implementation falls back to regular synthesis.
        Providers that support streaming should override this method.
        
        Args:
            request: TTS request parameters
        
        Yields:
            Audio data chunks
        """
        response = await self.synthesize(request)
        yield response.audio_data
    
    async def validate_request(self, request: TTSRequest) -> None:
        """
        Validate a TTS request against provider capabilities.
        
        Args:
            request: TTS request to validate
        
        Raises:
            TTSError: If request is invalid
        """
        capabilities = await self.get_capabilities()
        
        # Check text length
        if len(request.text) > capabilities.max_text_length:
            raise TextTooLongError(
                f"Text length {len(request.text)} exceeds maximum {capabilities.max_text_length}",
                error_code="TEXT_TOO_LONG",
                provider=self.provider_id
            )
        
        # Check format support
        # Note: audio_format may be a string value due to Pydantic use_enum_values=True
        supported_formats = [fmt if isinstance(fmt, str) else fmt.value for fmt in capabilities.supported_formats]
        request_format = request.audio_format if isinstance(request.audio_format, str) else request.audio_format.value
        if request_format not in supported_formats:
            raise TTSError(
                f"Audio format {request_format} not supported",
                error_code="UNSUPPORTED_FORMAT",
                provider=self.provider_id
            )
        
        # Check sample rate support
        if request.sample_rate not in capabilities.supported_sample_rates:
            raise TTSError(
                f"Sample rate {request.sample_rate} not supported",
                error_code="UNSUPPORTED_SAMPLE_RATE",
                provider=self.provider_id
            )
    
    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and available.
        
        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            await self.get_capabilities()
            return True
        except Exception:
            return False
    
    def __str__(self) -> str:
        return f"{self.display_name} ({self.provider_id})"
    
    def __repr__(self) -> str:
        return f"<TTSProvider: {self.provider_id}>"