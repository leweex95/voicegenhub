"""Simple VoiceGenHub TTS Engine."""

from typing import List, Dict, Optional, Any
import asyncio

from ..providers.base import TTSProvider, TTSRequest, TTSResponse, AudioFormat, TTSError
from ..providers.factory import provider_factory
from .voice import VoiceSelector
from ..utils.logger import get_logger

logger = get_logger(__name__)


class VoiceGenHub:
    """TTS engine supporting multiple providers."""
    
    def __init__(self, provider: str = "edge"):
        """Initialize VoiceGenHub engine."""
        self._provider: Optional[TTSProvider] = None
        self._voice_selector: Optional[VoiceSelector] = None
        self._initialized = False
        self._provider_id = provider
    
    async def initialize(self) -> None:
        """Initialize the engine and load the specified provider."""
        if self._initialized:
            return
        
        logger.info("Initializing VoiceGenHub engine")
        
        # Discover and register providers
        await provider_factory.discover_and_register_providers()
        
        # Create provider instance
        try:
            self._provider = await provider_factory.create_provider(self._provider_id)
            logger.info(f"Loaded provider: {self._provider.display_name}")
        except Exception as e:
            raise TTSError(f"Failed to load {self._provider_id} provider: {e}")
        
        # Initialize voice selector
        self._voice_selector = VoiceSelector([self._provider])
        
        self._initialized = True
        logger.info(f"VoiceGenHub initialized with {self._provider_id} provider")
    
    async def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        audio_format: Optional[AudioFormat] = None,
        sample_rate: Optional[int] = None,
        speed: float = 1.0,
        **kwargs
    ) -> TTSResponse:
        """
        Generate speech from text.
        
        Args:
            text: Text to synthesize
            voice: Voice ID or name
            language: Language code (e.g., 'en', 'en-US')
            audio_format: Output audio format
            sample_rate: Audio sample rate
            speed: Speech speed (0.5-2.0)
            **kwargs: Additional parameters
            
        Returns:
            TTS response with audio data
        """
        await self.initialize()
        
        # Prepare request
        request = TTSRequest(
            text=text,
            voice_id=voice or "en-US-AriaNeural",
            language=language,
            audio_format=audio_format or AudioFormat.MP3,
            sample_rate=sample_rate or 22050,
            speed=speed,
        )
        
        # Generate audio
        logger.info(f"Generating audio with Edge TTS")
        response = await self._provider.synthesize(request)
        
        logger.info(f"Successfully generated {response.duration:.2f}s of audio")
        return response
    
    async def get_voices(
        self,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get available voices.
        
        Args:
            language: Filter by language
            
        Returns:
            List of voice information
        """
        await self.initialize()
        
        voices = await self._voice_selector.get_all_voices()
        
        if language:
            voices = [v for v in voices if v.language == language or v.locale.startswith(language)]
        
        # Convert to dict format
        return [
            {
                "id": voice.id,
                "name": voice.name,
                "language": voice.language,
                "locale": voice.locale,
                "gender": voice.gender.value,
                "provider": voice.provider,
            }
            for voice in voices
        ]