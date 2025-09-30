"""
Simple Voice Selection

Provides basic voice selection functionality for TTS providers.
"""

from typing import List, Optional
import asyncio

from ..providers.base import Voice, TTSProvider
from ..utils.logger import get_logger

logger = get_logger(__name__)


class VoiceSelector:
    """Simple voice selector for TTS providers."""
    
    def __init__(self, providers: List[TTSProvider] = None):
        self.providers = providers or []
        self._voice_cache: dict = {}
    
    async def get_all_voices(self) -> List[Voice]:
        """Get all available voices from all providers."""
        all_voices = []
        for provider in self.providers:
            try:
                voices = await provider.get_voices()
                all_voices.extend(voices)
            except Exception as e:
                logger.error(f"Failed to get voices from {provider.provider_id}: {e}")
        return all_voices
    
    async def select_voice(self, language: Optional[str] = None) -> Optional[Voice]:
        """Select the first suitable voice for the given language."""
        voices = await self.get_all_voices()
        
        if not voices:
            return None
        
        # If language specified, try to find a matching voice
        if language:
            lang_code = language.lower().split('-')[0]  # Extract base language
            for voice in voices:
                if voice.language.lower().startswith(lang_code):
                    return voice
        
        # Return first available voice as fallback
        return voices[0] if voices else None