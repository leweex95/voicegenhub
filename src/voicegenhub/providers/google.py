"""
Google Cloud TTS Provider

High-quality TTS provider using Google Cloud Text-to-Speech API.
Supports multiple voices, languages, and neural voices.
"""

import asyncio
import os
from typing import List, Dict, Optional, Any, AsyncGenerator
from io import BytesIO
import tempfile

try:
    from google.cloud import texttospeech
    GOOGLE_AVAILABLE = True
except ImportError:
    texttospeech = None
    GOOGLE_AVAILABLE = False

from .base import (
    TTSProvider, Voice, VoiceGender, VoiceType, AudioFormat,
    TTSRequest, TTSResponse, ProviderCapabilities,
    TTSError, VoiceNotFoundError, TextTooLongError, AuthenticationError
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GoogleTTSProvider(TTSProvider):
    """
    Google Cloud TTS provider implementation.
    
    Uses the google-cloud-texttospeech library to provide high-quality 
    text-to-speech synthesis with Google's neural voices.
    """
    
    def __init__(self, name: str = "google", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self._client: Optional[Any] = None
        self._voices_cache: Optional[List[Voice]] = None
        
        if not GOOGLE_AVAILABLE:
            raise TTSError(
                "Google Cloud TTS not available. Install with: pip install google-cloud-texttospeech",
                error_code="DEPENDENCY_MISSING",
                provider=self.provider_id
            )
    
    @property
    def provider_id(self) -> str:
        return "google"
    
    @property
    def display_name(self) -> str:
        return "Google Cloud TTS"
    
    async def initialize(self) -> None:
        """Initialize the Google TTS provider."""
        try:
            # Check for credentials
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            
            if not creds_path and not creds_json:
                logger.warning("No Google Cloud credentials found. Provider will be unavailable.")
                return
            
            # If JSON credentials provided, create temp file
            if creds_json and not creds_path:
                import json
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(json.loads(creds_json), f)
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
            
            # Initialize client
            self._client = texttospeech.TextToSpeechClient()
            
            # Test connection by listing voices
            await self._test_connection()
            
            logger.info("Google Cloud TTS provider initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google TTS provider: {e}")
            raise TTSError(
                f"Google TTS initialization failed: {str(e)}",
                error_code="PROVIDER_INIT_FAILED",
                provider=self.provider_id
            )
    
    async def _test_connection(self) -> None:
        """Test the connection to Google Cloud TTS."""
        if not self._client:
            raise TTSError("Client not initialized", provider=self.provider_id)
        
        try:
            # Try to list voices as a connection test
            voices = self._client.list_voices()
            if not voices.voices:
                raise TTSError("No voices returned from Google TTS", provider=self.provider_id)
        except Exception as e:
            if "authentication" in str(e).lower():
                raise AuthenticationError(f"Google TTS authentication failed: {e}", provider=self.provider_id)
            raise TTSError(f"Google TTS connection test failed: {e}", provider=self.provider_id)
    
    async def get_voices(self, language: Optional[str] = None) -> List[Voice]:
        """
        Get available voices from Google Cloud TTS.
        
        Args:
            language: Optional language filter
            
        Returns:
            List of available voices
        """
        if not self._client:
            raise TTSError("Provider not initialized", provider=self.provider_id)
        
        try:
            # Use cached voices if available
            if self._voices_cache:
                voices = self._voices_cache
            else:
                # Get voices from Google TTS
                response = self._client.list_voices()
                
                voices = []
                for voice in response.voices:
                    # Parse voice information
                    parsed_voice = self._parse_google_voice(voice)
                    voices.append(parsed_voice)
                
                # Cache the results
                self._voices_cache = voices
            
            # Apply language filter if specified
            if language:
                filtered_voices = []
                for voice in voices:
                    if voice.language.startswith(language.lower()) or voice.locale.startswith(language):
                        filtered_voices.append(voice)
                voices = filtered_voices
            
            return voices
            
        except Exception as e:
            logger.error(f"Failed to get Google TTS voices: {e}")
            raise TTSError(
                f"Failed to retrieve voices: {str(e)}",
                error_code="VOICE_LIST_FAILED",
                provider=self.provider_id
            )
    
    def _parse_google_voice(self, voice) -> Voice:
        """Parse Google TTS voice data into our Voice format."""
        if not texttospeech:
            raise TTSError("Google Cloud TTS not available", provider=self.provider_id)
            
        # Get primary language code
        language_codes = voice.language_codes
        primary_locale = language_codes[0] if language_codes else "en-US"
        language = primary_locale.split("-")[0] if primary_locale else "en"
        
        # Map gender
        gender = VoiceGender.NEUTRAL
        if voice.ssml_gender == texttospeech.SsmlVoiceGender.MALE:
            gender = VoiceGender.MALE
        elif voice.ssml_gender == texttospeech.SsmlVoiceGender.FEMALE:
            gender = VoiceGender.FEMALE
        
        # Determine voice type
        voice_type = VoiceType.NEURAL if "Neural" in voice.name else VoiceType.STANDARD
        
        return Voice(
            id=voice.name,
            name=voice.name,
            language=language,
            locale=primary_locale,
            gender=gender,
            voice_type=voice_type,
            provider=self.provider_id,
            sample_rate=24000,  # Google TTS default
            description=f"Google Cloud TTS - {voice.name}",
            quality_score=0.9 if voice_type == VoiceType.NEURAL else 0.7
        )
    
    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """
        Synthesize speech using Google Cloud TTS.
        
        Args:
            request: TTS request parameters
            
        Returns:
            TTS response with audio data
        """
        if not self._client:
            raise TTSError("Provider not initialized", provider=self.provider_id)
        
        if not texttospeech:
            raise TTSError("Google Cloud TTS not available", provider=self.provider_id)
        
        await self.validate_request(request)
        
        try:
            # Prepare synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=request.text)
            
            # Configure voice
            voice_config = texttospeech.VoiceSelectionParams(
                name=request.voice_id,
                language_code=request.language or "en-US"
            )
            
            # Configure audio format
            audio_encoding = self._get_google_audio_format(request.audio_format)
            audio_config = texttospeech.AudioConfig(
                audio_encoding=audio_encoding,
                sample_rate_hertz=request.sample_rate,
                speaking_rate=request.speed,
                pitch=request.pitch,
                volume_gain_db=self._volume_to_db(request.volume)
            )
            
            # Synthesize speech
            response = self._client.synthesize_speech(
                input=synthesis_input,
                voice=voice_config,
                audio_config=audio_config
            )
            
            # Estimate duration
            duration = self._estimate_duration(request.text, request.speed)
            
            tts_response = TTSResponse(
                audio_data=response.audio_content,
                format=request.audio_format,
                sample_rate=request.sample_rate,
                duration=duration,
                voice_used=request.voice_id,
                metadata={
                    "provider": self.provider_id,
                    "original_format": request.audio_format.value
                }
            )
            
            logger.info(f"Successfully generated {duration:.2f}s of audio using Google TTS")
            return tts_response
            
        except Exception as e:
            logger.error(f"Google TTS synthesis failed: {e}")
            if "not found" in str(e).lower():
                raise VoiceNotFoundError(
                    f"Voice {request.voice_id} not found",
                    error_code="VOICE_NOT_FOUND",
                    provider=self.provider_id
                )
            raise TTSError(
                f"Speech synthesis failed: {str(e)}",
                error_code="SYNTHESIS_FAILED",
                provider=self.provider_id
            )
    
    def _get_google_audio_format(self, format: AudioFormat) -> Any:
        """Convert our AudioFormat to Google's AudioEncoding."""
        if not texttospeech:
            raise TTSError("Google Cloud TTS not available", provider=self.provider_id)
            
        format_map = {
            AudioFormat.MP3: texttospeech.AudioEncoding.MP3,
            AudioFormat.WAV: texttospeech.AudioEncoding.LINEAR16,
            AudioFormat.OGG: texttospeech.AudioEncoding.OGG_OPUS,
        }
        return format_map.get(format, texttospeech.AudioEncoding.MP3)
    
    def _volume_to_db(self, volume: float) -> float:
        """Convert volume ratio to decibels."""
        if volume <= 0:
            return -96.0  # Very quiet
        elif volume == 1.0:
            return 0.0    # No change
        else:
            import math
            return 20.0 * math.log10(volume)
    
    def _estimate_duration(self, text: str, speed: float = 1.0) -> float:
        """Estimate audio duration based on text length and speed."""
        # Rough estimation: ~150 words per minute for normal speech
        words = len(text.split())
        base_duration = (words / 150.0) * 60.0  # seconds
        
        # Adjust for speed
        adjusted_duration = base_duration / speed
        
        return max(adjusted_duration, 0.1)  # Minimum 0.1 seconds
    
    async def synthesize_streaming(self, request: TTSRequest) -> AsyncGenerator[bytes, None]:
        """Google TTS doesn't support streaming, so we return the full audio."""
        response = await self.synthesize(request)
        yield response.audio_data
    
    async def health_check(self) -> bool:
        """Check if Google Cloud TTS service is available."""
        try:
            if not self._client:
                return False
            
            # Try to list voices as health check
            voices = self._client.list_voices()
            return len(voices.voices) > 0
            
        except Exception as e:
            logger.error(f"Google TTS health check failed: {e}")
            return False
    
    async def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities."""
        return ProviderCapabilities(
            supports_ssml=True,
            supports_emotions=False,
            supports_styles=False,
            supports_speed_control=True,
            supports_pitch_control=True,
            supports_volume_control=True,
            supports_streaming=False,
            max_text_length=5000,
            rate_limit_per_minute=60,
            supported_formats=[AudioFormat.MP3, AudioFormat.WAV, AudioFormat.OGG],
            supported_sample_rates=[8000, 16000, 22050, 24000, 44100, 48000]
        )
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities."""
        return ProviderCapabilities(
            supports_ssml=True,
            supports_emotions=False,
            supports_styles=False,
            supports_speed_control=True,
            supports_pitch_control=True,
            supports_volume_control=True,
            supports_streaming=False,
            max_text_length=5000,
            rate_limit_per_minute=60,
            supported_formats=[AudioFormat.MP3, AudioFormat.WAV, AudioFormat.OGG],
            supported_sample_rates=[8000, 16000, 22050, 24000, 44100, 48000]
        )