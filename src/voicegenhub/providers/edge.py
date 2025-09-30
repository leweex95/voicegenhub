"""
Microsoft Edge TTS Provider

High-quality TTS provider using Microsoft Edge's Text-to-Speech service.
Supports multiple voices, languages, and SSML markup.
"""

import asyncio
import re
from typing import List, Dict, Optional, Any, AsyncGenerator
from io import BytesIO
import tempfile
import os

import edge_tts

from .base import (
    TTSProvider, Voice, VoiceGender, VoiceType, AudioFormat,
    TTSRequest, TTSResponse, ProviderCapabilities,
    TTSError, VoiceNotFoundError, TextTooLongError
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EdgeTTSProvider(TTSProvider):
    """
    Microsoft Edge TTS provider implementation.
    
    Uses the edge-tts library to provide high-quality text-to-speech
    synthesis with Microsoft's neural voices.
    """
    
    def __init__(self, name: str = "edge", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self._voices_cache: Optional[List[Voice]] = None
        self._rate_limit_delay = config.get("rate_limit_delay", 0.1) if config else 0.1
    
    @property
    def provider_id(self) -> str:
        return "edge"
    
    @property
    def display_name(self) -> str:
        return "Microsoft Edge TTS"
    
    async def initialize(self) -> None:
        """Initialize the Edge TTS provider."""
        try:
            # Test that we can access the edge-tts API
            await edge_tts.list_voices()
            logger.info("Edge TTS provider initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Edge TTS provider: {e}")
            raise TTSError(
                f"Edge TTS initialization failed: {str(e)}",
                error_code="PROVIDER_INIT_FAILED",
                provider=self.provider_id
            )
            raise TTSError(
                f"Edge TTS initialization failed: {str(e)}",
                error_code="PROVIDER_INIT_FAILED",
                provider=self.provider_id
            )
    
    async def get_voices(self, language: Optional[str] = None) -> List[Voice]:
        """
        Get available voices from Edge TTS.
        
        Args:
            language: Optional language filter
            
        Returns:
            List of available voices
        """
        try:
            # Use cached voices if available
            if self._voices_cache:
                voices = self._voices_cache
            else:
                # Get raw voices from Edge TTS using new API
                raw_voices = await edge_tts.list_voices()
                
                voices = []
                for raw_voice in raw_voices:
                    # Parse voice information
                    voice = self._parse_edge_voice(raw_voice)
                    voices.append(voice)
                
                # Cache the results
                self._voices_cache = voices
            
            # Apply language filter if specified
            if language:
                filtered_voices = []
                for voice in voices:
                    if voice.language.startswith(language.lower()) or voice.locale.startswith(language):
                        filtered_voices.append(voice)
                voices = filtered_voices
            
            logger.info(f"Retrieved {len(voices)} voices from Edge TTS")
            return voices
            
        except Exception as e:
            logger.error(f"Failed to get voices from Edge TTS: {e}")
            raise TTSError(
                f"Failed to retrieve voices: {str(e)}",
                error_code="VOICE_RETRIEVAL_FAILED",
                provider=self.provider_id
            )
    
    async def get_capabilities(self) -> ProviderCapabilities:
        """Get Edge TTS provider capabilities."""
        return ProviderCapabilities(
            supports_ssml=True,
            supports_emotions=True,
            supports_styles=True,
            supports_speed_control=True,
            supports_pitch_control=True,
            supports_volume_control=True,
            supports_streaming=True,
            max_text_length=10000,  # Edge TTS has generous limits
            rate_limit_per_minute=60,
            supported_formats=[AudioFormat.MP3, AudioFormat.WAV],
            supported_sample_rates=[16000, 22050, 24000, 48000]
        )
    
    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """
        Synthesize speech using Edge TTS.
        
        Args:
            request: TTS request parameters
            
        Returns:
            TTS response with audio data
        """
        await self.validate_request(request)
        
        try:
            # Get voice information
            voice_info = await self._get_voice_info(request.voice_id)
            if not voice_info:
                raise VoiceNotFoundError(
                    f"Voice {request.voice_id} not found",
                    error_code="VOICE_NOT_FOUND",
                    provider=self.provider_id
                )
            
            # Prepare text with SSML if needed
            ssml_text = self._prepare_text(request, voice_info)
            
            # Create TTS communication
            communicate = edge_tts.Communicate(ssml_text, voice_info["Name"])
            
            # Generate audio
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            if not audio_data:
                raise TTSError(
                    "No audio data generated",
                    error_code="NO_AUDIO_GENERATED",
                    provider=self.provider_id
                )
            
            # Convert format if needed
            if request.audio_format != AudioFormat.MP3:
                audio_data = await self._convert_audio_format(
                    audio_data, AudioFormat.MP3, request.audio_format
                )
            
            # Calculate duration (approximate)
            duration = self._estimate_duration(request.text, request.speed)
            
            response = TTSResponse(
                audio_data=audio_data,
                format=request.audio_format,
                sample_rate=request.sample_rate,
                duration=duration,
                voice_used=voice_info["Name"],
                metadata={
                    "provider": self.provider_id,
                    "voice_locale": voice_info.get("Locale"),
                    "voice_gender": voice_info.get("Gender"),
                    "original_format": AudioFormat.MP3.value
                }
            )
            
            # Rate limiting
            if self._rate_limit_delay > 0:
                await asyncio.sleep(self._rate_limit_delay)
            
            return response
            
        except Exception as e:
            if isinstance(e, TTSError):
                raise
            
            logger.error(f"Edge TTS synthesis failed: {e}")
            raise TTSError(
                f"Speech synthesis failed: {str(e)}",
                error_code="SYNTHESIS_FAILED",
                provider=self.provider_id
            )
    
    async def synthesize_streaming(
        self, 
        request: TTSRequest
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize speech with streaming response.
        
        Args:
            request: TTS request parameters
            
        Yields:
            Audio data chunks
        """
        await self.validate_request(request)
        
        try:
            voice_info = await self._get_voice_info(request.voice_id)
            if not voice_info:
                raise VoiceNotFoundError(
                    f"Voice {request.voice_id} not found",
                    error_code="VOICE_NOT_FOUND",
                    provider=self.provider_id
                )
            
            ssml_text = self._prepare_text(request, voice_info)
            communicate = edge_tts.Communicate(ssml_text, voice_info["Name"])
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
            
            # Rate limiting
            if self._rate_limit_delay > 0:
                await asyncio.sleep(self._rate_limit_delay)
                
        except Exception as e:
            if isinstance(e, TTSError):
                raise
            
            logger.error(f"Edge TTS streaming failed: {e}")
            raise TTSError(
                f"Streaming synthesis failed: {str(e)}",
                error_code="STREAMING_FAILED",
                provider=self.provider_id
            )
    
    async def health_check(self) -> bool:
        """Check if Edge TTS service is available."""
        try:
            # Try to get voices as health check
            voices = await edge_tts.list_voices()
            return len(voices) > 0
            
        except Exception as e:
            logger.error(f"Edge TTS health check failed: {e}")
            return False
    
    def _parse_edge_voice(self, raw_voice: Dict) -> Voice:
        """Parse Edge TTS voice data into our Voice format."""
        # Extract voice metadata
        name = raw_voice.get("Name", "")
        short_name = raw_voice.get("ShortName", name)
        locale = raw_voice.get("Locale", "")
        language = locale.split("-")[0] if locale else ""
        gender_str = raw_voice.get("Gender", "").lower()
        
        # Map gender
        gender = VoiceGender.NEUTRAL
        if gender_str == "male":
            gender = VoiceGender.MALE
        elif gender_str == "female":
            gender = VoiceGender.FEMALE
        
        # Determine voice type (Edge voices are neural)
        voice_type = VoiceType.NEURAL
        
        # Extract styles and emotions if available
        voice_tag = raw_voice.get("VoiceTag", {})
        content_categories = voice_tag.get("ContentCategories", [])
        voice_personalities = voice_tag.get("VoicePersonalities", [])
        
        # Estimate quality score based on available features
        quality_score = 0.8  # Base score for Edge TTS
        if content_categories:
            quality_score += 0.1
        if voice_personalities:
            quality_score += 0.1
        
        return Voice(
            id=short_name,
            name=short_name,
            language=language,
            locale=locale,
            gender=gender,
            voice_type=voice_type,
            provider=self.provider_id,
            sample_rate=24000,  # Edge TTS default
            description=f"Microsoft Edge TTS - {name}",
            emotions=None,  # Edge TTS doesn't explicitly list emotions
            styles=voice_personalities if voice_personalities else None,
            age_group=self._determine_age_group(voice_personalities),
            quality_score=min(quality_score, 1.0)
        )
    
    def _determine_age_group(self, personalities: List[str]) -> Optional[str]:
        """Determine age group from voice personalities."""
        if not personalities:
            return None
        
        personalities_str = " ".join(personalities).lower()
        if any(word in personalities_str for word in ["child", "young", "kid"]):
            return "child"
        elif any(word in personalities_str for word in ["adult", "mature"]):
            return "adult"
        elif any(word in personalities_str for word in ["elderly", "senior", "old"]):
            return "elderly"
        
        return "adult"  # Default
    
    async def _get_voice_info(self, voice_id: str) -> Optional[Dict]:
        """Get detailed voice information by ID."""
        try:
            voices = await edge_tts.list_voices()
            for voice in voices:
                if voice.get("ShortName") == voice_id or voice.get("Name") == voice_id:
                    return voice
            return None
        except Exception as e:
            logger.error(f"Failed to get voice info for {voice_id}: {e}")
            return None
    
    def _prepare_text(self, request: TTSRequest, voice_info: Dict) -> str:
        """Prepare text with SSML markup."""
        text = request.text.strip()
        
        # If already SSML, return as-is
        if text.startswith("<speak>") and text.endswith("</speak>"):
            return text
        
        # Build SSML
        ssml_parts = ['<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{}">'.format(
            voice_info.get("Locale", "en-US")
        )]
        
        # Add voice element with prosody controls
        prosody_attrs = []
        if request.speed != 1.0:
            speed_percent = int((request.speed - 1.0) * 100)
            prosody_attrs.append(f'rate="{speed_percent:+d}%"')
        
        if request.pitch != 1.0:
            pitch_percent = int((request.pitch - 1.0) * 100)
            prosody_attrs.append(f'pitch="{pitch_percent:+d}%"')
        
        if request.volume != 1.0:
            volume_level = "loud" if request.volume > 1.0 else "soft"
            prosody_attrs.append(f'volume="{volume_level}"')
        
        if prosody_attrs:
            ssml_parts.append(f'<prosody {" ".join(prosody_attrs)}>')
        
        # Add emotion/style if supported and specified
        if request.emotion or request.style:
            style = request.emotion or request.style
            ssml_parts.append(f'<mstts:express-as style="{style}">')
        
        # Add the actual text
        ssml_parts.append(self._escape_ssml(text))
        
        # Close tags
        if request.emotion or request.style:
            ssml_parts.append('</mstts:express-as>')
        
        if prosody_attrs:
            ssml_parts.append('</prosody>')
        
        ssml_parts.append('</speak>')
        
        return "".join(ssml_parts)
    
    def _escape_ssml(self, text: str) -> str:
        """Escape special characters for SSML."""
        # Basic XML escaping
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&apos;")
        return text
    
    async def _convert_audio_format(
        self, 
        audio_data: bytes, 
        from_format: AudioFormat, 
        to_format: AudioFormat
    ) -> bytes:
        """Convert audio between formats using pydub."""
        if from_format == to_format:
            return audio_data
        
        try:
            from pydub import AudioSegment
            
            # Load audio data
            with tempfile.NamedTemporaryFile(suffix=f".{from_format.value}") as temp_in:
                temp_in.write(audio_data)
                temp_in.flush()
                
                # Load with pydub
                audio = AudioSegment.from_file(temp_in.name, format=from_format.value)
                
                # Export to target format
                with tempfile.NamedTemporaryFile(suffix=f".{to_format.value}") as temp_out:
                    audio.export(temp_out.name, format=to_format.value)
                    temp_out.seek(0)
                    return temp_out.read()
        
        except ImportError:
            logger.warning("pydub not available, cannot convert audio format")
            return audio_data
        except Exception as e:
            logger.error(f"Audio format conversion failed: {e}")
            return audio_data
    
    def _estimate_duration(self, text: str, speed: float = 1.0) -> float:
        """Estimate audio duration based on text length and speed."""
        # Rough estimation: ~150 words per minute for normal speech
        words = len(text.split())
        base_duration = (words / 150.0) * 60.0  # seconds
        
        # Adjust for speed
        adjusted_duration = base_duration / speed
        
        return max(adjusted_duration, 0.1)  # Minimum 0.1 seconds