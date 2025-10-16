"""
Coqui TTS Provider

High-quality Text-to-Speech using Coqui TTS models.
Supports multiple languages with state-of-the-art deep learning models.
"""

import asyncio
import os
import sys
from typing import List, Dict, Optional, Any
from io import BytesIO
import tempfile

from .base import (
    TTSProvider, Voice, VoiceGender, VoiceType, AudioFormat,
    TTSRequest, TTSResponse, ProviderCapabilities,
    TTSError, VoiceNotFoundError
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CoquiTTSProvider(TTSProvider):
    """
    Coqui TTS Provider - State-of-the-art neural TTS engine.
    
    Coqui TTS is a deep learning TTS framework that provides
    high-quality, natural-sounding speech synthesis.
    """
    
    def __init__(self, provider_id: str = "coqui", config: Dict[str, Any] = None):
        super().__init__(provider_id, config)
        self.tts = None
        self.model_name = None
        self._default_voices = {}
        self._initialized = False
    
    @property
    def provider_id(self) -> str:
        """Unique identifier for this provider."""
        return "coqui"
    
    @property
    def display_name(self) -> str:
        """Human-readable display name for this provider."""
        return "Coqui TTS"
    
    async def initialize(self) -> None:
        """Initialize the Coqui TTS provider."""
        try:
            from TTS.api import TTS
            
            # Get model name from config or use default
            self.model_name = self.config.get("model_name") or "tts_models/en/ljspeech/tacotron2-DDC"
            
            logger.info(f"Initializing Coqui TTS with model: {self.model_name}")
            
            # Initialize TTS model
            # Use GPU if available, otherwise CPU
            try:
                self.tts = TTS(model_name=self.model_name, gpu=True)
                logger.info("Coqui TTS initialized with GPU support")
            except Exception as gpu_error:
                logger.warning(f"GPU initialization failed, falling back to CPU: {gpu_error}")
                self.tts = TTS(model_name=self.model_name, gpu=False)
                logger.info("Coqui TTS initialized with CPU")
            
            self._initialized = True
            logger.info("Coqui TTS provider initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Coqui TTS dependencies not available: {e}")
            if sys.platform == "win32":
                logger.warning("On Windows, TTS requires Microsoft Visual C++ Build Tools. Install with:")
                logger.warning("  1. Install Microsoft C++ Build Tools from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
                logger.warning("  2. Then run: pip install TTS")
            else:
                logger.warning("Coqui TTS provider will be disabled. Install with: pip install TTS")
            self._initialized = False
        except Exception as e:
            logger.warning(f"Failed to initialize Coqui TTS: {str(e)}")
            logger.warning("Coqui TTS provider will be disabled")
            self._initialized = False
    
    async def get_voices(self, language: Optional[str] = None) -> List[Voice]:
        """Get available Coqui voices."""
        # Return a curated list of commonly available Coqui voices
        voices = [
            Voice(
                id="tacotron2-en",
                name="Tacotron2 (English)",
                language="en",
                locale="en-US",
                gender=VoiceGender.FEMALE,
                voice_type=VoiceType.NEURAL,
                provider=self.provider_id,
                sample_rate=22050,
                description="High-quality English voice using Tacotron2"
            ),
            Voice(
                id="glow-tts-en",
                name="Glow-TTS (English)",
                language="en",
                locale="en-US",
                gender=VoiceGender.NEUTRAL,
                voice_type=VoiceType.NEURAL,
                provider=self.provider_id,
                sample_rate=22050,
                description="Fast and expressive English voice using Glow-TTS"
            ),
            Voice(
                id="speedy-speech-en",
                name="Speedy Speech (English)",
                language="en",
                locale="en-US",
                gender=VoiceGender.NEUTRAL,
                voice_type=VoiceType.NEURAL,
                provider=self.provider_id,
                sample_rate=22050,
                description="Fast and lightweight English voice"
            ),
            Voice(
                id="glow-tts-ru",
                name="Glow-TTS (Russian)",
                language="ru",
                locale="ru-RU",
                gender=VoiceGender.FEMALE,
                voice_type=VoiceType.NEURAL,
                provider=self.provider_id,
                sample_rate=22050,
                description="High-quality Russian voice using Glow-TTS"
            ),
        ]
        
        # Filter by language if provided
        if language:
            voices = [v for v in voices if v.language == language or v.locale.startswith(language)]
        
        return voices
    
    async def get_capabilities(self) -> ProviderCapabilities:
        """Get Coqui TTS provider capabilities."""
        return ProviderCapabilities(
            supports_ssml=False,
            supports_emotions=False,
            supports_styles=False,
            supports_speed_control=True,
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_streaming=False,
            max_text_length=10000,
            rate_limit_per_minute=600,
            supported_formats=[AudioFormat.WAV],
            supported_sample_rates=[22050]
        )
    
    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Synthesize speech using Coqui TTS."""
        if not self._initialized:
            raise TTSError(
                "Coqui TTS provider not initialized. Dependencies may not be available on this platform.",
                error_code="NOT_INITIALIZED",
                provider=self.provider_id
            )
        
        try:
            # Validate request
            await self.validate_request(request)
            
            # Map voice ID to model if needed
            model_name = self._get_model_for_voice(request.voice_id)
            
            # Synthesize audio to temporary file
            # Coqui TTS synthesize returns WAV file path
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Synthesize speech
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self.tts.tts_to_file,
                    request.text,
                    None,  # speaker (not used for most models)
                    tmp_path
                )
                
                # Read the generated WAV file
                with open(tmp_path, 'rb') as f:
                    audio_data = f.read()
                
                # Calculate duration (approximate)
                # WAV format: 44 bytes header + audio data
                # Sample rate: 22050 Hz, 16-bit = 2 bytes per sample
                samples = (len(audio_data) - 44) // 2
                duration = samples / 22050.0 if samples > 0 else 0.0
                
                return TTSResponse(
                    audio_data=audio_data,
                    format=AudioFormat.WAV,
                    sample_rate=22050,
                    duration=duration,
                    voice_used=request.voice_id,
                    metadata={
                        "provider": self.provider_id,
                        "model": model_name,
                        "quality": "high"
                    }
                )
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception as e:
                        logger.warning(f"Failed to clean up temporary file {tmp_path}: {e}")
            
        except Exception as e:
            raise TTSError(
                f"Coqui TTS synthesis failed: {str(e)}",
                error_code="SYNTHESIS_ERROR",
                provider=self.provider_id
            )
    
    def _get_model_for_voice(self, voice_id: str) -> str:
        """Get the model name for a voice ID."""
        # Map voice IDs to Coqui model names
        model_mapping = {
            "tacotron2-en": "tts_models/en/ljspeech/tacotron2-DDC",
            "glow-tts-en": "tts_models/en/ljspeech/glow-tts",
            "speedy-speech-en": "tts_models/en/ljspeech/speedy-speech",
            "glow-tts-ru": "tts_models/ru/cv/glow-tts",
        }
        
        return model_mapping.get(voice_id, self.model_name)
