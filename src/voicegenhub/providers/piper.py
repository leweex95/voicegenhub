"""
Piper TTS Provider

Fast and local neural text-to-speech using Piper models.
Supports multiple languages and voices with excellent quality.
"""

import asyncio
import os
import sys
import json
from typing import List, Dict, Optional, Any
from io import BytesIO
import tempfile
from pathlib import Path

from .base import (
    TTSProvider, Voice, VoiceGender, VoiceType, AudioFormat,
    TTSRequest, TTSResponse, ProviderCapabilities,
    TTSError, VoiceNotFoundError
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PiperTTSProvider(TTSProvider):
    """
    Piper TTS Provider - Fast local neural TTS engine.
    
    Piper is a fast and local neural text-to-speech engine that runs
    completely offline after downloading voice models.
    """
    
    def __init__(self, name: str = "piper", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.piper_voice = None
        self.model_path = None
        self._voices_cache: Optional[List[Voice]] = None
        self._initialization_failed = False
    
    @property
    def provider_id(self) -> str:
        """Unique identifier for this provider."""
        return "piper"
    
    @property
    def display_name(self) -> str:
        """Human-readable display name for this provider."""
        return "Piper TTS"
    
    async def initialize(self) -> None:
        """Initialize the Piper TTS provider."""
        try:
            from piper import PiperVoice
            
            # Try to import onnxruntime to check if it's available
            import onnxruntime
            
            # Try to get model path from config or environment
            self.model_path = self.config.get("model_path") or os.environ.get("PIPER_MODEL_PATH")
            
            if not self.model_path:
                # Look for a default model in common locations
                possible_paths = [
                    os.path.expanduser("~/.local/share/piper/models/en_US-lessac-medium.onnx"),
                    os.path.expanduser("~/.cache/piper/models/en_US-lessac-medium.onnx"),
                    "/usr/share/piper/models/en_US-lessac-medium.onnx",
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        self.model_path = path
                        break
            
            # If no model found, we'll try to download it on first use
            if not self.model_path:
                logger.warning("No Piper model path configured. Models will be downloaded on first use.")
            else:
                logger.info(f"Using Piper model: {self.model_path}")
            
            self._initialization_failed = False
            logger.info("Piper TTS provider initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Piper TTS dependencies not available: {e}")
            logger.warning("Piper TTS provider will be disabled. Install with: pip install piper-tts")
            self._initialization_failed = True
        except Exception as e:
            logger.warning(f"Failed to initialize Piper TTS: {str(e)}")
            logger.warning("Piper TTS provider will be disabled")
            self._initialization_failed = True
    
    async def get_voices(self, language: Optional[str] = None) -> List[Voice]:
        """Get available Piper voices by fetching from voices.json."""
        # Return cached voices if available
        if self._voices_cache:
            voices = self._voices_cache
        else:
            try:
                from piper.download import get_voices as fetch_piper_voices
                
                # Get or create download directory
                download_dir = self.config.get("download_dir") or os.path.expanduser("~/.local/share/piper")
                os.makedirs(download_dir, exist_ok=True)
                
                # Fetch voices from voices.json (embedded or downloaded)
                voices_data = fetch_piper_voices(download_dir, update_voices=False)
                
                voices = []
                for voice_id, voice_info in voices_data.items():
                    # Parse voice information from Piper's voice.json format
                    try:
                        language_code = voice_info.get("language", {}).get("code", "en")
                        language_name = voice_info.get("language", {}).get("name_english", "English")
                        
                        # Determine gender from speaker name or default
                        speaker_name = voice_info.get("name", voice_id)
                        gender = VoiceGender.FEMALE if any(x in speaker_name.lower() for x in ["she", "female", "woman"]) else VoiceGender.MALE
                        
                        # Parse locale from language code
                        locale_parts = language_code.split("_")
                        locale = f"{locale_parts[0]}-{locale_parts[1].upper()}" if len(locale_parts) > 1 else language_code
                        
                        parsed_voice = Voice(
                            id=voice_id,
                            name=speaker_name,
                            language=locale_parts[0],
                            locale=locale,
                            gender=gender,
                            voice_type=VoiceType.NEURAL,
                            provider=self.provider_id,
                            sample_rate=voice_info.get("sample_rate", 22050),
                            description=voice_info.get("description", "Piper neural voice")
                        )
                        voices.append(parsed_voice)
                    except Exception as e:
                        logger.warning(f"Could not parse voice {voice_id}: {e}")
                        continue
                
                # Cache for future calls
                self._voices_cache = voices
                logger.info(f"Loaded {len(voices)} Piper voices")
                
            except Exception as e:
                logger.warning(f"Could not fetch dynamic Piper voices: {e}. Using fallback hardcoded voices.")
                # Fallback to hardcoded voices
                voices = [
                    Voice(
                        id="en_US-lessac-medium",
                        name="Lessac (US English)",
                        language="en",
                        locale="en-US",
                        gender=VoiceGender.NEUTRAL,
                        voice_type=VoiceType.NEURAL,
                        provider=self.provider_id,
                        sample_rate=22050,
                        description="High-quality US English voice"
                    ),
                    Voice(
                        id="en_US-libritts-high",
                        name="LibriTTS (US English)",
                        language="en",
                        locale="en-US",
                        gender=VoiceGender.NEUTRAL,
                        voice_type=VoiceType.NEURAL,
                        provider=self.provider_id,
                        sample_rate=22050,
                        description="LibriTTS-based US English voice"
                    ),
                    Voice(
                        id="en_GB-alan-medium",
                        name="Alan (UK English)",
                        language="en",
                        locale="en-GB",
                        gender=VoiceGender.MALE,
                        voice_type=VoiceType.NEURAL,
                        provider=self.provider_id,
                        sample_rate=22050,
                        description="Male UK English voice"
                    ),
                    Voice(
                        id="ru_RU-irene-medium",
                        name="Irene (Russian)",
                        language="ru",
                        locale="ru-RU",
                        gender=VoiceGender.FEMALE,
                        voice_type=VoiceType.NEURAL,
                        provider=self.provider_id,
                        sample_rate=22050,
                        description="Female Russian voice"
                    ),
                ]
                self._voices_cache = voices
        
        # Filter by language if provided
        if language:
            voices = [v for v in voices if v.language == language or v.locale.startswith(language)]
        
        return voices
    
    async def get_capabilities(self) -> ProviderCapabilities:
        """Get Piper TTS provider capabilities."""
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
        """Synthesize speech using Piper TTS."""
        if self._initialization_failed:
            raise TTSError(
                "Piper TTS provider not initialized. Dependencies may not be available on this platform.",
                error_code="NOT_INITIALIZED",
                provider=self.provider_id
            )
        
        try:
            from piper import PiperVoice
            
            # Validate request
            await self.validate_request(request)
            
            # Get voice
            voice_id = request.voice_id
            model_path = self._get_model_path(voice_id)
            
            if not os.path.exists(model_path):
                raise VoiceNotFoundError(
                    f"Piper model not found: {model_path}. Download it first.",
                    error_code="MODEL_NOT_FOUND",
                    provider=self.provider_id
                )
            
            # Load voice model
            voice = PiperVoice.load(model_path)
            
            # Synthesize audio
            wav_buffer = BytesIO()
            voice.synthesize(request.text, wav_buffer)
            audio_data = wav_buffer.getvalue()
            
            # Apply speed adjustment if needed
            if request.speed != 1.0:
                # Piper doesn't support speed directly, but we can resample
                # For now, we'll just note this limitation
                logger.warning(f"Piper TTS doesn't support speed control. Using default speed (ignoring {request.speed})")
            
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
                voice_used=voice_id,
                metadata={
                    "provider": self.provider_id,
                    "model": voice_id,
                    "quality": "high"
                }
            )
            
        except VoiceNotFoundError:
            raise
        except Exception as e:
            raise TTSError(
                f"Piper TTS synthesis failed: {str(e)}",
                error_code="SYNTHESIS_ERROR",
                provider=self.provider_id
            )
    
    def _get_model_path(self, voice_id: str) -> str:
        """Get the file path for a Piper model."""
        # Map voice IDs to model filenames
        model_mapping = {
            "en_US-lessac-medium": "en_US-lessac-medium.onnx",
            "en_US-libritts-high": "en_US-libritts-high.onnx",
            "en_GB-alan-medium": "en_GB-alan-medium.onnx",
            "ru_RU-irene-medium": "ru_RU-irene-medium.onnx",
        }
        
        model_file = model_mapping.get(voice_id)
        if not model_file:
            model_file = f"{voice_id}.onnx"
        
        # Try configured path first
        if self.model_path and os.path.isdir(self.model_path):
            return os.path.join(self.model_path, model_file)
        
        # Try common locations
        possible_dirs = [
            os.path.expanduser("~/.local/share/piper/models"),
            os.path.expanduser("~/.cache/piper/models"),
            "/usr/share/piper/models",
        ]
        
        for dir_path in possible_dirs:
            full_path = os.path.join(dir_path, model_file)
            if os.path.exists(full_path):
                return full_path
        
        # Return default location
        return os.path.expanduser(f"~/.local/share/piper/models/{model_file}")
