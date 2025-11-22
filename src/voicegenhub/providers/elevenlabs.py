"""
ElevenLabs TTS Provider

High-quality TTS provider using ElevenLabs Text-to-Speech API.
Supports multiple voices and languages with natural-sounding speech.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from elevenlabs.client import ElevenLabs

    ELEVENLABS_AVAILABLE = True
except ImportError:
    ElevenLabs = None
    ELEVENLABS_AVAILABLE = False

from ..utils.logger import get_logger
from .base import (
    AudioFormat,
    AuthenticationError,
    ProviderCapabilities,
    TTSError,
    TTSProvider,
    TTSRequest,
    TTSResponse,
    Voice,
    VoiceGender,
    VoiceNotFoundError,
    VoiceType,
)

logger = get_logger(__name__)


class ElevenLabsTTSProvider(TTSProvider):
    """
    ElevenLabs TTS provider implementation.

    Uses the ElevenLabs API to provide high-quality text-to-speech synthesis
    with natural-sounding voices and multilingual support.
    """

    def __init__(self, name: str = "elevenlabs", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self._client: Optional[Any] = None
        self._voices_cache: Optional[List[Voice]] = None
        self._initialization_failed = False

        if not ELEVENLABS_AVAILABLE:
            logger.warning(
                "ElevenLabs not available. Install with: pip install elevenlabs"
            )
            self._initialization_failed = True

    @property
    def provider_id(self) -> str:
        return "elevenlabs"

    @property
    def display_name(self) -> str:
        return "ElevenLabs TTS"

    async def initialize(self) -> None:
        """Initialize the ElevenLabs TTS provider."""
        if self._initialization_failed:
            raise TTSError(
                "ElevenLabs TTS provider is not available",
                error_code="PROVIDER_UNAVAILABLE",
                provider=self.provider_id,
            )

        try:
            # Get API key from environment or config file
            api_key = os.getenv("ELEVENLABS_API_KEY")

            if not api_key:
                # Try loading from config file
                config_path = (
                    Path(__file__).parent.parent.parent.parent
                    / "config"
                    / "elevenlabs-api-key.json"
                )
                if config_path.exists():
                    try:
                        with open(config_path, "r") as f:
                            config_data = json.load(f)
                            api_key = config_data.get("ELEVENLABS_API_KEY")
                    except Exception as e:
                        logger.debug(f"Failed to load config from {config_path}: {e}")

            if not api_key:
                logger.warning(
                    "ELEVENLABS_API_KEY not found in environment or config file. Provider will be unavailable."
                )
                self._initialization_failed = True
                raise AuthenticationError(
                    "ElevenLabs API key not provided",
                    error_code="MISSING_API_KEY",
                    provider=self.provider_id,
                )

            # Initialize client
            self._client = ElevenLabs(api_key=api_key)

            # Test connection by listing voices
            await self._test_connection()

            logger.info("ElevenLabs TTS provider initialized successfully")

        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ElevenLabs TTS provider: {e}")
            self._initialization_failed = True
            raise TTSError(
                f"ElevenLabs initialization failed: {str(e)}",
                error_code="INITIALIZATION_FAILED",
                provider=self.provider_id,
            )

    async def _test_connection(self) -> None:
        """Test connection to ElevenLabs API by fetching voices."""
        try:
            if self._client is None:
                raise TTSError(
                    "Client not initialized",
                    error_code="CLIENT_ERROR",
                    provider=self.provider_id,
                )

            # Test by fetching available voices
            try:
                voices = self._client.voices.get_all()
                if not voices:
                    logger.warning("No voices returned from ElevenLabs API")
            except Exception as e:
                # If voices_read permission is missing, we can still try synthesis
                if (
                    "missing_permissions" in str(e).lower()
                    and "voices_read" in str(e).lower()
                ):
                    logger.warning(f"Limited ElevenLabs permissions: {e}")
                else:
                    raise

        except Exception as e:
            raise TTSError(
                f"Failed to connect to ElevenLabs API: {str(e)}",
                error_code="CONNECTION_FAILED",
                provider=self.provider_id,
            )

    async def get_voices(self, language: Optional[str] = None) -> List[Voice]:
        """Get available ElevenLabs voices."""
        if self._initialization_failed:
            raise TTSError(
                "ElevenLabs provider is not available",
                error_code="PROVIDER_UNAVAILABLE",
                provider=self.provider_id,
            )

        if self._voices_cache and not language:
            return self._voices_cache

        try:
            if self._client is None:
                raise TTSError(
                    "Client not initialized",
                    error_code="CLIENT_ERROR",
                    provider=self.provider_id,
                )

            voices_data = self._client.voices.get_all()
            voices = []

            for voice in voices_data:
                # Map gender if available
                gender = VoiceGender.NEUTRAL
                if hasattr(voice, "labels") and voice.labels:
                    labels_str = str(voice.labels).lower()
                    if "female" in labels_str:
                        gender = VoiceGender.FEMALE
                    elif "male" in labels_str:
                        gender = VoiceGender.MALE

                # Extract language from labels
                voice_language = "en"
                voice_locale = "en-US"
                if hasattr(voice, "labels") and voice.labels:
                    labels_str = str(voice.labels).lower()
                    if "british" in labels_str:
                        voice_locale = "en-GB"
                    elif "american" in labels_str:
                        voice_locale = "en-US"
                    elif "australian" in labels_str:
                        voice_locale = "en-AU"

                voices.append(
                    Voice(
                        id=f"elevenlabs-{voice.voice_id}",
                        name=voice.name,
                        language=voice_language,
                        locale=voice_locale,
                        gender=gender,
                        voice_type=VoiceType.NEURAL,
                        provider="elevenlabs",
                        sample_rate=24000,
                        description=f"ElevenLabs voice: {voice.name}",
                    )
                )

            if not language:
                self._voices_cache = voices
                return voices

            # Filter by language
            return [
                v
                for v in voices
                if v.language == language or v.locale.startswith(language)
            ]

        except TTSError:
            raise
        except Exception as e:
            raise TTSError(
                f"Failed to get voices: {str(e)}",
                error_code="VOICE_LIST_ERROR",
                provider=self.provider_id,
            )

    async def get_capabilities(self) -> ProviderCapabilities:
        """Get ElevenLabs provider capabilities."""
        return ProviderCapabilities(
            supports_ssml=False,
            supports_emotions=False,
            supports_styles=False,
            supports_speed_control=False,
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_streaming=True,
            max_text_length=5000,
            rate_limit_per_minute=60,
            supported_formats=[AudioFormat.MP3],
            supported_sample_rates=[24000],
        )

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Synthesize speech using ElevenLabs."""
        if self._initialization_failed:
            raise TTSError(
                "ElevenLabs provider is not available",
                error_code="PROVIDER_UNAVAILABLE",
                provider=self.provider_id,
            )

        try:
            await self.validate_request(request)

            if self._client is None:
                raise TTSError(
                    "Client not initialized",
                    error_code="CLIENT_ERROR",
                    provider=self.provider_id,
                )

            # Parse voice ID - extract the actual voice ID (e.g., "EXAVITQu4vr4xnSDxMaL" from "elevenlabs-EXAVITQu4vr4xnSDxMaL")
            if not request.voice_id.startswith("elevenlabs-"):
                raise VoiceNotFoundError(
                    f"Invalid ElevenLabs voice ID: {request.voice_id}",
                    error_code="VOICE_NOT_FOUND",
                    provider=self.provider_id,
                )

            voice_id = request.voice_id[11:]  # Remove "elevenlabs-" prefix

            # Generate audio
            audio_data = self._client.text_to_speech.convert(
                voice_id=voice_id,
                text=request.text,
                model_id="eleven_multilingual_v2",
                output_format="mp3_24000_48",
            )

            # Collect audio bytes
            audio_bytes = b"".join(audio_data)

            if not audio_bytes:
                raise TTSError(
                    "No audio data generated",
                    error_code="SYNTHESIS_ERROR",
                    provider=self.provider_id,
                )

            # Calculate duration (approximately)
            duration = len(request.text) / 150  # rough estimate

            return TTSResponse(
                audio_data=audio_bytes,
                format=AudioFormat.MP3,
                sample_rate=24000,
                duration=duration,
                voice_used=request.voice_id,
                metadata={"provider": "elevenlabs", "model": "eleven_monolingual_v1"},
            )

        except VoiceNotFoundError:
            raise
        except Exception as e:
            raise TTSError(
                f"ElevenLabs synthesis failed: {str(e)}",
                error_code="SYNTHESIS_ERROR",
                provider=self.provider_id,
            )
