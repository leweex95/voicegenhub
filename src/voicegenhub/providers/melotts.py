"""
MeloTTS Provider

Self-hosted MeloTTS model from MyShell AI for high-quality speech synthesis.
Models are loaded on-demand to avoid bloating the package size.
"""

import asyncio
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger
from .base import (
    AudioFormat,
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


class MeloTTSProvider(TTSProvider):
    """
    MeloTTS Provider - High-quality self-hosted neural TTS.

    MeloTTS is a fast and expressive neural text-to-speech engine
    that runs locally after downloading voice models.
    """

    def __init__(self, name: str = "melotts", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.model = None
        self._voices_cache: Optional[List[Voice]] = None
        self._initialization_failed = False

    @property
    def provider_id(self) -> str:
        """Unique identifier for this provider."""
        return "melotts"

    @property
    def display_name(self) -> str:
        """Human-readable display name for this provider."""
        return "MeloTTS"

    async def initialize(self) -> None:
        """Initialize the MeloTTS provider."""
        try:
            from melo.api import TTS as MeloTTS  # noqa: F401

            # Models are loaded on-demand via get_voices
            self._initialization_failed = False
            logger.info("MeloTTS provider initialized successfully")

        except ImportError as e:
            logger.warning(f"MeloTTS dependencies not available: {e}")
            logger.warning(
                "MeloTTS provider will be disabled. Install with: pip install voicegenhub[melotts]"
            )
            self._initialization_failed = True
        except Exception as e:
            logger.warning(f"Failed to initialize MeloTTS: {str(e)}")
            logger.warning("MeloTTS provider will be disabled")
            self._initialization_failed = True

    async def get_voices(self, language: Optional[str] = None) -> List[Voice]:
        """Get available MeloTTS voices."""
        if self._initialization_failed:
            raise TTSError(
                "MeloTTS provider is not available",
                error_code="PROVIDER_UNAVAILABLE",
                provider=self.provider_id,
            )

        if self._voices_cache:
            voices = self._voices_cache
        else:
            # MeloTTS supports multiple languages and English variants
            # Define available voices with their language configurations
            available_voices = {
                "EN-US": ("en", "en-US", "English (US)"),
                "EN-BR": ("en", "en-GB", "English (British)"),
                "EN-AU": ("en", "en-AU", "English (Australian)"),
                "EN-IN": ("en", "en-IN", "English (Indian)"),
                "ES": ("es", "es-ES", "Spanish"),
                "FR": ("fr", "fr-FR", "French"),
                "ZH": ("zh", "zh-CN", "Chinese"),
                "JP": ("ja", "ja-JP", "Japanese"),
                "KO": ("ko", "ko-KR", "Korean"),
            }

            voices = []
            for voice_code, (lang_code, locale, lang_name) in available_voices.items():
                voices.append(
                    Voice(
                        id=f"melotts-{voice_code}",
                        name=f"MeloTTS {lang_name}",
                        language=lang_code,
                        locale=locale,
                        gender=VoiceGender.NEUTRAL,
                        voice_type=VoiceType.NEURAL,
                        provider="melotts",
                        sample_rate=44100,
                        description=f"MeloTTS neural voice for {lang_name}",
                    )
                )

            self._voices_cache = voices

        if language:
            return [
                v
                for v in voices
                if v.language == language or v.locale.startswith(language)
            ]

        return voices

    async def get_capabilities(self) -> ProviderCapabilities:
        """Get MeloTTS provider capabilities."""
        return ProviderCapabilities(
            supports_ssml=False,
            supports_emotions=False,
            supports_styles=False,
            supports_speed_control=True,
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_streaming=False,
            max_text_length=1000,
            rate_limit_per_minute=60,
            supported_formats=[AudioFormat.WAV],
            supported_sample_rates=[44100],
        )

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Synthesize speech using MeloTTS."""
        if self._initialization_failed:
            raise TTSError(
                "MeloTTS provider is not available",
                error_code="PROVIDER_UNAVAILABLE",
                provider=self.provider_id,
            )

        try:
            await self.validate_request(request)

            # Parse voice ID
            if not request.voice_id.startswith("melotts-"):
                raise VoiceNotFoundError(
                    f"Invalid MeloTTS voice ID: {request.voice_id}",
                    error_code="VOICE_NOT_FOUND",
                    provider=self.provider_id,
                )

            # Parse voice ID - extract speaker name
            voice_parts = request.voice_id.split("-", 1)  # Split on first dash only
            if len(voice_parts) < 2:
                raise VoiceNotFoundError(
                    f"Invalid MeloTTS voice ID format: {request.voice_id}",
                    error_code="VOICE_NOT_FOUND",
                    provider=self.provider_id,
                )

            speaker_name = voice_parts[1].upper()  # e.g., "EN-US", "EN-BR", "ES"

            # Lazy load model
            from melo.api import TTS as MeloTTS

            if self.model is None:
                # Extract language from speaker name (EN for English variants, ES for Spanish, etc.)
                language_code = (
                    speaker_name.split("-")[0] if "-" in speaker_name else speaker_name
                )
                device = self.config.get("device", "cpu")
                self.model = MeloTTS(language=language_code, device=device)
                logger.info(f"Loaded MeloTTS model for {language_code}")

            # Synthesize
            speaker_ids = self.model.hps.data.spk2id

            # Map voice codes to MeloTTS speaker names
            speaker_mapping = {
                "EN-US": "EN-US",
                "EN-BR": "EN-BR",
                "EN-AU": "EN-AU",
                "EN-IN": "EN_INDIA",
                "ES": "ES",
                "FR": "FR",
                "ZH": "ZH",
                "JP": "JP",
                "KO": "KR",
            }

            # Get the actual speaker name from mapping
            actual_speaker = speaker_mapping.get(speaker_name, speaker_name)
            if actual_speaker not in speaker_ids:
                available_speakers = list(speaker_ids.keys())
                raise VoiceNotFoundError(
                    f"MeloTTS speaker '{actual_speaker}' not found. Available: {available_speakers}",
                    error_code="VOICE_NOT_FOUND",
                    provider=self.provider_id,
                )

            speaker_id = speaker_ids[actual_speaker]

            # Run synthesis in thread pool to avoid blocking
            def _synthesize():
                output_path = None
                try:
                    import tempfile

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        output_path = f.name

                    self.model.tts_to_file(
                        request.text,
                        speaker_id,
                        output_path,
                        speed=request.speed,
                    )

                    with open(output_path, "rb") as f:
                        audio_data = f.read()

                    return audio_data
                finally:
                    if output_path:
                        import os

                        try:
                            os.unlink(output_path)
                        except Exception:
                            pass

            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(None, _synthesize)

            # Calculate duration (approximately)
            duration = len(request.text) / 150  # rough estimate

            return TTSResponse(
                audio_data=audio_data,
                format=AudioFormat.WAV,
                sample_rate=44100,
                duration=duration,
                voice_used=request.voice_id,
                metadata={"provider": "melotts"},
            )

        except VoiceNotFoundError:
            raise
        except Exception as e:
            raise TTSError(
                f"MeloTTS synthesis failed: {str(e)}",
                error_code="SYNTHESIS_ERROR",
                provider=self.provider_id,
            )
