"""
Kokoro TTS Provider

Self-hosted Kokoro-82M model from Hexgrad for fast speech synthesis.
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


class KokoroTTSProvider(TTSProvider):
    """
    Kokoro TTS Provider - Fast and lightweight self-hosted TTS.

    Kokoro is a compact yet high-quality neural text-to-speech engine
    that runs locally with minimal computational requirements.
    """

    def __init__(self, name: str = "kokoro", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.model = None
        self.voices_list = None
        self._voices_cache: Optional[List[Voice]] = None
        self._initialization_failed = False

    @property
    def provider_id(self) -> str:
        """Unique identifier for this provider."""
        return "kokoro"

    @property
    def display_name(self) -> str:
        """Human-readable display name for this provider."""
        return "Kokoro TTS"

    async def initialize(self) -> None:
        """Initialize the Kokoro TTS provider."""
        # Set cache directory BEFORE any kokoro imports
        import os
        import warnings

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self._local_cache_dir = os.path.join(project_root, 'cache', 'kokoro')
        os.environ['HF_HUB_CACHE'] = self._local_cache_dir
        os.environ['TRANSFORMERS_CACHE'] = self._local_cache_dir
        logger.info(f"Configured Kokoro cache: {self._local_cache_dir}")

        try:
            # Suppress HuggingFace hub warnings during import
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Defaulting repo_id.*")
                import kokoro  # noqa: F401

            # Models are loaded on-demand via get_voices
            self._initialization_failed = False
            logger.info("Kokoro TTS provider initialized successfully")

        except Exception:
            # Re-raise the exception instead of catching it
            raise

    async def get_voices(self, language: Optional[str] = None) -> List[Voice]:
        """Get available Kokoro voices."""
        if self._initialization_failed:
            raise TTSError(
                "Kokoro TTS provider is not available",
                error_code="PROVIDER_UNAVAILABLE",
                provider=self.provider_id,
            )

        if self._voices_cache:
            voices = self._voices_cache
        else:
            # Load available voices from Kokoro
            try:
                import warnings
                import sys
                from io import StringIO

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*Defaulting repo_id.*")
                    old_stderr = sys.stderr
                    sys.stderr = StringIO()
                    try:
                        import kokoro  # noqa: F401
                    finally:
                        sys.stderr = old_stderr

                # Get available voices - these are the actual voice files from the repo
                available_voice_names = [
                    # American Female
                    "af_alloy",
                    "af_aoede",
                    "af_bella",
                    "af_heart",
                    "af_jessica",
                    "af_kore",
                    "af_nicole",
                    "af_nova",
                    "af_river",
                    "af_sarah",
                    "af_sky",
                    # American Male
                    "am_adam",
                    "am_echo",
                    "am_eric",
                    "am_fenrir",
                    "am_liam",
                    "am_michael",
                    "am_onyx",
                    "am_puck",
                    "am_santa",
                    # British Female
                    "bf_alice",
                    "bf_emma",
                    "bf_isabella",
                    "bf_lily",
                    # British Male
                    "bm_daniel",
                    "bm_fable",
                    "bm_george",
                    "bm_lewis",
                    # Spanish Female
                    "ef_dora",
                    # Spanish Male
                    "em_alex",
                    "em_santa",
                    # French Female
                    "ff_siwis",
                    # Hindi Female
                    "hf_alpha",
                    "hf_beta",
                    # Hindi Male
                    "hm_omega",
                    "hm_psi",
                    # Italian Female
                    "if_sara",
                    # Italian Male
                    "im_nicola",
                    # Japanese Female
                    "jf_alpha",
                    "jf_gongitsune",
                    "jf_nezumi",
                    "jf_tebukuro",
                    # Japanese Male
                    "jm_kumo",
                    # Portuguese Female
                    "pf_dora",
                    # Portuguese Male
                    "pm_alex",
                    "pm_santa",
                    # Mandarin Female
                    "zf_xiaobei",
                    "zf_xiaoni",
                    "zf_xiaoxiao",
                    "zf_xiaoyi",
                    # Mandarin Male
                    "zm_yunjian",
                    "zm_yunxi",
                    "zm_yunxia",
                    "zm_yunyang",
                ]
            except Exception as e:
                logger.warning(f"Could not load Kokoro voices: {e}")
                available_voice_names = [
                    "af",
                    "am",
                    "ar",
                    "bg",
                    "bn",
                    "ca",
                    "cs",
                    "cy",
                    "da",
                    "de",
                    "el",
                    "en",
                    "es",
                    "et",
                    "fa",
                    "fi",
                    "fr",
                    "gu",
                    "he",
                    "hi",
                    "hr",
                    "hu",
                    "id",
                    "it",
                    "ja",
                    "jv",
                    "kn",
                    "ko",
                    "ml",
                    "mr",
                    "nb",
                    "nl",
                    "pa",
                    "pl",
                    "pt",
                    "ro",
                    "ru",
                    "sk",
                    "sl",
                    "so",
                    "sv",
                    "sw",
                    "ta",
                    "te",
                    "tg",
                    "th",
                    "tr",
                    "uk",
                    "ur",
                    "vi",
                    "yo",
                    "zh",
                ]

            voices = []
            for voice_name in available_voice_names:
                # Map voice prefixes to language codes and names
                prefix_map = {
                    "af": ("en", "en-US", "American Female", VoiceGender.FEMALE),
                    "am": ("en", "en-US", "American Male", VoiceGender.MALE),
                    "bf": ("en", "en-GB", "British Female", VoiceGender.FEMALE),
                    "bm": ("en", "en-GB", "British Male", VoiceGender.MALE),
                    "ef": ("es", "es-ES", "Spanish Female", VoiceGender.FEMALE),
                    "em": ("es", "es-ES", "Spanish Male", VoiceGender.MALE),
                    "ff": ("fr", "fr-FR", "French Female", VoiceGender.FEMALE),
                    "hf": ("hi", "hi-IN", "Hindi Female", VoiceGender.FEMALE),
                    "hm": ("hi", "hi-IN", "Hindi Male", VoiceGender.MALE),
                    "if": ("it", "it-IT", "Italian Female", VoiceGender.FEMALE),
                    "im": ("it", "it-IT", "Italian Male", VoiceGender.MALE),
                    "jf": ("ja", "ja-JP", "Japanese Female", VoiceGender.FEMALE),
                    "jm": ("ja", "ja-JP", "Japanese Male", VoiceGender.MALE),
                    "pf": ("pt", "pt-BR", "Portuguese Female", VoiceGender.FEMALE),
                    "pm": ("pt", "pt-BR", "Portuguese Male", VoiceGender.MALE),
                    "zf": ("zh", "zh-CN", "Mandarin Female", VoiceGender.FEMALE),
                    "zm": ("zh", "zh-CN", "Mandarin Male", VoiceGender.MALE),
                }

                prefix = voice_name.split("_")[0]
                lang_code, locale, lang_name, gender = prefix_map.get(
                    prefix,
                    (
                        voice_name[:2],
                        f"{voice_name[:2]}-{voice_name[:2].upper()}",
                        voice_name[:2].upper(),
                        VoiceGender.NEUTRAL,
                    ),
                )

                # Create a readable name from the voice name
                display_name = voice_name.replace("_", " ").title()

                voices.append(
                    Voice(
                        id=f"kokoro-{voice_name}",
                        name=f"Kokoro {display_name}",
                        language=lang_code,
                        locale=locale,
                        gender=gender,
                        voice_type=VoiceType.NEURAL,
                        provider="kokoro",
                        sample_rate=22050,
                        description=f"Kokoro neural voice: {display_name}",
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
        """Get Kokoro provider capabilities."""
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
            supported_sample_rates=[22050],
        )

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Synthesize speech using Kokoro."""
        import time

        if self._initialization_failed:
            raise TTSError(
                "Kokoro TTS provider is not available",
                error_code="PROVIDER_UNAVAILABLE",
                provider=self.provider_id,
            )

        try:
            await self.validate_request(request)

            # Parse voice ID - extract the actual voice name (e.g., "af_bella" from "kokoro-af_bella")
            if not request.voice_id.startswith("kokoro-"):
                raise VoiceNotFoundError(
                    f"Invalid Kokoro voice ID: {request.voice_id}",
                    error_code="VOICE_NOT_FOUND",
                    provider=self.provider_id,
                )

            voice_name = request.voice_id[7:]  # Remove "kokoro-" prefix

            # Get language code from voice prefix for pipeline
            prefix = voice_name.split("_")[0]
            lang_code_map = {
                "af": "a",
                "am": "a",  # American English
                "bf": "b",
                "bm": "b",  # British English
                "ef": "e",
                "em": "e",  # Spanish
                "ff": "f",  # French
                "hf": "h",
                "hm": "h",  # Hindi
                "if": "i",
                "im": "i",  # Italian
                "jf": "j",
                "jm": "j",  # Japanese
                "pf": "p",
                "pm": "p",  # Portuguese
                "zf": "z",
                "zm": "z",  # Mandarin
            }

            kokoro_lang = lang_code_map.get(prefix, "a")  # Default to American English

            # Cache directory should already be set in initialize()
            if not hasattr(self, '_local_cache_dir'):
                raise TTSError(
                    "Kokoro provider not properly initialized - _local_cache_dir not set",
                    error_code="PROVIDER_NOT_INITIALIZED",
                    provider=self.provider_id,
                )
            logger.info(f"Using local Kokoro cache: {self._local_cache_dir}")

            # Import kokoro (cache should already be configured)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Defaulting repo_id.*")
                import kokoro

            setup_start = time.perf_counter()
            if self.model is None:
                device = self.config.get("device", "cpu")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*Defaulting repo_id.*")
                    self.model = kokoro.KPipeline(lang_code=kokoro_lang, device=device)
                logger.info(f"Loaded Kokoro pipeline for {kokoro_lang} ({prefix})")
            setup_end = time.perf_counter()

            # Run synthesis in thread pool to avoid blocking
            def _synthesize():
                import tempfile

                inference_start = time.perf_counter()
                output_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        output_path = f.name

                    # Kokoro uses KPipeline
                    results = list(
                        self.model(request.text, voice=voice_name, speed=request.speed)
                    )

                    if not results:
                        raise Exception("No audio generated")

                    # Get the first result's audio
                    audio_samples = results[0].audio

                    # Save to WAV - Kokoro outputs at 22050 Hz
                    import soundfile as sf

                    sf.write(output_path, audio_samples, 22050)

                    with open(output_path, "rb") as f:
                        audio_data = f.read()

                    inference_end = time.perf_counter()
                    logger.debug(f"Kokoro inference time: {inference_end - inference_start:.3f}s")

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

            logger.info(f"Kokoro setup time: {setup_end - setup_start:.3f}s")

            # Calculate duration (approximately)
            duration = len(request.text) / 150  # rough estimate

            return TTSResponse(
                audio_data=audio_data,
                format=AudioFormat.WAV,
                sample_rate=22050,
                duration=duration,
                voice_used=request.voice_id,
                metadata={"provider": "kokoro"},
            )

        except VoiceNotFoundError:
            raise
        except Exception as e:
            raise TTSError(
                f"Kokoro synthesis failed: {str(e)}",
                error_code="SYNTHESIS_ERROR",
                provider=self.provider_id,
            )
