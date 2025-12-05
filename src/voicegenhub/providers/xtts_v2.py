"""
XTTS-v2 TTS Provider

Coqui TTS XTTS-v2 model - State-of-the-art multilingual speech synthesis
with voice cloning capabilities.
"""

import asyncio
import os
import warnings
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

# Suppress jieba pkg_resources deprecation warning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning, module="jieba")
# Suppress XTTS attention mask warning
warnings.filterwarnings("ignore", message="The attention mask is not set and cannot be inferred", category=UserWarning)
warnings.filterwarnings("ignore", message=".*attention mask.*", category=UserWarning)
# Suppress all transformers warnings during TTS operations
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

logger = get_logger(__name__)


class XTTSv2Provider(TTSProvider):
    """
    XTTS-v2 TTS Provider - Coqui TTS multilingual synthesis.

    Supports 16 languages with excellent naturalness and voice cloning.
    """

    def __init__(self, name: str = "xtts_v2", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.model = None
        self._voices_cache: Optional[List[Voice]] = None
        self._initialization_failed = False

        # Set up local cache directory for XTTS models (similar to Kokoro)
        import pathlib
        project_root = pathlib.Path(__file__).parent.parent.parent.parent
        self._local_cache_dir = os.path.join(project_root, 'cache', 'xtts')
        os.environ['HF_HUB_CACHE'] = self._local_cache_dir
        os.environ['TRANSFORMERS_CACHE'] = self._local_cache_dir
        logger.info(f"Configured XTTS cache: {self._local_cache_dir}")

    @property
    def provider_id(self) -> str:
        return "xtts_v2"

    @property
    def display_name(self) -> str:
        return "XTTS-v2 (Coqui)"

    async def initialize(self) -> None:
        """Initialize the XTTS-v2 TTS provider."""
        try:
            import torch
            import torch.serialization

            # Patch torch.load for PyTorch 2.6+ compatibility with XTTS models
            _original_torch_load = torch.load

            def _patched_torch_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return _original_torch_load(*args, **kwargs)

            torch.load = _patched_torch_load

            # Add safe globals for XTTS config
            try:
                from TTS.tts.configs.xtts_config import XttsConfig
                torch.serialization.add_safe_globals([XttsConfig])
            except ImportError:
                pass

            from TTS.api import TTS  # noqa: F401

            # Patch GPT2InferenceModel to inherit from GenerationMixin for transformers v4.50+
            try:
                from transformers import GenerationMixin
                from TTS.tts.layers.xtts.gpt_inference import GPT2InferenceModel

                # Make GPT2InferenceModel inherit from GenerationMixin if it doesn't already
                if not issubclass(GPT2InferenceModel, GenerationMixin):
                    GPT2InferenceModel.__bases__ = (GenerationMixin,) + GPT2InferenceModel.__bases__
            except ImportError:
                pass

            self._initialization_failed = False
            logger.info("XTTS-v2 provider initialized successfully")

        except ImportError as e:
            logger.warning(f"XTTS-v2 dependencies not available: {e}")
            logger.warning("Install with: pip install TTS")
            self._initialization_failed = True
        except Exception as e:
            logger.warning(f"Failed to initialize XTTS-v2: {str(e)}")
            self._initialization_failed = True

    async def get_voices(self, language: Optional[str] = None) -> List[Voice]:
        """Get available XTTS-v2 voices."""
        if self._initialization_failed:
            raise TTSError(
                "XTTS-v2 provider is not available",
                error_code="PROVIDER_UNAVAILABLE",
                provider=self.provider_id,
            )

        if self._voices_cache:
            voices = self._voices_cache
        else:
            # XTTS-v2 supports multiple languages with generic voices
            available_voices = {
                "en": ("en", "en-US", "English (US)"),
                "es": ("es", "es-ES", "Spanish"),
                "fr": ("fr", "fr-FR", "French"),
                "de": ("de", "de-DE", "German"),
                "it": ("it", "it-IT", "Italian"),
                "pt": ("pt", "pt-BR", "Portuguese"),
                "pl": ("pl", "pl-PL", "Polish"),
                "tr": ("tr", "tr-TR", "Turkish"),
                "ru": ("ru", "ru-RU", "Russian"),
                "nl": ("nl", "nl-NL", "Dutch"),
                "cs": ("cs", "cs-CZ", "Czech"),
                "ar": ("ar", "ar-AR", "Arabic"),
                "zh": ("zh", "zh-CN", "Chinese"),
                "ja": ("ja", "ja-JP", "Japanese"),
                "ko": ("ko", "ko-KR", "Korean"),
                "hu": ("hu", "hu-HU", "Hungarian"),
            }

            voices = []
            for voice_code, (lang_code, locale, lang_name) in available_voices.items():
                voices.append(
                    Voice(
                        id=f"xtts_v2-{voice_code}",
                        name=f"XTTS-v2 {lang_name}",
                        language=lang_code,
                        locale=locale,
                        gender=VoiceGender.NEUTRAL,
                        voice_type=VoiceType.NEURAL,
                        provider="xtts_v2",
                        sample_rate=22050,
                        description=f"XTTS-v2 multilingual voice for {lang_name}",
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
        """Get XTTS-v2 provider capabilities."""
        return ProviderCapabilities(
            supports_ssml=False,
            supports_emotions=False,
            supports_styles=False,
            supports_speed_control=True,
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_streaming=False,
            max_text_length=1000,
            rate_limit_per_minute=10,
            supported_formats=[AudioFormat.WAV],
            supported_sample_rates=[22050],
        )

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Synthesize speech using XTTS-v2."""
        if self._initialization_failed:
            raise TTSError(
                "XTTS-v2 provider is not available",
                error_code="PROVIDER_UNAVAILABLE",
                provider=self.provider_id,
            )

        try:
            await self.validate_request(request)

            # Parse voice ID - support both direct language codes and full voice IDs
            voice_id = request.voice_id

            # If it's already in xtts_v2-{code} format, extract the language code
            if voice_id.startswith("xtts_v2-"):
                lang_code = voice_id[8:]  # Remove "xtts_v2-" prefix
            else:
                # Try to find matching voice by name
                available_voices = await self.get_voices()
                matching_voice = next(
                    (v for v in available_voices if v.name == voice_id or v.id == voice_id),
                    None
                )

                if matching_voice:
                    lang_code = matching_voice.language
                else:
                    # Fall back to treating it as a language code
                    lang_code = voice_id.split("-")[0] if "-" in voice_id else voice_id

                    # Validate it's a supported language
                    supported_langs = {
                        "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
                        "nl", "cs", "ar", "zh", "ja", "ko", "hu"
                    }
                    if lang_code not in supported_langs:
                        raise VoiceNotFoundError(
                            f"Invalid XTTS-v2 voice ID: {request.voice_id}. "
                            f"Supported: {', '.join(sorted(supported_langs))}",
                            error_code="VOICE_NOT_FOUND",
                            provider=self.provider_id,
                        )

            def _synthesize():
                import warnings
                from TTS.api import TTS
                import numpy as np

                # Suppress warnings during synthesis
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)

                    # Use the new coqui-tts API
                    tts = TTS(
                        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                        gpu=self.config.get("gpu", False),
                        progress_bar=False,
                    )

                    # Use a default speaker for the language
                    speaker_map = {
                        "en": "Claribel Dervla",
                        "es": "Ana Florence",
                        "fr": "Gracie Wise",
                        "de": "Alison Dietlinde",
                        "it": "Henriette Usha",
                        "pt": "Tammie Ema",
                        "pl": "Gitta Nikolina",
                        "tr": "Tanja Adelina",
                        "ru": "Vjollca Johnnie",
                        "nl": "Royston Min",
                        "cs": "Viktor Eka",
                        "ar": "Abrahan Mack",
                        "zh": "Adde Michal",
                        "ja": "Baldur Sanjin",
                        "ko": "Craig Gutsy",
                        "hu": "Damien Black",
                    }
                    speaker = speaker_map.get(lang_code, "Claribel Dervla")  # Default to English

                    # Generate audio using the high-level API
                    wav = tts.tts(
                        text=request.text,
                        speaker_wav=None,
                        language=lang_code,
                        speaker=speaker,
                    )

                wav = np.array(wav)
                # Ensure it's 1D for mono audio
                if wav.ndim > 1:
                    wav = wav.flatten()

                # Convert to float32 for better compatibility
                wav = wav.astype(np.float32)

                return wav

            loop = asyncio.get_event_loop()
            audio_array = await loop.run_in_executor(None, _synthesize)

            # Convert to bytes
            import soundfile as sf
            import io

            wav_io = io.BytesIO()
            sf.write(wav_io, audio_array, 22050, format='WAV', subtype='PCM_16')
            audio_data = wav_io.getvalue()

            duration = len(request.text) / 150

            return TTSResponse(
                audio_data=audio_data,
                format=AudioFormat.WAV,
                sample_rate=22050,
                duration=duration,
                voice_used=request.voice_id,
                metadata={"provider": "xtts_v2"},
            )

        except VoiceNotFoundError:
            raise
        except Exception as e:
            raise TTSError(
                f"XTTS-v2 synthesis failed: {str(e)}",
                error_code="SYNTHESIS_ERROR",
                provider=self.provider_id,
            )
