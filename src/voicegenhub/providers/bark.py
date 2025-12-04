"""
Bark TTS Provider

Suno's Bark model - Outstanding naturalness with prosody markers.
"""

import asyncio
import pickle
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


class BarkProvider(TTSProvider):
    """
    Bark TTS Provider - Suno's natural speech synthesis.

    Outstanding naturalness with psychoacoustic optimization.
    """

    def __init__(self, name: str = "bark", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.model = None
        self._voices_cache: Optional[List[Voice]] = None
        self._initialization_failed = False

    @property
    def provider_id(self) -> str:
        return "bark"

    @property
    def display_name(self) -> str:
        return "Bark (Suno)"

    async def initialize(self) -> None:
        """Initialize the Bark TTS provider."""
        try:
            import torch
            import torch.serialization

            # Patch torch.load for PyTorch 2.6+ compatibility with Bark models
            _original_torch_load = torch.load

            def _patched_torch_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return _original_torch_load(*args, **kwargs)

            torch.load = _patched_torch_load

            # Add safe globals for Bark models
            try:
                import numpy
                torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
            except ImportError:
                pass

            from bark import SAMPLE_RATE  # noqa: F401

            self._initialization_failed = False
            logger.info("Bark provider initialized successfully")

        except ImportError as e:
            logger.warning(f"Bark dependencies not available: {e}")
            logger.warning("Install with: pip install bark-model")
            self._initialization_failed = True
        except Exception as e:
            logger.warning(f"Failed to initialize Bark: {str(e)}")
            self._initialization_failed = True

    async def get_voices(self, language: Optional[str] = None) -> List[Voice]:
        """Get available Bark voices."""
        if self._initialization_failed:
            raise TTSError(
                "Bark provider is not available",
                error_code="PROVIDER_UNAVAILABLE",
                provider=self.provider_id,
            )

        if self._voices_cache:
            voices = self._voices_cache
        else:
            # Bark supports multiple speaker presets (en_speaker_0 to en_speaker_9, etc.)
            voices = []

            speaker_presets = ["en_speaker_0", "en_speaker_1", "en_speaker_2", "en_speaker_3", "en_speaker_4"]

            for preset in speaker_presets:
                speaker_num = preset.split("_")[-1]
                voices.append(
                    Voice(
                        id=f"bark-{preset}",
                        name=f"Bark Speaker {speaker_num}",
                        language="en",
                        locale="en-US",
                        gender=VoiceGender.NEUTRAL,
                        voice_type=VoiceType.NEURAL,
                        provider="bark",
                        sample_rate=24000,
                        description=f"Bark natural speaker preset {speaker_num}",
                    )
                )

            self._voices_cache = voices

        if language:
            return [v for v in voices if v.language == language]

        return voices

    async def get_capabilities(self) -> ProviderCapabilities:
        """Get Bark provider capabilities."""
        return ProviderCapabilities(
            supports_ssml=False,
            supports_emotions=False,
            supports_styles=False,
            supports_speed_control=False,
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_streaming=False,
            max_text_length=1000,
            rate_limit_per_minute=5,
            supported_formats=[AudioFormat.WAV],
            supported_sample_rates=[24000],
        )

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Synthesize speech using Bark."""
        if self._initialization_failed:
            raise TTSError(
                "Bark provider is not available",
                error_code="PROVIDER_UNAVAILABLE",
                provider=self.provider_id,
            )

        try:
            await self.validate_request(request)

            # Parse voice ID
            if not request.voice_id.startswith("bark-"):
                raise VoiceNotFoundError(
                    f"Invalid Bark voice ID: {request.voice_id}",
                    error_code="VOICE_NOT_FOUND",
                    provider=self.provider_id,
                )

            speaker_preset = request.voice_id[5:]  # Remove "bark-" prefix

            def _synthesize():
                from bark import generate_audio, preload_models, SAMPLE_RATE
                import numpy as np
                import torch

                # Patch torch.load to handle weights_only issue with newer PyTorch
                try:
                    preload_models()
                except (RuntimeError, pickle.UnpicklingError) as e:
                    if "weights_only" in str(e):
                        # Retry with weights_only=False
                        import torch.serialization
                        original_load = torch.load

                        def patched_load(f, *args, **kwargs):
                            kwargs['weights_only'] = False
                            return original_load(f, *args, **kwargs)

                        torch.load = patched_load
                        preload_models()
                        torch.load = original_load
                    else:
                        raise

                audio = generate_audio(
                    request.text,
                    history_prompt=speaker_preset,
                    text_temp=0.7,
                    waveform_temp=0.8,
                )

                return audio, SAMPLE_RATE

            loop = asyncio.get_event_loop()
            audio_array, sample_rate = await loop.run_in_executor(None, _synthesize)

            # Convert to bytes
            import soundfile as sf
            import io
            import numpy as np

            audio_array = np.array(audio_array)
            wav_io = io.BytesIO()
            sf.write(wav_io, audio_array, sample_rate, format='WAV')
            audio_data = wav_io.getvalue()

            duration = len(request.text) / 150

            return TTSResponse(
                audio_data=audio_data,
                format=AudioFormat.WAV,
                sample_rate=sample_rate,
                duration=duration,
                voice_used=request.voice_id,
                metadata={"provider": "bark", "preset": speaker_preset},
            )

        except VoiceNotFoundError:
            raise
        except Exception as e:
            raise TTSError(
                f"Bark synthesis failed: {str(e)}",
                error_code="SYNTHESIS_ERROR",
                provider=self.provider_id,
            )
