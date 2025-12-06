"""Chatterbox TTS Provider - MIT Licensed, Multilingual, Emotion Control."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger
from .base import (
    ProviderCapabilities,
    TTSError,
    TTSProvider,
    TTSRequest,
    TTSResponse,
    Voice,
    VoiceGender,
    VoiceType,
)

# Suppress deprecated pkg_resources warning from perth watermarking
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning, module="perth.*")

logger = get_logger(__name__)


def _patch_cuda_on_cpu():
    """Monkey-patch torch.cuda and Chatterbox CUDA graph utilities to work on CPU."""
    try:
        import torch

        # Only patch if we're on CPU
        if not torch.cuda.is_available():
            # Patch CUDAGraph
            class CPUCUDAGraphCompat:
                """Compatibility wrapper for CUDA graph on CPU."""
                def __init__(self, *args, **kwargs):
                    pass

                def capture_begin(self, *args, **kwargs):
                    pass

                def capture_end(self, *args, **kwargs):
                    pass

                def replay(self, *args, **kwargs):
                    pass

                def reset(self, *args, **kwargs):
                    pass

                def __enter__(self, *args, **kwargs):
                    return self

                def __exit__(self, *args, **kwargs):
                    pass

            # Replace CUDAGraph with CPU-compatible version
            torch.cuda.CUDAGraph = CPUCUDAGraphCompat
            logger.info("CUDA graph compatibility patched for CPU mode")

            # Patch other CUDA-related stream classes
            try:
                class CPUStreamCompat:
                    """Compatibility wrapper for CUDA Stream on CPU."""
                    def __init__(self, device=None, priority=0, **kwargs):
                        self.device = device or 'cpu'
                        self.priority = priority

                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        pass

                    def synchronize(self):
                        pass

                    def wait_event(self, event):
                        pass

                    def record_event(self, event=None):
                        return event

                torch.cuda.Stream = CPUStreamCompat
                logger.info("CUDA Stream compatibility patched for CPU mode")
            except Exception:
                pass

            # Patch Chatterbox's CUDA graph wrapper
            try:
                from chatterbox.models.t3 import t3_cuda_graphs

                class CPUCUDAGraphWrapper:
                    """CPU-compatible wrapper for T3 CUDA graph sampler."""
                    def __init__(self, *args, **kwargs):
                        self.graphs = {}

                    def __call__(self, *args, **kwargs):
                        # Just run normally without graph capture
                        if 'model' in kwargs:
                            return kwargs['model'](*args, **{k: v for k, v in kwargs.items() if k != 'model'})
                        return None

                    def __getattr__(self, name):
                        return lambda *args, **kwargs: None

                t3_cuda_graphs.T3StepCUDAGraphWrapper = CPUCUDAGraphWrapper
                logger.info("Chatterbox T3 CUDA graph wrapper patched for CPU mode")
            except Exception as e:
                logger.debug(f"Could not patch Chatterbox CUDA wrapper: {e}")

            # Patch torch._C._cuda to prevent CUDA initialization
            try:
                if hasattr(torch, '_C'):
                    torch._C._cuda_init = lambda: None
                    torch._C._is_available = lambda: False
            except Exception:
                pass

    except Exception as e:
        logger.debug(f"Could not patch CUDA compatibility: {e}")


class ChatterboxProvider(TTSProvider):
    """Chatterbox TTS Provider - Resemble AI's production-grade multilingual TTS.

    Features:
    - 23 languages supported
    - Emotion/exaggeration control (0.0-1.0 intensity)
    - Zero-shot voice cloning
    - State-of-the-art quality
    - MIT License - fully commercial compatible
    """

    def __init__(self, name: str = "chatterbox", config: Dict[str, Any] = None):
        """Initialize Chatterbox provider."""
        super().__init__(name, config)
        self._model = None
        self._multilingual_model = None
        self.device = "cpu"
        self._voices_cache = []

    @property
    def provider_id(self) -> str:
        """Return provider identifier."""
        return "chatterbox"

    @property
    def provider_name(self) -> str:
        """Return human-readable provider name."""
        return "Chatterbox TTS (Resemble AI)"

    @property
    def display_name(self) -> str:
        """Return display name for provider."""
        return "Chatterbox TTS (MIT Licensed)"

    async def initialize(self):
        """Initialize the Chatterbox TTS provider."""
        try:
            logger.info("Initializing Chatterbox TTS provider...")

            # Disable CUDA graph capture for CPU inference
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            os.environ['CHATTERBOX_NO_GRAPH'] = '1'

            # Patch CUDA graph compatibility for CPU mode BEFORE importing chatterbox
            _patch_cuda_on_cpu()

            # Set cache directories for local storage
            project_root = Path(__file__).parent.parent.parent.parent
            cache_root = project_root / 'cache'
            cache_chatterbox = cache_root / 'chatterbox'
            cache_chatterbox.mkdir(parents=True, exist_ok=True)

            os.environ['HF_HUB_CACHE'] = str(cache_root / 'huggingface')
            os.environ['TORCH_HOME'] = str(cache_root / 'torch')

            logger.info(f"Cache directory: {cache_chatterbox}")

            # Suppress deprecated pkg_resources warning from perth watermarking
            warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

            # Import here after env vars are set
            from chatterbox.tts import ChatterboxTTS

            # Detect device
            self.device = self._detect_device()
            logger.info(f"Using device: {self.device}")

            # Set torch map location for loading models onto correct device
            if self.device == "cpu":
                import torch
                torch.set_default_device("cpu")

            # Load English model only (multilingual has device mapping issues)
            logger.info("Loading English Chatterbox model...")
            self._model = ChatterboxTTS.from_pretrained(device=self.device)
            logger.info("English model loaded successfully")

            # Multilingual model disabled on CPU - use English only or handle manually
            self._multilingual_model = None
            logger.info("Multilingual support disabled on CPU (use voice cloning instead)")

            logger.info("Chatterbox provider initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Chatterbox: {str(e)}")
            raise TTSError(f"Chatterbox initialization failed: {str(e)}")

    def is_voice_id_valid(self, voice_id: str, language: Optional[str] = None) -> bool:
        """Validate voice ID format."""
        # Chatterbox doesn't use predefined voice IDs; uses audio prompts
        # Valid formats: "chatterbox-default", "chatterbox-<language>"
        return voice_id.startswith("chatterbox-")

    def get_supported_languages(self) -> list:
        """Return list of supported languages."""
        return [
            "en", "ar", "da", "de", "el", "es", "fi", "fr", "he", "hi", "it",
            "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"
        ]

    async def _synthesize(
        self,
        text: str,
        voice_id: str = "chatterbox-default",
        language: str = "en",
        **kwargs
    ) -> tuple[bytes, int]:
        """Synthesize speech using Chatterbox.

        Args:
            text: Text to synthesize
            voice_id: Voice ID (format: "chatterbox-<language>" or "chatterbox-default")
            language: Language code (en, es, fr, zh, etc.)

        Keyword Args:
            exaggeration: Emotion intensity (0.0-1.0, default 0.5)
            cfg_weight: Classifier-free guidance weight (0.0-1.0, default 0.5)
            audio_prompt_path: Path to audio file for voice cloning (optional)

        Returns:
            Tuple of (audio_bytes, sample_rate)
        """
        import torch

        try:
            exaggeration = kwargs.get("exaggeration", 0.5)
            cfg_weight = kwargs.get("cfg_weight", 0.5)
            audio_prompt_path = kwargs.get("audio_prompt_path", None)

            logger.info(f"Generating {language} audio with text: {text[:50]}...")

            # Validate parameters
            exaggeration = max(0.0, min(1.0, exaggeration))
            cfg_weight = max(0.0, min(1.0, cfg_weight))

            # Disable CUDA graph capture on CPU
            if self.device == "cpu":
                # Use no_grad and disable graph capture
                torch.cuda.is_available = lambda: False  # Fake CUDA availability check

            # Select model based on language
            if language.lower() == "en" or self._multilingual_model is None:
                model = self._model
                logger.info(f"Using English model with exaggeration={exaggeration}, cfg_weight={cfg_weight}")

                with torch.no_grad():
                    t3_params = {"generate_token_backend": "eager"} if self.device == "cpu" else {}
                    wav = model.generate(
                        text,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        audio_prompt_path=audio_prompt_path,
                        t3_params=t3_params
                    )
            else:
                if self._multilingual_model is None:
                    logger.warning("Multilingual support not available on CPU, falling back to English")
                    model = self._model
                    with torch.no_grad():
                        t3_params = {"generate_token_backend": "eager"} if self.device == "cpu" else {}
                        wav = model.generate(
                            text,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                            audio_prompt_path=audio_prompt_path,
                            t3_params=t3_params
                        )
                else:
                    model = self._multilingual_model
                    language_id = language.lower()
                    logger.info(f"Using Multilingual model for {language_id}")

                    with torch.no_grad():
                        t3_params = {"generate_token_backend": "eager"} if self.device == "cpu" else {}
                        wav = model.generate(
                            text,
                            language_id=language_id,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                            audio_prompt_path=audio_prompt_path,
                            t3_params=t3_params
                        )

            # Convert to bytes
            sample_rate = model.sr

            # Ensure wav is 1D or 2D
            if wav.dim() == 3:
                wav = wav.squeeze(0)  # Remove batch dimension if present
            if wav.dim() == 2:
                wav = wav.squeeze(0) if wav.shape[0] == 1 else wav[0]  # Take first channel if stereo/multi

            logger.info(f"Audio tensor shape after processing: {wav.shape}")

            # Normalize audio
            if wav.abs().max() > 1.0:
                wav = wav / wav.abs().max()

            # Convert to 16-bit PCM
            wav_int16 = (wav * 32767).short().cpu().numpy() if wav.is_cuda else (wav * 32767).short().numpy()

            # Create WAV file manually to avoid torchaudio issues on CPU
            import struct
            import io

            buffer = io.BytesIO()

            # WAV header
            n_channels = 1
            n_samples = len(wav_int16)
            byte_rate = sample_rate * n_channels * 2
            block_align = n_channels * 2

            # RIFF header
            buffer.write(b'RIFF')
            buffer.write(struct.pack('<I', 36 + n_samples * n_channels * 2))
            buffer.write(b'WAVE')

            # fmt subchunk
            buffer.write(b'fmt ')
            buffer.write(struct.pack('<I', 16))  # Subchunk1Size
            buffer.write(struct.pack('<H', 1))   # AudioFormat (1=PCM)
            buffer.write(struct.pack('<H', n_channels))
            buffer.write(struct.pack('<I', sample_rate))
            buffer.write(struct.pack('<I', byte_rate))
            buffer.write(struct.pack('<H', block_align))
            buffer.write(struct.pack('<H', 16))  # BitsPerSample

            # data subchunk
            buffer.write(b'data')
            buffer.write(struct.pack('<I', n_samples * n_channels * 2))
            buffer.write(wav_int16.tobytes())

            audio_bytes = buffer.getvalue()
            logger.info(f"Successfully generated {len(audio_bytes)} bytes of audio")
            return audio_bytes, sample_rate

        except Exception as e:
            logger.error(f"Audio generation failed: {str(e)}")
            raise TTSError(f"Chatterbox synthesis failed: {str(e)}")

    def _detect_device(self) -> str:
        """Detect available device (cuda or cpu)."""
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("CUDA device detected")
                return "cuda"
        except ImportError:
            pass
        logger.info("Using CPU for inference")
        return "cpu"

    async def get_voices(self, language: Optional[str] = None) -> List[Voice]:
        """Get available Chatterbox voices."""
        if not self._voices_cache:
            # Chatterbox uses default voice for all languages
            self._voices_cache = [
                Voice(
                    id="chatterbox-default",
                    name="Chatterbox Default",
                    language="en",
                    locale="en-US",
                    gender=VoiceGender.NEUTRAL,
                    voice_type=VoiceType.NEURAL,
                    provider="chatterbox",
                    sample_rate=24000,
                    description="Chatterbox default multilingual voice with emotion control",
                )
            ]

        if language:
            return [v for v in self._voices_cache if v.language.startswith(language)]

        return self._voices_cache

    async def get_capabilities(self) -> ProviderCapabilities:
        """Get Chatterbox provider capabilities."""
        return ProviderCapabilities(
            supports_ssml=False,
            supports_emotions=True,  # Exaggeration/intensity control
            supports_styles=False,
            supports_speed_control=False,
            supports_pitch_control=False,
            supports_voice_cloning=True,  # Zero-shot voice cloning with audio prompts
            supported_languages=self.get_supported_languages(),
            max_text_length=1000,
        )

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Synthesize speech using Chatterbox."""
        try:
            # Extract parameters from request
            language = request.language or "en"
            exaggeration = 0.5  # Default exaggeration
            cfg_weight = 0.5    # Default classifier-free guidance weight

            # Call _synthesize
            audio_bytes, sample_rate = await self._synthesize(
                request.text,
                voice_id=request.voice_id,
                language=language,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )

            # Calculate duration (WAV files have header, so we estimate from PCM data)
            # WAV header is typically 44 bytes, then PCM data follows
            estimated_pcm_samples = len(audio_bytes) // 2  # 16-bit samples
            duration = estimated_pcm_samples / sample_rate

            return TTSResponse(
                audio_data=audio_bytes,
                format=request.audio_format or "wav",
                sample_rate=sample_rate,
                duration=duration,
                voice_used=request.voice_id or "chatterbox-default",
            )
        except Exception as e:
            raise TTSError(f"Chatterbox synthesis error: {str(e)}")
