"""Chatterbox TTS Provider - MIT Licensed, Multilingual, Emotion Control."""

import os
import warnings
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

# Set attention implementation to eager before any imports to prevent SDPA warnings
os.environ['TRANSFORMERS_ATTENTION_IMPLEMENTATION'] = 'eager'

# Suppress warnings from dependencies to keep output clean
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message=".*LoRACompatibleLinear.*deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*torch\.backends\.cuda\.sdp_kernel.*deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*LlamaModel is using LlamaSdpaAttention.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*past_key_values.*deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*scaled_dot_product_attention.*", category=UserWarning)

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

            # Patch torch.load to handle CUDA-saved models on CPU
            original_load = torch.load

            def patched_load(*args, **kwargs):
                if 'map_location' not in kwargs:
                    kwargs['map_location'] = 'cpu'
                return original_load(*args, **kwargs)
            torch.load = patched_load
            logger.info("torch.load patched for CPU mode")

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

            # Patch S3Tokenizer to avoid Double/Float mismatch on CPU
            try:
                from chatterbox.models.s3tokenizer.s3tokenizer import S3Tokenizer
                original_log_mel = S3Tokenizer.log_mel_spectrogram

                def patched_log_mel(self, wav):
                    # Ensure wav is float32 for compatibility with S3Tokenizer
                    import torch
                    if wav.dtype == torch.float64:
                        wav = wav.to(torch.float32)

                    # Ensure mel filters match the input type
                    if hasattr(self, '_mel_filters'):
                        self._mel_filters = self._mel_filters.to(wav.dtype)
                    return original_log_mel(self, wav)

                S3Tokenizer.log_mel_spectrogram = patched_log_mel
                logger.info("S3Tokenizer patched for CPU float32 compatibility")
            except Exception as e:
                logger.debug(f"Could not patch S3Tokenizer: {e}")

            # Patch ChatterboxTurboTTS to ensure float32 audio for cloning
            try:
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                import torch

                original_prepare = ChatterboxTurboTTS.prepare_conditionals

                def patched_prepare(self, audio_prompt_path, **kwargs):
                    # This is a bit tricky because we need to ensure the loaded wav is float32
                    # We can't easily intercept the load inside prepare_conditionals without patching torchaudio.load
                    # but we can patch the method and then ensure the model's internal tensors are float32
                    return original_prepare(self, audio_prompt_path, **kwargs)

                # Actually, patching the VoiceEncoder might be more effective
                from chatterbox.models.voice_encoder.voice_encoder import VoiceEncoder
                original_embeds_from_wavs = VoiceEncoder.embeds_from_wavs

                def patched_embeds_from_wavs(self, wavs, *args, **kwargs):
                    import numpy as np
                    processed_wavs = []
                    for wav in wavs:
                        if isinstance(wav, np.ndarray) and wav.dtype == np.float64:
                            processed_wavs.append(wav.astype(np.float32))
                        else:
                            processed_wavs.append(wav)
                    return original_embeds_from_wavs(self, processed_wavs, *args, **kwargs)

                VoiceEncoder.embeds_from_wavs = patched_embeds_from_wavs
                logger.info("VoiceEncoder patched for CPU float32 compatibility")
            except Exception as e:
                logger.debug(f"Could not patch VoiceEncoder: {e}")

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
        self._turbo_model = None
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
        return "Chatterbox TTS"

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

            # Set environment variables to suppress transformers warnings
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
            os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

            # Detect device
            self.device = self._detect_device()
            logger.info(f"Using device: {self.device}")

            self._initialized = True
            logger.info("Chatterbox provider initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Chatterbox: {str(e)}")
            raise TTSError(f"Chatterbox initialization failed: {str(e)}")

    def _load_model(self, model_type: str):
        """Lazy-load a specific Chatterbox model.

        Args:
            model_type: One of 'default', 'turbo', 'multilingual'
        """
        import torch
        import warnings

        # Suppress deprecated pkg_resources warning from perth watermarking
        warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

        # Additional warning suppressions for transformers
        warnings.filterwarnings("ignore", message=r".*LlamaModel is using LlamaSdpaAttention.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=r".*past_key_values.*deprecated", category=FutureWarning)
        warnings.filterwarnings("ignore", message=r".*scaled_dot_product_attention.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=r".*We detected that you are passing.*past_key_values.*", category=UserWarning)

        if model_type == "default" and self._model is None:
            from chatterbox.tts import ChatterboxTTS
            logger.info("Loading English Chatterbox model...")
            self._model = ChatterboxTTS.from_pretrained(device=self.device)
            logger.info("English model loaded successfully")

        elif model_type == "turbo" and self._turbo_model is None:
            try:
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                logger.info("Loading Chatterbox Turbo model...")
                self._turbo_model = ChatterboxTurboTTS.from_pretrained(device=self.device)
                logger.info("Chatterbox Turbo model loaded successfully")
            except ImportError:
                logger.warning("Chatterbox Turbo support not found in package")
                self._turbo_model = None
            except Exception as e:
                logger.warning(f"Failed to load Turbo model: {e}")
                self._turbo_model = None

        elif model_type == "multilingual" and self._multilingual_model is None:
            try:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                torch_device = torch.device(self.device)
                logger.info("Loading Multilingual Chatterbox model...")
                self._multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=torch_device)
                logger.info("Multilingual model loaded successfully")
            except ImportError:
                logger.warning("Multilingual support not found in chatterbox package")
                self._multilingual_model = None
            except Exception as e:
                logger.warning(f"Failed to load Multilingual model: {e}")
                self._multilingual_model = None

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
        **kwargs
    ) -> tuple[bytes, int]:
        """Synthesize speech using Chatterbox.

        Args:
            text: Text to synthesize
            voice_id: Voice ID (chatterbox-default, chatterbox-turbo, or chatterbox-<lang>)

        Keyword Args:
            exaggeration: Emotion intensity (0.0-1.0, default 0.5)
            cfg_weight: Classifier-free guidance weight (0.0-1.0, default 0.5)
            audio_prompt_path: Path to audio file for voice cloning (optional)

        Returns:
            Tuple of (audio_bytes, sample_rate)
        """
        import asyncio

        def _sync_synthesize():
            import torch

            try:
                exaggeration = kwargs.get("exaggeration")
                cfg_weight = kwargs.get("cfg_weight")
                audio_prompt_path = kwargs.get("audio_prompt_path", None)
                temp_audio_path = None  # For cleanup

                # Process audio prompt for channel normalization
                temp_audio_path = None
                if audio_prompt_path:
                    import torchaudio
                    import tempfile
                    import os

                    # Load the reference audio
                    audio_tensor, sample_rate = torchaudio.load(audio_prompt_path)
                    logger.info(f"Loaded reference audio: shape {audio_tensor.shape}, sample_rate {sample_rate}")

                    # Normalize to stereo (2 channels) as expected by Chatterbox
                    if audio_tensor.shape[0] == 1:
                        # Convert mono to stereo by duplicating channel
                        audio_tensor = audio_tensor.repeat(2, 1)
                        logger.info("Converted mono reference audio to stereo")
                    elif audio_tensor.shape[0] > 2:
                        # If more than 2 channels, take first 2
                        audio_tensor = audio_tensor[:2]
                        logger.info("Reduced reference audio to first 2 channels")

                    # Validate channel count
                    if audio_tensor.shape[0] != 2:
                        raise ValueError(f"Reference audio must have 2 channels after processing, got {audio_tensor.shape[0]}")

                    # Save normalized audio to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_audio_path = temp_file.name
                        torchaudio.save(temp_audio_path, audio_tensor, sample_rate)
                        logger.info(f"Saved normalized reference audio to {temp_audio_path}")

                    audio_prompt_path = temp_audio_path

                logger.info(f"Generating audio with voice {voice_id}: {text[:50]}...")

                # Determine model type from voice_id
                if voice_id == "chatterbox-turbo":
                    model_type = "Turbo (English-only)"
                elif voice_id == "chatterbox-default":
                    model_type = "Default (English-only)"
                else:
                    lang_code = voice_id.split("-")[1]
                    model_type = f"Multilingual ({lang_code.upper()})"

                logger.info(f"Chatterbox model selected: {model_type}")

                # Validate parameters
                exaggeration = max(0.0, min(1.0, exaggeration))
                cfg_weight = max(0.0, min(1.0, cfg_weight))

                # Disable CUDA graph capture on CPU
                if self.device == "cpu":
                    # Use no_grad and disable graph capture
                    torch.cuda.is_available = lambda: False  # Fake CUDA availability check

                # Select model based on voice_id
                if voice_id == "chatterbox-turbo":
                    self._load_model("turbo")
                    if self._turbo_model is None:
                        raise TTSError("Chatterbox Turbo model not available")
                    model = self._turbo_model
                    logger.info("Using Chatterbox Turbo model")

                    with torch.no_grad(), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        wav = model.generate(
                            text,
                            audio_prompt_path=audio_prompt_path
                        )
                elif voice_id == "chatterbox-default":
                    self._load_model("default")
                    if self._model is None:
                        raise TTSError("Chatterbox English model not available")
                    model = self._model
                    logger.info(f"Using English model with exaggeration={exaggeration}, cfg_weight={cfg_weight}")

                    with torch.no_grad(), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        wav = model.generate(
                            text,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                            audio_prompt_path=audio_prompt_path
                        )
                else:
                    # Multilingual model
                    self._load_model("multilingual")
                    if self._multilingual_model is None:
                        raise TTSError("Chatterbox Multilingual model not available")
                    model = self._multilingual_model
                    language_id = voice_id.split("-")[1]
                    logger.info(f"Using Multilingual model for {language_id} with exaggeration={exaggeration}")

                    with torch.no_grad(), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        wav = model.generate(
                            text,
                            language_id=language_id,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                            audio_prompt_path=audio_prompt_path
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
            finally:
                # Clean up temporary audio file
                if 'temp_audio_path' in locals() and temp_audio_path and os.path.exists(temp_audio_path):
                    try:
                        os.unlink(temp_audio_path)
                        logger.debug(f"Cleaned up temporary audio file: {temp_audio_path}")
                    except Exception as cleanup_e:
                        logger.warning(f"Failed to clean up temporary audio file: {cleanup_e}")

        # Run the synchronous synthesis in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_synthesize)

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
            # Add default voice
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
                    description="Chatterbox standard English voice with emotion control",
                ),
                Voice(
                    id="chatterbox-turbo",
                    name="Chatterbox Turbo",
                    language="en",
                    locale="en-US",
                    gender=VoiceGender.NEUTRAL,
                    voice_type=VoiceType.NEURAL,
                    provider="chatterbox",
                    sample_rate=24000,
                    description="Chatterbox turbo English voice for faster generation",
                )
            ]

            # Add default voice for each supported language to indicate multilingual support
            for lang in self.get_supported_languages():
                if lang == "en":
                    continue
                self._voices_cache.append(
                    Voice(
                        id=f"chatterbox-{lang}",
                        name=f"Chatterbox {lang.upper()}",
                        language=lang,
                        locale=lang,
                        gender=VoiceGender.NEUTRAL,
                        voice_type=VoiceType.NEURAL,
                        provider="chatterbox",
                        sample_rate=24000,
                        description=f"Chatterbox {lang.upper()} multilingual voice",
                    )
                )

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
            # Get parameters from extra_params or defaults
            exaggeration = request.extra_params.get("exaggeration", 0.5)
            cfg_weight = request.extra_params.get("cfg_weight", 0.5)
            audio_prompt_path = request.extra_params.get("audio_prompt_path")

            # Call _synthesize
            audio_bytes, sample_rate = await self._synthesize(
                request.text,
                voice_id=request.voice_id,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                audio_prompt_path=audio_prompt_path,
            )

            # Calculate duration
            estimated_pcm_samples = len(audio_bytes) // 2  # 16-bit samples
            duration = estimated_pcm_samples / sample_rate

            return TTSResponse(
                audio_data=audio_bytes,
                format=request.audio_format or "wav",
                sample_rate=sample_rate,
                duration=duration,
                voice_used=request.voice_id,
            )
        except Exception as e:
            raise TTSError(f"Chatterbox synthesis error: {str(e)}")
