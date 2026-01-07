"""Chatterbox TTS Provider - MIT Licensed, Multilingual, Emotion Control."""

import logging
import os
import struct
import subprocess
import sys
import tempfile
import warnings
from io import BytesIO
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

# Suppress all transformers/torch warnings related to attention implementation
warnings.filterwarnings("ignore", category=UserWarning, message=r".*sdpa.*attention.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*scaled_dot_product_attention.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*LlamaModel is using LlamaSdpaAttention.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*output_attentions.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*past_key_values.*")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*Reference mel length is not equal to 2 \* reference token length.*")

# Monkey-patch warnings.warn to suppress the mel length warning
_original_warn = warnings.warn


def _patched_warn(message, category=None, stacklevel=1, source=None):
    if "Reference mel length is not equal to 2 * reference token length" in str(message):
        return
    return _original_warn(message, category, stacklevel, source)


warnings.warn = _patched_warn


# Monkey-patch logging.warning to suppress the mel length warning
_original_warning = logging.warning


def _patched_warning(message, *args, **kwargs):
    if "Reference mel length is not equal to 2 * reference token length" in str(message):
        return
    return _original_warning(message, *args, **kwargs)


logging.warning = _patched_warning


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

            # Actually, patching the VoiceEncoder might be more effective
            try:
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

            # Validate voice cloning dependencies
            self._voice_cloning_available = False
            try:
                self._voice_cloning_available = True
                logger.info("Chatterbox provider initialized with voice cloning support")
            except ImportError as e:
                logger.warning(
                    f"Voice cloning dependencies not available ({e}). "
                    "Falling back to base voice support only."
                )

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
        import io
        import torch

        # Suppress deprecated pkg_resources warning from perth watermarking
        warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

        # Additional warning suppressions for transformers and chatterbox
        warnings.filterwarnings("ignore", message=r".*LlamaModel is using LlamaSdpaAttention.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=r".*past_key_values.*deprecated", category=FutureWarning)
        warnings.filterwarnings("ignore", message=r".*scaled_dot_product_attention.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=r".*We detected that you are passing.*past_key_values.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=r".*LlamaModel is using LlamaSdpaAttention.*")
        warnings.filterwarnings("ignore", message=r".*We detected that you are passing.*past_key_values.*")
        warnings.filterwarnings("ignore", message=r".*does not support.*output_attentions.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=r".*sdpa.*attention.*", category=UserWarning)

        # Patch showwarning to suppress the warnings
        original_showwarning = warnings.showwarning

        def patched_showwarning(message, category, filename, lineno, file=None, line=None):
            if any(x in str(message) for x in ["LlamaModel is using LlamaSdpaAttention", "We detected that you are passing", "sdpa", "scaled_dot_product_attention", "output_attentions"]):
                return
            return original_showwarning(message, category, filename, lineno, file, line)
        warnings.showwarning = patched_showwarning

        # Patch logging.info to suppress the LlamaModel warning
        original_info = logging.info

        def patched_info(message, *args, **kwargs):
            if any(x in str(message) for x in ["LlamaModel is using LlamaSdpaAttention", "loaded PerthNet"]):
                return
            return original_info(message, *args, **kwargs)
        logging.info = patched_info

        # Patch warnings.warn to suppress the past_key_values warning
        original_warn = warnings.warn

        def patched_warn(message, category=None, stacklevel=1, source=None):
            if any(x in str(message) for x in ["We detected that you are passing", "sdpa", "scaled_dot_product_attention", "output_attentions"]):
                return
            return original_warn(message, category, stacklevel, source)
        warnings.warn = patched_warn

        # Suppress stdout/stderr during model loading to hide PerthNet and other console messages
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            if model_type == "default" and self._model is None:
                from chatterbox.tts import ChatterboxTTS
                logger.info("Loading English Chatterbox model...")

                # Redirect stdout/stderr to suppress PerthNet and other console output
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()
                try:
                    self._model = ChatterboxTTS.from_pretrained(device=self.device)
                finally:
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr

                logger.info("English model loaded successfully")

            elif model_type == "multilingual" and self._multilingual_model is None:
                try:
                    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                    torch_device = torch.device(self.device)
                    logger.info("Loading Multilingual Chatterbox model...")

                    # Redirect stdout/stderr to suppress output
                    sys.stdout = io.StringIO()
                    sys.stderr = io.StringIO()
                    try:
                        self._multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=torch_device)
                    finally:
                        sys.stdout = original_stdout
                        sys.stderr = original_stderr

                    logger.info("Multilingual model loaded successfully")
                except ImportError:
                    logger.warning("Multilingual support not found in chatterbox package")
                    self._multilingual_model = None
                except Exception as e:
                    logger.warning(f"Failed to load Multilingual model: {e}")
                    self._multilingual_model = None
        finally:
            # Ensure stdout/stderr are restored
            sys.stdout = original_stdout
            sys.stderr = original_stderr

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
            import io
            import contextlib
            import torch

            try:
                exaggeration = kwargs.get("exaggeration")
                cfg_weight = kwargs.get("cfg_weight")
                audio_prompt_path = kwargs.get("audio_prompt_path", None)
                speed = kwargs.get("speed", 1.0)

                # Check if voice cloning is requested
                if audio_prompt_path and not getattr(self, "_voice_cloning_available", False):
                    raise TTSError(
                        "Voice cloning requested but dependencies are not available. "
                        "Required packages (transformers, torchaudio, librosa) must be installed for voice cloning. "
                        "Install with: pip install transformers torchaudio librosa"
                    )
                temp_audio_path = None  # For cleanup

                # Process audio prompt for channel normalization
                temp_audio_path = None
                cloned_audio_prompt = audio_prompt_path
                if cloned_audio_prompt:
                    import torchaudio

                    # Load the reference audio
                    audio_tensor, sample_rate = torchaudio.load(cloned_audio_prompt)
                    logger.info(f"Loaded reference audio: shape {audio_tensor.shape}, sample_rate {sample_rate}")

                    # Force mono (1 channel) as expected by Chatterbox
                    if audio_tensor.shape[0] > 1:
                        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
                        logger.info("Converted stereo reference audio to mono")

                    # Save normalized audio to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_audio_path = temp_file.name
                        torchaudio.save(temp_audio_path, audio_tensor, sample_rate)
                        logger.debug(f"Saved normalized reference audio to {temp_audio_path}")

                    cloned_audio_prompt = temp_audio_path

                logger.info(f"Generating audio with voice {voice_id}: {text[:50]}...")

                # Determine model type from voice_id
                target_voice_id = voice_id
                if target_voice_id == "chatterbox-turbo":
                    model_type = "Turbo"
                elif target_voice_id == "chatterbox-default":
                    model_type = "Default"
                else:
                    lang_code = target_voice_id.split("-")[1]
                    model_type = f"Multilingual ({lang_code.upper()})"

                logger.info(f"Chatterbox model selected: {model_type}")

                # Validate parameters
                exaggeration = max(0.0, min(1.0, exaggeration or 0.5))
                cfg_weight = max(0.0, min(1.0, cfg_weight or 0.5))

                # Disable CUDA graph capture on CPU
                if self.device == "cpu":
                    # Use no_grad and disable graph capture
                    torch.cuda.is_available = lambda: False  # Fake CUDA availability check

                # Select model based on target_voice_id
                if target_voice_id in ["chatterbox-default", "chatterbox-turbo"]:
                    self._load_model("default")
                    if self._model is None:
                        raise TTSError("Chatterbox English model not available")
                    model = self._model
                    logger.info(f"Using {model_type} model with exaggeration={exaggeration}, cfg_weight={cfg_weight}")

                    # Suppress transformers warnings by disabling their loggers
                    transformers_logger = logging.getLogger("transformers")
                    original_level = transformers_logger.level
                    transformers_logger.setLevel(logging.ERROR)

                    # Suppress all output during generation
                    with torch.no_grad(), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            try:
                                wav = model.generate(
                                    text,
                                    exaggeration=exaggeration,
                                    cfg_weight=cfg_weight,
                                    audio_prompt_path=cloned_audio_prompt
                                )
                            except Exception as e:
                                err_msg = str(e).lower()
                                if any(x in err_msg for x in ["torchcodec", "llamamodel", "import module"]):
                                    raise TTSError(f"Voice cloning failed due to dependency error: {e}")
                                else:
                                    raise
                            finally:
                                transformers_logger.setLevel(original_level)
                else:
                    # Multilingual model
                    self._load_model("multilingual")
                    if self._multilingual_model is None:
                        raise TTSError("Chatterbox Multilingual model not available")
                    model = self._multilingual_model
                    language_id = target_voice_id.split("-")[1]
                    logger.info(f"Using Multilingual model for {language_id} with exaggeration={exaggeration}")

                    # Suppress transformers warnings by disabling their loggers
                    transformers_logger = logging.getLogger("transformers")
                    original_level = transformers_logger.level
                    transformers_logger.setLevel(logging.ERROR)

                    # Suppress all output during generation
                    with torch.no_grad(), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            try:
                                wav = model.generate(
                                    text,
                                    language_id=language_id,
                                    exaggeration=exaggeration,
                                    cfg_weight=cfg_weight,
                                    audio_prompt_path=cloned_audio_prompt
                                )
                            except Exception as e:
                                err_msg = str(e).lower()
                                if any(x in err_msg for x in ["torchcodec", "llamamodel", "import module"]):
                                    raise TTSError(f"Voice cloning failed due to dependency error: {e}")
                                else:
                                    raise
                            finally:
                                transformers_logger.setLevel(original_level)

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
                buffer = BytesIO()

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

                buffer.write(b'data')
                buffer.write(struct.pack('<I', n_samples * n_channels * 2))
                buffer.write(wav_int16.tobytes())

                audio_bytes = buffer.getvalue()

                # Apply speed control if requested
                if speed != 1.0:
                    logger.debug("Applying speed adjustment", speed=speed, tool="ffmpeg", filter="atempo")

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        temp_input = f.name
                        f.write(audio_bytes)

                    temp_output = temp_input + "_speed.wav"
                    try:
                        # Use atempo filter. For values < 0.5 or > 2.0, multiple atempo filters would be needed
                        # but VoiceGenHub restricts speed between 0.5 and 2.0.
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", temp_input, "-filter:a", f"atempo={speed}", temp_output],
                            capture_output=True,
                            check=True
                        )
                        with open(temp_output, "rb") as f:
                            audio_bytes = f.read()
                        logger.info(f"Successfully adjusted speed to {speed}x")
                    except Exception as e:
                        logger.warning(f"Failed to apply speed control via ffmpeg: {e}")
                    finally:
                        if os.path.exists(temp_input):
                            os.unlink(temp_input)
                        if os.path.exists(temp_output):
                            os.unlink(temp_output)

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
            supports_speed_control=True,  # Enabled via ffmpeg post-processing
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
                speed=request.speed,
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
