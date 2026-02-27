"""Qwen 3 TTS Provider - Apache 2.0 Licensed, Multilingual, Advanced Control."""

import warnings
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

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
    VoiceType,
)

# Suppress transformers warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = get_logger(__name__)

# Speaker metadata: name -> (language, locale, gender, description).
# The model's get_supported_speakers() only returns names; this dict supplies
# the additional info needed to build a Voice object.  New speakers returned
# by the model but absent here fall back to neutral/multilingual defaults.
_SPEAKER_META: Dict[str, tuple] = {
    "Ryan":     ("en", "en-US", VoiceGender.MALE,   "Dynamic male, strong rhythmic drive — English native"),
    "Aiden":    ("en", "en-US", VoiceGender.MALE,   "Sunny American male, clear midrange — English native"),
    "Vivian":   ("zh", "zh-CN", VoiceGender.FEMALE, "Bright, slightly edgy young female — Chinese native"),
    "Serena":   ("zh", "zh-CN", VoiceGender.FEMALE, "Warm, gentle young female — Chinese native"),
    "Uncle_Fu": ("zh", "zh-CN", VoiceGender.MALE,   "Seasoned male, low mellow timbre — Chinese native"),
    "Dylan":    ("zh", "zh-CN", VoiceGender.MALE,   "Youthful Beijing male, natural timbre — Chinese native"),
    "Eric":     ("zh", "zh-CN", VoiceGender.MALE,   "Lively Chengdu male, slightly husky — Chinese native"),
    "Ono_Anna": ("ja", "ja-JP", VoiceGender.FEMALE, "Playful female, light and nimble — Japanese native"),
    "Sohee":    ("ko", "ko-KR", VoiceGender.FEMALE, "Warm female with rich emotion — Korean native"),
}


class QwenTTSProvider(TTSProvider):
    """
    Qwen 3 TTS provider with maximum configurability.

    Supports three generation modes:
    1. CustomVoice: Use predefined speaker voices
    2. VoiceDesign: Use natural language instructions to design voice
    3. VoiceClone: Clone voice from reference audio

    Configuration parameters:
    - model_name_or_path: HuggingFace model path (default: "Qwen/Qwen3-TTS-CustomVoice")
    - device: Device to use ("cuda", "cpu", or "auto")
    - dtype: Model precision ("float32", "float16", "bfloat16")
    - attn_implementation: Attention implementation ("eager", "sdpa", "flash_attention_2")
    - generation_mode: Mode to use ("custom_voice", "voice_design", "voice_clone")
    - speaker: Default speaker name for custom_voice mode
    - instruct: Default instruction for voice_design mode
    - ref_audio: Reference audio path for voice_clone mode
    - ref_text: Reference text for voice_clone mode
    - x_vector_only_mode: Use only x-vector for voice clone (bool)
    - non_streaming_mode: Use non-streaming mode (bool)
    - temperature: Sampling temperature (float)
    - top_p: Nucleus sampling parameter (float)
    - top_k: Top-k sampling parameter (int)
    - repetition_penalty: Repetition penalty (float)
    - max_new_tokens: Maximum tokens to generate (int)
    - do_sample: Whether to use sampling (bool)
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self._model = None
        self._processor = None
        self._model_loaded = False

        # Configuration with defaults
        self.model_name_or_path = self.config.get(
            "model_name_or_path", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        )
        self.device = self.config.get("device", "auto")
        self.dtype_str = self.config.get("dtype", "bfloat16")
        self.attn_implementation = self.config.get("attn_implementation", "eager")
        self.generation_mode = self.config.get("generation_mode", "custom_voice")

        # Default generation parameters
        self.default_speaker = self.config.get("speaker", None)
        self.default_instruct = self.config.get("instruct", "")
        self.default_ref_audio = self.config.get("ref_audio", None)
        self.default_ref_text = self.config.get("ref_text", None)
        self.x_vector_only_mode = self.config.get("x_vector_only_mode", False)
        self.non_streaming_mode = self.config.get("non_streaming_mode", True)

        # Sampling parameters
        self.temperature = self.config.get("temperature", 1.0)
        self.top_p = self.config.get("top_p", 0.95)
        self.top_k = self.config.get("top_k", 50)
        self.repetition_penalty = self.config.get("repetition_penalty", 1.0)
        self.max_new_tokens = self.config.get("max_new_tokens", 2048)
        self.do_sample = self.config.get("do_sample", True)

    @property
    def provider_id(self) -> str:
        return "qwen"

    @property
    def display_name(self) -> str:
        return "Qwen 3 TTS"

    def _get_dtype(self):
        """Convert dtype string to torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.dtype_str, torch.float32)

    def _determine_device(self):
        """Determine the device to use."""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    async def initialize(self) -> None:
        """Initialize the Qwen TTS model."""
        if self._model_loaded:
            return

        try:
            # Apply CPU compatibility patches
            try:
                from ..utils.compatibility import apply_cpu_compatibility_patches
                apply_cpu_compatibility_patches()
            except ImportError:
                pass

            from qwen_tts import Qwen3TTSModel

            logger.info(
                f"Loading Qwen 3 TTS model: {self.model_name_or_path}",
                model=self.model_name_or_path
            )

            device = self._determine_device()
            dtype = self._get_dtype()

            # Load model with specified configuration
            load_kwargs = {
                "torch_dtype": dtype,
                "attn_implementation": self.attn_implementation,
            }

            if device == "cuda":
                load_kwargs["device_map"] = "auto"

            self._model = Qwen3TTSModel.from_pretrained(
                self.model_name_or_path,
                **load_kwargs
            )

            if device == "cpu":
                self._model.model = self._model.model.to("cpu")

            self._model_loaded = True

            logger.info(
                "Qwen 3 TTS model loaded successfully",
                device=device,
                dtype=self.dtype_str,
                mode=self.generation_mode
            )

        except ImportError as e:
            raise TTSError(
                "Qwen TTS not available. Install with: pip install qwen-tts",
                error_code="IMPORT_ERROR",
                provider=self.provider_id,
            ) from e
        except Exception as e:
            raise TTSError(
                f"Failed to initialize Qwen TTS: {str(e)}",
                error_code="INIT_ERROR",
                provider=self.provider_id,
            ) from e

    async def get_voices(self, language: Optional[str] = None) -> List[Voice]:
        """Return Qwen3-TTS CustomVoice speakers by querying the loaded model.

        Speakers are enriched with language/gender metadata from _SPEAKER_META.
        Speakers not in _SPEAKER_META fall back to neutral/multilingual defaults.
        If *language* is provided (e.g. 'en', 'zh', 'ja', 'ko'), only voices whose
        native language matches are returned.  If no match, the full list is returned.
        """
        await self.initialize()
        speakers = self._model.model.get_supported_speakers() or []

        if not speakers:
            return [Voice(
                id="default",
                name="Default",
                language="multilingual",
                locale="multilingual",
                gender=VoiceGender.NEUTRAL,
                voice_type=VoiceType.NEURAL,
                provider="qwen",
            )]

        voices: List[Voice] = []
        for speaker in speakers:
            meta = _SPEAKER_META.get(speaker)
            if meta:
                lang, locale, gender, desc = meta
            else:
                lang, locale, gender, desc = (
                    "multilingual", "multilingual", VoiceGender.NEUTRAL,
                    f"{speaker} speaker",
                )
            voices.append(Voice(
                id=speaker,
                name=speaker,
                language=lang,
                locale=locale,
                gender=gender,
                voice_type=VoiceType.NEURAL,
                provider="qwen",
                description=desc,
            ))

        if language is None:
            return voices

        lang_filter = language.lower().split("-")[0]  # normalise "en-US" → "en"
        filtered = [v for v in voices if v.language == lang_filter]
        return filtered if filtered else voices

    def _get_native_speaker_for_language(self, language: str) -> str:
        """Get native speaker for a given language."""
        native_speakers = {
            "english": "Ryan",  # Dynamic male voice, English native
            "chinese": "Serena",  # Warm, gentle female, Chinese native
            "japanese": "Ono_Anna",  # Playful female, Japanese native
            "korean": "Sohee",  # Warm female, Korean native
            # For other languages, use Ryan as default (supports all languages)
            "french": "Ryan",
            "german": "Ryan",
            "italian": "Ryan",
            "portuguese": "Ryan",
            "russian": "Ryan",
            "spanish": "Ryan",
        }
        return native_speakers.get(language.lower(), "Ryan")

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Synthesize speech using Qwen TTS."""
        await self.initialize()

        try:
            # Map language codes to Qwen's expected format
            language_map = {
                "en": "english",
                "zh": "chinese",
                "fr": "french",
                "de": "german",
                "it": "italian",
                "ja": "japanese",
                "ko": "korean",
                "pt": "portuguese",
                "ru": "russian",
                "es": "spanish",
            }

            # Extract generation parameters from request
            lang_code = request.language or "en"
            language = language_map.get(lang_code.lower(), "auto")

            logger.info(f"Using language: {language} (from code: {lang_code})")

            # Merge extra params with defaults
            generate_kwargs = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repetition_penalty": self.repetition_penalty,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
            }
            generate_kwargs.update(request.extra_params)

            # Seed support: pin torch seed before generation for reproducibility
            seed = generate_kwargs.pop("seed", None)
            if seed is not None:
                seed = int(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                logger.info(f"Pinned random seed: {seed}")

            # Generate based on mode
            if self.generation_mode == "custom_voice":
                speaker = generate_kwargs.pop("speaker", self.default_speaker or request.voice_id)

                # If speaker is still None or "default", select native speaker for language
                if not speaker or speaker == "default":
                    speaker = self._get_native_speaker_for_language(language)
                    logger.info(f"Auto-selected native speaker '{speaker}' for language '{language}'")

                instruct = generate_kwargs.pop("instruct", self.default_instruct)

                if instruct:
                    wavs, sample_rate = self._model.generate_custom_voice(
                        text=request.text,
                        speaker=speaker,
                        language=language,
                        instruct=instruct,
                        non_streaming_mode=self.non_streaming_mode,
                        **generate_kwargs
                    )
                else:
                    wavs, sample_rate = self._model.generate_custom_voice(
                        text=request.text,
                        speaker=speaker,
                        language=language,
                        non_streaming_mode=self.non_streaming_mode,
                        **generate_kwargs
                    )

            elif self.generation_mode == "voice_design":
                instruct = generate_kwargs.pop("instruct", self.default_instruct)

                if not instruct:
                    raise TTSError(
                        "VoiceDesign mode requires 'instruct' parameter",
                        error_code="MISSING_PARAM",
                        provider=self.provider_id,
                    )

                wavs, sample_rate = self._model.generate_voice_design(
                    text=request.text,
                    instruct=instruct,
                    language=language,
                    non_streaming_mode=self.non_streaming_mode,
                    **generate_kwargs
                )

            elif self.generation_mode == "voice_clone":
                ref_audio = generate_kwargs.pop("ref_audio", self.default_ref_audio)
                ref_text = generate_kwargs.pop("ref_text", self.default_ref_text)
                x_vector_only = generate_kwargs.pop("x_vector_only_mode", self.x_vector_only_mode)

                wavs, sample_rate = self._model.generate_voice_clone(
                    text=request.text,
                    language=language,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only,
                    non_streaming_mode=self.non_streaming_mode,
                    **generate_kwargs
                )

            else:
                raise TTSError(
                    f"Unknown generation mode: {self.generation_mode}",
                    error_code="INVALID_MODE",
                    provider=self.provider_id,
                )

            # Convert numpy array to audio bytes
            wav = wavs[0]  # Take first result

            # Normalize audio to int16 range
            wav = np.clip(wav, -1.0, 1.0)
            wav_int16 = (wav * 32767).astype(np.int16)

            # Write to bytes buffer
            buffer = BytesIO()
            sf.write(buffer, wav_int16, sample_rate, format="WAV")
            audio_data = buffer.getvalue()

            # Calculate duration
            duration = len(wav) / sample_rate

            return TTSResponse(
                audio_data=audio_data,
                format=AudioFormat.WAV,
                sample_rate=sample_rate,
                duration=duration,
                voice_used=request.voice_id,
                metadata={
                    "provider": self.provider_id,
                    "model": self.model_name_or_path,
                    "generation_mode": self.generation_mode,
                    "language": language,
                },
            )

        except Exception as e:
            raise TTSError(
                f"Qwen TTS synthesis failed: {str(e)}",
                error_code="SYNTHESIS_ERROR",
                provider=self.provider_id,
            ) from e

    async def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities."""
        return ProviderCapabilities(
            supports_ssml=False,
            supports_emotions=True,  # Via instruct parameter
            supports_styles=True,     # Via instruct parameter
            supports_speed_control=False,  # Speed controlled via generation params
            supports_pitch_control=False,   # Pitch controlled via generation params
            supports_volume_control=False,
            supports_streaming=False,  # Qwen TTS doesn't support true streaming yet
            max_text_length=5000,
            rate_limit_per_minute=60,
            supported_formats=[AudioFormat.WAV],
            supported_sample_rates=[24000],
        )

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_loaded = False

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Qwen TTS provider cleaned up")
