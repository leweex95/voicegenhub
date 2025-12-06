"""Simple VoiceGenHub TTS Engine."""

from typing import Any, Dict, List, Optional

from ..providers.base import (
    AudioFormat,
    TTSError,
    TTSProvider,
    TTSRequest,
    TTSResponse,
    VoiceNotFoundError,
)
from ..providers.factory import provider_factory
from ..utils.logger import get_logger
from .voice import VoiceSelector

logger = get_logger(__name__)


class VoiceGenHub:
    """TTS engine supporting multiple providers."""

    def __init__(self, provider: str = "edge"):
        """Initialize VoiceGenHub engine."""
        self._provider: Optional[TTSProvider] = None
        self._voice_selector: Optional[VoiceSelector] = None
        self._initialized = False
        self._provider_id = provider

    async def initialize(self) -> None:
        """Initialize the engine and load the specified provider."""
        if self._initialized:
            return

        logger.info("Initializing VoiceGenHub engine")

        # Discover only the requested provider
        await provider_factory.discover_provider(self._provider_id)

        # Create provider instance
        try:
            self._provider = await provider_factory.create_provider(self._provider_id)
            logger.info(f"Loaded provider: {self._provider.display_name}")
        except Exception as e:
            raise TTSError(f"Failed to load {self._provider_id} provider: {e}")

        # Initialize voice selector
        self._voice_selector = VoiceSelector([self._provider])

        self._initialized = True
        logger.info(f"VoiceGenHub initialized with {self._provider_id} provider")

    async def get_available_providers(self) -> List[str]:
        """Get list of available provider IDs."""
        all_providers = ["edge", "piper", "melotts", "kokoro", "elevenlabs", "bark"]
        providers = []

        for provider_id in all_providers:
            await provider_factory.discover_provider(provider_id)

        if provider_factory._edge_provider_class:
            providers.append("edge")
        if provider_factory._google_provider_class:
            providers.append("google")
        if provider_factory._piper_provider_class:
            providers.append("piper")
        if provider_factory._melotts_provider_class:
            providers.append("melotts")
        if provider_factory._kokoro_provider_class:
            providers.append("kokoro")
        if provider_factory._elevenlabs_provider_class:
            providers.append("elevenlabs")

        return providers

    async def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        audio_format: Optional[AudioFormat] = None,
        sample_rate: Optional[int] = None,
        speed: float = 1.0,
        pitch: float = 1.0,
        **kwargs,
    ) -> TTSResponse:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            voice: Voice ID or name
            language: Language code (e.g., 'en', 'en-US')
            audio_format: Output audio format
            sample_rate: Audio sample rate
            speed: Speech speed (0.5-2.0)
            pitch: Speech pitch (0.5-2.0)
            **kwargs: Additional parameters

        Returns:
            TTS response with audio data
        """
        # Validate parameters
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string")

        if speed < 0.5 or speed > 2.0:
            raise ValueError("Speed must be between 0.5 and 2.0")

        if pitch < 0.5 or pitch > 2.0:
            raise ValueError("Pitch must be between 0.5 and 2.0")

        await self.initialize()

        # Auto-detect SSML content
        is_ssml = text.strip().startswith("<") and (
            "</speak>" in text
            or "<speak" in text
            or "<prosody" in text
            or "<voice" in text
        )

        # Get provider capabilities to determine appropriate defaults
        capabilities = await self._provider.get_capabilities()

        # Use provider's preferred sample rate if none specified
        default_sample_rate = sample_rate
        if default_sample_rate is None and capabilities.supported_sample_rates:
            default_sample_rate = capabilities.supported_sample_rates[
                0
            ]  # Use first supported rate
        elif default_sample_rate is None:
            default_sample_rate = 22050  # Fallback

        # Use provider's preferred format if none specified
        default_format = audio_format
        if default_format is None and capabilities.supported_formats:
            default_format = capabilities.supported_formats[0]
        elif default_format is None:
            default_format = AudioFormat.MP3

        # Check provider capabilities and warn about unsupported parameters
        if speed != 1.0 and not capabilities.supports_speed_control:
            logger.warning(
                f"Provider {self._provider.display_name} does not support speed control. "
                "Speed parameter will be ignored."
            )

        if pitch != 1.0 and not capabilities.supports_pitch_control:
            logger.warning(
                f"Provider {self._provider.display_name} does not support pitch control. "
                "Pitch parameter will be ignored."
            )

        # Get available voices to determine appropriate default
        available_voices = await self._voice_selector.get_all_voices()
        voice_ids = [v.id for v in available_voices]

        # Use provided voice or find appropriate default based on language
        if not voice:
            # Try to find a voice matching the requested or default language
            target_lang = (language or "en").lower()
            matching_voices = [
                v for v in available_voices
                if v.language.lower().startswith(target_lang)
            ]

            if matching_voices:
                voice = matching_voices[0].id
                logger.info(f"Using {target_lang.upper()} voice: {voice}")
            elif voice_ids:
                voice = voice_ids[0]
                logger.info(f"No {target_lang.upper()} voice available, using: {voice}")
            else:
                voice = "en-US-AriaNeural"  # Fallback if no voices available

        # Prepare request
        request = TTSRequest(
            text=text,
            voice_id=voice,
            language=language,
            audio_format=default_format,
            sample_rate=default_sample_rate,
            speed=speed,
            pitch=pitch,
            ssml=is_ssml,
        )

        # Verify voice is available
        if request.voice_id not in voice_ids:
            # Try to provide helpful suggestions
            error_msg = f"Voice '{request.voice_id}' not found"

            # Extract language from voice name (common patterns: en-US-*, zh-CN-*, etc.)
            detected_lang = self._extract_language_from_voice_name(request.voice_id)
            if detected_lang:
                # Filter voices by detected language
                lang_voices = [
                    v
                    for v in available_voices
                    if v.language.lower().startswith(detected_lang.lower())
                ]
                if lang_voices:
                    error_msg += f"\n\nAvailable {detected_lang} voices:"
                    for voice in lang_voices[:10]:  # Show first 10
                        error_msg += f"\n  {voice.id} - {voice.name}"
                    if len(lang_voices) > 10:
                        error_msg += f"\n  ... and {len(lang_voices) - 10} more"
                else:
                    # No voices for detected language, show all available
                    error_msg += (
                        f"\n\nNo {detected_lang} voices found. Available voices:"
                    )
                    for voice in available_voices[:10]:
                        error_msg += f"\n  {voice.id} - {voice.name} ({voice.language})"
                    if len(available_voices) > 10:
                        error_msg += f"\n  ... and {len(available_voices) - 10} more"
            else:
                # Could not detect language, show all available
                error_msg += "\n\nAvailable voices:"
                for voice in available_voices[:10]:
                    error_msg += f"\n  {voice.id} - {voice.name} ({voice.language})"
                if len(available_voices) > 10:
                    error_msg += f"\n  ... and {len(available_voices) - 10} more"

            raise VoiceNotFoundError(
                error_msg, error_code="VOICE_NOT_FOUND", provider=self._provider_id
            )

        # Generate audio
        logger.info(f"Generating audio with {self._provider.display_name}")
        response = await self._provider.synthesize(request)

        logger.info(f"Successfully generated {response.duration:.2f}s of audio")
        return response

    def _extract_language_from_voice_name(self, voice_name: str) -> Optional[str]:
        """
        Extract language code from voice name.

        Common patterns:
        - en-US-Name -> en
        - zh-CN-Name -> zh
        - fr-FR-Name -> fr
        - Name_en -> en
        - Name_en_US -> en
        """
        import re

        # Pattern 1: Name_locale (e.g., voice_en, voice_en_US, tts_zh_CN)
        locale_match = re.search(r"_([a-zA-Z_]+)$", voice_name)
        if locale_match:
            locale = locale_match.group(1).lower()
            # For compound locales like zh_cn, extract the base language
            base_lang = locale.split("_")[0]
            return base_lang

        # Pattern 2: LOCALE-Name (e.g., en-US-AriaNeural)
        locale_match = re.match(r"^([a-z]{2}(?:-[A-Z]{2})?)", voice_name)
        if locale_match:
            locale = locale_match.group(1)
            # Extract base language (en from en-US)
            return locale.split("-")[0].lower()

        # Pattern 3: Look for common language prefixes
        voice_lower = voice_name.lower()
        for lang in ["en", "zh", "fr", "de", "es", "it", "pt", "ru", "ja", "ko"]:
            if voice_lower.startswith(lang + "-") or voice_lower.startswith(lang + "_"):
                return lang

        return None

    async def get_voices(self, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get available voices.

        Args:
            language: Filter by language

        Returns:
            List of voice information
        """
        await self.initialize()

        voices = await self._voice_selector.get_all_voices()

        if language:
            voices = [
                v
                for v in voices
                if v.language == language or v.locale.startswith(language)
            ]

        # Convert to dict format
        return [
            {
                "id": voice.id,
                "name": voice.name,
                "language": voice.language,
                "locale": voice.locale,
                "gender": voice.gender.value,
                "provider": voice.provider,
            }
            for voice in voices
        ]
