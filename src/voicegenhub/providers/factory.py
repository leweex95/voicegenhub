"""Provider factory for TTS providers."""

from typing import Any, Dict, Optional

from ..utils.logger import get_logger
from .base import TTSError, TTSProvider

logger = get_logger(__name__)


class ProviderFactory:
    """Factory for creating TTS providers."""

    def __init__(self):
        self._edge_provider_class = None
        self._google_provider_class = None
        self._piper_provider_class = None
        self._melotts_provider_class = None
        self._kokoro_provider_class = None
        self._elevenlabs_provider_class = None

    async def discover_and_register_providers(self) -> None:
        """Discover and register available TTS providers."""
        # Discover Edge TTS provider
        try:
            from .edge import EdgeTTSProvider

            self._edge_provider_class = EdgeTTSProvider
            logger.info("Discovered Edge TTS provider")
        except ImportError as e:
            logger.error(f"Could not import Edge TTS provider: {e}")

        # Discover Google TTS provider
        try:
            from .google import GoogleTTSProvider

            self._google_provider_class = GoogleTTSProvider
            logger.info("Discovered Google TTS provider")
        except ImportError as e:
            logger.warning(f"Google TTS provider not available: {e}")

        # Discover Piper TTS provider
        try:
            from .piper import PiperTTSProvider

            self._piper_provider_class = PiperTTSProvider
            logger.info("Discovered Piper TTS provider")
        except ImportError as e:
            logger.warning(f"Piper TTS provider not available: {e}")

        # Discover MeloTTS provider
        try:
            from .melotts import MeloTTSProvider

            self._melotts_provider_class = MeloTTSProvider
            logger.info("Discovered MeloTTS provider")
        except ImportError as e:
            logger.warning(f"MeloTTS provider not available: {e}")

        # Discover Kokoro TTS provider
        try:
            from .kokoro import KokoroTTSProvider

            self._kokoro_provider_class = KokoroTTSProvider
            logger.info("Discovered Kokoro TTS provider")
        except ImportError as e:
            logger.warning(f"Kokoro TTS provider not available: {e}")

        # Discover ElevenLabs TTS provider
        try:
            from .elevenlabs import ElevenLabsTTSProvider

            self._elevenlabs_provider_class = ElevenLabsTTSProvider
            logger.info("Discovered ElevenLabs TTS provider")
        except ImportError as e:
            logger.warning(f"ElevenLabs TTS provider not available: {e}")

    async def create_provider(
        self, provider_id: str, config: Optional[Dict[str, Any]] = None
    ) -> TTSProvider:
        """
        Create TTS provider instance.

        Args:
            provider_id: Provider ID ("edge", "google", or "piper")
            config: Optional configuration

        Returns:
            Initialized TTS provider instance
        """
        if provider_id == "edge":
            if self._edge_provider_class is None:
                raise TTSError("Edge TTS provider not available")

            provider = self._edge_provider_class(name=provider_id, config=config)
            await provider.initialize()
            return provider

        elif provider_id == "google":
            if self._google_provider_class is None:
                raise TTSError(
                    "Google TTS provider not available. Install with: pip install google-cloud-texttospeech"
                )

            provider = self._google_provider_class(name=provider_id, config=config)
            await provider.initialize()
            return provider

        elif provider_id == "piper":
            if self._piper_provider_class is None:
                raise TTSError(
                    "Piper TTS provider not available. Install with: pip install voicegenhub[piper]"
                )

            provider = self._piper_provider_class(name=provider_id, config=config)
            await provider.initialize()
            return provider

        elif provider_id == "melotts":
            if self._melotts_provider_class is None:
                raise TTSError(
                    "MeloTTS provider not available. Install with: pip install voicegenhub[melotts]"
                )

            provider = self._melotts_provider_class(name=provider_id, config=config)
            await provider.initialize()
            return provider

        elif provider_id == "kokoro":
            if self._kokoro_provider_class is None:
                raise TTSError(
                    "Kokoro TTS provider not available. Install with: pip install voicegenhub[kokoro]"
                )

            provider = self._kokoro_provider_class(name=provider_id, config=config)
            await provider.initialize()
            return provider

        elif provider_id == "elevenlabs":
            if self._elevenlabs_provider_class is None:
                raise TTSError(
                    "ElevenLabs TTS provider not available. Install with: pip install voicegenhub[elevenlabs]"
                )

            provider = self._elevenlabs_provider_class(name=provider_id, config=config)
            await provider.initialize()
            return provider

        else:
            raise TTSError(
                f"Unsupported provider: '{provider_id}'. Available: edge, google, piper, melotts, kokoro, elevenlabs"
            )


# Global provider factory instance
provider_factory = ProviderFactory()
