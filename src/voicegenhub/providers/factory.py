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

    async def discover_provider(self, provider_id: str) -> None:
        """Discover and register a specific TTS provider."""
        if provider_id == "edge":
            try:
                from .edge import EdgeTTSProvider
                self._edge_provider_class = EdgeTTSProvider
            except ImportError:
                pass
        elif provider_id == "google":
            try:
                from .google import GoogleTTSProvider
                self._google_provider_class = GoogleTTSProvider
            except ImportError:
                pass
        elif provider_id == "piper":
            try:
                from .piper import PiperTTSProvider
                self._piper_provider_class = PiperTTSProvider
            except ImportError:
                pass
        elif provider_id == "melotts":
            try:
                from .melotts import MeloTTSProvider
                self._melotts_provider_class = MeloTTSProvider
            except ImportError:
                pass
        elif provider_id == "kokoro":
            try:
                from .kokoro import KokoroTTSProvider
                self._kokoro_provider_class = KokoroTTSProvider
            except ImportError:
                pass
        elif provider_id == "elevenlabs":
            try:
                from .elevenlabs import ElevenLabsTTSProvider
                self._elevenlabs_provider_class = ElevenLabsTTSProvider
            except ImportError:
                pass

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
