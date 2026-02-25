"""Provider factory for TTS providers."""

from typing import Any, Dict, Optional

from ..utils.logger import get_logger
from .base import TTSError, TTSProvider

logger = get_logger(__name__)


class ProviderFactory:
    """Factory for creating TTS providers."""

    def __init__(self):
        self._edge_provider_class = None
        self._kokoro_provider_class = None
        self._elevenlabs_provider_class = None
        self._bark_provider_class = None
        self._chatterbox_provider_class = None
        self._qwen_provider_class = None

    async def discover_provider(self, provider_id: str) -> None:
        """Discover and register a specific TTS provider."""
        if provider_id == "edge":
            try:
                from .edge import EdgeTTSProvider
                self._edge_provider_class = EdgeTTSProvider
            except ImportError as e:
                logger.debug(f"Edge provider discovery failed: {e}")
        elif provider_id == "kokoro":
            try:
                from .kokoro import KokoroTTSProvider
                self._kokoro_provider_class = KokoroTTSProvider
            except ImportError as e:
                logger.debug(f"Kokoro provider discovery failed: {e}")
        elif provider_id == "elevenlabs":
            try:
                from .elevenlabs import ElevenLabsTTSProvider
                self._elevenlabs_provider_class = ElevenLabsTTSProvider
            except ImportError as e:
                logger.debug(f"ElevenLabs provider discovery failed: {e}")
        elif provider_id == "bark":
            try:
                from .bark import BarkProvider
                self._bark_provider_class = BarkProvider
            except ImportError as e:
                logger.debug(f"Bark provider discovery failed: {e}")
        elif provider_id == "chatterbox":
            try:
                from .chatterbox import ChatterboxProvider
                self._chatterbox_provider_class = ChatterboxProvider
            except ImportError as e:
                logger.debug(f"Chatterbox provider discovery failed: {e}")
        elif provider_id == "qwen":
            try:
                from .qwen import QwenTTSProvider
                self._qwen_provider_class = QwenTTSProvider
            except ImportError as e:
                logger.debug(f"Qwen provider discovery failed: {e}")

    async def create_provider(
        self, provider_id: str, config: Optional[Dict[str, Any]] = None
    ) -> TTSProvider:
        """
        Create TTS provider instance.

        Args:
            provider_id: Provider ID ("edge", "kokoro", "elevenlabs", "bark", "chatterbox", "qwen")
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

        elif provider_id == "bark":
            if self._bark_provider_class is None:
                raise TTSError(
                    "Bark provider not available. Install with: pip install bark-model"
                )

            provider = self._bark_provider_class(name=provider_id, config=config)
            await provider.initialize()
            return provider

        elif provider_id == "chatterbox":
            if self._chatterbox_provider_class is None:
                raise TTSError(
                    "Chatterbox provider not available. Install with: "
                    "poetry run pip install git+https://github.com/rsxdalv/chatterbox.git@faster"
                )

            provider = self._chatterbox_provider_class(name=provider_id, config=config)
            await provider.initialize()
            return provider

        elif provider_id == "qwen":
            if self._qwen_provider_class is None:
                raise TTSError(
                    "Qwen TTS provider not available. Install with: pip install qwen-tts"
                )

            provider = self._qwen_provider_class(name=provider_id, config=config)
            await provider.initialize()
            return provider

        else:
            raise TTSError(
                f"Unsupported provider: '{provider_id}'. "
                "Available: edge, kokoro, elevenlabs, bark, chatterbox, qwen"
            )


# Global provider factory instance
provider_factory = ProviderFactory()
