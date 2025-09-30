"""Simple provider factory for Edge TTS."""

from typing import Optional, Dict, Any

from .base import TTSProvider, TTSError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ProviderFactory:
    """Simple factory for creating Edge TTS provider."""
    
    def __init__(self):
        self._edge_provider_class = None
    
    async def discover_and_register_providers(self) -> None:
        """Discover and register Edge TTS provider."""
        try:
            from .edge import EdgeTTSProvider
            self._edge_provider_class = EdgeTTSProvider
            logger.info("Discovered Edge TTS provider")
        except ImportError as e:
            logger.error(f"Could not import Edge TTS provider: {e}")
    
    async def create_provider(
        self, 
        provider_id: str, 
        config: Optional[Dict[str, Any]] = None
    ) -> TTSProvider:
        """
        Create Edge TTS provider instance.
        
        Args:
            provider_id: Must be "edge"
            config: Optional configuration (unused)
        
        Returns:
            Initialized Edge TTS provider instance
        """
        if provider_id != "edge":
            raise TTSError(f"Only 'edge' provider is supported, got '{provider_id}'")
        
        if self._edge_provider_class is None:
            raise TTSError("Edge TTS provider not available")
        
        try:
            provider = self._edge_provider_class(provider_id)
            await provider.initialize()
            return provider
        except Exception as e:
            raise TTSError(f"Failed to create Edge TTS provider: {str(e)}")


# Global provider factory instance
provider_factory = ProviderFactory()