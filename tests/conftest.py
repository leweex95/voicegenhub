"""Pytest configuration and fixtures."""
import pytest


@pytest.fixture(autouse=True)
def mock_provider_initialize(request, monkeypatch):
    """Mock all provider initialize methods to avoid slow setup, except for integration tests."""
    # Skip mocking for integration tests or tests that explicitly request real initialization
    if "integration" in request.keywords or "no_mock" in request.keywords:
        return

    from voicegenhub.providers.edge import EdgeTTSProvider
    from voicegenhub.providers.kokoro import KokoroTTSProvider
    # Add other providers as they are added
    try:
        from voicegenhub.providers.chatterbox import ChatterboxProvider
        from voicegenhub.providers.bark import BarkProvider
        from voicegenhub.providers.qwen import QwenTTSProvider
        providers = [
            EdgeTTSProvider,
            KokoroTTSProvider,
            ChatterboxProvider,
            BarkProvider,
            QwenTTSProvider
        ]
    except ImportError:
        providers = [
            EdgeTTSProvider,
            KokoroTTSProvider,
        ]

    async def mock_initialize(self):
        self._initialization_failed = False
        self._initialized = True

    for provider_class in providers:
        if hasattr(provider_class, 'initialize'):
            monkeypatch.setattr(provider_class, 'initialize', mock_initialize)
