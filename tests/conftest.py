"""Pytest configuration and fixtures."""
import pytest


@pytest.fixture(autouse=True)
def mock_provider_initialize(monkeypatch):
    """Mock all provider initialize methods to avoid slow setup."""
    from voicegenhub.providers.edge import EdgeTTSProvider
    from voicegenhub.providers.kokoro import KokoroTTSProvider

    async def mock_initialize(self):
        self._initialization_failed = False

    providers = [
        EdgeTTSProvider,
        KokoroTTSProvider,
    ]

    for provider_class in providers:
        monkeypatch.setattr(provider_class, 'initialize', mock_initialize)
