import pytest
import os
import asyncio
from voicegenhub.providers.factory import provider_factory
from voicegenhub.providers.base import TTSError, TTSRequest, AudioFormat

@pytest.mark.asyncio
@pytest.mark.integration
class TestProviderLoading:
    """
    Tests focused on verifying that providers can be discovered and initialized
    in the current environment. This helps catch missing dependencies or
    import-time crashes (like the torchcodec issue).
    """

    async def _test_provider_init(self, provider_id):
        """Helper to test discovery and initialization of a provider."""
        print(f"\nTesting discovery of '{provider_id}'...")
        await provider_factory.discover_provider(provider_id)

        # Check if it was discovered
        class_attr = f"_{provider_id}_provider_class"
        provider_class = getattr(provider_factory, class_attr, None)

        if provider_class is None:
            pytest.skip(f"Provider '{provider_id}' dependencies not installed in this environment.")
            return

        print(f"Initializing '{provider_id}'...")
        try:
            provider = await provider_factory.create_provider(provider_id)
            # create_provider calls initialize()
            assert provider is not None
            print(f"Successfully initialized '{provider_id}'.")
            return provider
        except (ImportError, TTSError) as e:
            pytest.fail(f"Failed to initialize discovered provider '{provider_id}': {e}")

    async def test_chatterbox_load(self):
        """Test that Chatterbox can be loaded and initialized."""
        await self._test_provider_init("chatterbox")

    async def test_qwen_load(self):
        """Test that Qwen can be loaded and initialized."""
        await self._test_provider_init("qwen")

    async def test_kokoro_load(self):
        """Test that Kokoro can be loaded and initialized."""
        await self._test_provider_init("kokoro")

    async def test_edge_load(self):
        """Test that Edge TTS can be loaded and initialized."""
        await self._test_provider_init("edge")

    def test_torchcodec_compatibility_mock(self):
        """Verify that our torchcodec compatibility mock works if torchcodec is missing."""
        import sys
        import importlib.metadata
        from voicegenhub.utils.compatibility import apply_cpu_compatibility_patches

        # We can't easily uninstall it here, but we can verify the patched state
        # if it was already applied by __init__.py
        try:
            version = importlib.metadata.version("torchcodec")
            assert version is not None
        except importlib.metadata.PackageNotFoundError:
            # If it's missing, apply the patch now
            apply_cpu_compatibility_patches()
            version = importlib.metadata.version("torchcodec")
            assert version == "0.9.1"
            assert "torchcodec" in sys.modules

@pytest.mark.asyncio
@pytest.mark.integration
class TestProviderExecution:
    """
    Smoke tests to ensure that initialized providers can actually generate audio
    on the current system (especially CPU).
    """

    @pytest.mark.slow
    async def test_chatterbox_generate_smoke(self):
        """Small generation test for Chatterbox on CPU."""
        await provider_factory.discover_provider("chatterbox")
        if not provider_factory._chatterbox_provider_class:
            pytest.skip("Chatterbox not installed")

        provider = await provider_factory.create_provider("chatterbox", config={"device": "cpu"})
        request = TTSRequest(text="Test.", voice_id="chatterbox-default")

        response = await provider.synthesize(request)
        assert response.audio_data is not None
        assert len(response.audio_data) > 0

    @pytest.mark.slow
    async def test_qwen_generate_smoke(self):
        """Small generation test for Qwen on CPU."""
        await provider_factory.discover_provider("qwen")
        if not provider_factory._qwen_provider_class:
            pytest.skip("Qwen not installed")

        provider = await provider_factory.create_provider("qwen", config={"device": "cpu"})
        request = TTSRequest(text="Test.", voice_id="default")

        response = await provider.synthesize(request)
        assert response.audio_data is not None
        assert len(response.audio_data) > 0
