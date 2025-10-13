"""Integration tests for VoiceGenHub."""
import pytest
import asyncio
from unittest.mock import patch

from voicegenhub import VoiceGenHub


class TestVoiceGenHub:
    """Integration tests for VoiceGenHub."""

    @pytest.mark.asyncio
    async def test_edge_tts_generation(self):
        """Test Edge TTS generation through VoiceGenHub."""
        tts = VoiceGenHub(provider='edge')
        await tts.initialize()

        response = await tts.generate(
            text='nightly test of edge tts provider',
            voice='en-US-AriaNeural'
        )

        assert len(response.audio_data) > 0, 'no audio data generated'
        assert response.duration > 0, 'invalid audio duration'
        assert response.metadata is not None

    @pytest.mark.asyncio
    async def test_google_tts_generation(self):
        """Test Google TTS generation through VoiceGenHub."""
        # Skip if no Google credentials
        import os
        if 'GOOGLE_APPLICATION_CREDENTIALS_JSON' not in os.environ:
            pytest.skip("Google credentials not available")

        tts = VoiceGenHub(provider='google')
        await tts.initialize()

        response = await tts.generate(
            text='nightly test of google tts provider',
            voice='en-US-Standard-A',
            language='en-US'
        )

        assert len(response.audio_data) > 0, 'no audio data generated'
        assert response.duration > 0, 'invalid audio duration'
        assert response.metadata is not None

    @pytest.mark.asyncio
    async def test_invalid_provider(self):
        """Test invalid provider raises error."""
        with pytest.raises(Exception):  # TTSError inherits from Exception
            tts = VoiceGenHub(provider='invalid')
            await tts.initialize()

    @pytest.mark.asyncio
    async def test_generate_without_initialize(self):
        """Test generate works even without explicit initialization."""
        tts = VoiceGenHub(provider='edge')

        # Should work because generate calls initialize internally
        response = await tts.generate(text="test")
        assert len(response.audio_data) > 0

    @pytest.mark.asyncio
    async def test_generate_with_custom_params(self):
        """Test generation with custom parameters."""
        tts = VoiceGenHub(provider='edge')
        await tts.initialize()

        response = await tts.generate(
            text='Custom test message',
            voice='en-US-AriaNeural',  # Use a known working voice
            audio_format='mp3'
        )

        assert len(response.audio_data) > 0
        assert response.duration > 0
        assert response.metadata['provider'] == 'edge'
        assert 'voice_locale' in response.metadata


# Test configuration
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment."""
    # This could be used to set up any global test state
    pass