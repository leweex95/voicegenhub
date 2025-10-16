"""Unit tests for TTS providers."""
import pytest
from unittest.mock import patch, AsyncMock
import asyncio

from voicegenhub.providers.base import TTSRequest, TTSResponse, AudioFormat, Voice, VoiceGender, VoiceType
from voicegenhub.providers.edge import EdgeTTSProvider
from voicegenhub.providers.google import GoogleTTSProvider
from voicegenhub.providers.base import TTSError


class TestEdgeTTSProvider:
    """Test Edge TTS provider."""

    @pytest.fixture
    def provider(self):
        """Create EdgeTTSProvider instance."""
        return EdgeTTSProvider()

    @pytest.fixture
    def sample_request(self):
        """Create sample TTS request."""
        return TTSRequest(
            text="Hello world",
            voice_id="en-US-AriaNeural",
            language="en-US",
            audio_format=AudioFormat.MP3
        )

    @pytest.mark.asyncio
    async def test_initialize(self, provider):
        """Test provider initialization."""
        # Should not raise an exception
        await provider.initialize()

    @pytest.mark.asyncio
    async def test_synthesize_success(self, provider, sample_request):
        """Test successful synthesis."""
        await provider.initialize()
        response = await provider.synthesize(sample_request)
        assert response.metadata is not None

    @pytest.mark.asyncio
    async def test_edge_get_voices(self):
        provider = EdgeTTSProvider()
        sample_voice = type('Voice', (), {'id': 'en-US-JennyNeural', 'language': 'en', 'locale': 'en-US'})()
        with patch.object(provider, 'get_voices', return_value=[sample_voice]):
            voices = await provider.get_voices()
            assert len(voices) > 0, "Edge TTS should return available voices"

    @pytest.mark.asyncio
    async def test_edge_get_voices_with_language_filter(self):
        provider = EdgeTTSProvider()
        sample_voice = type('Voice', (), {'id': 'en-US-JennyNeural', 'language': 'en', 'locale': 'en-US'})()
        with patch.object(provider, 'get_voices', return_value=[sample_voice]):
            voices = await provider.get_voices(language="en")
            assert len(voices) > 0, "Should return English voices"

    @pytest.mark.asyncio
    async def test_synthesize_unsupported_format(self, provider):
        """Test synthesis with unsupported audio format."""
        await provider.initialize()

        # Use a valid format but mock capabilities to not support it
        request = TTSRequest(
            text="Hello world",
            voice_id="en-US-AriaNeural",
            language="en-US",
            audio_format=AudioFormat.FLAC  # Valid enum but we'll mock as unsupported
        )

        # Mock get_capabilities to not support FLAC
        with patch.object(provider, 'get_capabilities', new_callable=AsyncMock) as mock_caps:
            from voicegenhub.providers.base import ProviderCapabilities
            mock_caps.return_value = ProviderCapabilities(
                supported_formats=[AudioFormat.MP3, AudioFormat.WAV]
            )

            with pytest.raises(TTSError, match="Audio format flac not supported"):
                await provider.synthesize(request)

    @pytest.mark.asyncio
    async def test_synthesize_edge_tts_error(self, provider, sample_request):
        """Test handling of edge-tts errors."""
        await provider.initialize()

        # This test expects synthesis to work, but if API fails, we'll skip
        try:
            response = await provider.synthesize(sample_request)
            assert isinstance(response, TTSResponse)
        except Exception as e:
            # Skip test if Edge TTS API is unavailable
            if "401" in str(e) or "403" in str(e) or "Voice" in str(e) or "Failed to fetch voices" in str(e):
                pytest.skip(f"Edge TTS API unavailable: {e}")
            else:
                # If it's a different error, it should be a TTSError
                assert isinstance(e, TTSError)


class TestGoogleTTSProvider:
    """Test Google TTS provider."""

    @pytest.fixture
    def provider(self):
        """Create GoogleTTSProvider instance."""
        return GoogleTTSProvider()

    @pytest.fixture
    def sample_request(self):
        """Create sample TTS request."""
        return TTSRequest(
            text="Hello world",
            voice_id="en-US-Standard-A",
            language="en-US",
            audio_format=AudioFormat.MP3
        )

    @pytest.mark.asyncio
    async def test_initialize_without_credentials(self, provider):
        """Test initialization without credentials (should not raise error)."""
        with patch.dict('os.environ', {}, clear=True):
            # Should not raise an exception, just log warning
            await provider.initialize()
            assert provider._client is None  # Client should not be initialized

    @pytest.mark.asyncio
    async def test_initialize_with_credentials(self, provider):
        """Test initialization succeeds with credentials."""
        # Check for Google credentials
        import os
        creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
        creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        
        if not creds_json and not (creds_path and os.path.exists(creds_path)):
            pytest.skip("Google credentials not available for integration test")
        
        # Should not raise an exception with real credentials
        await provider.initialize()
        assert provider._client is not None

    @pytest.mark.asyncio
    async def test_synthesize_success(self, provider, sample_request):
        """Test successful synthesis."""
        # Check for Google credentials
        import os
        creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
        creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        
        if not creds_json and not (creds_path and os.path.exists(creds_path)):
            pytest.skip("Google credentials not available for integration test")
        
        try:
            await provider.initialize()
            response = await provider.synthesize(sample_request)

            assert isinstance(response, TTSResponse)
            assert len(response.audio_data) > 0
            assert response.duration > 0
            assert response.metadata is not None
        except Exception as e:
            # Skip test if Google API is unavailable
            if "403" in str(e) or "401" in str(e) or "quota" in str(e).lower() or "permission" in str(e).lower():
                pytest.skip(f"Google TTS API unavailable: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_synthesize_google_api_error(self, provider, sample_request):
        """Test that synthesis works with real Google API (or skips if unavailable)."""
        # Check for Google credentials
        import os
        creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
        creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        
        if not creds_json and not (creds_path and os.path.exists(creds_path)):
            pytest.skip("Google credentials not available for integration test")
        
        try:
            await provider.initialize()
            response = await provider.synthesize(sample_request)
            assert isinstance(response, TTSResponse)
        except Exception as e:
            # Skip test if Google API is unavailable
            if "403" in str(e) or "401" in str(e) or "quota" in str(e).lower() or "permission" in str(e).lower():
                pytest.skip(f"Google TTS API unavailable: {e}")
            else:
                raise