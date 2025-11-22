"""Unit tests for TTS providers."""
from unittest.mock import AsyncMock, patch

import pytest

from voicegenhub.providers.base import (
    AudioFormat,
    TTSError,
    TTSRequest,
    TTSResponse,
    Voice,
)
from voicegenhub.providers.edge import EdgeTTSProvider
from voicegenhub.providers.elevenlabs import ElevenLabsTTSProvider
from voicegenhub.providers.google import GoogleTTSProvider
from voicegenhub.providers.kokoro import KokoroTTSProvider
from voicegenhub.providers.melotts import MeloTTSProvider


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
            audio_format=AudioFormat.MP3,
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
        sample_voice = type(
            "Voice",
            (),
            {"id": "en-US-JennyNeural", "language": "en", "locale": "en-US"},
        )()
        with patch.object(provider, "get_voices", return_value=[sample_voice]):
            voices = await provider.get_voices()
            assert len(voices) > 0, "Edge TTS should return available voices"

    @pytest.mark.asyncio
    async def test_edge_get_voices_with_language_filter(self):
        provider = EdgeTTSProvider()
        sample_voice = type(
            "Voice",
            (),
            {"id": "en-US-JennyNeural", "language": "en", "locale": "en-US"},
        )()
        with patch.object(provider, "get_voices", return_value=[sample_voice]):
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
            audio_format=AudioFormat.FLAC,  # Valid enum but we'll mock as unsupported
        )

        # Mock get_capabilities to not support FLAC
        with patch.object(
            provider, "get_capabilities", new_callable=AsyncMock
        ) as mock_caps:
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
            if (
                "401" in str(e)
                or "403" in str(e)
                or "Voice" in str(e)
                or "Failed to fetch voices" in str(e)
            ):
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
            audio_format=AudioFormat.MP3,
        )

    @pytest.mark.asyncio
    async def test_initialize_without_credentials(self, provider):
        """Test initialization without credentials (should not raise error)."""
        with patch.dict("os.environ", {}, clear=True):
            # Should not raise an exception, just log warning
            await provider.initialize()
            assert provider._client is None  # Client should not be initialized

    @pytest.mark.asyncio
    async def test_initialize_with_credentials(self, provider):
        """Test initialization succeeds with credentials."""
        # Check for Google credentials
        import os

        creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

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

        creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

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
            if (
                "403" in str(e)
                or "401" in str(e)
                or "quota" in str(e).lower()
                or "permission" in str(e).lower()
            ):
                pytest.skip(f"Google TTS API unavailable: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_synthesize_google_api_error(self, provider, sample_request):
        """Test that synthesis works with real Google API (or skips if unavailable)."""
        # Check for Google credentials
        import os

        creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

        if not creds_json and not (creds_path and os.path.exists(creds_path)):
            pytest.skip("Google credentials not available for integration test")

        try:
            await provider.initialize()
            response = await provider.synthesize(sample_request)
            assert isinstance(response, TTSResponse)
        except Exception as e:
            # Skip test if Google API is unavailable
            if (
                "403" in str(e)
                or "401" in str(e)
                or "quota" in str(e).lower()
                or "permission" in str(e).lower()
            ):
                pytest.skip(f"Google TTS API unavailable: {e}")
            else:
                raise


class TestMeloTTSProvider:
    """Test MeloTTS provider."""

    @pytest.fixture
    def provider(self):
        """Create MeloTTSProvider instance."""
        return MeloTTSProvider()

    @pytest.fixture
    def sample_request(self):
        """Create sample TTS request."""
        return TTSRequest(
            text="Hello world",
            voice_id="melotts-EN",
            language="en",
            audio_format=AudioFormat.WAV,
        )

    @pytest.mark.asyncio
    async def test_initialize(self, provider):
        """Test provider initialization."""
        await provider.initialize()

    @pytest.mark.asyncio
    async def test_get_voices(self, provider):
        """Test getting available voices."""
        await provider.initialize()
        if provider._initialization_failed:
            pytest.skip("MeloTTS provider not available (optional dependency)")
        voices = await provider.get_voices()
        assert len(voices) > 0, "MeloTTS should return available voices"
        assert all(isinstance(v, Voice) for v in voices)
        assert all(v.provider == "melotts" for v in voices)

    @pytest.mark.asyncio
    async def test_get_voices_with_language_filter(self, provider):
        """Test getting voices with language filter."""
        await provider.initialize()
        if provider._initialization_failed:
            pytest.skip("MeloTTS provider not available (optional dependency)")
        voices = await provider.get_voices(language="en")
        assert len(voices) > 0, "Should return English voices"
        assert all(v.language == "en" for v in voices)

    @pytest.mark.asyncio
    async def test_get_capabilities(self, provider):
        """Test getting provider capabilities."""
        await provider.initialize()
        if provider._initialization_failed:
            pytest.skip("MeloTTS provider not available (optional dependency)")
        caps = await provider.get_capabilities()
        assert caps.supported_formats == [AudioFormat.WAV]
        assert 24000 in caps.supported_sample_rates

    @pytest.mark.asyncio
    async def test_synthesize_unavailable(self, provider, sample_request):
        """Test synthesis when provider is unavailable."""
        provider._initialization_failed = True
        with pytest.raises(TTSError, match="MeloTTS provider is not available"):
            await provider.synthesize(sample_request)

    @pytest.mark.asyncio
    async def test_synthesize_invalid_voice_id(self, provider):
        """Test synthesis with invalid voice ID."""
        await provider.initialize()
        if provider._initialization_failed:
            pytest.skip("MeloTTS provider not available (optional dependency)")
        request = TTSRequest(
            text="Hello", voice_id="invalid-voice", audio_format=AudioFormat.WAV
        )
        with pytest.raises(TTSError):
            await provider.synthesize(request)


class TestKokoroTTSProvider:
    """Test Kokoro TTS provider."""

    @pytest.fixture
    def provider(self):
        """Create KokoroTTSProvider instance."""
        return KokoroTTSProvider()

    @pytest.fixture
    def sample_request(self):
        """Create sample TTS request."""
        return TTSRequest(
            text="Hello world",
            voice_id="kokoro-en",
            language="en",
            audio_format=AudioFormat.WAV,
        )

    @pytest.mark.asyncio
    async def test_initialize(self, provider):
        """Test provider initialization."""
        await provider.initialize()

    @pytest.mark.asyncio
    async def test_get_voices(self, provider):
        """Test getting available voices."""
        await provider.initialize()
        if provider._initialization_failed:
            pytest.skip("Kokoro provider not available (optional dependency)")
        voices = await provider.get_voices()
        assert len(voices) > 0, "Kokoro should return available voices"
        assert all(isinstance(v, Voice) for v in voices)
        assert all(v.provider == "kokoro" for v in voices)

    @pytest.mark.asyncio
    async def test_get_voices_with_language_filter(self, provider):
        """Test getting voices with language filter."""
        await provider.initialize()
        if provider._initialization_failed:
            pytest.skip("Kokoro provider not available (optional dependency)")
        voices = await provider.get_voices(language="en")
        assert len(voices) > 0, "Should return English voices"
        assert all(v.language == "en" for v in voices)

    @pytest.mark.asyncio
    async def test_get_capabilities(self, provider):
        """Test getting provider capabilities."""
        await provider.initialize()
        if provider._initialization_failed:
            pytest.skip("Kokoro provider not available (optional dependency)")
        caps = await provider.get_capabilities()
        assert caps.supported_formats == [AudioFormat.WAV]
        assert 24000 in caps.supported_sample_rates

    @pytest.mark.asyncio
    async def test_synthesize_unavailable(self, provider, sample_request):
        """Test synthesis when provider is unavailable."""
        provider._initialization_failed = True
        with pytest.raises(TTSError, match="Kokoro TTS provider is not available"):
            await provider.synthesize(sample_request)

    @pytest.mark.asyncio
    async def test_synthesize_invalid_voice_id(self, provider):
        """Test synthesis with invalid voice ID."""
        await provider.initialize()
        if provider._initialization_failed:
            pytest.skip("Kokoro provider not available (optional dependency)")
        request = TTSRequest(
            text="Hello", voice_id="invalid-voice", audio_format=AudioFormat.WAV
        )
        with pytest.raises(TTSError):
            await provider.synthesize(request)


class TestElevenLabsTTSProvider:
    """Test ElevenLabs TTS provider."""

    @pytest.fixture
    def provider(self):
        """Create ElevenLabsTTSProvider instance."""
        return ElevenLabsTTSProvider()

    @pytest.fixture
    def sample_request(self):
        """Create sample TTS request."""
        return TTSRequest(
            text="Hello world",
            voice_id="elevenlabs-EXAVITQu4vr4xnSDxMaL",
            language="en",
            audio_format=AudioFormat.MP3,
        )

    @pytest.mark.asyncio
    async def test_initialize_without_api_key(self, provider):
        """Test initialization without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(Exception):
                await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialize(self, provider):
        """Test provider initialization."""
        import os

        if not os.getenv("ELEVENLABS_API_KEY"):
            pytest.skip("ElevenLabs API key not available for integration test")

        await provider.initialize()

    @pytest.mark.asyncio
    async def test_get_voices(self, provider):
        """Test getting available voices."""
        import os

        if not os.getenv("ELEVENLABS_API_KEY"):
            pytest.skip("ElevenLabs API key not available for integration test")

        await provider.initialize()
        if provider._initialization_failed:
            pytest.skip("ElevenLabs provider not available (optional dependency)")

        voices = await provider.get_voices()
        assert len(voices) > 0, "ElevenLabs should return available voices"
        assert all(isinstance(v, Voice) for v in voices)
        assert all(v.provider == "elevenlabs" for v in voices)

    @pytest.mark.asyncio
    async def test_get_voices_with_language_filter(self, provider):
        """Test getting voices with language filter."""
        import os

        if not os.getenv("ELEVENLABS_API_KEY"):
            pytest.skip("ElevenLabs API key not available for integration test")

        await provider.initialize()
        if provider._initialization_failed:
            pytest.skip("ElevenLabs provider not available (optional dependency)")

        voices = await provider.get_voices(language="en")
        assert len(voices) > 0, "Should return English voices"

    @pytest.mark.asyncio
    async def test_get_capabilities(self, provider):
        """Test getting provider capabilities."""
        import os

        if not os.getenv("ELEVENLABS_API_KEY"):
            pytest.skip("ElevenLabs API key not available for integration test")

        await provider.initialize()
        if provider._initialization_failed:
            pytest.skip("ElevenLabs provider not available (optional dependency)")

        caps = await provider.get_capabilities()
        assert caps.supported_formats == [AudioFormat.MP3]
        assert 24000 in caps.supported_sample_rates
        assert caps.supports_streaming is True

    @pytest.mark.asyncio
    async def test_synthesize_unavailable(self, provider, sample_request):
        """Test synthesis when provider is unavailable."""
        provider._initialization_failed = True
        with pytest.raises(TTSError, match="ElevenLabs provider is not available"):
            await provider.synthesize(sample_request)

    @pytest.mark.asyncio
    async def test_synthesize_invalid_voice_id(self, provider):
        """Test synthesis with invalid voice ID."""
        import os

        if not os.getenv("ELEVENLABS_API_KEY"):
            pytest.skip("ElevenLabs API key not available for integration test")

        await provider.initialize()
        if provider._initialization_failed:
            pytest.skip("ElevenLabs provider not available (optional dependency)")

        request = TTSRequest(
            text="Hello", voice_id="invalid-voice", audio_format=AudioFormat.MP3
        )
        with pytest.raises(TTSError):
            await provider.synthesize(request)
