"""Unit tests for TTS providers."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

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

        # Mock edge-tts and voice fetching
        with patch('voicegenhub.providers.edge.EdgeTTSProvider._fetch_voices_with_retry', new_callable=AsyncMock), \
             patch('voicegenhub.providers.edge.EdgeTTSProvider._get_voice_info', new_callable=AsyncMock) as mock_get_voice, \
             patch('edge_tts.Communicate') as mock_communicate:
            
            # Mock voice info
            mock_get_voice.return_value = {"ShortName": "en-US-AriaNeural", "Name": "Aria"}
            
            mock_instance = MagicMock()
            mock_communicate.return_value = mock_instance

            # Mock the stream method to return audio chunks
            async def mock_stream():
                yield {"type": "audio", "data": b"fake_audio_data"}
            
            mock_instance.stream = mock_stream

            response = await provider.synthesize(sample_request)

            assert isinstance(response, TTSResponse)
            assert len(response.audio_data) > 0
            assert response.duration > 0
            assert response.metadata is not None

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

        with patch('voicegenhub.providers.edge.EdgeTTSProvider._fetch_voices_with_retry', new_callable=AsyncMock), \
             patch('voicegenhub.providers.edge.EdgeTTSProvider._get_voice_info', new_callable=AsyncMock) as mock_get_voice, \
             patch('edge_tts.Communicate') as mock_communicate:
            
            # Mock voice info
            mock_get_voice.return_value = {"ShortName": "en-US-AriaNeural", "Name": "Aria"}
            
            mock_instance = MagicMock()
            mock_communicate.return_value = mock_instance
            mock_instance.__aenter__ = AsyncMock(side_effect=Exception("Edge TTS failed"))
            mock_instance.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(TTSError, match="Synthesis failed after 3 attempts"):
                await provider.synthesize(sample_request)


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
        fake_creds = '{"type": "service_account"}'
        with patch.dict('os.environ', {'GOOGLE_APPLICATION_CREDENTIALS_JSON': fake_creds}):
            with patch('google.cloud.texttospeech.TextToSpeechClient') as mock_client_class:
                await provider.initialize()
                mock_client_class.assert_called_once()
                assert provider._client is not None

    @pytest.mark.asyncio
    async def test_synthesize_success(self, provider, sample_request):
        """Test successful synthesis."""
        fake_creds = '{"type": "service_account"}'
        with patch.dict('os.environ', {'GOOGLE_APPLICATION_CREDENTIALS_JSON': fake_creds}):
            with patch('google.cloud.texttospeech.TextToSpeechClient') as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client

                # Mock the response
                mock_response = MagicMock()
                mock_response.audio_content = b'fake_audio_data'
                mock_client.synthesize_speech.return_value = mock_response

                await provider.initialize()
                response = await provider.synthesize(sample_request)

                assert isinstance(response, TTSResponse)
                assert response.audio_data == b'fake_audio_data'
                assert response.duration > 0
                assert response.metadata is not None

    @pytest.mark.asyncio
    async def test_synthesize_google_api_error(self, provider, sample_request):
        """Test handling of Google API errors."""
        fake_creds = '{"type": "service_account"}'
        with patch.dict('os.environ', {'GOOGLE_APPLICATION_CREDENTIALS_JSON': fake_creds}):
            with patch('google.cloud.texttospeech.TextToSpeechClient') as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                mock_client.synthesize_speech.side_effect = Exception("Google API error")

                await provider.initialize()

                with pytest.raises(TTSError, match="Speech synthesis failed"):
                    await provider.synthesize(sample_request)