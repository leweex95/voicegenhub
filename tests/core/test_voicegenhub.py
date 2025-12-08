"""Integration tests for VoiceGenHub."""
import asyncio
import sys

import pytest

# Fix for Windows: ensure SelectorEventLoop is used for aiodns compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from voicegenhub import VoiceGenHub


class TestVoiceGenHubUnit:
    """Unit tests for VoiceGenHub with mocked providers."""

    @pytest.fixture
    def mock_provider_factory(self, mocker, mock_provider):
        """Mock provider factory."""
        mock_factory = mocker.patch("voicegenhub.core.engine.provider_factory")
        mock_factory.discover_provider = mocker.AsyncMock()
        mock_factory.create_provider = mocker.AsyncMock(return_value=mock_provider)
        return mock_factory

    @pytest.fixture
    def mock_provider(self, mocker):
        """Mock TTS provider."""
        from voicegenhub.providers.base import (
            AudioFormat,
            ProviderCapabilities,
            TTSResponse,
            Voice,
            VoiceGender,
            VoiceType,
        )

        mock_prov = mocker.AsyncMock()
        mock_prov.provider_id = "mock"
        mock_prov.display_name = "Mock Provider"
        mock_prov.get_capabilities.return_value = ProviderCapabilities(
            supported_formats=[AudioFormat.MP3, AudioFormat.WAV],
            supported_sample_rates=[22050, 44100],
            supports_speed_control=True,
            supports_pitch_control=True,
        )
        mock_prov.get_voices.return_value = [
            Voice(
                id="en-US-MockVoice",
                name="Mock Voice",
                language="en",
                locale="en-US",
                gender=VoiceGender.FEMALE,
                voice_type=VoiceType.NEURAL,
                provider="mock",
            )
        ]
        mock_prov.synthesize.return_value = TTSResponse(
            audio_data=b"fake_audio_data",
            format=AudioFormat.MP3,
            sample_rate=22050,
            duration=1.5,
            voice_used="en-US-MockVoice",
            metadata={"test": "data"},
        )
        return mock_prov

    @pytest.fixture
    def mock_voice_selector(self, mocker, mock_provider):
        """Mock voice selector."""
        mock_selector = mocker.patch("voicegenhub.core.voice.VoiceSelector")
        mock_selector.return_value.get_all_voices.return_value = mock_provider.get_voices.return_value
        return mock_selector

    def test_init(self):
        """Test VoiceGenHub initialization."""
        tts = VoiceGenHub(provider="edge")
        assert tts._provider_id == "edge"
        assert tts._provider is None
        assert not tts._initialized

    @pytest.mark.asyncio
    async def test_initialize(self, mock_provider_factory, mock_provider):
        """Test engine initialization."""
        mock_provider_factory.discover_provider.return_value = None
        mock_provider_factory.create_provider.return_value = mock_provider

        tts = VoiceGenHub(provider="edge")
        await tts.initialize()

        assert tts._initialized
        assert tts._provider == mock_provider
        mock_provider_factory.discover_provider.assert_called_once_with("edge")
        mock_provider_factory.create_provider.assert_called_once_with("edge")

    @pytest.mark.asyncio
    async def test_generate_text_validation(self, mock_provider_factory, mock_provider, mock_voice_selector):
        """Test text parameter validation."""
        mock_provider_factory.discover_provider.return_value = None
        mock_provider_factory.create_provider.return_value = mock_provider

        tts = VoiceGenHub(provider="edge")

        # Empty text
        with pytest.raises(ValueError, match="Text must be a non-empty string"):
            await tts.generate(text="")

        # None text
        with pytest.raises(ValueError, match="Text must be a non-empty string"):
            await tts.generate(text=None)

        # Whitespace only
        with pytest.raises(ValueError, match="Text must be a non-empty string"):
            await tts.generate(text="   ")

    @pytest.mark.asyncio
    async def test_generate_speed_validation(self, mock_provider_factory, mock_provider, mock_voice_selector):
        """Test speed parameter validation."""
        mock_provider_factory.discover_provider.return_value = None
        mock_provider_factory.create_provider.return_value = mock_provider

        tts = VoiceGenHub(provider="edge")

        with pytest.raises(ValueError, match="Speed must be between 0.5 and 2.0"):
            await tts.generate(text="test", speed=0.4)

        with pytest.raises(ValueError, match="Speed must be between 0.5 and 2.0"):
            await tts.generate(text="test", speed=2.1)

    @pytest.mark.asyncio
    async def test_generate_pitch_validation(self, mock_provider_factory, mock_provider, mock_voice_selector):
        """Test pitch parameter validation."""
        mock_provider_factory.discover_provider.return_value = None
        mock_provider_factory.create_provider.return_value = mock_provider

        tts = VoiceGenHub(provider="edge")

        with pytest.raises(ValueError, match="Pitch must be between 0.5 and 2.0"):
            await tts.generate(text="test", pitch=0.4)

        with pytest.raises(ValueError, match="Pitch must be between 0.5 and 2.0"):
            await tts.generate(text="test", pitch=2.1)

    @pytest.mark.asyncio
    async def test_generate_ssml_detection(self, mock_provider_factory, mock_provider, mock_voice_selector):
        """Test SSML detection in text."""
        mock_provider_factory.discover_provider.return_value = None
        mock_provider_factory.create_provider.return_value = mock_provider

        tts = VoiceGenHub(provider="edge")
        await tts.generate(text="<speak>Hello world</speak>")

        # Check that synthesize was called with ssml=True
        call_args = mock_provider.synthesize.call_args
        assert call_args[0][0].ssml is True

    @pytest.mark.asyncio
    async def test_generate_voice_selection_default(self, mock_provider_factory, mock_provider, mock_voice_selector):
        """Test default voice selection."""
        mock_provider_factory.discover_provider.return_value = None
        mock_provider_factory.create_provider.return_value = mock_provider

        tts = VoiceGenHub(provider="edge")
        await tts.generate(text="test")

        # Should use the mock voice
        call_args = mock_provider.synthesize.call_args
        assert call_args[0][0].voice_id == "en-US-MockVoice"

    @pytest.mark.asyncio
    async def test_generate_voice_selection_with_language(self, mock_provider_factory, mock_provider, mock_voice_selector):
        """Test voice selection with language preference."""
        mock_provider_factory.discover_provider.return_value = None
        mock_provider_factory.create_provider.return_value = mock_provider

        tts = VoiceGenHub(provider="edge")
        await tts.generate(text="test", language="en")

        call_args = mock_provider.synthesize.call_args
        assert call_args[0][0].voice_id == "en-US-MockVoice"

    @pytest.mark.asyncio
    async def test_generate_custom_parameters(self, mock_provider_factory, mock_provider, mock_voice_selector):
        """Test generation with custom parameters."""
        tts = VoiceGenHub(provider="edge")
        response = await tts.generate(
            text="test",
            voice="en-US-MockVoice",  # Use the mock voice
            audio_format="wav",
            sample_rate=44100,
            speed=1.5,
            pitch=0.8,
        )

        call_args = mock_provider.synthesize.call_args
        request = call_args[0][0]
        assert request.voice_id == "en-US-MockVoice"
        assert request.audio_format == "wav"
        assert request.sample_rate == 44100
        assert request.speed == 1.5
        assert request.pitch == 0.8

        assert response.audio_data == b"fake_audio_data"
        assert response.duration == 1.5

    @pytest.mark.asyncio
    async def test_generate_invalid_voice_error(self, mock_provider_factory, mock_provider, mock_voice_selector):
        """Test error when voice not found."""
        mock_provider_factory.discover_provider.return_value = None
        mock_provider_factory.create_provider.return_value = mock_provider

        # Mock empty voices list
        mock_provider.get_voices.return_value = []

        tts = VoiceGenHub(provider="edge")
        with pytest.raises(Exception):  # VoiceNotFoundError
            await tts.generate(text="test", voice="nonexistent-voice")

    @pytest.mark.asyncio
    async def test_get_voices(self, mock_provider_factory, mock_provider, mock_voice_selector):
        """Test get_voices method."""
        mock_provider_factory.discover_provider.return_value = None
        mock_provider_factory.create_provider.return_value = mock_provider

        tts = VoiceGenHub(provider="edge")
        voices = await tts.get_voices()

        assert len(voices) == 1
        assert voices[0]["id"] == "en-US-MockVoice"
        assert voices[0]["language"] == "en"

    @pytest.mark.asyncio
    async def test_get_voices_with_language_filter(self, mock_provider_factory, mock_provider, mock_voice_selector):
        """Test get_voices with language filter."""
        mock_provider_factory.discover_provider.return_value = None
        mock_provider_factory.create_provider.return_value = mock_provider

        tts = VoiceGenHub(provider="edge")
        voices = await tts.get_voices(language="en")

        assert len(voices) == 1

    @pytest.mark.asyncio
    async def test_extract_language_from_voice_name(self):
        """Test language extraction from voice names."""
        tts = VoiceGenHub(provider="edge")

        # Test various patterns
        assert tts._extract_language_from_voice_name("en-US-AriaNeural") == "en"
        assert tts._extract_language_from_voice_name("zh-CN-YunxiNeural") == "zh"
        assert tts._extract_language_from_voice_name("fr-FR-DeniseNeural") == "fr"
        assert tts._extract_language_from_voice_name("kokoro_en_adam") == "en"
        assert tts._extract_language_from_voice_name("voice_zh_CN") == "zh"
        assert tts._extract_language_from_voice_name("123-voice") is None
