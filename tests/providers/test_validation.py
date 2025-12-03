"""
Tests for request validation and error handling.
Fast unit tests suitable for pre-commit hooks.
"""
import pytest
from voicegenhub.providers.base import TTSRequest, AudioFormat


class TestTTSRequestValidation:
    """Unit tests for TTSRequest validation."""

    def test_valid_request_creation(self):
        """Test creating valid TTS request."""
        request = TTSRequest(
            text="Hello world",
            voice_id="en-US-AriaNeural",
            audio_format=AudioFormat.WAV,
        )
        assert request.text == "Hello world"
        assert request.voice_id == "en-US-AriaNeural"
        assert request.audio_format == "wav"  # Enum values as strings

    def test_request_with_all_parameters(self):
        """Test request with all optional parameters."""
        request = TTSRequest(
            text="Test",
            voice_id="kokoro-am_adam",
            audio_format=AudioFormat.WAV,
            speed=0.8,
            pitch=1.2,
            language="en",
        )
        assert request.speed == 0.8
        assert request.pitch == 1.2
        assert request.language == "en"

    def test_request_default_audio_format(self):
        """Test that default audio format is MP3."""
        request = TTSRequest(
            text="Hello",
            voice_id="kokoro-am_adam",
        )
        assert request.audio_format == AudioFormat.MP3

    def test_request_missing_voice_id_fails(self):
        """Test that missing voice_id raises validation error."""
        with pytest.raises(Exception):
            TTSRequest(
                text="Hello",
                audio_format=AudioFormat.WAV,
            )

    def test_request_invalid_speed_bounds(self):
        """Test speed parameter bounds."""
        # Extreme values should be stored but provider may reject
        request = TTSRequest(
            text="Test",
            voice_id="kokoro-am_adam",
            audio_format=AudioFormat.WAV,
            speed=0.5,
        )
        assert request.speed == 0.5

        request2 = TTSRequest(
            text="Test",
            voice_id="kokoro-am_adam",
            audio_format=AudioFormat.WAV,
            speed=2.0,
        )
        assert request2.speed == 2.0

    def test_request_invalid_pitch_bounds(self):
        """Test pitch parameter bounds."""
        request = TTSRequest(
            text="Test",
            voice_id="kokoro-am_adam",
            audio_format=AudioFormat.WAV,
            pitch=0.5,
        )
        assert request.pitch == 0.5

    def test_request_language_validation(self):
        """Test language parameter."""
        request = TTSRequest(
            text="Test",
            voice_id="kokoro-am_adam",
            audio_format=AudioFormat.WAV,
            language="en",
        )
        assert request.language == "en"

    def test_request_ssml_parameter(self):
        """Test SSML parameter."""
        request = TTSRequest(
            text="<speak>Hello</speak>",
            voice_id="kokoro-am_adam",
            ssml=True,
        )
        assert request.ssml is True


class TestAudioFormatValidation:
    """Unit tests for audio format handling."""

    def test_wav_format(self):
        """Test WAV format enum."""
        assert AudioFormat.WAV.value == "wav"

    def test_mp3_format(self):
        """Test MP3 format enum."""
        assert AudioFormat.MP3.value == "mp3"

    def test_request_with_wav_format(self):
        """Test request with WAV format."""
        request = TTSRequest(
            text="Test",
            voice_id="kokoro-am_adam",
            audio_format=AudioFormat.WAV,
        )
        assert request.audio_format == "wav"

    def test_request_with_mp3_format(self):
        """Test request with MP3 format."""
        request = TTSRequest(
            text="Test",
            voice_id="kokoro-am_adam",
            audio_format=AudioFormat.MP3,
        )
        assert request.audio_format == "mp3"

    def test_all_supported_formats(self):
        """Test all supported audio formats."""
        formats = [AudioFormat.WAV, AudioFormat.MP3, AudioFormat.OGG, AudioFormat.FLAC, AudioFormat.AAC]
        for fmt in formats:
            request = TTSRequest(
                text="Test",
                voice_id="test-voice",
                audio_format=fmt,
            )
            assert request.audio_format == fmt.value


class TestProviderFactoryUnit:
    """Unit tests for provider factory."""

    def test_engine_initialization(self):
        """Test VoiceGenHub engine can be initialized."""
        from voicegenhub.core.engine import VoiceGenHub

        engine = VoiceGenHub(provider="edge")
        assert engine is not None

    def test_engine_with_kokoro(self):
        """Test VoiceGenHub can initialize with Kokoro."""
        from voicegenhub.core.engine import VoiceGenHub

        engine = VoiceGenHub(provider="kokoro")
        assert engine is not None


class TestVoiceIDValidation:
    """Unit tests for voice ID format validation."""

    def test_edge_voice_id_format(self):
        """Test Edge voice ID format."""
        request = TTSRequest(
            text="Test",
            voice_id="en-US-AriaNeural",
            audio_format=AudioFormat.WAV,
        )
        assert "en-US" in request.voice_id

    def test_kokoro_voice_id_format(self):
        """Test Kokoro voice ID format."""
        request = TTSRequest(
            text="Test",
            voice_id="kokoro-am_adam",
            audio_format=AudioFormat.WAV,
        )
        assert "kokoro-" in request.voice_id

    def test_voice_id_with_provider_prefix(self):
        """Test voice ID parsing with provider prefix."""
        voice_id = "kokoro-af_alloy"
        parts = voice_id.split("-", 1)
        assert parts[0] == "kokoro"
        assert parts[1] == "af_alloy"


class TestErrorHandlingBasic:
    """Basic error handling unit tests."""

    def test_invalid_audio_format_handling(self):
        """Test handling of invalid audio formats."""
        # Pydantic will validate enum
        with pytest.raises(Exception):
            TTSRequest(
                text="Test",
                voice_id="kokoro-am_adam",
                audio_format="invalid_format",
            )
