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


class TestVoiceGenHubIntegration:
    """Integration tests for VoiceGenHub."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_edge_tts_generation(self):
        """Test Edge TTS generation through VoiceGenHub."""
        tts = VoiceGenHub(provider="edge")

        try:
            await tts.initialize()
            response = await tts.generate(
                text="nightly test of edge tts provider", voice="en-US-AriaNeural"
            )

            assert len(response.audio_data) > 0, "no audio data generated"
            assert response.duration > 0, "invalid audio duration"
            assert response.metadata is not None
        except Exception as e:
            # Skip test if Edge TTS API is unavailable (401/403 errors, network issues, etc.)
            if (
                "401" in str(e)
                or "403" in str(e)
                or "Voice" in str(e)
                or "Failed to fetch voices" in str(e)
            ):
                pytest.skip(f"Edge TTS API unavailable: {e}")
            else:
                raise

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_google_tts_generation(self):
        """Test Google TTS generation through VoiceGenHub."""
        # Check for Google credentials in order of preference:
        # 1. GOOGLE_APPLICATION_CREDENTIALS (set by CI after creating temp file)
        # 2. GOOGLE_APPLICATION_CREDENTIALS_JSON (CI secret, create temp file)
        # 3. Local config file
        import json
        import os
        import tempfile

        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

        if not credentials_path:
            # Check for JSON content from CI secret
            creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            if creds_json:
                # Create temp file from JSON content (CI scenario)
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    json.dump(json.loads(creds_json), f)
                    credentials_path = f.name
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            else:
                # Fall back to local config file
                project_root = os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
                config_credentials = os.path.join(
                    project_root, "config", "google-credentials.json"
                )
                if os.path.exists(config_credentials):
                    credentials_path = config_credentials
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
                else:
                    pytest.skip(
                        "Google credentials not available - no GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_APPLICATION_CREDENTIALS_JSON, or config/google-credentials.json found"
                    )

        print(f"GOOGLE_APPLICATION_CREDENTIALS: {credentials_path}")
        print(f"Credentials file exists: {os.path.exists(credentials_path)}")

        tts = VoiceGenHub(provider="google")
        try:
            await tts.initialize()
        except Exception as e:
            pytest.skip(
                f"Google TTS provider failed to initialize: {str(e)}"
            )

        # Get available voices and use the first English one that doesn't require a model
        try:
            voices = await tts.get_voices(language="en")
        except Exception as e:
            pytest.skip(
                f"Failed to get Google TTS voices: {str(e)}"
            )
        print(f"Available English voices: {len(voices)}")
        if not voices:
            pytest.skip("No English voices available from Google TTS")

        # Try to find a standard voice that doesn't require model specification
        voice_to_use = None
        for voice in voices:
            voice_name = voice["name"]
            # Prefer standard voices over neural ones for testing
            if "Standard" in voice_name and "en-US" in voice_name:
                voice_to_use = voice["id"]
                break

        # Fallback to first voice if no standard voice found
        if not voice_to_use:
            voice_to_use = voices[0]["id"]

        print(f"Using voice: {voice_to_use}")

        response = await tts.generate(
            text="nightly test of google tts provider",
            voice=voice_to_use,
            language="en-US",
        )

        assert len(response.audio_data) > 0, "no audio data generated"
        assert response.duration > 0, "invalid audio duration"
        assert response.metadata is not None

    @pytest.mark.asyncio
    async def test_invalid_provider(self):
        """Test invalid provider raises error."""
        with pytest.raises(Exception):  # TTSError inherits from Exception
            tts = VoiceGenHub(provider="invalid")
            await tts.initialize()

    @pytest.mark.asyncio
    async def test_generate_without_initialize(self):
        """Test generate works even without explicit initialization."""
        tts = VoiceGenHub(provider="edge")

        try:
            # Should work because generate calls initialize internally
            response = await tts.generate(text="test")
            assert len(response.audio_data) > 0
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
                raise

    @pytest.mark.asyncio
    async def test_generate_with_custom_params(self):
        """Test generation with custom parameters."""
        tts = VoiceGenHub(provider="edge")

        try:
            await tts.initialize()

            response = await tts.generate(
                text="Custom test message",
                voice="en-US-AriaNeural",  # Use a known working voice
                audio_format="mp3",
            )

            assert len(response.audio_data) > 0
            assert response.duration > 0
            assert response.metadata["provider"] == "edge"
            assert "voice_locale" in response.metadata
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
                raise

    @pytest.mark.asyncio
    async def test_edge_get_voices(self):
        """Test Edge TTS get_voices functionality."""
        tts = VoiceGenHub(provider="edge")

        try:
            await tts.initialize()
            voices = await tts.get_voices()

            assert len(voices) > 0, "Edge TTS should return available voices"
            assert all(
                isinstance(v, dict) for v in voices
            ), "Voices should be dictionaries"
            assert all(
                "id" in v and "name" in v for v in voices
            ), "Voices should have id and name"
        except Exception as e:
            if "401" in str(e) or "403" in str(e) or "Failed to fetch voices" in str(e):
                pytest.skip(f"Edge TTS API unavailable: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_edge_get_voices_with_language_filter(self):
        """Test Edge TTS get_voices with language filter."""
        tts = VoiceGenHub(provider="edge")

        try:
            await tts.initialize()
            voices_en = await tts.get_voices(language="en")

            assert len(voices_en) > 0, "Should return English voices"
            # Verify all returned voices are English
            for voice in voices_en:
                assert voice["language"].startswith("en") or voice["locale"].startswith(
                    "en"
                ), f"Voice {voice} should be English"
        except Exception as e:
            if "401" in str(e) or "403" in str(e) or "Failed to fetch voices" in str(e):
                pytest.skip(f"Edge TTS API unavailable: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_edge_invalid_voice(self):
        """Test Edge TTS with invalid voice raises error."""
        tts = VoiceGenHub(provider="edge")

        try:
            await tts.initialize()

            with pytest.raises(Exception):
                await tts.generate(text="test", voice="invalid-voice-id-12345")
        except Exception as e:
            if "401" in str(e) or "403" in str(e) or "Failed to fetch voices" in str(e):
                pytest.skip(f"Edge TTS API unavailable: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_edge_voice_caching(self):
        """Test Edge TTS caches voices properly."""
        tts = VoiceGenHub(provider="edge")

        try:
            await tts.initialize()

            # First call fetches voices
            voices1 = await tts.get_voices()
            # Second call should use cache (same object)
            voices2 = await tts.get_voices()

            assert len(voices1) == len(voices2)
        except Exception as e:
            if "401" in str(e) or "403" in str(e) or "Failed to fetch voices" in str(e):
                pytest.skip(f"Edge TTS API unavailable: {e}")
            else:
                raise


# Test configuration
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment."""
    # This could be used to set up any global test state
    pass
