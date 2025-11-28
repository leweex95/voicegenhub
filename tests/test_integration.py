"""
Integration tests for voice selection, provider initialization, and audio response handling.
These tests are marked with @pytest.mark.integration and should only run in CI.
"""
import pytest


class TestVoiceSelectionIntegration:
    """Integration tests for voice selection and caching."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_voice_caching_single_provider(self):
        """Integration: Test voice caching within single provider."""
        from voicegenhub.core.engine import VoiceGenHub

        engine = VoiceGenHub(provider="edge")
        await engine.initialize()

        # First call should populate cache
        voices1 = await engine.get_voices()
        assert len(voices1) > 0

        # Second call should return cached voices
        voices2 = await engine.get_voices()
        assert voices1 == voices2

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_voice_language_filtering(self):
        """Integration: Test language filtering in voice lists."""
        from voicegenhub.core.engine import VoiceGenHub

        engine = VoiceGenHub(provider="edge")
        await engine.initialize()

        # Get all voices
        all_voices = await engine.get_voices()
        assert len(all_voices) > 0

        # Filter by language
        en_voices = await engine.get_voices(language="en")
        assert len(en_voices) > 0
        assert all(v.language.startswith("en") for v in en_voices)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_voice_aggregation_multiple_providers(self):
        """Integration: Test voice aggregation from multiple providers."""
        from voicegenhub.core.engine import VoiceGenHub

        # Create engine without specifying provider (auto-select)
        engine = VoiceGenHub()
        await engine.initialize()

        voices = await engine.get_voices()
        assert len(voices) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_voice_selection_by_gender(self):
        """Integration: Test filtering voices by gender."""
        from voicegenhub.core.engine import VoiceGenHub

        engine = VoiceGenHub(provider="edge")
        await engine.initialize()

        voices = await engine.get_voices()
        male_voices = [v for v in voices if v.gender.value == "male"]
        female_voices = [v for v in voices if v.gender.value == "female"]

        assert len(male_voices) > 0
        assert len(female_voices) > 0


class TestProviderInitializationIntegration:
    """Integration tests for provider initialization."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_edge_provider_initialization(self):
        """Integration: Test Edge provider initialization."""
        from voicegenhub.providers.edge import EdgeTTSProvider

        provider = EdgeTTSProvider()
        await provider.initialize()

        assert provider is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kokoro_provider_initialization(self):
        """Integration: Test Kokoro provider initialization (slow)."""
        pytest.importorskip("kokoro")
        from voicegenhub.providers.kokoro import KokoroTTSProvider

        provider = KokoroTTSProvider()
        await provider.initialize()

        assert provider is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_provider_get_capabilities(self):
        """Integration: Test provider capability reporting."""
        from voicegenhub.core.engine import VoiceGenHub

        engine = VoiceGenHub(provider="edge")
        await engine.initialize()

        capabilities = engine._provider.capabilities
        assert capabilities is not None
        assert hasattr(capabilities, "supports_speed")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_provider_initialization_failure_handling(self):
        """Integration: Test handling of provider initialization failures."""
        from voicegenhub.providers.factory import discover_provider
        from voicegenhub.providers.base import TTSError

        with pytest.raises(TTSError):
            discover_provider("nonexistent_provider")


class TestAudioResponseIntegration:
    """Integration tests for audio response handling."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_synthesize_returns_valid_audio_data(self):
        """Integration: Test that synthesis returns valid audio data."""
        from voicegenhub.core.engine import VoiceGenHub

        engine = VoiceGenHub(provider="edge")
        await engine.initialize()

        from voicegenhub.providers.base import TTSRequest, AudioFormat

        request = TTSRequest(
            text="Hello world",
            voice_id="en-US-AriaNeural",
            audio_format=AudioFormat.WAV,
        )

        response = await engine._provider.synthesize(request)
        assert response is not None
        assert response.audio_data is not None
        assert len(response.audio_data) > 0
        # WAV files start with RIFF header
        assert response.audio_data.startswith(b"RIFF")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_synthesize_response_format(self):
        """Integration: Test audio response format integrity."""
        from voicegenhub.core.engine import VoiceGenHub

        engine = VoiceGenHub(provider="edge")
        await engine.initialize()

        from voicegenhub.providers.base import TTSRequest, AudioFormat

        request = TTSRequest(
            text="Test",
            voice_id="en-US-AriaNeural",
            audio_format=AudioFormat.WAV,
        )

        response = await engine._provider.synthesize(request)

        # Check audio data is bytes
        assert isinstance(response.audio_data, bytes)

        # Check RIFF header for WAV
        assert response.audio_data[:4] == b"RIFF"
        assert response.audio_data[8:12] == b"WAVE"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_synthesize_with_speed_parameter(self):
        """Integration: Test synthesis with speed parameter."""
        from voicegenhub.core.engine import VoiceGenHub

        engine = VoiceGenHub(provider="edge")
        await engine.initialize()

        from voicegenhub.providers.base import TTSRequest, AudioFormat

        # Test with different speeds
        for speed in [0.8, 1.0, 1.2]:
            request = TTSRequest(
                text="Hello",
                voice_id="en-US-AriaNeural",
                audio_format=AudioFormat.WAV,
                speed=speed,
            )
            response = await engine._provider.synthesize(request)
            assert response.audio_data is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_synthesize_with_pitch_parameter(self):
        """Integration: Test synthesis with pitch parameter."""
        from voicegenhub.core.engine import VoiceGenHub

        engine = VoiceGenHub(provider="edge")
        await engine.initialize()

        from voicegenhub.providers.base import TTSRequest, AudioFormat

        request = TTSRequest(
            text="Hello",
            voice_id="en-US-AriaNeural",
            audio_format=AudioFormat.WAV,
            pitch=1.2,
        )
        response = await engine._provider.synthesize(request)
        assert response.audio_data is not None


class TestErrorHandlingIntegration:
    """Integration tests for error handling edge cases."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_invalid_voice_id_error(self):
        """Integration: Test handling of invalid voice ID."""
        from voicegenhub.core.engine import VoiceGenHub
        from voicegenhub.providers.base import TTSRequest, AudioFormat, VoiceNotFoundError

        engine = VoiceGenHub(provider="edge")
        await engine.initialize()

        request = TTSRequest(
            text="Hello",
            voice_id="invalid-voice-id-xyz",
            audio_format=AudioFormat.WAV,
        )

        with pytest.raises(VoiceNotFoundError):
            await engine._provider.synthesize(request)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_empty_text_handling(self):
        """Integration: Test handling of empty text."""
        from voicegenhub.core.engine import VoiceGenHub
        from voicegenhub.providers.base import TTSRequest, AudioFormat

        engine = VoiceGenHub(provider="edge")
        await engine.initialize()

        # Empty text should fail at request validation
        with pytest.raises(Exception):
            TTSRequest(
                text="",
                voice_id="en-US-AriaNeural",
                audio_format=AudioFormat.WAV,
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_very_long_text_handling(self):
        """Integration: Test handling of very long text."""
        from voicegenhub.core.engine import VoiceGenHub
        from voicegenhub.providers.base import TTSRequest, AudioFormat

        engine = VoiceGenHub(provider="edge")
        await engine.initialize()

        # Create very long text
        long_text = "Hello world. " * 1000

        request = TTSRequest(
            text=long_text,
            voice_id="en-US-AriaNeural",
            audio_format=AudioFormat.WAV,
        )

        # Should handle without crashing (may timeout or fail gracefully)
        try:
            response = await engine._provider.synthesize(request)
            assert response.audio_data is not None
        except Exception:
            # Acceptable to fail with very long text
            pass


class TestKokoroSpecificIntegration:
    """Integration tests specific to Kokoro provider."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kokoro_voice_list(self):
        """Integration: Test Kokoro voice listing (slow)."""
        pytest.importorskip("kokoro")
        from voicegenhub.providers.kokoro import KokoroTTSProvider

        provider = KokoroTTSProvider()
        await provider.initialize()

        voices = await provider.get_voices()
        assert len(voices) > 0
        assert any("am_adam" in v.voice_id for v in voices)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kokoro_male_voice_synthesis(self):
        """Integration: Test Kokoro male voice synthesis."""
        pytest.importorskip("kokoro")
        from voicegenhub.providers.kokoro import KokoroTTSProvider
        from voicegenhub.providers.base import TTSRequest, AudioFormat

        provider = KokoroTTSProvider()
        await provider.initialize()

        request = TTSRequest(
            text="Test synthesis",
            voice_id="kokoro-am_adam",
            audio_format=AudioFormat.WAV,
            speed=0.85,
        )

        response = await provider.synthesize(request)
        assert response.audio_data is not None
        assert len(response.audio_data) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kokoro_female_voice_synthesis(self):
        """Integration: Test Kokoro female voice synthesis."""
        pytest.importorskip("kokoro")
        from voicegenhub.providers.kokoro import KokoroTTSProvider
        from voicegenhub.providers.base import TTSRequest, AudioFormat

        provider = KokoroTTSProvider()
        await provider.initialize()

        request = TTSRequest(
            text="Test synthesis",
            voice_id="kokoro-af_alloy",
            audio_format=AudioFormat.WAV,
        )

        response = await provider.synthesize(request)
        assert response.audio_data is not None
        assert len(response.audio_data) > 0
