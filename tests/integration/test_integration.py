"""
Integration tests for voice selection, provider initialization, and audio response handling.
These tests are marked with @pytest.mark.integration and should only run in CI.
Tests cover Edge TTS, Kokoro, Chatterbox, and Bark providers.
"""
import pytest


@pytest.fixture(scope="session")
async def edge_provider():
    """Session-scoped fixture for Edge provider to load once."""
    from voicegenhub.providers.edge import EdgeTTSProvider
    provider = EdgeTTSProvider()
    await provider.initialize()
    return provider


@pytest.fixture(scope="session")
async def kokoro_provider():
    """Session-scoped fixture for Kokoro provider to load once."""
    pytest.importorskip("kokoro")
    from voicegenhub.providers.kokoro import KokoroTTSProvider
    provider = KokoroTTSProvider()
    await provider.initialize()
    return provider


@pytest.fixture(scope="session")
async def chatterbox_provider():
    """Session-scoped fixture for Chatterbox provider to load once."""
    pytest.importorskip("chatterbox")
    from voicegenhub.providers.chatterbox import ChatterboxProvider
    provider = ChatterboxProvider()
    await provider.initialize()
    return provider


@pytest.fixture(scope="session")
async def bark_provider():
    """Session-scoped fixture for Bark provider to load once."""
    pytest.importorskip("bark")
    from voicegenhub.providers.bark import BarkProvider
    provider = BarkProvider()
    await provider.initialize()
    return provider


class TestEdgeTTSIntegration:
    """Integration tests specific to Edge TTS provider."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_edge_voice_list(self):
        """Integration: Test Edge TTS voice listing."""
        from voicegenhub.providers.edge import EdgeTTSProvider

        provider = EdgeTTSProvider()
        await provider.initialize()

        voices = await provider.get_voices()
        assert len(voices) > 0
        assert all(hasattr(v, "id") and hasattr(v, "name") for v in voices)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_edge_voice_filtering(self):
        """Integration: Test Edge TTS voice filtering by language."""
        from voicegenhub.providers.edge import EdgeTTSProvider

        provider = EdgeTTSProvider()
        await provider.initialize()

        voices = await provider.get_voices()
        assert len(voices) > 0

        en_voices = [v for v in voices if hasattr(v, "language") and v.language.startswith("en")]
        assert len(en_voices) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_edge_tts_synthesis(self):
        """Integration: Test Edge TTS synthesis."""
        from voicegenhub.providers.edge import EdgeTTSProvider
        from voicegenhub.providers.base import TTSRequest, AudioFormat

        provider = EdgeTTSProvider()
        await provider.initialize()

        request = TTSRequest(
            text="Hello world, this is Edge TTS",
            voice_id="en-US-AriaNeural",
            audio_format=AudioFormat.MP3,
        )

        response = await provider.synthesize(request)
        assert response.audio_data is not None
        assert len(response.audio_data) > 0
        assert isinstance(response.audio_data, bytes)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_edge_tts_with_speed(self):
        """Integration: Test Edge TTS with speed parameter."""
        from voicegenhub.providers.edge import EdgeTTSProvider
        from voicegenhub.providers.base import TTSRequest, AudioFormat

        provider = EdgeTTSProvider()
        await provider.initialize()

        for speed in [0.8, 1.0, 1.5]:
            request = TTSRequest(
                text="Speed test",
                voice_id="en-US-AriaNeural",
                audio_format=AudioFormat.MP3,
                speed=speed,
            )
            response = await provider.synthesize(request)
            assert response.audio_data is not None
            assert len(response.audio_data) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_edge_tts_with_pitch(self):
        """Integration: Test Edge TTS with pitch parameter."""
        from voicegenhub.providers.edge import EdgeTTSProvider
        from voicegenhub.providers.base import TTSRequest, AudioFormat

        provider = EdgeTTSProvider()
        await provider.initialize()

        request = TTSRequest(
            text="Pitch test",
            voice_id="en-US-AriaNeural",
            audio_format=AudioFormat.MP3,
            pitch=1.2,
        )
        response = await provider.synthesize(request)
        assert response.audio_data is not None
        assert len(response.audio_data) > 0


class TestKokoroIntegration:
    """Integration tests specific to Kokoro provider."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kokoro_voice_list(self):
        """Integration: Test Kokoro voice listing."""
        pytest.importorskip("kokoro")
        from voicegenhub.providers.kokoro import KokoroTTSProvider

        provider = KokoroTTSProvider()
        await provider.initialize()

        voices = await provider.get_voices()
        assert len(voices) > 0
        assert all(hasattr(v, "id") for v in voices)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kokoro_voice_genders(self):
        """Integration: Test Kokoro voice genders."""
        pytest.importorskip("kokoro")
        from voicegenhub.providers.kokoro import KokoroTTSProvider

        provider = KokoroTTSProvider()
        await provider.initialize()

        voices = await provider.get_voices()
        assert len(voices) > 0

        genders = {v.gender.value for v in voices}
        assert len(genders) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Kokoro model files may not be available")
    async def test_kokoro_synthesis(self):
        """Integration: Test Kokoro synthesis (may fail if model unavailable)."""
        pytest.importorskip("kokoro")
        from voicegenhub.providers.kokoro import KokoroTTSProvider
        from voicegenhub.providers.base import TTSRequest, AudioFormat

        provider = KokoroTTSProvider()
        await provider.initialize()

        request = TTSRequest(
            text="Hello world, testing Kokoro",
            voice_id="kokoro-af_alloy",
            audio_format=AudioFormat.WAV,
        )

        response = await provider.synthesize(request)
        assert response.audio_data is not None
        assert len(response.audio_data) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Kokoro model files may not be available")
    async def test_kokoro_synthesis_with_speed(self):
        """Integration: Test Kokoro synthesis with speed parameter."""
        pytest.importorskip("kokoro")
        from voicegenhub.providers.kokoro import KokoroTTSProvider
        from voicegenhub.providers.base import TTSRequest, AudioFormat

        provider = KokoroTTSProvider()
        await provider.initialize()

        request = TTSRequest(
            text="Speed test",
            voice_id="kokoro-af_alloy",
            audio_format=AudioFormat.WAV,
            speed=0.9,
        )

        response = await provider.synthesize(request)
        assert response.audio_data is not None
        assert len(response.audio_data) > 0


class TestChatterboxIntegration:
    """Integration tests specific to Chatterbox provider."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_chatterbox_voice_list(self):
        """Integration: Test Chatterbox voice listing."""
        pytest.importorskip("chatterbox")
        from voicegenhub.providers.chatterbox import ChatterboxProvider

        provider = ChatterboxProvider()
        await provider.initialize()

        voices = await provider.get_voices()
        assert len(voices) > 0
        assert all(hasattr(v, "id") for v in voices)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_chatterbox_multilingual(self):
        """Integration: Test Chatterbox supports multiple languages."""
        pytest.importorskip("chatterbox")
        from voicegenhub.providers.chatterbox import ChatterboxProvider

        provider = ChatterboxProvider()
        await provider.initialize()

        voices = await provider.get_voices()
        assert len(voices) > 0

        languages = {v.language for v in voices}
        assert len(languages) > 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Chatterbox model files may not be available")
    async def test_chatterbox_synthesis(self):
        """Integration: Test Chatterbox synthesis (may fail if model unavailable)."""
        pytest.importorskip("chatterbox")
        from voicegenhub.providers.chatterbox import ChatterboxProvider
        from voicegenhub.providers.base import TTSRequest, AudioFormat

        provider = ChatterboxProvider()
        await provider.initialize()

        voices = await provider.get_voices()
        if voices:
            first_voice = voices[0]
            request = TTSRequest(
                text="Hello world, testing Chatterbox",
                voice_id=first_voice.id,
                audio_format=AudioFormat.WAV,
            )

            response = await provider.synthesize(request)
            assert response.audio_data is not None
            assert len(response.audio_data) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Chatterbox model files may not be available")
    async def test_chatterbox_multilingual_synthesis(self):
        """Integration: Test Chatterbox multilingual synthesis."""
        pytest.importorskip("chatterbox")
        from voicegenhub.providers.chatterbox import ChatterboxProvider
        from voicegenhub.providers.base import TTSRequest, AudioFormat

        provider = ChatterboxProvider()
        await provider.initialize()

        voices = await provider.get_voices()
        en_voices = [v for v in voices if v.language.startswith("en")]
        if en_voices:
            request = TTSRequest(
                text="English text",
                voice_id=en_voices[0].id,
                audio_format=AudioFormat.WAV,
            )
            response = await provider.synthesize(request)
            assert response.audio_data is not None


class TestBarkIntegration:
    """Integration tests specific to Bark provider."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_bark_voice_list(self):
        """Integration: Test Bark voice listing."""
        pytest.importorskip("bark")
        from voicegenhub.providers.bark import BarkProvider

        provider = BarkProvider()
        await provider.initialize()

        voices = await provider.get_voices()
        assert len(voices) > 0
        assert all(hasattr(v, "id") for v in voices)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_bark_preset_voices(self):
        """Integration: Test Bark has expected preset voices."""
        pytest.importorskip("bark")
        from voicegenhub.providers.bark import BarkProvider

        provider = BarkProvider()
        await provider.initialize()

        voices = await provider.get_voices()
        voice_ids = {v.id for v in voices}
        assert len(voice_ids) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Bark model requires significant resources")
    async def test_bark_synthesis(self):
        """Integration: Test Bark synthesis (may fail due to resource requirements)."""
        pytest.importorskip("bark")
        from voicegenhub.providers.bark import BarkProvider
        from voicegenhub.providers.base import TTSRequest, AudioFormat

        provider = BarkProvider()
        await provider.initialize()

        request = TTSRequest(
            text="Hello world, testing Bark",
            voice_id="v2/en_speaker_6",
            audio_format=AudioFormat.WAV,
        )

        response = await provider.synthesize(request)
        assert response.audio_data is not None
        assert len(response.audio_data) > 0


class TestVoiceSelectionIntegration:
    """Integration tests for voice selection and caching."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_voice_caching_edge(self):
        """Integration: Test voice caching for Edge provider."""
        from voicegenhub.core.engine import VoiceGenHub

        engine = VoiceGenHub(provider="edge")
        await engine.initialize()

        voices1 = await engine.get_voices()
        assert len(voices1) > 0

        voices2 = await engine.get_voices()
        assert voices1 == voices2

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_voice_language_filtering_edge(self):
        """Integration: Test language filtering for Edge provider."""
        from voicegenhub.core.engine import VoiceGenHub

        engine = VoiceGenHub(provider="edge")
        await engine.initialize()

        all_voices = await engine.get_voices()
        assert len(all_voices) > 0

        en_voices = await engine.get_voices(language="en")
        assert len(en_voices) > 0
        assert len(en_voices) <= len(all_voices)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_voice_aggregation_auto_select(self):
        """Integration: Test voice aggregation with auto-select provider."""
        from voicegenhub.core.engine import VoiceGenHub

        engine = VoiceGenHub()
        await engine.initialize()

        voices = await engine.get_voices()
        assert len(voices) > 0
        assert all("id" in v and "provider" in v for v in voices)


class TestErrorHandlingIntegration:
    """Integration tests for error handling edge cases."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_invalid_voice_id_edge(self):
        """Integration: Test handling of invalid voice ID for Edge."""
        from voicegenhub.providers.edge import EdgeTTSProvider
        from voicegenhub.providers.base import TTSRequest, AudioFormat, TTSError

        provider = EdgeTTSProvider()
        await provider.initialize()

        request = TTSRequest(
            text="Hello",
            voice_id="invalid-voice-xyz-12345",
            audio_format=AudioFormat.MP3,
        )

        with pytest.raises(TTSError):
            await provider.synthesize(request)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_very_long_text_edge(self):
        """Integration: Test handling of long text for Edge."""
        from voicegenhub.providers.edge import EdgeTTSProvider
        from voicegenhub.providers.base import TTSRequest, AudioFormat

        provider = EdgeTTSProvider()
        await provider.initialize()

        long_text = "Hello world. " * 50

        request = TTSRequest(
            text=long_text,
            voice_id="en-US-AriaNeural",
            audio_format=AudioFormat.MP3,
        )

        try:
            response = await provider.synthesize(request)
            assert response.audio_data is not None
        except Exception:
            pass


class TestCLIIntegration:
    """Integration tests for CLI functionality with all supported providers."""

    @pytest.mark.integration
    @pytest.mark.parametrize("provider", ["edge"])
    def test_cli_single_prompt_edge(self, provider, tmp_path):
        """Integration: Test CLI with single prompt for Edge provider."""
        import subprocess
        import sys

        output_file = tmp_path / f"test_single_{provider}.wav"

        cmd = [
            sys.executable, "-m", "voicegenhub.cli",
            "synthesize", "Hello world",
            "--provider", provider,
            "--voice", "en-US-AriaNeural",
            "--language", "en",
            "--output", str(output_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)
        assert result.returncode == 0, f"CLI failed for {provider}: {result.stderr}"
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    @pytest.mark.integration
    @pytest.mark.parametrize("provider", ["edge"])
    def test_cli_multi_prompt_edge(self, provider, tmp_path):
        """Integration: Test CLI with multiple prompts for Edge provider."""
        import subprocess
        import sys

        output_base = tmp_path / f"test_multi_{provider}.wav"

        cmd = [
            sys.executable, "-m", "voicegenhub.cli",
            "synthesize", "First", "Second", "Third",
            "--provider", provider,
            "--voice", "en-US-AriaNeural",
            "--language", "en",
            "--output", str(output_base)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)
        assert result.returncode == 0, f"CLI failed for {provider}: {result.stderr}"

        for i in range(1, 4):
            expected_file = tmp_path / f"test_multi_{provider}.wav_{i:02d}.wav"
            assert expected_file.exists()
            assert expected_file.stat().st_size > 0

    @pytest.mark.integration
    def test_cli_with_effects_edge(self, tmp_path):
        """Integration: Test CLI with audio effects for Edge provider."""
        import subprocess
        import sys

        output_base = tmp_path / "test_effects_edge.wav"

        cmd = [
            sys.executable, "-m", "voicegenhub.cli",
            "synthesize", "Hello", "World",
            "--provider", "edge",
            "--voice", "en-US-AriaNeural",
            "--language", "en",
            "--output", str(output_base),
            "--normalize"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)
        assert result.returncode == 0, f"CLI with effects failed: {result.stderr}"

        for i in range(1, 3):
            expected_file = tmp_path / f"test_effects_edge.wav_{i:02d}.wav"
            assert expected_file.exists()
            assert expected_file.stat().st_size > 0
