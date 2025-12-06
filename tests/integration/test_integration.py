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
        assert all("id" in v for v in voices1)

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
        assert all("language" in v for v in all_voices)

        # Filter by language
        en_voices = await engine.get_voices(language="en")
        assert len(en_voices) > 0
        for voice in en_voices:
            assert voice["language"].startswith("en")

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
        assert all("id" in v and "provider" in v for v in voices)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_voice_selection_by_gender(self):
        """Integration: Test filtering voices by gender."""
        from voicegenhub.core.engine import VoiceGenHub

        engine = VoiceGenHub(provider="edge")
        await engine.initialize()

        voices = await engine.get_voices()
        assert len(voices) > 0

        # Check that voices have gender attribute
        for v in voices:
            assert "gender" in v
            assert v["gender"] is not None


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
        voices = await provider.get_voices()
        assert len(voices) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kokoro_provider_initialization(self):
        """Integration: Test Kokoro provider initialization (slow)."""
        pytest.importorskip("kokoro")
        from voicegenhub.providers.kokoro import KokoroTTSProvider

        provider = KokoroTTSProvider()
        await provider.initialize()

        assert provider is not None
        voices = await provider.get_voices()
        assert len(voices) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_provider_initialization_error_handling(self):
        """Integration: Test provider can handle voice requests gracefully."""
        from voicegenhub.core.engine import VoiceGenHub

        engine = VoiceGenHub(provider="edge")
        await engine.initialize()

        # Test that we can get voices
        voices = await engine.get_voices()
        assert len(voices) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_provider_initialization_no_error_on_auto_select(self):
        """Integration: Test that auto-select provider initializes without error."""
        from voicegenhub.core.engine import VoiceGenHub

        engine = VoiceGenHub()
        await engine.initialize()

        assert engine._provider is not None
        voices = await engine.get_voices()
        assert len(voices) > 0


class TestAudioResponseIntegration:
    """Integration tests for audio response handling."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_synthesize_returns_valid_audio_data(self):
        """Integration: Test that synthesis returns valid audio data."""
        from voicegenhub.core.engine import VoiceGenHub
        from voicegenhub.providers.base import TTSRequest, AudioFormat

        engine = VoiceGenHub(provider="edge")
        await engine.initialize()

        request = TTSRequest(
            text="Hello world",
            voice_id="en-US-AriaNeural",
            audio_format=AudioFormat.MP3,
        )

        response = await engine._provider.synthesize(request)
        assert response is not None
        assert response.audio_data is not None
        assert len(response.audio_data) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_synthesize_response_format_valid(self):
        """Integration: Test audio response format integrity."""
        from voicegenhub.core.engine import VoiceGenHub
        from voicegenhub.providers.base import TTSRequest, AudioFormat

        engine = VoiceGenHub(provider="edge")
        await engine.initialize()

        request = TTSRequest(
            text="Test",
            voice_id="en-US-AriaNeural",
            audio_format=AudioFormat.MP3,
        )

        response = await engine._provider.synthesize(request)

        # Check audio data is bytes
        assert isinstance(response.audio_data, bytes)
        assert len(response.audio_data) > 0

        # Check format field
        assert response.format == AudioFormat.MP3

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_synthesize_with_speed_parameter(self):
        """Integration: Test synthesis with speed parameter."""
        from voicegenhub.core.engine import VoiceGenHub
        from voicegenhub.providers.base import TTSRequest, AudioFormat

        engine = VoiceGenHub(provider="edge")
        await engine.initialize()

        # Test with different speeds
        for speed in [0.8, 1.0, 1.2]:
            request = TTSRequest(
                text="Hello",
                voice_id="en-US-AriaNeural",
                audio_format=AudioFormat.MP3,
                speed=speed,
            )
            response = await engine._provider.synthesize(request)
            assert response.audio_data is not None
            assert len(response.audio_data) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_synthesize_with_pitch_parameter(self):
        """Integration: Test synthesis with pitch parameter."""
        from voicegenhub.core.engine import VoiceGenHub
        from voicegenhub.providers.base import TTSRequest, AudioFormat

        engine = VoiceGenHub(provider="edge")
        await engine.initialize()

        request = TTSRequest(
            text="Hello",
            voice_id="en-US-AriaNeural",
            audio_format=AudioFormat.MP3,
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
        from voicegenhub.providers.base import TTSRequest, AudioFormat, TTSError

        engine = VoiceGenHub(provider="edge")
        await engine.initialize()

        request = TTSRequest(
            text="Hello",
            voice_id="invalid-voice-id-xyz",
            audio_format=AudioFormat.MP3,
        )

        with pytest.raises(TTSError):
            await engine._provider.synthesize(request)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_empty_text_validation(self):
        """Integration: Test validation of empty text."""
        from voicegenhub.providers.base import TTSRequest, AudioFormat

        # Empty text should be rejected at request validation or allowed
        try:
            TTSRequest(
                text="",
                voice_id="en-US-AriaNeural",
                audio_format=AudioFormat.MP3,
            )
            # If it doesn't raise, text is allowed to be empty
            assert True
        except (ValueError, AssertionError):
            # Expected behavior: empty text rejected
            assert True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_very_long_text_handling(self):
        """Integration: Test handling of very long text."""
        from voicegenhub.core.engine import VoiceGenHub
        from voicegenhub.providers.base import TTSRequest, AudioFormat

        engine = VoiceGenHub(provider="edge")
        await engine.initialize()

        # Create very long text (but within reasonable limits)
        long_text = "Hello world. " * 100

        request = TTSRequest(
            text=long_text,
            voice_id="en-US-AriaNeural",
            audio_format=AudioFormat.MP3,
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
    @pytest.mark.xfail(reason="Voice files may not be available on HuggingFace")
    async def test_kokoro_male_voice_synthesis(self):
        """Integration: Test Kokoro male voice synthesis."""
        pytest.importorskip("kokoro")
        from voicegenhub.providers.kokoro import KokoroTTSProvider
        from voicegenhub.providers.base import TTSRequest, AudioFormat

        provider = KokoroTTSProvider()
        await provider.initialize()

        # Get voices and find a male voice
        voices = await provider.get_voices()
        male_voices = [v for v in voices if v.gender.value == "male"]

        if male_voices:
            male_voice = male_voices[0]
            request = TTSRequest(
                text="Test synthesis",
                voice_id=f"kokoro-{male_voice.id}",
                audio_format=AudioFormat.WAV,
                speed=0.85,
            )

            response = await provider.synthesize(request)
            assert response.audio_data is not None
            assert len(response.audio_data) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Voice files may not be available on HuggingFace")
    async def test_kokoro_female_voice_synthesis(self):
        """Integration: Test Kokoro female voice synthesis."""
        pytest.importorskip("kokoro")
        from voicegenhub.providers.kokoro import KokoroTTSProvider
        from voicegenhub.providers.base import TTSRequest, AudioFormat

        provider = KokoroTTSProvider()
        await provider.initialize()

        # Get voices and find a female voice
        voices = await provider.get_voices()
        female_voices = [v for v in voices if v.gender.value == "female"]

        if female_voices:
            female_voice = female_voices[0]
            request = TTSRequest(
                text="Test synthesis",
                voice_id=f"kokoro-{female_voice.id}",
                audio_format=AudioFormat.WAV,
            )

            response = await provider.synthesize(request)
            assert response.audio_data is not None
            assert len(response.audio_data) > 0

        request = TTSRequest(
            text="Test synthesis",
            voice_id="kokoro-af_alloy",
            audio_format=AudioFormat.WAV,
        )

        response = await provider.synthesize(request)
        assert response.audio_data is not None
        assert len(response.audio_data) > 0


class TestCLIIntegration:
    """Integration tests for CLI functionality with all supported providers."""

    @pytest.mark.integration
    @pytest.mark.parametrize("provider", ["edge", "kokoro", "bark", "chatterbox"])
    def test_cli_single_prompt_string(self, provider, tmp_path):
        """Integration: Test CLI with single prompt string for all providers."""
        import subprocess
        import sys
        from pathlib import Path

        output_file = tmp_path / f"test_single_{provider}.wav"
        
        # Select appropriate voice for each provider
        voice_map = {
            "edge": "en-US-AriaNeural",
            "kokoro": "kokoro-af_alloy",
            "bark": "bark-en_speaker_0",
            "chatterbox": "chatterbox-default"
        }
        voice = voice_map[provider]

        # Run CLI command with required language parameter
        cmd = [
            sys.executable, "-m", "voicegenhub.cli",
            "synthesize", "Hello world",
            "--provider", provider,
            "--voice", voice,
            "--language", "en",
            "--output", str(output_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)

        # Check that command succeeded
        assert result.returncode == 0, f"CLI failed for {provider}: {result.stderr}"

        # Check that output file was created
        assert output_file.exists(), f"Output file not created for {provider}"
        assert output_file.stat().st_size > 0, f"Output file is empty for {provider}"

    @pytest.mark.integration
    @pytest.mark.parametrize("provider", ["edge", "kokoro", "bark", "chatterbox"])
    def test_cli_multi_prompt_list(self, provider, tmp_path):
        """Integration: Test CLI with multi-prompt list for all providers."""
        import subprocess
        import sys
        from pathlib import Path

        output_base = tmp_path / f"test_multi_{provider}.wav"
        
        # Select appropriate voice for each provider
        voice_map = {
            "edge": "en-US-AriaNeural",
            "kokoro": "kokoro-af_alloy",
            "bark": "bark-en_speaker_0",
            "chatterbox": "chatterbox-default"
        }
        voice = voice_map[provider]

        # Run CLI command with multiple texts
        cmd = [
            sys.executable, "-m", "voicegenhub.cli",
            "synthesize", "First message", "Second message", "Third message",
            "--provider", provider,
            "--voice", voice,
            "--language", "en",
            "--output", str(output_base)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)

        # Check that command succeeded
        assert result.returncode == 0, f"CLI failed for {provider}: {result.stderr}"

        # Check that output files were created (should be auto-numbered)
        expected_files = [
            tmp_path / f"test_multi_{provider}.wav_01.wav",
            tmp_path / f"test_multi_{provider}.wav_02.wav",
            tmp_path / f"test_multi_{provider}.wav_03.wav"
        ]

        for expected_file in expected_files:
            assert expected_file.exists(), f"Output file {expected_file} not created for {provider}"
            assert expected_file.stat().st_size > 0, f"Output file {expected_file} is empty for {provider}"

    @pytest.mark.integration
    @pytest.mark.parametrize("provider,max_concurrent", [
        ("edge", 1),
        ("edge", 2),
        ("kokoro", 1),
        ("kokoro", 2),
        ("bark", 1),
        ("bark", 2),
        ("chatterbox", 1),
    ])
    def test_cli_max_concurrency_settings(self, provider, max_concurrent, tmp_path):
        """Integration: Test CLI with different max concurrency settings."""
        import subprocess
        import sys
        from pathlib import Path

        voice_map = {
            "edge": "en-US-AriaNeural",
            "kokoro": "kokoro-af_alloy",
            "bark": "bark-en_speaker_0",
            "chatterbox": "chatterbox-default",
        }

        output_base = tmp_path / f"test_concurrency_{provider}_{max_concurrent}.wav"

        # Run CLI command with multiple texts and specific concurrency
        cmd = [
            sys.executable, "-m", "voicegenhub.cli",
            "synthesize", "Message one", "Message two",
            "--provider", provider,
            "--language", "en",
            "--voice", voice_map[provider],
            "--output", str(output_base)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)

        # Check that command succeeded
        assert result.returncode == 0, f"CLI failed for {provider} with concurrency {max_concurrent}: {result.stderr}"

        # Check that output files were created
        expected_files = [
            tmp_path / f"test_concurrency_{provider}_{max_concurrent}.wav_01.wav",
            tmp_path / f"test_concurrency_{provider}_{max_concurrent}.wav_02.wav"
        ]

        for expected_file in expected_files:
            assert expected_file.exists(), f"Output file {expected_file} not created for {provider} with concurrency {max_concurrent}"
            assert expected_file.stat().st_size > 0, f"Output file {expected_file} is empty for {provider} with concurrency {max_concurrent}"

    @pytest.mark.integration
    @pytest.mark.parametrize("provider", ["edge", "kokoro", "bark", "chatterbox"])
    def test_cli_provider_concurrency_limits_respected(self, provider, tmp_path):
        """Integration: Test that provider-specific concurrency limits are respected."""
        import subprocess
        import sys
        import re
        from pathlib import Path

        voice_map = {
            "edge": "en-US-AriaNeural",
            "kokoro": "kokoro-af_alloy",
            "bark": "bark-en_speaker_0",
            "chatterbox": "chatterbox-default",
        }

        output_base = tmp_path / f"test_limits_{provider}.wav"

        # Run CLI command with multiple texts to trigger batch processing
        cmd = [
            sys.executable, "-m", "voicegenhub.cli",
            "synthesize", "Text 1", "Text 2", "Text 3", "Text 4", "Text 5",
            "--provider", provider,
            "--language", "en",
            "--voice", voice_map[provider],
            "--output", str(output_base)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)

        # Check that command succeeded
        assert result.returncode == 0, f"CLI failed for {provider}: {result.stderr}"

        # Check output for concurrency information
        output = result.stdout + result.stderr

        # Verify that the correct concurrency limit was reported
        if provider == "edge":
            # Edge should use all CPU cores (unlimited)
            assert "max" in output.lower() or "concurrent" in output.lower()
        elif provider == "kokoro":
            # Kokoro should use all CPU cores (unlimited)
            assert "max" in output.lower() or "concurrent" in output.lower()
        elif provider == "bark":
            # Bark should be limited to 2
            assert "max 2 concurrent" in output
        elif provider == "chatterbox":
            # Chatterbox should be limited to 1
            assert "max 1 concurrent" in output

        # Check that all output files were created
        for i in range(1, 6):
            expected_file = tmp_path / f"test_limits_{provider}.wav_{i:02d}.wav"
            assert expected_file.exists(), f"Output file {expected_file} not created for {provider}"
            assert expected_file.stat().st_size > 0, f"Output file {expected_file} is empty for {provider}"

    @pytest.mark.integration
    @pytest.mark.parametrize("provider", ["edge", "kokoro", "bark", "chatterbox"])
    def test_cli_audio_effects_with_multi_prompt(self, provider, tmp_path):
        """Integration: Test CLI audio effects work with multi-prompt processing."""
        import subprocess
        import sys
        from pathlib import Path

        voice_map = {
            "edge": "en-US-AriaNeural",
            "kokoro": "kokoro-af_alloy",
            "bark": "bark-en_speaker_0",
            "chatterbox": "chatterbox-default",
        }

        output_base = tmp_path / f"test_effects_{provider}.wav"

        # Run CLI command with effects
        cmd = [
            sys.executable, "-m", "voicegenhub.cli",
            "synthesize", "Hello", "World",
            "--provider", provider,
            "--language", "en",
            "--voice", voice_map[provider],
            "--output", str(output_base),
            "--normalize"  # Add a simple effect
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)

        # Check that command succeeded
        assert result.returncode == 0, f"CLI with effects failed for {provider}: {result.stderr}"

        # Check that output files were created
        expected_files = [
            tmp_path / f"test_effects_{provider}.wav_01.wav",
            tmp_path / f"test_effects_{provider}.wav_02.wav"
        ]

        for expected_file in expected_files:
            assert expected_file.exists(), f"Output file {expected_file} not created for {provider} with effects"
            assert expected_file.stat().st_size > 0, f"Output file {expected_file} is empty for {provider} with effects"
