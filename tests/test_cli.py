"""Unit tests for VoiceGenHub CLI."""
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from voicegenhub.cli import cli


class TestCLI:
    """Test CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "VoiceGenHub" in result.output
        assert "synthesize" in result.output
        assert "voices" in result.output

    def test_cli_synthesize_help(self, runner):
        """Test synthesize command help."""
        result = runner.invoke(cli, ["synthesize", "--help"])
        assert result.exit_code == 0
        assert "Generate speech from text" in result.output
        assert "--voice" in result.output
        assert "--provider" in result.output
        assert "--rate" in result.output
        assert "--pitch" in result.output

    def test_cli_voices_help(self, runner):
        """Test voices command help."""
        result = runner.invoke(cli, ["voices", "--help"])
        assert result.exit_code == 0
        assert "List available voices" in result.output
        assert "--provider" in result.output
        assert "--language" in result.output

    def test_cli_rejects_unsupported_provider_synthesize(self, runner):
        """Test CLI rejects unsupported provider in synthesize command."""
        result = runner.invoke(
            cli, ["synthesize", "hello world", "--provider", "coqui"]
        )
        assert result.exit_code == 1
        assert "Unsupported provider 'coqui'" in result.output
        assert "edge, google, piper, melotts, kokoro" in result.output

    def test_cli_rejects_unsupported_provider_voices(self, runner):
        """Test CLI rejects unsupported provider in voices command."""
        result = runner.invoke(cli, ["voices", "--provider", "coqui"])
        assert result.exit_code == 1
        assert "Unsupported provider 'coqui'" in result.output
        assert "edge, google, piper, melotts, kokoro" in result.output

    def test_cli_accepts_supported_providers_synthesize(self, runner, tmp_path):
        """Test CLI accepts supported providers in synthesize command."""
        for provider in ["edge", "google", "piper", "melotts", "kokoro"]:
            result = runner.invoke(cli, ["synthesize", "hello", "--provider", provider, "--output", str(tmp_path / "dummy.wav")])
            # Should fail due to provider issues, not validation
            assert "Unsupported provider" not in result.output

    def test_cli_accepts_supported_providers_voices(self, runner):
        """Test CLI accepts supported providers in voices command."""
        for provider in ["edge", "google", "piper", "melotts", "kokoro"]:
            result = runner.invoke(cli, ["voices", "--provider", provider])
            # Should fail due to provider issues, not validation
            assert "Unsupported provider" not in result.output

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_cli_synthesize_basic_call(self, mock_tts_class, runner, tmp_path):
        """Test synthesize command makes correct calls."""
        # Mock the TTS instance
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake_audio")
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(
            cli, ["synthesize", "hello world", "--voice", "en-US-AriaNeural", "--output", str(tmp_path / "speech.wav")]
        )
        assert result.exit_code == 0

        # Verify TTS was called correctly
        mock_tts_class.assert_called_once_with(provider=None)  # Auto-select provider
        mock_tts.generate.assert_called_once()
        call_args = mock_tts.generate.call_args
        assert call_args[1]["text"] == "hello world"
        assert call_args[1]["voice"] == "en-US-AriaNeural"
        assert call_args[1]["speed"] == 1.0  # default rate
        assert call_args[1]["pitch"] == 1.0  # default pitch

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_cli_synthesize_with_params(self, mock_tts_class, runner, tmp_path):
        """Test synthesize command with custom parameters."""
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake_audio")
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(
            cli,
            [
                "synthesize",
                "hello world",
                "--voice",
                "en-US-AriaNeural",
                "--provider",
                "edge",
                "--rate",
                "1.5",
                "--pitch",
                "0.8",
                "--format",
                "wav",
                "--output",
                str(tmp_path / "test.wav"),
            ],
        )
        assert result.exit_code == 0

        # Verify parameters were passed correctly
        mock_tts_class.assert_called_once_with(provider="edge")
        call_args = mock_tts.generate.call_args
        assert call_args[1]["speed"] == 1.5
        assert call_args[1]["pitch"] == 0.8
        assert call_args[1]["audio_format"].value == "wav"

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_cli_voices_basic_call(self, mock_tts_class, runner):
        """Test voices command makes correct calls."""
        mock_tts = AsyncMock()
        mock_tts.get_voices.return_value = [
            {
                "id": "en-US-AriaNeural",
                "name": "Aria",
                "language": "en",
                "locale": "en-US",
                "gender": "female",
            }
        ]
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(cli, ["voices"])
        assert result.exit_code == 0
        assert "en-US-AriaNeural" in result.output
        assert "Aria" in result.output

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_cli_voices_json_format(self, mock_tts_class, runner):
        """Test voices command with JSON output."""
        mock_tts = AsyncMock()
        mock_tts.get_voices.return_value = [
            {
                "id": "en-US-AriaNeural",
                "name": "Aria",
                "language": "en",
                "locale": "en-US",
                "gender": "female",
            }
        ]
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(cli, ["voices", "--format", "json"])
        assert result.exit_code == 0
        assert '"id": "en-US-AriaNeural"' in result.output
        assert '"language": "en-US"' in result.output

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_cli_voices_with_provider(self, mock_tts_class, runner):
        """Test voices command with specific provider."""
        mock_tts = AsyncMock()
        mock_tts.get_voices.return_value = [
            {
                "id": "en-US-AriaNeural",
                "name": "Aria",
                "language": "en",
                "locale": "en-US",
                "gender": "female",
            }
        ]
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(cli, ["voices", "--provider", "edge"])
        assert result.exit_code == 0
        mock_tts_class.assert_called_once_with(provider="edge")

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_cli_synthesize_error_handling(self, mock_tts_class, runner):
        """Test synthesize command error handling."""
        mock_tts = AsyncMock()
        mock_tts.generate.side_effect = Exception("Test error")
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(cli, ["synthesize", "hello world"])
        assert result.exit_code == 1
        assert "Error: Test error" in result.output

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_cli_voices_error_handling(self, mock_tts_class, runner):
        """Test voices command error handling."""
        mock_tts = AsyncMock()
        mock_tts.get_voices.side_effect = Exception("Test error")
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(cli, ["voices"])
        assert result.exit_code == 1
        assert "Error: Test error" in result.output


class TestCLIVoiceNotFoundErrors:
    """Test CLI voice not found error messages with suggestions."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_voice_not_found_english_suggestions(self, mock_tts_class, runner):
        """Test voice not found shows English voice suggestions."""
        from voicegenhub.providers.base import VoiceNotFoundError

        mock_tts = AsyncMock()
        # Mock available voices with English voices
        mock_tts._voice_selector.get_all_voices.return_value = [
            type(
                "Voice",
                (),
                {
                    "id": "en-US-AriaNeural",
                    "name": "Aria Neural",
                    "language": "en",
                    "locale": "en-US",
                    "gender": type("Gender", (), {"value": "female"})(),
                },
            )(),
            type(
                "Voice",
                (),
                {
                    "id": "en-GB-SoniaNeural",
                    "name": "Sonia Neural",
                    "language": "en",
                    "locale": "en-GB",
                    "gender": type("Gender", (), {"value": "female"})(),
                },
            )(),
        ]
        # Mock generate to raise VoiceNotFoundError with suggestions
        mock_tts.generate.side_effect = VoiceNotFoundError(
            "Voice 'en-US-NonExistentVoice' not found\n\nAvailable en voices:\n  en-US-AriaNeural - Aria Neural\n  en-GB-SoniaNeural - Sonia Neural",
            error_code="VOICE_NOT_FOUND",
            provider="edge",
        )
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(
            cli, ["synthesize", "hello", "--voice", "en-US-NonExistentVoice"]
        )
        assert result.exit_code == 1
        assert "Voice 'en-US-NonExistentVoice' not found" in result.output
        assert "Available en voices:" in result.output
        assert "en-US-AriaNeural" in result.output
        assert "en-GB-SoniaNeural" in result.output

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_voice_not_found_all_voices_fallback(self, mock_tts_class, runner):
        """Test voice not found shows all voices when language not detected."""
        from voicegenhub.providers.base import VoiceNotFoundError

        mock_tts = AsyncMock()
        # Mock available voices
        mock_tts._voice_selector.get_all_voices.return_value = [
            type(
                "Voice",
                (),
                {
                    "id": "en-US-AriaNeural",
                    "name": "Aria Neural",
                    "language": "en",
                    "locale": "en-US",
                    "gender": type("Gender", (), {"value": "female"})(),
                },
            )(),
            type(
                "Voice",
                (),
                {
                    "id": "fr-FR-DeniseNeural",
                    "name": "Denise Neural",
                    "language": "fr",
                    "locale": "fr-FR",
                    "gender": type("Gender", (), {"value": "female"})(),
                },
            )(),
        ]
        # Mock generate to raise VoiceNotFoundError with all voices
        mock_tts.generate.side_effect = VoiceNotFoundError(
            "Voice 'SomeRandomVoice' not found\n\nAvailable voices:\n  en-US-AriaNeural - Aria Neural (en)\n  fr-FR-DeniseNeural - Denise Neural (fr)",
            error_code="VOICE_NOT_FOUND",
            provider="edge",
        )
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(
            cli, ["synthesize", "hello", "--voice", "SomeRandomVoice"]
        )
        assert result.exit_code == 1
        assert "Voice 'SomeRandomVoice' not found" in result.output
        assert "Available voices:" in result.output
        assert "en-US-AriaNeural" in result.output
        assert "fr-FR-DeniseNeural" in result.output

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_voice_not_found_no_voices_available(self, mock_tts_class, runner):
        """Test voice not found when no voices are available."""
        from voicegenhub.providers.base import VoiceNotFoundError

        mock_tts = AsyncMock()
        mock_tts._voice_selector.get_all_voices.return_value = []
        mock_tts.generate.side_effect = VoiceNotFoundError(
            "Voice 'en-US-AriaNeural' not found",
            error_code="VOICE_NOT_FOUND",
            provider="edge",
        )
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(
            cli, ["synthesize", "hello", "--voice", "en-US-AriaNeural"]
        )
        assert result.exit_code == 1
        assert "Voice 'en-US-AriaNeural' not found" in result.output
        # Should not show suggestions when no voices available


class TestLanguageDetection:
    """Test language detection functionality."""

    def test_extract_language_from_voice_name(self):
        """Test language extraction from various voice name patterns."""
        from voicegenhub.core.engine import VoiceGenHub

        # Create a temporary instance to access the method
        tts = VoiceGenHub.__new__(VoiceGenHub)

        # Test various patterns
        assert tts._extract_language_from_voice_name("en-US-AriaNeural") == "en"
        assert tts._extract_language_from_voice_name("zh-CN-YunxiNeural") == "zh"
        assert tts._extract_language_from_voice_name("fr-FR-DeniseNeural") == "fr"
        assert tts._extract_language_from_voice_name("de-DE-KatjaNeural") == "de"
        assert tts._extract_language_from_voice_name("ja-JP-NanamiNeural") == "ja"
        assert tts._extract_language_from_voice_name("ko-KR-SunHiNeural") == "ko"

        # Test underscore patterns
        assert tts._extract_language_from_voice_name("voice_en") == "en"
        assert tts._extract_language_from_voice_name("tts_zh_CN") == "zh"

        # Test no language detected
        assert tts._extract_language_from_voice_name("SomeRandomVoice") is None
        assert tts._extract_language_from_voice_name("CustomVoice123") is None

        # Test edge cases
        assert tts._extract_language_from_voice_name("") is None
        assert tts._extract_language_from_voice_name("en") == "en"
        assert tts._extract_language_from_voice_name("zh") == "zh"
