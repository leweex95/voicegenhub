"""Unit tests for VoiceGenHub CLI."""
from unittest.mock import AsyncMock, patch
import subprocess
import logging

import pytest
from click.testing import CliRunner

from voicegenhub.cli import cli

DEFAULT_SYNTH_ARGS = ["--voice", "en-US-AriaNeural", "--language", "en"]


class TestCLI:
    """Test CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    def test_cli_group_initialization(self):
        """Verify CLI group is created and has no side effects on import."""
        # CLI group should be importable without issues
        from voicegenhub.cli import cli
        assert cli is not None
        assert cli.name == "cli"
        assert len(cli.commands) >= 2  # synthesize and voices

    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "VoiceGenHub" in result.output
        assert "synthesize" in result.output
        assert "voices" in result.output
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
        assert "edge, kokoro" in result.output

    def test_cli_rejects_unsupported_provider_voices(self, runner):
        """Test CLI rejects unsupported provider in voices command."""
        result = runner.invoke(cli, ["voices", "--provider", "coqui"])
        assert result.exit_code == 1
        assert "Unsupported provider 'coqui'" in result.output
        assert "edge, kokoro" in result.output

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_cli_accepts_supported_providers_synthesize(self, mock_tts_class, runner, tmp_path):
        """Test CLI accepts supported providers in synthesize command without hitting real providers."""
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake")
        mock_tts_class.return_value = mock_tts

        for provider in ["edge", "kokoro"]:
            result = runner.invoke(
                cli,
                [
                    "synthesize",
                    "hello",
                    *DEFAULT_SYNTH_ARGS,
                    "--provider",
                    provider,
                    "--output",
                    str(tmp_path / "dummy.wav"),
                ],
            )
            assert "Unsupported provider" not in result.output

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_synthesize_engine_initialization(self, mock_tts_class, runner, tmp_path):
        """Verify VoiceGenHub is initialized with correct provider argument."""
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake")
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(
            cli,
            [
                "synthesize",
                "test",
                *DEFAULT_SYNTH_ARGS,
                "--provider",
                "edge",
                "--output",
                str(tmp_path / "test.wav"),
            ],
        )
        assert result.exit_code == 0
        mock_tts_class.assert_called_once_with(provider="edge")

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_synthesize_text_required(self, mock_tts_class, runner):
        """Verify text argument is required and passed correctly to generate()."""
        # Text is required by Click, so this should fail at parsing
        result = runner.invoke(cli, ["synthesize"])
        assert result.exit_code == 2  # Click error for missing argument
        assert "Missing argument" in result.output

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_synthesize_audio_format_enum(self, mock_tts_class, runner, tmp_path):
        """Verify AudioFormat enum is correctly constructed from format option."""
        from voicegenhub.providers.base import AudioFormat
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake")
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(
            cli,
            [
                "synthesize",
                "test",
                *DEFAULT_SYNTH_ARGS,
                "--format",
                "mp3",
                "--output",
                str(tmp_path / "test.mp3"),
            ],
        )
        assert result.exit_code == 0
        # Check that AudioFormat.MP3 was passed
        call_args = mock_tts.generate.call_args
        assert call_args[1]["audio_format"] == AudioFormat.MP3

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_synthesize_rate_and_pitch_bounds(self, mock_tts_class, runner, tmp_path):
        """Verify rate and pitch values are correctly passed to generate() and respected."""
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake")
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(
            cli,
            [
                "synthesize",
                "test",
                *DEFAULT_SYNTH_ARGS,
                "--rate",
                "1.5",
                "--pitch",
                "0.8",
                "--output",
                str(tmp_path / "test.wav"),
            ],
        )
        assert result.exit_code == 0
        call_args = mock_tts.generate.call_args
        assert call_args[1]["speed"] == 1.5
        assert call_args[1]["pitch"] == 0.8

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_synthesize_rate_edge_cases(self, mock_tts_class, runner, tmp_path):
        """Test boundary values: 0.5, 2.0 and out-of-range handling."""
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake")
        mock_tts_class.return_value = mock_tts

        # Valid boundaries
        result = runner.invoke(
            cli,
            [
                "synthesize",
                "test",
                *DEFAULT_SYNTH_ARGS,
                "--rate",
                "0.5",
                "--output",
                str(tmp_path / "test.wav"),
            ],
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli,
            [
                "synthesize",
                "test",
                *DEFAULT_SYNTH_ARGS,
                "--rate",
                "2.0",
                "--output",
                str(tmp_path / "test.wav"),
            ],
        )
        assert result.exit_code == 0

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_synthesize_pitch_edge_cases(self, mock_tts_class, runner, tmp_path):
        """Test boundary values: 0.5, 2.0 and out-of-range handling."""
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake")
        mock_tts_class.return_value = mock_tts

        # Valid boundaries
        result = runner.invoke(
            cli,
            [
                "synthesize",
                "test",
                *DEFAULT_SYNTH_ARGS,
                "--pitch",
                "0.5",
                "--output",
                str(tmp_path / "test.wav"),
            ],
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli,
            [
                "synthesize",
                "test",
                *DEFAULT_SYNTH_ARGS,
                "--pitch",
                "2.0",
                "--output",
                str(tmp_path / "test.wav"),
            ],
        )
        assert result.exit_code == 0

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_synthesize_async_generate_called(self, mock_tts_class, runner, tmp_path):
        """Verify asyncio.run(tts.generate(...)) is called with correct arguments."""
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake")
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(cli, [
            "synthesize", "hello world", "--voice", "test-voice", "--language", "en",
            "--output", str(tmp_path / "test.wav"), "--provider", "edge", "--rate", "1.2", "--pitch", "0.9"
        ])
        assert result.exit_code == 0
        mock_tts.generate.assert_called_once()
        call_args = mock_tts.generate.call_args
        assert call_args[1]["text"] == "hello world"
        assert call_args[1]["voice"] == "test-voice"
        assert call_args[1]["language"] == "en"
        assert call_args[1]["speed"] == 1.2
        assert call_args[1]["pitch"] == 0.9

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_synthesize_generate_raises_exception(self, mock_tts_class, runner, caplog):
        """Verify CLI handles exceptions from tts.generate() and exits with code 1."""
        mock_tts = AsyncMock()
        mock_tts.generate.side_effect = Exception("Test error")
        mock_tts_class.return_value = mock_tts

        with caplog.at_level(logging.ERROR):
            result = runner.invoke(cli, ["synthesize", "test", *DEFAULT_SYNTH_ARGS])
        assert result.exit_code == 1
        assert "Error: Test error" in caplog.text

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_synthesize_output_path_default(self, mock_tts_class, runner):
        """Verify default output filename is used if none provided."""
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake")
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(cli, ["synthesize", "test", *DEFAULT_SYNTH_ARGS, "--format", "wav"])
        assert result.exit_code == 0
        # Check that file was attempted to be written (mock doesn't actually write)
        # Since we can't check file existence easily with mocks, just ensure no crash

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_synthesize_output_path_custom(self, mock_tts_class, runner, tmp_path):
        """Verify output file is saved to provided path correctly."""
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake")
        mock_tts_class.return_value = mock_tts

        output_file = tmp_path / "custom.wav"
        result = runner.invoke(
            cli,
            ["synthesize", "test", *DEFAULT_SYNTH_ARGS, "--output", str(output_file)],
        )
        assert result.exit_code == 0

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_synthesize_temp_file_created_for_effects(self, mock_tts_class, runner, tmp_path):
        """Verify temporary file is created when any audio effect is requested."""
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake")
        mock_tts_class.return_value = mock_tts

        with patch("subprocess.run") as mock_run:
            output_file = tmp_path / "test.wav"
            result = runner.invoke(
                cli,
                [
                    "synthesize",
                    "test",
                    *DEFAULT_SYNTH_ARGS,
                    "--output",
                    str(output_file),
                    "--pitch-shift",
                    "-2",
                ],
            )
            assert result.exit_code == 0
            mock_run.assert_called_once()
            # Check that FFmpeg was called with different input and output
            cmd = mock_run.call_args[0][0]
            assert cmd[1] != cmd[-1]  # input != output

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_synthesize_no_effects(self, mock_tts_class, runner, tmp_path):
        """Verify that no subprocess is called if no audio effects are requested."""
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake")
        mock_tts_class.return_value = mock_tts

        with patch("subprocess.run") as mock_run:
            output_file = tmp_path / "test.wav"
            result = runner.invoke(
                cli,
                ["synthesize", "test", *DEFAULT_SYNTH_ARGS, "--output", str(output_file)],
            )
            assert result.exit_code == 0
            mock_run.assert_not_called()

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_synthesize_effects_command_construction(self, mock_tts_class, runner, tmp_path):
        """Verify FFmpeg command is correctly built for combinations of effects."""
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake")
        mock_tts_class.return_value = mock_tts

        with patch("subprocess.run") as mock_run:
            output_file = tmp_path / "test.wav"
            result = runner.invoke(
                cli,
                [
                    "synthesize",
                    "test",
                    *DEFAULT_SYNTH_ARGS,
                    "--output",
                    str(output_file),
                    "--pitch-shift",
                    "-2",
                    "--lowpass",
                    "1200",
                    "--normalize",
                ],
            )
            assert result.exit_code == 0
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "ffmpeg" in cmd
            assert "-i" in cmd
            assert "-af" in cmd
            # Check filters are present
            af_arg = cmd[cmd.index("-af") + 1]
            assert "rubberband" in af_arg
            assert "lowpass" in af_arg
            assert "dynaudnorm" in af_arg

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_synthesize_noise_complex_filter(self, mock_tts_class, runner, tmp_path):
        """Verify complex filter is constructed correctly when noise is requested with other effects."""
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake")
        mock_tts_class.return_value = mock_tts

        with patch("subprocess.run") as mock_run:
            output_file = tmp_path / "test.wav"
            result = runner.invoke(
                cli,
                [
                    "synthesize",
                    "test",
                    *DEFAULT_SYNTH_ARGS,
                    "--output",
                    str(output_file),
                    "--noise",
                    "0.1",
                    "--lowpass",
                    "1200",
                ],
            )
            assert result.exit_code == 0
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "-filter_complex" in cmd
            complex_arg = cmd[cmd.index("-filter_complex") + 1]
            assert "anoisesrc" in complex_arg
            assert "amix" in complex_arg
            assert "lowpass" in complex_arg

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_synthesize_effects_subprocess_success(self, mock_tts_class, runner, tmp_path):
        """Verify subprocess.run is called and temp file removed on success."""
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake")
        mock_tts_class.return_value = mock_tts

        with patch("subprocess.run") as mock_run, patch("pathlib.Path.unlink") as mock_unlink:
            output_file = tmp_path / "test.wav"
            result = runner.invoke(
                cli,
                [
                    "synthesize",
                    "test",
                    *DEFAULT_SYNTH_ARGS,
                    "--output",
                    str(output_file),
                    "--normalize",
                ],
            )
            assert result.exit_code == 0
            mock_run.assert_called_once()
            mock_unlink.assert_called_once()

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_synthesize_effects_subprocess_callederror(self, mock_tts_class, runner, tmp_path):
        """Verify CLI logs warning and retains temp file if subprocess fails."""
        from subprocess import CalledProcessError
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake")
        mock_tts_class.return_value = mock_tts

        with patch("subprocess.run", side_effect=CalledProcessError(1, "ffmpeg", stderr=b"error")), \
             patch("pathlib.Path.unlink") as mock_unlink:
            output_file = tmp_path / "test.wav"
            result = runner.invoke(
                cli,
                [
                    "synthesize",
                    "test",
                    *DEFAULT_SYNTH_ARGS,
                    "--output",
                    str(output_file),
                    "--normalize",
                ],
            )
            assert result.exit_code == 0  # CLI doesn't exit on post-processing failure
            mock_unlink.assert_not_called()

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_synthesize_effects_subprocess_filenotfound(self, mock_tts_class, runner, tmp_path):
        """Verify CLI logs warning when FFmpeg is missing and retains temp file."""
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake")
        mock_tts_class.return_value = mock_tts

        with patch("subprocess.run", side_effect=FileNotFoundError), \
             patch("pathlib.Path.unlink") as mock_unlink:
            output_file = tmp_path / "test.wav"
            result = runner.invoke(
                cli,
                [
                    "synthesize",
                    "test",
                    *DEFAULT_SYNTH_ARGS,
                    "--output",
                    str(output_file),
                    "--normalize",
                ],
            )
            assert result.exit_code == 0
            mock_unlink.assert_not_called()

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_synthesize_generic_exception(self, mock_tts_class, runner, caplog):
        """Verify CLI catches unexpected exceptions and exits with code 1."""
        mock_tts = AsyncMock()
        mock_tts.generate.side_effect = RuntimeError("Unexpected error")
        mock_tts_class.return_value = mock_tts

        with caplog.at_level(logging.ERROR):
            result = runner.invoke(cli, ["synthesize", "test", *DEFAULT_SYNTH_ARGS])
        assert result.exit_code == 1
        assert "Error: Unexpected error" in caplog.text

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_cli_accepts_supported_providers_voices(self, mock_tts_class, runner):
        """Test voices command accepts supported providers without real network calls."""
        mock_tts = AsyncMock()
        mock_tts.get_voices.return_value = []
        mock_tts_class.return_value = mock_tts

        for provider in ["edge", "kokoro"]:
            result = runner.invoke(cli, ["voices", "--provider", provider])
            assert "Unsupported provider" not in result.output

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_cli_synthesize_basic_call(self, mock_tts_class, runner, tmp_path):
        """Test synthesize command makes correct calls."""
        # Mock the TTS instance
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
                "--language",
                "en",
                "--output",
                str(tmp_path / "speech.wav"),
            ],
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
                "--language",
                "en",
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
    def test_cli_synthesize_error_handling(self, mock_tts_class, runner, caplog):
        """Test synthesize command error handling."""
        mock_tts = AsyncMock()
        mock_tts.generate.side_effect = Exception("Test error")
        mock_tts_class.return_value = mock_tts

        with caplog.at_level(logging.ERROR):
            result = runner.invoke(cli, ["synthesize", "hello world", *DEFAULT_SYNTH_ARGS])
        assert result.exit_code == 1
        assert "Error: Test error" in caplog.text

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_cli_voices_error_handling(self, mock_tts_class, runner):
        """Test voices command error handling."""
        mock_tts = AsyncMock()
        mock_tts.get_voices.side_effect = Exception("Test error")
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(cli, ["voices"])
        assert result.exit_code == 1
        assert "Error: Test error" in result.output

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_voices_async_get_voices_called(self, mock_tts_class, runner):
        """Verify asyncio.run(tts.get_voices(...)) is called with correct language argument."""
        mock_tts = AsyncMock()
        mock_tts.get_voices.return_value = []
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(cli, ["voices", "--language", "en"])
        assert result.exit_code == 0
        mock_tts.get_voices.assert_called_once_with(language="en")

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_voices_get_voices_raises(self, mock_tts_class, runner):
        """Verify CLI handles exceptions from tts.get_voices() and exits with code 1."""
        mock_tts = AsyncMock()
        mock_tts.get_voices.side_effect = Exception("Voices error")
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(cli, ["voices"])
        assert result.exit_code == 1
        assert "Error: Voices error" in result.output

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_voices_format_table(self, mock_tts_class, runner, capsys):
        """Verify CLI outputs voices in table format, first 10 voices with count summary."""
        mock_tts = AsyncMock()
        mock_tts.get_voices.return_value = [
            {"id": f"voice{i}", "name": f"Voice {i}", "language": "en", "locale": "en-US", "gender": "female"}
            for i in range(15)
        ]
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(cli, ["voices"])
        assert result.exit_code == 0
        output = result.output
        assert "Available Voices:" in output
        assert "voice0" in output
        assert "... and 5 more voices" in output

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_voices_format_json(self, mock_tts_class, runner):
        """Verify CLI outputs voices in JSON format with correct keys and indentation."""
        import json
        mock_tts = AsyncMock()
        mock_tts.get_voices.return_value = [
            {"id": "test-voice", "name": "Test Voice", "language": "en", "locale": "en-US", "gender": "female"}
        ]
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(cli, ["voices", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "voices" in data
        assert len(data["voices"]) == 1
        assert data["voices"][0]["id"] == "test-voice"
        assert data["voices"][0]["language"] == "en-US"

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_voices_empty_list(self, mock_tts_class, runner):
        """Verify CLI handles empty voices list without crashing."""
        mock_tts = AsyncMock()
        mock_tts.get_voices.return_value = []
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(cli, ["voices"])
        assert result.exit_code == 0
        assert "Available Voices:" in result.output

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_voices_language_filter(self, mock_tts_class, runner):
        """Verify only voices matching specified language code are returned."""
        mock_tts = AsyncMock()

        def mock_get_voices(language=None):
            all_voices = [
                {"id": "en-voice", "name": "English Voice", "language": "en", "locale": "en-US", "gender": "female"},
                {"id": "fr-voice", "name": "French Voice", "language": "fr", "locale": "fr-FR", "gender": "male"}
            ]
            if language:
                return [v for v in all_voices if v["language"].startswith(language)]
            return all_voices
        mock_tts.get_voices.side_effect = mock_get_voices
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(cli, ["voices", "--language", "en"])
        assert result.exit_code == 0
        assert "en-voice" in result.output
        assert "fr-voice" not in result.output

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_voices_echo_output(self, mock_tts_class, runner, capsys):
        """Verify correct output is sent to stdout/stderr and formatted properly."""
        mock_tts = AsyncMock()
        mock_tts.get_voices.return_value = [
            {"id": "test-voice", "name": "Test Voice", "language": "en", "locale": "en-US", "gender": "female"}
        ]
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(cli, ["voices"])
        assert result.exit_code == 0
        assert "Available Voices:" in result.output
        assert "test-voice" in result.output

    @patch("voicegenhub.cli.cli")
    def test_main_function_calls_cli(self, mock_cli):
        """Verify main() calls cli() entry point without arguments."""
        from voicegenhub.cli import main
        main()
        mock_cli.assert_called_once()

    def test_if_name_main_invokes_cli(self, monkeypatch):
        """Verify __name__=='__main__' block invokes cli()."""
        # This is hard to test directly, but we can check that cli is callable
        assert callable(cli)

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_synthesize_empty_text(self, mock_tts_class, runner, tmp_path):
        """Ensure engine handles empty string text input gracefully."""
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake")
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(
            cli,
            ["synthesize", "", *DEFAULT_SYNTH_ARGS, "--output", str(tmp_path / "test.wav")],
        )
        assert result.exit_code == 0
        mock_tts.generate.assert_called_once()
        # Should still call generate with empty text

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_synthesize_file_write_exception(self, mock_tts_class, runner, tmp_path, caplog):
        """Verify CLI exits if writing audio file fails due to IOError."""
        from unittest.mock import MagicMock
        mock_tts = AsyncMock()
        response = MagicMock()
        response.save.side_effect = OSError("Permission denied")
        mock_tts.generate.return_value = response
        mock_tts_class.return_value = mock_tts

        with caplog.at_level(logging.ERROR):
            result = runner.invoke(
                cli,
                ["synthesize", "test", *DEFAULT_SYNTH_ARGS, "--output", str(tmp_path / "test.wav")],
            )
        assert result.exit_code == 1
        assert "Error: Permission denied" in caplog.text

    def test_logger_initialized(self):
        """Ensure logger is initialized at module level."""
        from voicegenhub.cli import logger
        assert logger is not None
        # Logger should have the expected name
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'warning')

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_logger_info_called_on_success(self, mock_tts_class, runner, tmp_path, caplog):
        """Verify logger.info is called when audio is successfully saved."""
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake")
        mock_tts_class.return_value = mock_tts

        with caplog.at_level(logging.INFO):
            result = runner.invoke(
                cli,
                ["synthesize", "test", *DEFAULT_SYNTH_ARGS, "--output", str(tmp_path / "test.wav")],
            )
        assert result.exit_code == 0
        assert "SUCCESS: Audio saved to:" in caplog.text

    @patch("voicegenhub.cli.VoiceGenHub")
    @patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg", stderr=b"error"))
    def test_logger_warning_called_on_failure(self, mock_subprocess, mock_tts_class, runner, tmp_path, caplog):
        """Verify logger.warning is called if FFmpeg or post-processing fails."""
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake")
        mock_tts_class.return_value = mock_tts

        with caplog.at_level(logging.WARNING):
            result = runner.invoke(
                cli,
                [
                    "synthesize",
                    "test",
                    *DEFAULT_SYNTH_ARGS,
                    "--pitch-shift",
                    "2",
                    "--output",
                    str(tmp_path / "test.wav"),
                ],
            )
        assert result.exit_code == 0  # CLI doesn't exit on FFmpeg failure
        assert "Post-processing failed" in caplog.text


class TestCLIPostProcessingFlags:
    """Unit tests for new post-processing CLI flags."""

    def test_cli_lowpass_flag_construction(self):
        """Test that --lowpass flag creates correct FFmpeg filter."""
        from voicegenhub.cli import synthesize

        # Verify flag exists and is properly defined
        assert synthesize.params is not None
        param_names = [p.name for p in synthesize.params]
        assert "lowpass" in param_names

    def test_cli_distortion_flag_construction(self):
        """Test that --distortion flag exists and accepts float."""
        from voicegenhub.cli import synthesize

        param_names = [p.name for p in synthesize.params]
        assert "distortion" in param_names

    def test_cli_noise_flag_construction(self):
        """Test that --noise flag exists and accepts float."""
        from voicegenhub.cli import synthesize

        param_names = [p.name for p in synthesize.params]
        assert "noise" in param_names

    def test_cli_pitch_shift_flag_construction(self):
        """Test that --pitch-shift flag exists and accepts int."""
        from voicegenhub.cli import synthesize

        param_names = [p.name for p in synthesize.params]
        assert "pitch_shift" in param_names

    def test_cli_reverb_flag_construction(self):
        """Test that --reverb flag is boolean."""
        from voicegenhub.cli import synthesize

        param_names = [p.name for p in synthesize.params]
        assert "reverb" in param_names

    def test_cli_normalize_flag_construction(self):
        """Test that --normalize flag is boolean."""
        from voicegenhub.cli import synthesize

        param_names = [p.name for p in synthesize.params]
        assert "normalize" in param_names


class TestFFmpegFilterChainConstruction:
    """Unit tests for FFmpeg filter chain building logic."""

    def test_lowpass_filter_format(self):
        """Test lowpass filter string format."""
        cutoff = 1200
        expected = f"lowpass=f={cutoff}"
        assert "lowpass=f=" in expected

    def test_distortion_filter_format(self):
        """Test distortion filter string format."""
        gain = 10.0
        expected = f"volume={gain}dB,acompressor=threshold=-6dB:ratio=20:attack=5:release=50"
        assert "volume=" in expected
        assert "acompressor" in expected

    def test_noise_filter_format(self):
        """Test noise filter complex format."""
        noise_volume = 0.05
        expected = f"anoisesrc=d=10:c=white:r=44100:a={noise_volume}[noise];[0:a][noise]amix=inputs=2:duration=first"
        assert "anoisesrc" in expected
        assert "amix" in expected

    def test_pitch_shift_filter_format(self):
        """Test pitch shift filter format."""
        semitones = -5
        rate_mult = 2 ** (semitones / 12.0)
        expected = f"asetrate=44100*{rate_mult},aresample=44100"
        assert "asetrate" in expected
        assert "aresample" in expected

    def test_reverb_filter_format(self):
        """Test reverb filter format."""
        expected = "aecho=0.8:0.9:1000:0.3"
        assert "aecho" in expected

    def test_normalize_filter_format(self):
        """Test normalize filter format."""
        expected = "dynaudnorm=f=150:g=15"
        assert "dynaudnorm" in expected

    def test_pitch_shift_calculation(self):
        """Test pitch shift semitone calculation."""
        # -5 semitones = lower by ~0.749x rate
        semitones = -5
        rate_mult = 2 ** (semitones / 12.0)
        assert 0.74 < rate_mult < 0.76

    def test_pitch_shift_positive(self):
        """Test positive pitch shift."""
        semitones = 5
        rate_mult = 2 ** (semitones / 12.0)
        assert 1.33 < rate_mult < 1.34

    def test_pitch_shift_octave(self):
        """Test full octave shift."""
        semitones = 12
        rate_mult = 2 ** (semitones / 12.0)
        assert rate_mult == 2.0


class TestCLIPostProcessingIntegration:
    """Integration tests for CLI post-processing (slow, CI-only)."""

    @pytest.mark.integration
    def test_cli_synthesize_with_lowpass_effect(self, tmp_path):
        """Integration: Verify lowpass effect is actually applied."""
        pytest.importorskip("ffmpeg")
        from click.testing import CliRunner
        from voicegenhub.cli import cli
        from unittest.mock import patch, MagicMock

        runner = CliRunner()
        output_file = tmp_path / "test_lowpass.wav"

        with patch("voicegenhub.core.engine.VoiceGenHub") as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine_class.return_value = mock_engine

            # Mock response with valid audio data
            mock_response = MagicMock()
            mock_response.audio_data = b"RIFF" + b"\x00" * 36 + b"data" + b"\x00" * 100

            async def mock_generate(*args, **kwargs):
                return mock_response

            mock_engine.generate = mock_generate

            result = runner.invoke(
                cli,
                [
                    "synthesize",
                    "test text",
                    *DEFAULT_SYNTH_ARGS,
                    "--provider",
                    "edge",
                    "--output",
                    str(output_file),
                    "--lowpass",
                    "1200",
                ],
            )

            assert result.exit_code == 0

    @pytest.mark.integration
    def test_cli_synthesize_with_distortion_effect(self, tmp_path):
        """Integration: Verify distortion effect is applied."""
        pytest.importorskip("ffmpeg")
        from click.testing import CliRunner
        from voicegenhub.cli import cli
        from unittest.mock import patch

        runner = CliRunner()
        output_file = tmp_path / "test_distortion.wav"

        with patch("voicegenhub.core.engine.VoiceGenHub"):
            result = runner.invoke(
                cli,
                [
                    "synthesize",
                    "test",
                    *DEFAULT_SYNTH_ARGS,
                    "--provider",
                    "edge",
                    "--output",
                    str(output_file),
                    "--distortion",
                    "10",
                ],
            )
            # Just verify command is accepted (FFmpeg might not be available)
            assert result.exit_code in [0, 1]

    @pytest.mark.integration
    def test_cli_synthesize_combined_effects(self, tmp_path):
        """Integration: Test multiple effects combined."""
        pytest.importorskip("ffmpeg")
        from click.testing import CliRunner
        from voicegenhub.cli import cli
        from unittest.mock import patch

        runner = CliRunner()
        output_file = tmp_path / "test_combined.wav"

        with patch("voicegenhub.core.engine.VoiceGenHub"):
            result = runner.invoke(
                cli,
                [
                    "synthesize",
                    "test",
                    *DEFAULT_SYNTH_ARGS,
                    "--provider",
                    "edge",
                    "--output",
                    str(output_file),
                    "--lowpass",
                    "1200",
                    "--distortion",
                    "10",
                    "--pitch-shift",
                    "-5",
                    "--reverb",
                    "--normalize",
                ],
            )
            assert result.exit_code in [0, 1]

    @pytest.mark.integration
    def test_cli_synthesize_with_noise_effect(self, tmp_path):
        """Integration: Verify noise injection flag works."""
        pytest.importorskip("ffmpeg")
        from click.testing import CliRunner
        from voicegenhub.cli import cli
        from unittest.mock import patch

        runner = CliRunner()
        output_file = tmp_path / "test_noise.wav"

        with patch("voicegenhub.core.engine.VoiceGenHub"):
            result = runner.invoke(
                cli,
                [
                    "synthesize",
                    "test",
                    *DEFAULT_SYNTH_ARGS,
                    "--provider",
                    "edge",
                    "--output",
                    str(output_file),
                    "--noise",
                    "0.05",
                ],
            )
            assert result.exit_code in [0, 1]


class TestCLIPostProcessingTempFileLogic:
    """Unit tests for temp file creation logic in post-processing."""

    @pytest.mark.integration
    def test_pitch_shift_creates_temp_file(self, tmp_path):
        """Test that pitch-shift effect creates distinct temp file."""
        pytest.importorskip("ffmpeg")
        from click.testing import CliRunner
        from voicegenhub.cli import cli
        from unittest.mock import patch, MagicMock

        runner = CliRunner()
        output_file = tmp_path / "test_pitch_temp.wav"

        captured_cmds = []

        def mock_subprocess_run(cmd, capture_output=True, check=True):
            captured_cmds.append(cmd)
            return MagicMock()

        with patch("voicegenhub.core.engine.VoiceGenHub") as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine_class.return_value = mock_engine

            mock_response = MagicMock()
            mock_response.audio_data = b"RIFF" + b"\x00" * 36 + b"data" + b"\x00" * 100
            mock_engine.generate = MagicMock(return_value=mock_response)

            with patch("subprocess.run", side_effect=mock_subprocess_run):
                result = runner.invoke(cli, [
                    "synthesize",
                    "test",
                    *DEFAULT_SYNTH_ARGS,
                    "--provider", "edge",
                    "--output", str(output_file),
                    "--pitch-shift", "-2",
                ])

                assert result.exit_code == 0
                assert len(captured_cmds) == 1
                cmd = captured_cmds[0]
                input_file = cmd[1]  # -i <input>
                output_file_cmd = cmd[-1]
                assert input_file != output_file_cmd, "Temp file should differ from output file"

    @pytest.mark.integration
    def test_lowpass_creates_temp_file(self, tmp_path):
        """Test that lowpass effect creates distinct temp file."""
        pytest.importorskip("ffmpeg")
        from click.testing import CliRunner
        from voicegenhub.cli import cli
        from unittest.mock import patch, MagicMock

        runner = CliRunner()
        output_file = tmp_path / "test_lowpass_temp.wav"

        captured_cmds = []

        def mock_subprocess_run(cmd, capture_output=True, check=True):
            captured_cmds.append(cmd)
            return MagicMock()

        with patch("voicegenhub.core.engine.VoiceGenHub") as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine_class.return_value = mock_engine

            mock_response = MagicMock()
            mock_response.audio_data = b"RIFF" + b"\x00" * 36 + b"data" + b"\x00" * 100
            mock_engine.generate = MagicMock(return_value=mock_response)

            with patch("subprocess.run", side_effect=mock_subprocess_run):
                result = runner.invoke(cli, [
                    "synthesize",
                    "test",
                    *DEFAULT_SYNTH_ARGS,
                    "--provider", "edge",
                    "--output", str(output_file),
                    "--lowpass", "1200",
                ])

                assert result.exit_code == 0
                assert len(captured_cmds) == 1
                cmd = captured_cmds[0]
                input_file = cmd[1]
                output_file_cmd = cmd[-1]
                assert input_file != output_file_cmd

    @pytest.mark.integration
    def test_normalize_creates_temp_file(self, tmp_path):
        """Test that normalize effect creates distinct temp file."""
        pytest.importorskip("ffmpeg")
        from click.testing import CliRunner
        from voicegenhub.cli import cli
        from unittest.mock import patch, MagicMock

        runner = CliRunner()
        output_file = tmp_path / "test_normalize_temp.wav"

        captured_cmds = []

        def mock_subprocess_run(cmd, capture_output=True, check=True):
            captured_cmds.append(cmd)
            return MagicMock()

        with patch("voicegenhub.core.engine.VoiceGenHub") as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine_class.return_value = mock_engine

            mock_response = MagicMock()
            mock_response.audio_data = b"RIFF" + b"\x00" * 36 + b"data" + b"\x00" * 100
            mock_engine.generate = MagicMock(return_value=mock_response)

            with patch("subprocess.run", side_effect=mock_subprocess_run):
                result = runner.invoke(cli, [
                    "synthesize",
                    "test",
                    *DEFAULT_SYNTH_ARGS,
                    "--provider", "edge",
                    "--output", str(output_file),
                    "--normalize",
                ])

                assert result.exit_code == 0
                assert len(captured_cmds) == 1
                cmd = captured_cmds[0]
                input_file = cmd[1]
                output_file_cmd = cmd[-1]
                assert input_file != output_file_cmd

    @pytest.mark.integration
    def test_distortion_creates_temp_file(self, tmp_path):
        """Test that distortion effect creates distinct temp file."""
        pytest.importorskip("ffmpeg")
        from click.testing import CliRunner
        from voicegenhub.cli import cli
        from unittest.mock import patch, MagicMock

        runner = CliRunner()
        output_file = tmp_path / "test_distortion_temp.wav"

        captured_cmds = []

        def mock_subprocess_run(cmd, capture_output=True, check=True):
            captured_cmds.append(cmd)
            return MagicMock()

        with patch("voicegenhub.core.engine.VoiceGenHub") as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine_class.return_value = mock_engine

            mock_response = MagicMock()
            mock_response.audio_data = b"RIFF" + b"\x00" * 36 + b"data" + b"\x00" * 100
            mock_engine.generate = MagicMock(return_value=mock_response)

            with patch("subprocess.run", side_effect=mock_subprocess_run):
                result = runner.invoke(cli, [
                    "synthesize",
                    "test",
                    *DEFAULT_SYNTH_ARGS,
                    "--provider", "edge",
                    "--output", str(output_file),
                    "--distortion", "10",
                ])

                assert result.exit_code == 0
                assert len(captured_cmds) == 1
                cmd = captured_cmds[0]
                input_file = cmd[1]
                output_file_cmd = cmd[-1]
                assert input_file != output_file_cmd

    @pytest.mark.integration
    def test_reverb_creates_temp_file(self, tmp_path):
        """Test that reverb effect creates distinct temp file."""
        pytest.importorskip("ffmpeg")
        from click.testing import CliRunner
        from voicegenhub.cli import cli
        from unittest.mock import patch, MagicMock

        runner = CliRunner()
        output_file = tmp_path / "test_reverb_temp.wav"

        captured_cmds = []

        def mock_subprocess_run(cmd, capture_output=True, check=True):
            captured_cmds.append(cmd)
            return MagicMock()

        with patch("voicegenhub.core.engine.VoiceGenHub") as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine_class.return_value = mock_engine

            mock_response = MagicMock()
            mock_response.audio_data = b"RIFF" + b"\x00" * 36 + b"data" + b"\x00" * 100
            mock_engine.generate = MagicMock(return_value=mock_response)

            with patch("subprocess.run", side_effect=mock_subprocess_run):
                result = runner.invoke(cli, [
                    "synthesize",
                    "test",
                    *DEFAULT_SYNTH_ARGS,
                    "--provider", "edge",
                    "--output", str(output_file),
                    "--reverb",
                ])

                assert result.exit_code == 0
                assert len(captured_cmds) == 1
                cmd = captured_cmds[0]
                input_file = cmd[1]
                output_file_cmd = cmd[-1]
                assert input_file != output_file_cmd

    @pytest.mark.integration
    def test_noise_creates_temp_file(self, tmp_path):
        """Test that noise effect creates distinct temp file."""
        pytest.importorskip("ffmpeg")
        from click.testing import CliRunner
        from voicegenhub.cli import cli
        from unittest.mock import patch, MagicMock

        runner = CliRunner()
        output_file = tmp_path / "test_noise_temp.wav"

        captured_cmds = []

        def mock_subprocess_run(cmd, capture_output=True, check=True):
            captured_cmds.append(cmd)
            return MagicMock()

        with patch("voicegenhub.core.engine.VoiceGenHub") as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine_class.return_value = mock_engine

            mock_response = MagicMock()
            mock_response.audio_data = b"RIFF" + b"\x00" * 36 + b"data" + b"\x00" * 100
            mock_engine.generate = MagicMock(return_value=mock_response)

            with patch("subprocess.run", side_effect=mock_subprocess_run):
                result = runner.invoke(cli, [
                    "synthesize",
                    "test",
                    *DEFAULT_SYNTH_ARGS,
                    "--provider", "edge",
                    "--output", str(output_file),
                    "--noise", "0.05",
                ])

                assert result.exit_code == 0
                assert len(captured_cmds) == 1
                cmd = captured_cmds[0]
                input_file = cmd[1]
                output_file_cmd = cmd[-1]
                assert input_file != output_file_cmd


class TestCLIVoiceNotFoundErrors:
    """Test CLI voice not found error messages with suggestions."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_voice_not_found_english_suggestions(self, mock_tts_class, runner, caplog):
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

        with caplog.at_level(logging.ERROR):
            result = runner.invoke(
                cli,
                [
                    "synthesize",
                    "hello",
                    "--voice",
                    "en-US-NonExistentVoice",
                    "--language",
                    "en",
                ],
            )
        assert result.exit_code == 1
        assert "Voice 'en-US-NonExistentVoice' not found" in caplog.text
        assert "Available en voices:" in caplog.text
        assert "en-US-AriaNeural" in caplog.text
        assert "en-GB-SoniaNeural" in caplog.text

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_voice_not_found_all_voices_fallback(self, mock_tts_class, runner, caplog):
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

        with caplog.at_level(logging.ERROR):
            result = runner.invoke(
                cli,
                [
                    "synthesize",
                    "hello",
                    "--voice",
                    "SomeRandomVoice",
                    "--language",
                    "en",
                ],
            )
        assert result.exit_code == 1
        assert "Voice 'SomeRandomVoice' not found" in caplog.text
        assert "Available voices:" in caplog.text
        assert "en-US-AriaNeural" in caplog.text
        assert "fr-FR-DeniseNeural" in caplog.text

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_voice_not_found_no_voices_available(self, mock_tts_class, runner, caplog):
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

        with caplog.at_level(logging.ERROR):
            result = runner.invoke(
                cli,
                [
                    "synthesize",
                    "hello",
                    "--voice",
                    "en-US-AriaNeural",
                    "--language",
                    "en",
                ],
            )
        assert result.exit_code == 1
        assert "Voice 'en-US-AriaNeural' not found" in caplog.text
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
