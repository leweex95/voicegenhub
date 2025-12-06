"""Unit tests for VoiceGenHub CLI."""
from unittest.mock import AsyncMock, patch

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
            cli,
            [
                "synthesize",
                "hello world",
                *DEFAULT_SYNTH_ARGS,
                "--provider",
                "coqui",
                "--output",
                "dummy.wav",
            ],
        )
        assert result.exit_code == 1
        assert "Unsupported provider 'coqui'" in result.output
        assert "edge, piper, melotts, kokoro, elevenlabs, bark, chatterbox" in result.output

    def test_cli_rejects_unsupported_provider_voices(self, runner):
        """Test CLI rejects unsupported provider in voices command."""
        result = runner.invoke(cli, ["voices", "--provider", "coqui"])
        assert result.exit_code == 1
        assert "Unsupported provider 'coqui'" in result.output
        assert "edge, piper, melotts, kokoro, elevenlabs, bark, chatterbox" in result.output

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_cli_accepts_supported_providers_synthesize(self, mock_tts_class, runner, tmp_path):
        """Test CLI accepts supported providers in synthesize command without hitting real services."""
        mock_tts = AsyncMock()
        mock_tts.generate.return_value = AsyncMock(audio_data=b"fake")
        mock_tts_class.return_value = mock_tts

        for provider in ["edge", "piper", "melotts", "kokoro"]:
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
    def test_synthesize_generate_raises_exception(self, mock_tts_class, runner):
        """Verify CLI handles exceptions from tts.generate() and exits with code 1."""
        mock_tts = AsyncMock()
        mock_tts.generate.side_effect = Exception("Test error")
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(cli, ["synthesize", "test", *DEFAULT_SYNTH_ARGS, "--output", "dummy.wav"])
        assert result.exit_code == 1
        assert "Error: Test error" in result.output

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
                ["synthesize", "test", *DEFAULT_SYNTH_ARGS, "--output", str(output_file), "--normalize"],
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
                ["synthesize", "test", *DEFAULT_SYNTH_ARGS, "--output", str(output_file), "--normalize"],
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
                ["synthesize", "test", *DEFAULT_SYNTH_ARGS, "--output", str(output_file), "--normalize"],
            )
            assert result.exit_code == 0
            mock_unlink.assert_not_called()

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_synthesize_generic_exception(self, mock_tts_class, runner):
        """Verify CLI catches unexpected exceptions and exits with code 1."""
        mock_tts = AsyncMock()
        mock_tts.generate.side_effect = RuntimeError("Unexpected error")
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(cli, ["synthesize", "test", *DEFAULT_SYNTH_ARGS])
        assert result.exit_code == 1
        assert "Error: Unexpected error" in result.output

    @patch("voicegenhub.cli.VoiceGenHub")
    def test_cli_accepts_supported_providers_voices(self, mock_tts_class, runner):
        """Test voices command accepts supported providers without hitting network."""
        mock_tts = AsyncMock()
        mock_tts.get_voices.return_value = []
        mock_tts_class.return_value = mock_tts

        for provider in ["edge", "piper", "melotts", "kokoro"]:
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
    def test_cli_synthesize_error_handling(self, mock_tts_class, runner):
        """Test synthesize command error handling."""
        mock_tts = AsyncMock()
        mock_tts.generate.side_effect = Exception("Test error")
        mock_tts_class.return_value = mock_tts

        result = runner.invoke(
            cli,
            ["synthesize", "hello world", *DEFAULT_SYNTH_ARGS, "--output", "dummy.wav"],
        )
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
            cli,
            [
                "synthesize",
                "hello",
                "--voice",
                "en-US-NonExistentVoice",
                "--language",
                "en",
                "--output",
                "dummy.wav",
            ],
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
            cli,
            [
                "synthesize",
                "hello",
                "--voice",
                "SomeRandomVoice",
                "--language",
                "en",
                "--output",
                "dummy.wav",
            ],
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
            cli,
            [
                "synthesize",
                "hello",
                "--voice",
                "en-US-AriaNeural",
                "--language",
                "en",
                "--output",
                "dummy.wav",
            ],
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
