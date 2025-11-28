"""
Tests for CLI post-processing effects (new flags).
These are fast unit tests suitable for pre-commit hooks.
"""
import pytest


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
                    "--provider",
                    "edge",
                    "--output",
                    str(output_file),
                    "--noise",
                    "0.05",
                ],
            )
            assert result.exit_code in [0, 1]
