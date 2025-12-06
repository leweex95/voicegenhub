"""Unit tests for the StabilityAI sound effect generator."""
from unittest.mock import MagicMock, patch

import pytest

from voicegenhub.content.effect import (
    EffectConfigurationError,
    EffectGenerationError,
    StableAudioEffectGenerator,
)


def test_generator_requires_model_load_success(tmp_path):
    """Test that generator properly initializes the model."""
    with patch("voicegenhub.content.effect.get_pretrained_model") as mock_load:
        mock_model = MagicMock()
        mock_config = {"sample_rate": 48000, "sample_size": 65536}
        mock_load.return_value = (mock_model, mock_config)

        generator = StableAudioEffectGenerator()
        assert generator.model_id == "stabilityai/stable-audio-open-1.0"
        assert generator._sample_rate == 48000
        assert generator._sample_size == 65536


def test_generator_model_load_failure():
    """Test that generator raises error when model loading fails."""
    with patch("voicegenhub.content.effect.get_pretrained_model") as mock_load:
        mock_load.side_effect = RuntimeError("Model download failed")

        with pytest.raises(EffectConfigurationError):
            StableAudioEffectGenerator()


def test_generate_success(tmp_path):
    """Test successful audio generation."""
    with patch("voicegenhub.content.effect.get_pretrained_model") as mock_load, \
         patch("voicegenhub.content.effect.generate_diffusion_cond") as mock_gen, \
         patch("voicegenhub.content.effect.torchaudio.save") as mock_save, \
         patch("voicegenhub.content.effect.rearrange") as mock_rearrange, \
         patch("voicegenhub.content.effect.torch") as mock_torch:

        mock_model = MagicMock()
        mock_config = {"sample_rate": 48000, "sample_size": 65536}
        mock_load.return_value = (mock_model, mock_config)

        mock_output = MagicMock()
        mock_gen.return_value = mock_output
        mock_rearrange.return_value = mock_output

        # Mock the tensor operations properly
        mock_max_result = MagicMock()
        mock_max_result.item.return_value = 2.0  # Simulate max value > 1e-6
        mock_torch.max.return_value = mock_max_result

        mock_tensor = MagicMock()
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.div.return_value = mock_tensor
        mock_tensor.clamp.return_value = mock_tensor
        mock_tensor.mul.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.cpu.return_value = mock_tensor
        mock_tensor.numpy.return_value = MagicMock()  # Mock numpy array
        mock_output.to.return_value = mock_tensor

        generator = StableAudioEffectGenerator()
        output_file = tmp_path / "test.wav"

        result = generator.generate(
            prompt="artillery boom",
            output_path=output_file,
            duration=30,
            output_format="wav",
            guidance_scale=7.0,
            seed=42,
        )

        assert result.path == output_file
        assert result.metadata["model"] == "stabilityai/stable-audio-open-1.0"
        assert result.metadata["duration"] == 30
        assert result.metadata["guidance_scale"] == 7.0
        mock_save.assert_called_once()


def test_generate_empty_prompt_fails(tmp_path):
    """Test that empty prompt raises error."""
    with patch("voicegenhub.content.effect.get_pretrained_model") as mock_load:
        mock_model = MagicMock()
        mock_config = {"sample_rate": 48000, "sample_size": 65536}
        mock_load.return_value = (mock_model, mock_config)

        generator = StableAudioEffectGenerator()

        with pytest.raises(EffectGenerationError):
            generator.generate(
                prompt="",
                output_path=tmp_path / "test.wav",
            )
