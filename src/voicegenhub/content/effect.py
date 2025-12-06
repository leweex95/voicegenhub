"""Sound effect generation helpers for VoiceGenHub."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

from ..utils.logger import get_logger

logger = get_logger(__name__)


class EffectGenerationError(RuntimeError):
    """Raised when effect generation fails."""


class EffectConfigurationError(ValueError):
    """Raised when configuration for effect generation is invalid."""


@dataclass
class EffectGenerationResult:
    """Represents the outcome of a sound effect generation request."""

    path: Path
    metadata: Dict[str, Any]


class StableAudioEffectGenerator:
    """Client for StabilityAI's stable audio local inference."""

    def __init__(
        self,
        model_id: str = "stabilityai/stable-audio-open-1.0",
        token: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing StableAudioEffectGenerator on device: {self._device}")

        try:
            self._model, self._model_config = get_pretrained_model(model_id)
            self._model = self._model.to(self._device)
            self._sample_rate = self._model_config["sample_rate"]
            self._sample_size = self._model_config["sample_size"]
        except Exception as exc:
            raise EffectConfigurationError(f"Failed to load model {model_id}: {exc}") from exc

    def generate(
        self,
        prompt: str,
        output_path: Path,
        duration: int = 30,
        output_format: str = "wav",
        guidance_scale: float = 7.0,
        seed: Optional[int] = None,
    ) -> EffectGenerationResult:
        """Generate a sound effect audio clip using the configured model."""
        if not prompt or not prompt.strip():
            raise EffectGenerationError("Prompt cannot be empty.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Generating audio: {prompt} (duration={duration}s, cfg_scale={guidance_scale})")

            conditioning = [{
                "prompt": prompt,
                "seconds_start": 0,
                "seconds_total": duration,
            }]

            try:
                # Seed must be within int32 range (0 to 2^31-1) to avoid numpy random overflow
                # stable-audio-tools has a bug using 2^32-1 which overflows on Windows
                # Always provide a valid seed to avoid the buggy fallback
                safe_seed = seed if seed is not None else 0
                if safe_seed == -1 or safe_seed > 2**31 - 1:
                    safe_seed = 42  # Use fixed seed if invalid

                output = generate_diffusion_cond(
                    self._model,
                    steps=100,
                    cfg_scale=guidance_scale,
                    conditioning=conditioning,
                    sample_size=self._sample_size,
                    sigma_min=0.3,
                    sigma_max=500,
                    sampler_type="dpmpp-3m-sde",
                    device=self._device,
                    seed=safe_seed,
                )
                logger.info(f"Raw output - shape: {output.shape}, dtype: {output.dtype}")
            except Exception as e:
                logger.error(f"Generation failed: {e}", exc_info=True)
                raise

            output = rearrange(output, "b d n -> d (b n)")
            logger.info(f"Output shape: {output.shape}, dtype: {output.dtype}, device: {output.device}")

            output = output.to(torch.float32).cpu()
            logger.info(f"After to float32 cpu - shape: {output.shape}, dtype: {output.dtype}")

            max_val = torch.max(torch.abs(output)).item()
            logger.info(f"Audio max value: {max_val}, min: {torch.min(output).item()}")

            if max_val > 1e-6:
                output = output / max_val

            output = torch.clamp(output, -1.0, 1.0)
            logger.info(f"After clamp - shape: {output.shape}, dtype: {output.dtype}, device: {output.device}")
            logger.info(f"Clamped min: {torch.min(output).item()}, max: {torch.max(output).item()}")

            # Convert to float32 first, then CPU
            output = output.float()
            output_np = output.cpu().numpy()
            logger.info(f"Numpy shape: {output_np.shape}, dtype: {output_np.dtype}")
            logger.info(f"Numpy min: {output_np.min()}, max: {output_np.max()}")

            # Safe conversion to int16
            output_scaled = (output_np * 32767.0).astype("int16")

            torchaudio.save(str(output_path), torch.from_numpy(output_scaled).unsqueeze(0) if output_scaled.ndim == 1 else torch.from_numpy(output_scaled), self._sample_rate)

        except Exception as exc:
            raise EffectGenerationError(f"Sound effect generation failed: {exc}") from exc

        metadata = {
            "model": self.model_id,
            "duration": duration,
            "format": output_format,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "sample_rate": self._sample_rate,
        }
        logger.info("Sound effect generation complete: %s", output_path)
        return EffectGenerationResult(path=output_path, metadata=metadata)

    def close(self) -> None:
        """Clean up model resources."""
        if hasattr(self, "_model") and self._model is not None:
            del self._model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
