"""Run Chatterbox synthesis on a user-owned Kaggle GPU kernel."""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, Optional

from .base import AudioFormat, ProviderCapabilities, TTSError, TTSRequest, TTSResponse
from .chatterbox import ChatterboxProvider


class KaggleChatterboxProvider(ChatterboxProvider):
    """Chatterbox provider that submits one synthesis job to Kaggle per request."""

    def __init__(self, name: str = "chatterbox-kaggle", config: Dict[str, Any] = None):
        super().__init__(name, config)

    @property
    def provider_id(self) -> str:
        return "chatterbox-kaggle"

    @property
    def display_name(self) -> str:
        return "Chatterbox TTS (Kaggle GPU)"

    async def initialize(self) -> None:
        """Validate that Kaggle tooling and credentials are available."""
        try:
            import kaggle  # noqa: F401
            from kaggle_connector import JobManager  # noqa: F401
        except ImportError as exc:
            raise TTSError(
                "Kaggle support is optional. Install VoiceGenHub with its kaggle extra."
            ) from exc
        try:
            from kaggle import api
            username = api.config_values.get("username")
            if not username:
                raise ValueError("Kaggle username is not configured")
        except Exception as exc:
            raise TTSError(f"Kaggle credentials are not configured: {exc}") from exc
        self._initialized = True

    async def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_emotions=True,
            max_text_length=1000,
            supported_formats=[AudioFormat.WAV],
            supported_sample_rates=[24000],
        )

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        if not self._initialized:
            await self.initialize()
        if request.extra_params.get("audio_prompt_path"):
            raise TTSError("Voice cloning via audio_prompt_path is not yet supported on Kaggle")
        language = (request.language or "en").split("-")[0].lower()
        if language not in self.get_supported_languages():
            raise TTSError(f"Chatterbox does not support language '{language}'")
        audio, rate = await asyncio.to_thread(self._run_remote, request)
        try:
            with wave.open(io.BytesIO(audio), "rb") as wav_file:
                rate = wav_file.getframerate()
                duration = wav_file.getnframes() / rate
        except (wave.Error, EOFError) as exc:
            raise TTSError(f"Kaggle returned invalid WAV audio: {exc}") from exc
        return TTSResponse(
            audio_data=audio,
            format=AudioFormat.WAV,
            sample_rate=rate,
            duration=duration,
            voice_used=request.voice_id,
            metadata={"execution": "kaggle", "accelerator": self.config.get("accelerator") or os.getenv("VOICEGENHUB_KAGGLE_ACCELERATOR", "nvidia-t4-x2")},
        )

    def _run_remote(self, request: TTSRequest) -> tuple[bytes, int]:
        from kaggle import api
        from kaggle_connector import JobManager
        from kaggle_connector.utils.cli import get_kaggle_command

        username = api.config_values["username"]
        kernel_id = self.config.get("kernel_id") or os.getenv(
            "VOICEGENHUB_KAGGLE_KERNEL_ID", f"{username}/voicegenhub-chatterbox"
        )
        accelerator = self.config.get("accelerator") or os.getenv(
            "VOICEGENHUB_KAGGLE_ACCELERATOR", "nvidia-t4-x2"
        )
        with tempfile.TemporaryDirectory(prefix="voicegenhub-kaggle-") as tmp:
            root = Path(tmp)
            payload = {
                "text": request.text,
                "voice": request.voice_id,
                "language": request.language,
                "exaggeration": request.extra_params.get("exaggeration", 0.5),
                "cfg_weight": request.extra_params.get("cfg_weight", 0.5),
            }
            (root / "request.json").write_text(json.dumps(payload), encoding="utf-8")
            runner = Path(__file__).with_name("kaggle_runner.py")
            shutil.copy2(runner, root / "runner.py")
            manager = JobManager()
            manager.create_metadata(
                str(root), kernel_id=kernel_id, code_file="runner.py",
                kernel_type="script", enable_gpu=True, enable_internet=True,
                accelerator=accelerator,
            )
            manager.deploy(str(root), wait=True)
            status = JobManager(kernel_id).poll_until_complete(
                poll_interval=int(self.config.get("poll_interval_seconds", 30))
            )
            if "error" in status.lower():
                raise TTSError(f"Kaggle Chatterbox kernel failed: {status}")
            output_dir = root / "outputs"
            result = subprocess.run(
                [*get_kaggle_command(), "kernels", "output", kernel_id, "-p", str(output_dir)],
                capture_output=True, text=True, check=False,
            )
            if result.returncode:
                raise TTSError(f"Could not download Kaggle audio: {result.stderr.strip()}")
            audio_path = output_dir / "voicegenhub_output.wav"
            if not audio_path.is_file():
                raise TTSError("Kaggle job completed but did not produce voicegenhub_output.wav")
            return audio_path.read_bytes(), 24000
