"""Run Chatterbox synthesis on a user-owned Kaggle GPU kernel."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional

from .base import AudioFormat, ProviderCapabilities, TTSError, TTSRequest, TTSResponse
from .chatterbox import ChatterboxProvider

logger = logging.getLogger(__name__)


def _embed_request_payload(runner_code: str, payload: Dict[str, Any]) -> str:
    """Embed the request in the uploaded script (Kaggle doesn't upload sidecars)."""
    marker = "REQUEST_JSON = None"
    if marker not in runner_code:
        raise TTSError("Kaggle runner is missing its request payload marker")
    return runner_code.replace(marker, "REQUEST_JSON = " + repr(json.dumps(payload)), 1)


def _select_available_kernel(kernel_ids, get_status) -> str:
    """Select the first notebook with a known terminal status; fail closed otherwise."""
    terminal_markers = ("complete", "error", "cancel_acknowledged", "cancelled")
    busy = []
    for kernel_id in kernel_ids:
        status = get_status(kernel_id).lower()
        if status and any(marker in status for marker in terminal_markers):
            return kernel_id
        busy.append(f"{kernel_id}: {status or 'unknown'}")
    raise TTSError(
        "All configured Kaggle Chatterbox notebooks are busy; no new run was "
        "submitted (" + ", ".join(busy) + "). Configure a second notebook with "
        "VOICEGENHUB_KAGGLE_KERNEL_IDS to use Kaggle's other batch slot."
    )


def _kaggle_cli_env() -> Dict[str, str]:
    """Use UTF-8 for Kaggle CLI subprocesses (including remote log output)."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _download_kernel_logs(kernel_id: str, kaggle_command) -> str:
    """Fetch Kaggle's execution log without Windows console-encoding failures."""
    with tempfile.TemporaryDirectory(prefix="voicegenhub-kaggle-logs-") as tmp:
        result = subprocess.run(
            [*kaggle_command, "kernels", "output", kernel_id, "-p", tmp],
            capture_output=True, text=True, encoding="utf-8", check=False,
            env=_kaggle_cli_env(),
        )
        log_files = sorted(Path(tmp).rglob("*.log"))
        if log_files:
            logs = []
            for path in log_files:
                content = path.read_text(encoding="utf-8", errors="replace")
                try:
                    events = json.loads(content)
                except json.JSONDecodeError:
                    events = None
                if isinstance(events, list):
                    content = "".join(
                        f"[{event.get('stream_name', 'log')}] {event.get('data', '')}"
                        for event in events
                    )
                logs.append(f"--- {path.name} ---\n{content}")
            return "\n".join(logs)
        return result.stderr.strip()


def _wait_for_kernel(
    job, poll_interval_seconds: int, *, sleep=time.sleep, clock=time.monotonic
) -> str:
    """Poll until Kaggle reaches a terminal state; queued runs may wait indefinitely."""
    started_at = clock()
    last_status = None
    last_log_at = started_at
    active_markers = ("queued", "running", "pending", "initializing", "starting", "submitted")

    while True:
        status = job.get_status().lower()
        now = clock()
        if status != last_status:
            logger.info("Kaggle Chatterbox kernel status: %s", status or "unknown")
            last_status = status
            last_log_at = now

        if "complete" in status or "error" in status:
            logger.info("Kaggle Chatterbox kernel finished after %.0f seconds", now - started_at)
            return status
        if "cancel_acknowledged" in status or "cancelled" in status:
            raise TTSError(f"Kaggle Chatterbox kernel was cancelled (status: {status})")
        if not status or not any(marker in status for marker in active_markers):
            raise TTSError(
                f"Kaggle Chatterbox kernel returned an unknown status ({status or 'empty'}); "
                "stopping rather than treating it as safely queued or running."
            )

        if now - last_log_at >= 60:
            logger.info(
                "Still waiting for Kaggle Chatterbox kernel (%s; %.0f seconds elapsed)",
                status,
                now - started_at,
            )
            last_log_at = now
        sleep(poll_interval_seconds)


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
        default_kernel_id = self.config.get("kernel_id") or os.getenv(
            "VOICEGENHUB_KAGGLE_KERNEL_ID", f"{username}/voicegenhub-chatterbox"
        )
        configured_kernel_ids = self.config.get("kernel_ids") or os.getenv(
            "VOICEGENHUB_KAGGLE_KERNEL_IDS", ""
        )
        if isinstance(configured_kernel_ids, str):
            configured_kernel_ids = configured_kernel_ids.split(",")
        kernel_ids = list(dict.fromkeys(
            [default_kernel_id]
            + [kernel_id.strip() for kernel_id in configured_kernel_ids if kernel_id.strip()]
        ))
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
            runner = Path(__file__).with_name("kaggle_runner.py")
            runner_code = _embed_request_payload(runner.read_text(encoding="utf-8"), payload)
            (root / "runner.py").write_text(runner_code, encoding="utf-8")
            # The connector's resource check counts only running kernels, not
            # queued ones. Select a free notebook explicitly; the optional pool
            # lets callers use both Kaggle batch slots without overwriting the
            # status/output of a queued run on the first notebook.
            kernel_id = _select_available_kernel(
                kernel_ids, lambda candidate: JobManager(candidate).get_status()
            )

            manager = JobManager()
            manager.create_metadata(
                str(root), kernel_id=kernel_id, code_file="runner.py",
                kernel_type="script", enable_gpu=True, enable_internet=True,
                accelerator=accelerator,
            )
            # Do not rely on the connector's wait_for_resource(): it ignores
            # queued jobs and can therefore admit duplicate submissions.
            manager.deploy(str(root), wait=False)
            job = JobManager(kernel_id)
            poll_interval = max(1, int(self.config.get("poll_interval_seconds", 15)))
            status = _wait_for_kernel(job, poll_interval)
            if "error" in status:
                diagnostics = _download_kernel_logs(kernel_id, get_kaggle_command())
                detail = f"; logs: {diagnostics[-4000:]}" if diagnostics else ""
                raise TTSError(f"Kaggle Chatterbox kernel failed: {status}{detail}")
            output_dir = root / "outputs"
            result = subprocess.run(
                [*get_kaggle_command(), "kernels", "output", kernel_id, "-p", str(output_dir)],
                capture_output=True, text=True, encoding="utf-8", check=False,
                env=_kaggle_cli_env(),
            )
            if result.returncode:
                raise TTSError(f"Could not download Kaggle audio: {result.stderr.strip()}")
            audio_path = output_dir / "voicegenhub_output.wav"
            if not audio_path.is_file():
                raise TTSError("Kaggle job completed but did not produce voicegenhub_output.wav")
            return audio_path.read_bytes(), 24000
