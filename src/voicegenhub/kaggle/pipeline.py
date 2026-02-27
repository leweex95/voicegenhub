"""
Kaggle GPU Pipeline for Qwen3-TTS Remote Generation.

Pushes a notebook to Kaggle, runs it on a free P100 GPU,
polls for completion, and downloads the generated audio automatically.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import zipfile
from pathlib import Path
from typing import Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_SETTINGS_PATH = Path(__file__).parent / "config" / "kaggle_settings.json"
_KERNEL_SLUG = "voicegenhub-qwen3-tts"


def _load_settings() -> dict:
    """Load Kaggle pipeline settings from config JSON."""
    try:
        with open(_DEFAULT_SETTINGS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "deployment_timeout_minutes": 60,
            "polling_interval_seconds": 60,
            "retry_interval_seconds": 60,
        }


def _detect_kaggle_username() -> str:
    """Detect Kaggle username from credentials or env."""
    # 1. Environment variable
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]

    # 2. ~/.kaggle/kaggle.json
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            with open(kaggle_json) as f:
                creds = json.load(f)
            return creds.get("username", "")
        except Exception:
            pass

    raise RuntimeError(
        "Kaggle username not found. Set KAGGLE_USERNAME env var "
        "or ensure ~/.kaggle/kaggle.json exists with 'username' field."
    )


def _build_notebook_source(
    text: str,
    voice: str,
    language: str,
    model_id: str,
    dtype: str,
    output_filename: str,
) -> dict:
    """Build the Jupyter notebook content for Kaggle GPU execution."""

    # Language mapping (CLI code → Qwen language string)
    language_map = {
        "en": "English",
        "zh": "Chinese",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "pt": "Portuguese",
        "ru": "Russian",
        "es": "Spanish",
    }
    qwen_language = language_map.get(language.lower(), "English")

    install_code = textwrap.dedent("""\
        import subprocess, sys

        def pip_install(*packages):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

        pip_install("transformers>=4.40.0", "accelerate>=0.27.0", "tokenizers")
        pip_install("qwen-tts")
        try:
            pip_install("flash-attn", "--no-cache-dir")
        except Exception as e:
            print(f"flash-attn install skipped (non-fatal): {e}")
        pip_install("soundfile")
    """)

    gen_code = textwrap.dedent(f"""\
        import torch
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel

        MODEL_ID = {json.dumps(model_id)}
        OUTPUT_PATH = "/kaggle/working/{output_filename}"

        print(f"CUDA available: {{torch.cuda.is_available()}}")
        if torch.cuda.is_available():
            print(f"GPU: {{torch.cuda.get_device_name(0)}}")

        print(f"Loading model: {{MODEL_ID}}")
        model = Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16,
        )

        print("Generating speech...")
        wavs, sr = model.generate_custom_voice(
            text={json.dumps(text)},
            language={json.dumps(qwen_language)},
            speaker={json.dumps(voice)},
        )

        sf.write(OUTPUT_PATH, wavs[0], sr)
        print(f"Audio saved to {{OUTPUT_PATH}}")
        print(f"Sample rate: {{sr}} Hz, Duration: {{len(wavs[0])/sr:.2f}}s")
    """)

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "intro",
                "metadata": {},
                "source": [
                    "# VoiceGenHub — Qwen3-TTS GPU Generation\n",
                    f"**Model:** `{model_id}`  \n",
                    f"**Text:** {text[:120]}{'...' if len(text) > 120 else ''}  \n",
                    f"**Voice:** {voice}  **Language:** {qwen_language}\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "install",
                "metadata": {},
                "outputs": [],
                "source": install_code.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "generate",
                "metadata": {},
                "outputs": [],
                "source": gen_code.splitlines(keepends=True),
            },
        ],
    }
    return notebook


def _build_kernel_metadata(
    username: str, kernel_slug: str, notebook_filename: str, gpu_type: str = "p100"
) -> dict:
    """Build Kaggle kernel-metadata.json.

    IMPORTANT: The 'title' must slugify to exactly the same value as the slug
    portion of the 'id' field.  Kaggle derives the kernel slug from the title
    (spaces→hyphens, lowercase) and ignores the 'id' slug portion on creation.
    When they differ, every subsequent push hits a 409 Conflict because Kaggle
    already owns the title-derived slug.  Keep title = "VoiceGenHub Qwen3 TTS"
    so it slugifies to "voicegenhub-qwen3-tts", matching _KERNEL_SLUG.
    """
    return {
        "id": f"{username}/{kernel_slug}",
        "title": "VoiceGenHub Qwen3 TTS",
        "code_file": notebook_filename,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def _resolve_kaggle_executable() -> str:
    """
    Resolve the 'kaggle' CLI executable.

    Priority:
    1. Same directory as the current Python executable (venv Scripts/)
    2. System PATH
    """
    python_dir = Path(sys.executable).parent
    for candidate in ("kaggle.exe", "kaggle"):
        path = python_dir / candidate
        if path.exists():
            return str(path)

    # Fall back to PATH
    found = shutil.which("kaggle")
    if found:
        return found

    raise FileNotFoundError(
        "Could not find the 'kaggle' CLI executable. "
        "Install it with: pip install kaggle"
    )


def _extract_kernel_id_from_push(push_stdout: str, fallback: str) -> str:
    """
    Extract the actual kernel ID from the push output.

    The push command prints something like:
      "Kernel version 1 successfully pushed.  Please check progress at
       https://www.kaggle.com/code/leventecsibi/my-kernel-slug"

    We parse the URL path to get the actual slug Kaggle used.
    """
    import re
    match = re.search(r"kaggle\.com/code/([^/\s]+/[^/\s]+)", push_stdout)
    if match:
        return match.group(1)
    return fallback


def _run_cmd(args, capture=True, check=True):
    """Run a shell command. Resolves 'kaggle' to the correct venv executable."""
    resolved = list(args)
    if resolved and resolved[0] == "kaggle":
        resolved[0] = _resolve_kaggle_executable()
    logger.debug(f"Running: {' '.join(str(a) for a in resolved)}")
    result = subprocess.run(
        resolved,
        capture_output=capture,
        text=True,
        check=check,
    )
    return result


class KaggleQwenPipeline:
    """
    End-to-end pipeline: generate Qwen3-TTS audio on Kaggle P100 GPU
    and download the result locally.

    Workflow:
    1. Build a Jupyter notebook with the user's text/voice/model parameters.
    2. Push it to Kaggle with GPU enabled (P100).
    3. Poll until the kernel finishes.
    4. Download the output `.wav` file.
    5. Place it into a timestamped output directory.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        dtype: str = "float16",
        kernel_slug: str = _KERNEL_SLUG,
        settings_path: Optional[Path] = None,
        timeout_minutes: Optional[int] = None,
        poll_interval_seconds: Optional[int] = None,
    ):
        self.model_id = model_id
        self.dtype = dtype
        self.kernel_slug = kernel_slug
        self._settings = _load_settings() if settings_path is None else json.loads(Path(settings_path).read_text())
        # CLI-provided values take precedence over settings file
        self._timeout_minutes = timeout_minutes if timeout_minutes is not None else self._settings.get("deployment_timeout_minutes", 60)
        self._poll_interval = poll_interval_seconds if poll_interval_seconds is not None else self._settings.get("polling_interval_seconds", 60)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        text: str,
        voice: str = "Ryan",
        language: str = "en",
        output_dir: Optional[str] = None,
        output_filename: str = "qwen3_tts.wav",
        gpu_type: str = "p100",  # "p100" or "t4"
    ) -> Path:
        """
        Run the full Kaggle Qwen3-TTS pipeline.

        Args:
            text: Text to synthesize.
            voice: Speaker name (e.g. "Ryan", "Serena").
            language: ISO language code (e.g. "en", "zh").
            output_dir: Local directory for the downloaded audio file.
                        Defaults to a timestamped folder in the cwd.
            output_filename: Filename for the generated audio on Kaggle.
            gpu_type: Kaggle accelerator type ("p100", "t4").

        Returns:
            Path to the downloaded audio file.
        """
        username = _detect_kaggle_username()
        kernel_id = f"{username}/{self.kernel_slug}"

        if output_dir is None:
            from datetime import datetime
            output_dir = datetime.now().strftime("%Y%m%d")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting Kaggle Qwen3-TTS pipeline",
            kernel_id=kernel_id,
            model=self.model_id,
            voice=voice,
            language=language,
            gpu_type=gpu_type,
        )

        # 1. Build notebook + push
        with tempfile.TemporaryDirectory() as tmpdir:
            notebook_filename = "qwen3_tts.ipynb"
            notebook_path = Path(tmpdir) / notebook_filename
            metadata_path = Path(tmpdir) / "kernel-metadata.json"

            notebook = _build_notebook_source(
                text=text,
                voice=voice,
                language=language,
                model_id=self.model_id,
                dtype=self.dtype,
                output_filename=output_filename,
            )
            notebook_path.write_text(json.dumps(notebook, indent=2))

            metadata = _build_kernel_metadata(
                username, self.kernel_slug, notebook_filename, gpu_type=gpu_type
            )
            metadata_path.write_text(json.dumps(metadata, indent=2))

            logger.info(f"Pushing kernel to Kaggle: {kernel_id} (accelerator: {gpu_type})")
            try:
                # Accelerator flag to ensure correct resource allocation
                acc_flag = "nvidia-p100-1" if gpu_type == "p100" else "nvidia-t4-2"
                push_result = _run_cmd(
                    ["kaggle", "kernels", "push", "-p", tmpdir, "--accelerator", acc_flag],
                    capture=True,
                    check=True,
                )
                push_out = push_result.stdout.strip()
                logger.info(f"Push result: {push_out}")

                # Kaggle may create the kernel under a different slug than the
                # metadata 'id' field (it slugifies the 'title' instead when
                # they differ). Parse the actual URL from the push output.
                actual_kernel_id = _extract_kernel_id_from_push(push_out, kernel_id)
                if actual_kernel_id != kernel_id:
                    logger.info(
                        f"Kaggle resolved kernel slug: {actual_kernel_id} "
                        f"(metadata had: {kernel_id})"
                    )
                    kernel_id = actual_kernel_id
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"kaggle kernels push failed (exit {exc.returncode}).\n"
                    f"stdout: {exc.stdout.strip()}\n"
                    f"stderr: {exc.stderr.strip()}"
                ) from exc

        # 2. Poll until done
        self._poll_until_complete(kernel_id)

        # 3. Download output
        local_wav = self._download_output(kernel_id, output_path, output_filename)

        logger.info(
            "Kaggle Qwen3-TTS pipeline complete",
            output=str(local_wav),
        )
        return local_wav

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _poll_until_complete(self, kernel_id: str) -> None:
        """Poll Kaggle kernel status until it completes or times out."""
        timeout_seconds = self._timeout_minutes * 60
        elapsed = 0

        logger.info(
            f"Polling kernel status (timeout: {self._timeout_minutes}m, "
            f"interval: {self._poll_interval}s)…",
            kernel_id=kernel_id,
        )

        while elapsed < timeout_seconds:
            try:
                result = _run_cmd(
                    ["kaggle", "kernels", "status", kernel_id],
                    capture=True,
                    check=True,
                )
                status_line = result.stdout.strip()
                logger.info(f"Kernel status: {status_line}")

                status_lower = status_line.lower()
                if "complete" in status_lower:
                    logger.info("Kernel finished successfully.")
                    return
                elif "error" in status_lower or "cancel" in status_lower:
                    raise RuntimeError(
                        f"Kaggle kernel ended with non-successful status: {status_line}\n"
                        "Check the kernel logs at https://www.kaggle.com/code"
                    )
            except subprocess.CalledProcessError as e:
                logger.warning(f"Status check failed: {e.stderr.strip()}, retrying…")

            time.sleep(self._poll_interval)
            elapsed += self._poll_interval

        raise TimeoutError(
            f"Kaggle kernel did not complete within {self._timeout_minutes} minutes. "
            f"Check manually: https://www.kaggle.com/code/{kernel_id}"
        )

    def _download_output(
        self,
        kernel_id: str,
        output_path: Path,
        output_filename: str,
    ) -> Path:
        """Download kernel output and extract the audio file."""
        with tempfile.TemporaryDirectory() as dl_dir:
            logger.info(f"Downloading kernel outputs from {kernel_id}…")
            _run_cmd(
                ["kaggle", "kernels", "output", kernel_id, "-p", dl_dir],
                capture=True,
                check=True,
            )

            # Kaggle downloads a zip file; find and extract it
            dl_path = Path(dl_dir)
            wav_files = list(dl_path.rglob("*.wav"))
            zip_files = list(dl_path.rglob("*.zip"))

            # Extract zips first
            for zf in zip_files:
                logger.info(f"Extracting {zf.name}…")
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(dl_path)
                wav_files = list(dl_path.rglob("*.wav"))

            if not wav_files:
                # List what was downloaded for debugging
                all_files = list(dl_path.rglob("*"))
                file_list = ", ".join(f.name for f in all_files if f.is_file())
                raise FileNotFoundError(
                    f"No .wav file found in kernel output. Downloaded files: {file_list}\n"
                    f"Check kernel logs: https://www.kaggle.com/code/{kernel_id}"
                )

            # Find the right wav (matching output_filename if possible)
            target_wav = next(
                (f for f in wav_files if f.name == output_filename),
                wav_files[0],
            )

            dest = output_path / output_filename
            shutil.copy2(target_wav, dest)
            logger.info(f"Audio saved locally: {dest}")
            return dest
