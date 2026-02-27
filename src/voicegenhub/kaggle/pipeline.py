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
    texts: list,
    voice: str,
    language: str,
    model_id: str,
    dtype: str,
    seed: int = 42,
    temperature: float = 0.7,
    instruct: str = "",
) -> dict:
    """Build the Jupyter notebook content for Kaggle GPU batch execution.

    Generates one audio file per text entry (audio_001.wav, audio_002.wav, …)
    and writes a manifest.json that maps each filename to its source text.

    When *instruct* is non-empty, each text is generated via
    ``generate_custom_voice(… instruct=instruct)`` so Qwen3's voice-design /
    emotion capabilities are used even on remote Kaggle GPUs.
    """

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

    # Embed the texts list and metadata directly into the notebook cell
    gen_code = textwrap.dedent(f"""\
        import json
        import torch
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel

        MODEL_ID    = {json.dumps(model_id)}
        VOICE       = {json.dumps(voice)}
        LANGUAGE    = {json.dumps(qwen_language)}
        TEXTS       = {json.dumps(texts)}
        OUTPUT_DIR  = "/kaggle/working"
        SEED        = {seed}
        TEMPERATURE = {temperature}
        INSTRUCT    = {json.dumps(instruct)}

        # Pin global seed for reproducibility across runs
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f"CUDA available: {{torch.cuda.is_available()}}")
        if torch.cuda.is_available():
            print(f"GPU: {{torch.cuda.get_device_name(0)}}")
        print(f"Seed: {{SEED}}  Temperature: {{TEMPERATURE}}")
        if INSTRUCT:
            print(f"Instruct: {{INSTRUCT}}")

        print(f"Loading model: {{MODEL_ID}}")
        model = Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16,
        )

        manifest = []
        for i, text in enumerate(TEXTS, start=1):
            filename = f"audio_{{i:03d}}.wav"
            out_path = f"{{OUTPUT_DIR}}/{{filename}}"
            print(f"[{{i}}/{{len(TEXTS)}}] Generating: {{text[:80]}}")
            # Re-seed before each text so every audio is independently reproducible
            torch.manual_seed(SEED + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED + i)
            gen_kwargs = dict(
                text=text,
                language=LANGUAGE,
                speaker=VOICE,
                temperature=TEMPERATURE,
                top_p=0.9,
                repetition_penalty=1.1,
            )
            if INSTRUCT:
                gen_kwargs["instruct"] = INSTRUCT
            wavs, sr = model.generate_custom_voice(**gen_kwargs)
            sf.write(out_path, wavs[0], sr)
            duration = len(wavs[0]) / sr
            print(f"  -> {{filename}}  ({{duration:.2f}}s @ {{sr}} Hz)")
            manifest.append({{"index": i, "file": filename, "text": text, "duration_sec": round(duration, 2)}})

        manifest_path = f"{{OUTPUT_DIR}}/manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        print(f"\\nDone — {{len(TEXTS)}} audio files + manifest.json written to {{OUTPUT_DIR}}")
        for entry in manifest:
            print(f"  {{entry['file']}}  {{entry['duration_sec']}}s  {{entry['text'][:60]}}")
    """)

    summary_lines = [f"- `audio_{i:03d}.wav`: {t[:80]}{'…' if len(t) > 80 else ''}\n" for i, t in enumerate(texts, 1)]
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
                    "# VoiceGenHub — Qwen3-TTS GPU Batch Generation\n",
                    f"**Model:** `{model_id}`  **Voice:** {voice}  **Language:** {qwen_language}\n\n",
                    f"**{len(texts)} texts to synthesize:**\n",
                ] + summary_lines,
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
        texts,
        voice: str = "Ryan",
        language: str = "en",
        output_dir: Optional[str] = None,
        gpu_type: str = "p100",
        seed: int = 42,
        temperature: float = 0.7,
        instruct: str = "",
    ) -> list:
        """
        Run the full Kaggle Qwen3-TTS batch pipeline.

        Args:
            texts: A single text string or a list of text strings to synthesize.
                   Each text produces one audio file (audio_001.wav, …).
            voice: Speaker name (e.g. "Ryan", "Serena").
            language: ISO language code (e.g. "en", "zh").
            output_dir: Local directory for all downloaded files.
                        Defaults to a timestamped folder in the cwd.
            gpu_type: Kaggle accelerator type ("p100", "t4").
            seed: Random seed for reproducible generation.
            temperature: Sampling temperature.
            instruct: Qwen3 instruct string for voice design / emotion control.

        Returns:
            List of Paths to the downloaded audio files.
            A manifest.json is written alongside the audio files linking each
            filename to its source text.
        """
        # Normalise: accept both str and list[str]
        if isinstance(texts, str):
            texts = [texts]

        username = _detect_kaggle_username()
        kernel_id = f"{username}/{self.kernel_slug}"

        if output_dir is None:
            from datetime import datetime
            output_dir = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{gpu_type}"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting Kaggle Qwen3-TTS batch pipeline",
            kernel_id=kernel_id,
            model=self.model_id,
            voice=voice,
            language=language,
            gpu_type=gpu_type,
            num_texts=len(texts),
        )

        # 1. Build notebook + push
        with tempfile.TemporaryDirectory() as tmpdir:
            notebook_filename = "qwen3_tts.ipynb"
            notebook_path = Path(tmpdir) / notebook_filename
            metadata_path = Path(tmpdir) / "kernel-metadata.json"

            notebook = _build_notebook_source(
                texts=texts,
                voice=voice,
                language=language,
                model_id=self.model_id,
                dtype=self.dtype,
                seed=seed,
                temperature=temperature,
                instruct=instruct,
            )
            notebook_path.write_text(json.dumps(notebook, indent=2))

            # Save a copy of the submitted notebook to the output folder for traceability
            submitted_nb_dest = output_path / "submitted_notebook.ipynb"
            submitted_nb_dest.write_text(json.dumps(notebook, indent=2))
            logger.info(f"Submitted notebook saved: {submitted_nb_dest}")

            metadata = _build_kernel_metadata(
                username, self.kernel_slug, notebook_filename, gpu_type=gpu_type
            )
            metadata_path.write_text(json.dumps(metadata, indent=2))

            logger.info(f"Pushing kernel to Kaggle: {kernel_id} (accelerator: {gpu_type})")
            try:
                acc_flag = "nvidia-p100-1" if gpu_type == "p100" else "nvidia-t4-2"
                push_result = _run_cmd(
                    ["kaggle", "kernels", "push", "-p", tmpdir, "--accelerator", acc_flag],
                    capture=True,
                    check=True,
                )
                push_out = push_result.stdout.strip()
                logger.info(f"Push result: {push_out}")

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

        # 3. Download all output files (audio_*.wav + manifest.json)
        local_files = self._download_output(kernel_id, output_path, len(texts))

        logger.info(
            "Kaggle Qwen3-TTS batch pipeline complete",
            output_dir=str(output_path),
            num_files=len(local_files),
        )
        return local_files

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
        num_texts: int,
    ) -> list:
        """Download all kernel outputs (audio_*.wav + manifest.json) to output_path.

        Returns a list of Path objects for each downloaded audio file.
        """
        with tempfile.TemporaryDirectory() as dl_dir:
            logger.info(f"Downloading kernel outputs from {kernel_id}…")
            _run_cmd(
                ["kaggle", "kernels", "output", kernel_id, "-p", dl_dir],
                capture=True,
                check=True,
            )

            dl_path = Path(dl_dir)

            # Extract any zip archives first
            for zf in list(dl_path.rglob("*.zip")):
                logger.info(f"Extracting {zf.name}…")
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(dl_path)

            wav_files = sorted(dl_path.rglob("*.wav"))
            manifest_files = list(dl_path.rglob("manifest.json"))

            # Always copy logs and executed notebook — survive even if no wavs
            for log_file in dl_path.rglob("*.log"):
                dest = output_path / log_file.name
                shutil.copy2(log_file, dest)
                logger.info(f"Kernel log saved: {dest}  ({dest.stat().st_size:,} bytes)")

            for nb_file in dl_path.rglob("*.ipynb"):
                dest = output_path / "executed_notebook.ipynb"
                shutil.copy2(nb_file, dest)
                logger.info(f"Executed notebook saved: {dest}  ({dest.stat().st_size:,} bytes)")

            if not wav_files:
                all_files = list(dl_path.rglob("*"))
                file_list = ", ".join(f.name for f in all_files if f.is_file())
                raise FileNotFoundError(
                    f"No .wav files found in kernel output. Downloaded files: {file_list}\n"
                    f"Check kernel logs: https://www.kaggle.com/code/{kernel_id}"
                )

            # Copy all wav files
            local_wavs = []
            for wav in wav_files:
                dest = output_path / wav.name
                shutil.copy2(wav, dest)
                logger.info(f"Audio saved locally: {dest}  ({dest.stat().st_size:,} bytes)")
                local_wavs.append(dest)

            # Copy manifest.json if present
            if manifest_files:
                manifest_dest = output_path / "manifest.json"
                shutil.copy2(manifest_files[0], manifest_dest)
                logger.info(f"Manifest saved locally: {manifest_dest}")
            else:
                # Generate a minimal fallback manifest
                manifest_dest = output_path / "manifest.json"
                fallback = [
                    {"index": i + 1, "file": wav.name, "text": f"(text {i + 1} of {num_texts})"}
                    for i, wav in enumerate(local_wavs)
                ]
                manifest_dest.write_text(
                    json.dumps(fallback, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                logger.info(f"Fallback manifest written: {manifest_dest}")

            return local_wavs
