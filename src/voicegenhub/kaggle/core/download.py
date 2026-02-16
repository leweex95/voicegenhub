"""Download audio files from Kaggle kernel outputs."""

import logging
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path
from typing import List, Set
import sys

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".aac"}
ALLOWED_NON_AUDIO_FILES = {"cli_command.txt", "voicegenhub.log"}


def _get_kaggle_command() -> List[str]:
    """Get the appropriate command to invoke Kaggle CLI."""
    def find_pyproject_toml():
        current = Path.cwd()
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                return True
            current = current.parent
        return False

    if shutil.which("poetry") and find_pyproject_toml():
        try:
            result = subprocess.run(
                ["poetry", "run", "python", "-c", "import kaggle"],
                capture_output=True, check=False, timeout=10
            )
            if result.returncode == 0:
                return ["poetry", "run", "python", "-m", "kaggle.cli"]
        except Exception:
            pass

    return [sys.executable, "-m", "kaggle.cli"]


def _handle_remove_readonly(func, path, exc_info):
    """Force-remove read-only files during rmtree."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as err:
        logger.warning(f"Failed to delete artifact component {path}: {err}")


def _list_local_audio_files(dest_path: Path) -> Set[str]:
    """Scan local directory for audio files. Returns set of file names (not full paths)."""
    audio_files = set()
    for file_path in dest_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTENSIONS:
            audio_files.add(file_path.name)
    return audio_files


def download_from_kaggle(kernel_id: str, dest: str) -> bool:
    """Download ONLY audio files from Kaggle kernel by monitoring local directory.

    Starts the Kaggle CLI download process and monitors the destination directory.
    As soon as audio files are detected and stable (no new files for 5 seconds),
    immediately terminates the download process to prevent unnecessary files
    from being transferred.

    Args:
        kernel_id: Kaggle kernel ID (e.g., "username/kernel-name")
        dest: Destination directory for outputs

    Returns:
        True if audio files were successfully downloaded, False otherwise
    """
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting selective download from {kernel_id} to {dest_path}...")

    kaggle_cmd = _get_kaggle_command()

    # Start the download process
    process = subprocess.Popen(
        [*kaggle_cmd, "kernels", "output", kernel_id, "-p", str(dest_path).replace("\\", "/")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Monitor for audio files
    start_time = time.time()
    timeout_seconds = 120
    last_file_count = 0
    stable_count = 0
    stable_threshold = 3

    while process.poll() is None:
        time.sleep(2)

        if time.time() - start_time > timeout_seconds:
            logger.warning(f"Download timeout after {timeout_seconds} seconds")
            process.terminate()
            return False

        audio_files = _list_local_audio_files(dest_path)
        current_count = len(audio_files)

        if current_count > 0:
            logger.info(f"Found {current_count} audio file(s): {', '.join(sorted(audio_files))}")

            if current_count == last_file_count:
                stable_count += 1
                if stable_count >= stable_threshold:
                    logger.info("Audio files stable - terminating download early")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    return True
            else:
                stable_count = 0

            last_file_count = current_count

    # Process completed naturally
    audio_files = _list_local_audio_files(dest_path)
    if audio_files:
        logger.info(f"Download completed with {len(audio_files)} audio file(s)")
        return True
    else:
        logger.warning("Download completed but no audio files found")
        return False
