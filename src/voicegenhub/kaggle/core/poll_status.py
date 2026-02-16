"""Poll Kaggle kernel status until completion."""

import subprocess
import time
import re
import logging
import sys
from pathlib import Path
from voicegenhub.kaggle.config_loader import load_kaggle_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _get_kaggle_command():
    """Get the appropriate command to invoke Kaggle CLI."""
    import shutil

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


def poll_kernel_status(kernel_id=None, poll_interval=None):
    """Poll Kaggle kernel status until complete or error."""
    config = load_kaggle_config()
    kernel_id = kernel_id or config["kernel_id"]
    poll_interval = poll_interval or config["polling_interval_seconds"]

    consecutive_errors = 0
    max_consecutive_errors = 5

    while True:
        kaggle_cmd = _get_kaggle_command()
        try:
            result = subprocess.run(
                [*kaggle_cmd, "kernels", "status", kernel_id],
                capture_output=True, text=True, encoding='utf-8', timeout=30
            )
        except subprocess.TimeoutExpired:
            consecutive_errors += 1
            logging.warning(f"Status check timed out ({consecutive_errors}/{max_consecutive_errors})")
            if consecutive_errors >= max_consecutive_errors:
                return "unknown"
            time.sleep(poll_interval)
            continue

        if result.returncode != 0:
            consecutive_errors += 1
            logging.warning(f"Error fetching status ({consecutive_errors}/{max_consecutive_errors}): {result.stderr.strip()}")
            if consecutive_errors >= max_consecutive_errors:
                return "unknown"
            time.sleep(poll_interval)
            continue

        consecutive_errors = 0

        match = re.search(r'has status "(.*)"', result.stdout)
        status = match.group(1) if match else "unknown"
        logging.info("Kernel status: %s", status)

        if status.lower() == "unknown":
            logging.warning("Unable to parse kernel status from output: %s", result.stdout)
            time.sleep(poll_interval)
            continue

        if status.lower() in ["kernelworkerstatus.complete", "kernelworkerstatus.error"]:
            logging.info("Kernel finished with status: %s", status)
            return status.lower()

        time.sleep(poll_interval)
