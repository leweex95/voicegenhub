"""Deploy voice generation notebooks to Kaggle."""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from voicegenhub.kaggle.config_loader import load_kaggle_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _update_param(source_lines, param_name, value, is_list=False):
    """Update a parameter in notebook source code lines.

    Args:
        source_lines: List of source code lines
        param_name: Name of parameter (e.g., "TEXT", "VOICE_ID")
        value: New value
        is_list: True if value should be formatted as Python list

    Returns:
        Updated source lines
    """
    for i, line in enumerate(source_lines):
        if line.strip().startswith(f"{param_name} ="):
            if is_list:
                source_lines[i] = f"{param_name} = {value}\n"
            elif isinstance(value, str):
                source_lines[i] = f"{param_name} = \"{value}\"\n"
            elif isinstance(value, bool):
                source_lines[i] = f"{param_name} = {value}\n"
            else:
                source_lines[i] = f"{param_name} = {value}\n"
            break
    return source_lines


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


def deploy_to_kaggle(
    text: str,
    provider: str,
    voice: Optional[str] = None,
    output_file: str = "output.wav",
    notebook: Optional[str] = None,
    kernel_path: Optional[Path] = None,
):
    """Deploy a voice generation task to Kaggle GPU.

    Args:
        text: Text to synthesize
        provider: Provider name (qwen, chatterbox)
        voice: Voice ID (provider-specific)
        output_file: Output filename
        notebook: Notebook filename to use
        kernel_path: Path to kernel directory (containing notebook and metadata)

    Returns:
        kernel_id: The deployed kernel ID
    """
    logging.info(f"Deploying {provider} voice generation to Kaggle...")

    # Determine notebook and kernel path based on provider
    if notebook is None:
        if provider == "qwen":
            notebook = "kaggle-qwen.ipynb"
        elif provider == "chatterbox":
            notebook = "kaggle-chatterbox.ipynb"
        else:
            raise ValueError(f"No Kaggle notebook available for provider: {provider}")

    if kernel_path is None:
        kernel_path = Path(__file__).parent.parent / "notebooks"

    nb_path = kernel_path / notebook
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")

    # Load notebook
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Update notebook parameters
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = cell["source"] if isinstance(cell["source"], list) else [cell["source"]]

            # Update parameters
            source = _update_param(source, "TEXT", text)
            source = _update_param(source, "OUTPUT_FILE", output_file)
            if voice:
                source = _update_param(source, "VOICE_ID", voice)

            cell["source"] = source

    # Save updated notebook
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    # Update kernel metadata
    metadata_path = kernel_path / "kernel-metadata.json"
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            kernel_meta = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # Create default metadata
        config = load_kaggle_config()
        kernel_meta = {
            "id": config["kernel_id"],
            "title": "VoiceGenHub GPU",
            "code_file": "",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [],
            "metadata": {
                "kernelspec": {
                    "name": "python3",
                    "display_name": "Python 3",
                    "language": "python"
                }
            }
        }
        logging.warning(f"Kernel metadata not found, using defaults: {metadata_path}")

    # Update code_file to point to the correct notebook
    kernel_meta["code_file"] = notebook
    kernel_meta["enable_gpu"] = "true"

    # Save metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(kernel_meta, f, indent=2)

    # Push to Kaggle
    logging.info(f"Pushing kernel from {kernel_path}...")
    kaggle_cmd = _get_kaggle_command()

    result = subprocess.run(
        [*kaggle_cmd, "kernels", "push", "-p", str(kernel_path)],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to push kernel: {result.stderr}")

    logging.info("Kernel pushed successfully")
    logging.info(f"Output: {result.stdout}")

    # Extract kernel ID from metadata
    kernel_id = kernel_meta["id"]
    return kernel_id
