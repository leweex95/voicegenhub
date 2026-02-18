"""Main Kaggle GPU pipeline orchestration using kaggle-gpu-connector."""

import logging
from pathlib import Path
from typing import Optional

from kaggle_connector.jobs import JobManager
from voicegenhub.kaggle.config_loader import load_kaggle_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_kaggle_pipeline(
    text: str,
    provider: str,
    voice: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Run complete Kaggle GPU pipeline using kaggle-gpu-connector.

    Args:
        text: Text to synthesize
        provider: Provider name (qwen, chatterbox)
        voice: Voice ID (provider-specific)
        output_path: Desired local output path for the audio file

    Returns:
        Path to the downloaded audio file
    """
    # Load configuration
    config = load_kaggle_config()
    kernel_id = config.get("kernel_id")
    if not kernel_id:
        raise ValueError("kernel_id not found in Kaggle configuration.")

    # Determine output filename
    if output_path:
        output_file = output_path.name
        dest_dir = output_path.parent
    else:
        output_file = "output.wav"
        dest_dir = Path.cwd() / "kaggle_outputs"

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Determine notebook to use
    notebook_dir = Path(__file__).parent / "notebooks"
    if provider == "qwen":
        notebook_name = "kaggle-qwen.ipynb"
    elif provider == "chatterbox":
        notebook_name = "kaggle-chatterbox.ipynb"
    else:
        raise ValueError(f"No Kaggle notebook available for provider: {provider}")

    notebook_path = notebook_dir / notebook_name
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    logging.info("=" * 80)
    logging.info("KAGGLE GPU PIPELINE (via kaggle-gpu-connector)")
    logging.info(f"Provider: {provider}")
    logging.info(f"Voice: {voice}")
    logging.info(f"Text: {text[:50]}..." if len(text) > 50 else f"Text: {text}")
    logging.info(f"Output: {dest_dir / output_file}")
    logging.info("=" * 80)

    # Determine user and slug for unique identification
    if "/" in kernel_id:
        username, _ = kernel_id.split("/", 1)
    else:
        username = "leventecsibi"  # Default fallback

    # Create provider-specific slug to avoid collisions
    slug = f"vgh-{provider.replace('_', '-')}"
    unique_kernel_id = f"{username}/{slug}"

    # Initialize JobManager
    manager = JobManager(kernel_id=unique_kernel_id)

    # Step 1: Update notebook parameters
    logging.info(f"Step 1: Injecting parameters into {notebook_name}...")
    params = {
        "TEXT": text,
        "OUTPUT_FILE": output_file
    }
    if voice:
        params["VOICE_ID"] = voice

    manager.edit_notebook_params(str(notebook_path), params)

    # Step 2: Deploy to Kaggle
    logging.info("Step 2: Deploying to Kaggle...")
    # Use a temporary deployment directory to avoid file access conflicts and keep deployment clean
    import tempfile
    import shutil
    with tempfile.TemporaryDirectory() as deploy_tmp_dir:
        deploy_tmp_path = Path(deploy_tmp_dir)

        # Create metadata manually to ensure kernel_type="notebook" is set correctly
        # and specify the accelerator.
        logging.info(f"Creating metadata for {unique_kernel_id} in {deploy_tmp_path}...")
        manager.create_metadata(
            dest_dir=str(deploy_tmp_path),
            kernel_id=unique_kernel_id,
            code_file=notebook_name,
            kernel_type="notebook",
            enable_gpu=True,
            enable_internet=True,
            accelerator="nvidia-t4"
        )

        # Copy the notebook file to the temporary deployment directory
        shutil.copy2(notebook_path, deploy_tmp_path / notebook_name)
        logging.info(f"Copied {notebook_name} to deployment directory.")

        # Log metadata content for debugging
        with open(deploy_tmp_path / "kernel-metadata.json", "r") as f:
            logging.info(f"Metadata content: {f.read()}")

        # Deploy using the temp directory.
        # We don't pass kernel_id here so it uses the metadata we just created.
        manager.deploy(
            kernel_path=str(deploy_tmp_path),
            notebook_file=str(notebook_path),
            wait=True,
            timeout_min=config.get("deployment_timeout_minutes", 30)
        )

        # Step 3: Stream logs and wait for completion
        logging.info("Step 3: Streaming logs from Kaggle kernel...")

        # We wait until the kernel is actually running before we start streaming logs
        # to avoid the 'Failed to fetch logs' error when it's still queued.
        import time
        max_queued_wait = 120  # 2 minutes of queue wait
        poll_start_time = time.time()
        while time.time() - poll_start_time < max_queued_wait:
            status = manager.get_status().lower()
            if "running" in status or "complete" in status or "error" in status:
                break
            logging.info(f"Kernel is {status}, waiting to start streaming logs...")
            time.sleep(15)

        manager.stream_logs(polling_interval=15)

        status = manager.get_status().lower()
        if "complete" not in status:
            logging.error(f"Kernel failed with status: {status}")
            raise RuntimeError(f"Kernel failed with status: {status}. Check logs above for errors.")

        # Step 4: Download outputs
        logging.info("Step 4: Downloading results...")
        # Manually download for now as SelectiveDownloader defaults to images
        import subprocess
        import sys
        kaggle_cmd = [sys.executable, "-m", "kaggle.cli"]

        subprocess.run(
            [*kaggle_cmd, "kernels", "output", unique_kernel_id, "-p", str(dest_dir)],
            check=False
        )

    # Find the downloaded audio file
    audio_file = dest_dir / output_file
    if not audio_file.exists():
        # Try to find any audio file in the directory
        audio_files = list(dest_dir.glob("*.wav"))
        if audio_files:
            audio_file = audio_files[0]
        else:
            raise FileNotFoundError(f"Audio file not found in {dest_dir}")

    logging.info(f"Pipeline complete! Audio saved to: {audio_file}")
    return audio_file
