"""Main Kaggle GPU pipeline orchestration using kaggle-gpu-connector."""

import logging
import tempfile
import shutil
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

    # Determine output filename and destination directory
    # We must use a clean destination directory to avoid SelectiveDownloader
    # scanning the entire workspace (and its .venv) for "stable" file counts.
    if output_path:
        output_file = output_path.name
        # If output_path is just a filename, parent will be '.'
        # We'll use a specific subdirectory to avoid the workspace scan issue
        if str(output_path.parent) == ".":
            dest_dir = Path.cwd() / "kaggle_downloads"
        else:
            dest_dir = output_path.parent
    else:
        output_file = "voicegenhub_output.wav"
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
    import time
    timestamp = int(time.time()) % 10000
    slug = f"vgh-{provider.replace('_', '-')}-{timestamp}"
    unique_kernel_id = f"{username}/{slug}"

    # Initialize JobManager
    manager = JobManager(kernel_id=unique_kernel_id)

    # Step 1: Update notebook parameters
    logging.info(f"Step 1: Injecting parameters into {notebook_name}...")
    params = {
        "text": text,
        "output_file": output_file
    }
    if voice:
        params["voice"] = voice

    manager.edit_notebook_params(str(notebook_path), params)

    # Step 2: Deploy to Kaggle
    logging.info("Step 2: Deploying to Kaggle...")
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

        # Custom log streaming with timeout to avoid 10 min hangs
        logging.info("Step 3: Streaming logs from Kaggle kernel (with 5-minute guard)...")

        start_stream_time = time.time()
        timeout_limit = 330  # 5.5 minutes max execution time for a single sentence
        last_log_time = time.time()

        def log_callback(content: str):
            nonlocal last_log_time
            last_log_time = time.time()
            print(content, end="", flush=True)

        while True:
            status = manager.get_status().lower()
            current_time = time.time()

            # Use manager.get_logs() directly to check for new logs
            logs = manager.get_logs()
            if logs:
                # JobManager.stream_logs implementation detail:
                # We want to only print the NEW part.
                # Actually, let's just use JobManager.stream_logs logic but with a condition.
                pass

            # Since manager.stream_logs is blocking, we'll implement a custom poll loop here.
            # We'll use JobManager's get_logs() but keep track of last length.
            # To be safer, I'll just use a timeout on the loop and call manager.get_logs() manually.
            break # Jump to implementation below

        # Corrected Step 3 implementation:
        max_duration = 600 # 10 min
        poll_interval = 20
        last_logs = ""

        while time.time() - poll_start_time < max_duration:
            status = manager.get_status().lower()
            current_logs = manager.get_logs()

            if current_logs != last_logs:
                new_tokens = current_logs
                if last_logs and current_logs.startswith(last_logs):
                    new_tokens = current_logs[len(last_logs):]
                if new_tokens.strip():
                    print(new_tokens, end="", flush=True)
                    last_logs = current_logs

            if "complete" in status or "error" in status:
                break

            time.sleep(poll_interval)

        final_status = manager.get_status().lower()
        if "complete" not in final_status:
            # Check if we hit our internal timeout
            if time.time() - poll_start_time >= max_duration:
                logging.error(f"FATAL: Kernel timed out after {max_duration}s. Killing/Aborting pipeline.")
            else:
                logging.error(f"Kernel failed with status: {final_status}")
            raise RuntimeError(f"Kernel failed or timed out with status: {final_status}")

        # Step 4: Download outputs
        logging.info("Step 4: Downloading results...")
        # Since we are synthesizing a single audio file, we only need to download .wav files
        # and stop as soon as we have at least one.
        manager.download_results(
            dest=str(dest_dir),
            file_types={".wav"},
            expected_image_count=1,
            stable_count_patience=2  # Give it a bit of room to ensure it's written
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


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run VoiceGenHub Kaggle CPU/GPU pipeline.")
    parser.add_argument("--provider", type=str, required=True, help="Provider name (qwen, chatterbox)")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--voice", type=str, help="Voice ID (optional)")
    parser.add_argument("--output-file", type=str, default="qwen.wav", help="Local output filename")
    parser.add_argument("--gpu-only", action="store_true", help="Only run GPU (Kaggle) job")

    args = parser.parse_args()

    # Determine local output path
    target_path = Path.cwd() / "kaggle_downloads" / args.output_file

    try:
        run_kaggle_pipeline(
            text=args.text,
            provider=args.provider,
            voice=args.voice,
            output_path=target_path
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
