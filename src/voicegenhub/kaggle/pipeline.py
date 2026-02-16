"""Main Kaggle GPU pipeline orchestration."""

import logging
from pathlib import Path
from typing import Optional

from voicegenhub.kaggle.core.deploy import deploy_to_kaggle
from voicegenhub.kaggle.core.download import download_from_kaggle
from voicegenhub.kaggle.core.poll_status import poll_kernel_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_kaggle_pipeline(
    text: str,
    provider: str,
    voice: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Run complete Kaggle GPU pipeline: deploy -> poll -> download.

    Args:
        text: Text to synthesize
        provider: Provider name (qwen, chatterbox)
        voice: Voice ID (provider-specific)
        output_path: Desired local output path for the audio file

    Returns:
        Path to the downloaded audio file
    """
    # Determine output filename
    if output_path:
        output_file = output_path.name
        dest_dir = output_path.parent
    else:
        output_file = "output.wav"
        dest_dir = Path.cwd() / "kaggle_outputs"

    dest_dir.mkdir(parents=True, exist_ok=True)

    logging.info("="*80)
    logging.info("KAGGLE GPU PIPELINE")
    logging.info(f"Provider: {provider}")
    logging.info(f"Voice: {voice}")
    logging.info(f"Text: {text[:50]}..." if len(text) > 50 else f"Text: {text}")
    logging.info(f"Output: {dest_dir / output_file}")
    logging.info("="*80)

    # Step 1: Deploy to Kaggle
    logging.info("Step 1: Deploying to Kaggle...")
    kernel_id = deploy_to_kaggle(
        text=text,
        provider=provider,
        voice=voice,
        output_file=output_file,
    )

    # Step 2: Poll until complete
    logging.info("Step 2: Waiting for Kaggle kernel to complete...")
    status = poll_kernel_status(kernel_id=kernel_id)

    if status != "kernelworkerstatus.complete":
        raise RuntimeError(f"Kernel failed with status: {status}")

    # Step 3: Download outputs
    logging.info("Step 3: Downloading outputs...")
    success = download_from_kaggle(kernel_id=kernel_id, dest=str(dest_dir))

    if not success:
        raise RuntimeError("Failed to download audio outputs from Kaggle")

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
