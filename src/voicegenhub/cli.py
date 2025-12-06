"""
Simple Command Line Interface for VoiceGenHub.
"""

import asyncio
import json
import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import click

from .core.engine import VoiceGenHub
from .providers.base import AudioFormat
from .utils.logger import get_logger

logger = get_logger(__name__)


def _process_single(
    text: str,
    provider: str,
    voice: str,
    language: str,
    output: str,
    audio_format: str,
    speed: float,
    pitch: float,
    lowpass: int,
    normalize: bool,
    distortion: float,
    noise: float,
    reverb: bool,
    pitch_shift: int,
):
    """Process a single text with effects support."""
    try:
        # Initialize TTS
        tts = VoiceGenHub(provider=provider)
        asyncio.run(tts.initialize())

        # Generate audio
        response = asyncio.run(tts.generate(
            text=text,
            voice=voice,
            language=language,
            audio_format=AudioFormat(audio_format),
            speed=speed,
            pitch=pitch,
        ))

        output_path = Path(output)
        effects_requested = any([lowpass, normalize, distortion, noise, reverb, pitch_shift])

        if effects_requested:
            # Always use a true temp file for effects
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_format}") as tmp:
                temp_path = Path(tmp.name)
                tmp.write(response.audio_data)
        else:
            temp_path = output_path
            with open(temp_path, "wb") as f:
                f.write(response.audio_data)

        # Apply post-processing effects if requested
        if effects_requested:
            import subprocess

            # Build FFmpeg filter chain
            audio_filters = []
            complex_filter = None

            if pitch_shift:
                audio_filters.append(f"rubberband=pitch={2**(pitch_shift/12.0)}")
            if lowpass:
                audio_filters.append(f"lowpass=f={lowpass}")
            if distortion:
                audio_filters.append(f"volume={distortion}dB,acompressor=threshold=-6dB:ratio=20:attack=5:release=50")
            if reverb:
                audio_filters.append("aecho=0.8:0.9:1000:0.3")
            if normalize:
                audio_filters.append("dynaudnorm=f=150:g=15")

            # Noise requires filter_complex (separate stream)
            if noise:
                noise_filter = (
                    f"anoisesrc=d=10:c=white:r=44100:a={noise}[noise];"
                    f"[0:a][noise]amix=inputs=2:duration=first"
                )
                if audio_filters:
                    complex_filter = noise_filter + "[mixed];" + "[mixed]" + ",".join(audio_filters)
                else:
                    complex_filter = noise_filter

            cmd = ["ffmpeg", "-i", str(temp_path)]
            if complex_filter:
                cmd.extend(["-filter_complex", complex_filter])
            elif audio_filters:
                cmd.extend(["-af", ",".join(audio_filters)])
            cmd.extend(["-y", str(output_path)])

            try:
                subprocess.run(cmd, capture_output=True, check=True)
                if temp_path != output_path and temp_path.exists():
                    temp_path.unlink()  # Remove temp file
                logger.info(f"Audio with effects saved to: {output_path}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Post-processing failed: {e.stderr.decode()}")
                logger.info(f"Original audio saved to: {temp_path}")
            except FileNotFoundError:
                logger.warning("FFmpeg not found. Install FFmpeg for post-processing.")
                logger.info(f"Original audio saved to: {temp_path}")
        else:
            logger.info(f"Audio saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _process_batch(
    texts: list,
    provider: str,
    voice: str,
    language: str,
    output_base: str,
    audio_format: str,
    speed: float,
    pitch: float,
    effects_enabled: bool,
    lowpass: int,
    normalize: bool,
    distortion: float,
    noise: float,
    reverb: bool,
    pitch_shift: int,
):
    """Process multiple texts concurrently with provider-specific limits.
    
    Concurrency defaults are conservative (max 1-2 concurrent jobs):
    - Most providers use 1 to avoid memory exhaustion and API rate limiting
    - Only Bark uses 2 (tested and verified safe for concurrent processing)
    - This prevents issues like OOM errors and rate limit violations
    """
    # Provider-specific concurrency limits (conservative defaults for safety)
    provider_limits = {
        "edge": 1,  # Conservative: cloud API with rate limiting
        "piper": 1,  # Conservative: local model memory intensive
        "melotts": 1,  # Conservative: local model memory intensive
        "kokoro": 1,  # Conservative: local model memory intensive
        "elevenlabs": 1,  # Conservative: cloud API with rate limiting
        "bark": 2,  # Cautiously increased to 2 (tested and verified safe)
        "chatterbox": 1,  # Conservative: very heavy on resources
    }

    # Determine max concurrent jobs (defaults to provider's conservative limit for safety)
    limit = provider_limits.get(provider, 1)
    max_concurrent = max(1, limit)

    click.echo(f"Processing {len(texts)} texts with {provider} (max {max_concurrent} concurrent jobs)")

    # Create shared provider instance (loaded once, reused across jobs)
    click.echo(f"Initializing {provider} provider (this may take a moment for heavy models)...")
    
    async def init_provider():
        shared_tts = VoiceGenHub(provider=provider)
        await shared_tts.initialize()
        return shared_tts
    
    shared_tts = asyncio.run(init_provider())
    click.echo(f"{provider} provider ready for batch processing")

    # Use threading for concurrent processing
    results = []
    lock = threading.Lock()
    
    def process_item(index: int, text: str):
        """Process a single text item."""
        output_file = Path(f"{output_base}_{index + 1:02d}.{audio_format}")
        
        with lock:
            click.echo(f"Processing item {index + 1}/{len(texts)}: {text[:50]}...")
        
        try:
            # Run async generation in thread
            async def generate():
                return await shared_tts.generate(
                    text=text,
                    voice=voice,
                    language=language,
                    audio_format=AudioFormat(audio_format),
                    speed=speed,
                    pitch=pitch,
                )
            
            response = asyncio.run(generate())
            
            # Apply effects if requested (single processing with effects)
            if effects_enabled:
                _process_single(
                    text=text,
                    provider=provider,
                    voice=voice,
                    language=language,
                    output=str(output_file),
                    audio_format=audio_format,
                    speed=speed,
                    pitch=pitch,
                    lowpass=lowpass,
                    normalize=normalize,
                    distortion=distortion,
                    noise=noise,
                    reverb=reverb,
                    pitch_shift=pitch_shift,
                )
            else:
                # Save output directly
                with open(output_file, "wb") as f:
                    f.write(response.audio_data)
            with lock:
                click.echo(f"[SUCCESS] Saved to {output_file}")
            return True

        except Exception as e:
            with lock:
                click.echo(f"[FAILED] Item {index + 1}: {e}", err=True)
            return False

    # Run jobs with controlled concurrency
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [executor.submit(process_item, i, text) for i, text in enumerate(texts)]
        
        for future in as_completed(futures):
            results.append(future.result())

    successful = sum(1 for r in results if r is True)
    failed = len(results) - successful

    click.echo(f"\nBatch complete: {successful} successful, {failed} failed")
    
    if failed > 0:
        sys.exit(1)


@click.group()
def cli():
    """VoiceGenHub - Simple Text-to-Speech CLI."""
    pass


@cli.command()
@click.argument("texts", nargs=-1, required=True)
@click.option(
    "--voice",
    "-v",
    help="Voice ID (e.g., 'en-US-AriaNeural', 'kokoro-af_alloy', 'melotts-EN')",
)
@click.option("--language", "-l", help="Language code (e.g., 'en')")
@click.option("--output", "-o", type=click.Path(), help="Output file path (auto-numbered for multiple texts)")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["mp3", "wav"]),
    default="wav",
    help="Audio format",
)
@click.option(
    "--rate", "-r", type=float, default=1.0, help="Speech rate (0.5-2.0, default 1.0)"
)
@click.option(
    "--pitch", type=float, default=1.0, help="Speech pitch (0.5-2.0, default 1.0)"
)
@click.option("--provider", "-p", help="TTS provider")
@click.option(
    "--lowpass",
    type=int,
    help="Apply lowpass filter with cutoff frequency in Hz (e.g., 1200 for horror effect)",
)
@click.option(
    "--normalize",
    is_flag=True,
    help="Normalize audio loudness",
)
@click.option(
    "--distortion",
    type=float,
    help="Apply distortion/overdrive (gain 1-20, higher = more evil)",
)
@click.option(
    "--noise",
    type=float,
    help="Add white noise static (volume 0.0-1.0)",
)
@click.option(
    "--reverb",
    is_flag=True,
    help="Add reverb with delay effect",
)
@click.option(
    "--pitch-shift",
    type=int,
    help="Pitch shift in semitones (negative = lower/darker)",
)
def synthesize(
    texts, voice, language, output, format, rate, pitch, provider,
    lowpass, normalize, distortion, noise, reverb, pitch_shift
):
    """Generate speech from text(s)."""
    # Validate provider immediately
    supported_providers = [
        "edge", "piper", "melotts", "kokoro", "elevenlabs", "bark", "chatterbox"
    ]
    if provider and provider not in supported_providers:
        click.echo(
            f"Error: Unsupported provider '{provider}'. Supported providers: {', '.join(supported_providers)}",
            err=True,
        )
        sys.exit(1)

    # Validate required parameters - fail fast on missing inputs
    if not language:
        click.echo("Error: --language (-l) is required", err=True)
        sys.exit(1)
    
    if not voice:
        click.echo("Error: --voice (-v) is required", err=True)
        sys.exit(1)

    # Collect all texts
    all_texts = [t for t in texts if t.strip()]  # Filter out empty texts
    
    if not all_texts:
        click.echo("Error: No text provided", err=True)
        sys.exit(1)

    # Check if this is a batch operation (multiple texts)
    is_batch = len(all_texts) > 1
    
    if is_batch:
        # Batch processing with concurrency control
        _process_batch(
            texts=all_texts,
            provider=provider,
            voice=voice,
            language=language,
            output_base=output,
            audio_format=format,
            speed=rate,
            pitch=pitch,
            effects_enabled=any([lowpass, normalize, distortion, noise, reverb, pitch_shift]),
            lowpass=lowpass,
            normalize=normalize,
            distortion=distortion,
            noise=noise,
            reverb=reverb,
            pitch_shift=pitch_shift,
        )
    else:
        # Single text processing (original behavior)
        _process_single(
            text=all_texts[0],
            provider=provider,
            voice=voice,
            language=language,
            output=output,
            audio_format=format,
            speed=rate,
            pitch=pitch,
            lowpass=lowpass,
            normalize=normalize,
            distortion=distortion,
            noise=noise,
            reverb=reverb,
            pitch_shift=pitch_shift,
        )



@cli.command()
@click.option("--language", "-l", help="Filter by language")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
@click.option("--provider", "-p", help="TTS provider")
def voices(language: Optional[str], format: str, provider: str):
    """List available voices."""
    # Validate provider immediately
    supported_providers = ["edge", "piper", "melotts", "kokoro", "elevenlabs", "bark", "chatterbox"]
    if provider and provider not in supported_providers:
        click.echo(
            f"Error: Unsupported provider '{provider}'. Supported providers: {', '.join(supported_providers)}",
            err=True,
        )
        sys.exit(1)

    try:
        # Create engine (will auto-select provider if none specified)
        tts = VoiceGenHub(provider=provider)

        # Get voices - engine handles provider validation
        voices_data = asyncio.run(tts.get_voices(language=language))

        if format == "json":
            output = {
                "voices": [
                    {
                        "id": voice["id"],
                        "language": voice["locale"],
                        "gender": voice["gender"],
                    }
                    for voice in voices_data
                ]
            }
            click.echo(json.dumps(output, indent=2))
        else:
            click.echo("Available Voices:")
            click.echo("-" * 50)
            for voice in voices_data[:10]:  # Show first 10
                click.echo(f"{voice['id']} - {voice['name']} ({voice['language']})")

            if len(voices_data) > 10:
                click.echo(f"... and {len(voices_data) - 10} more voices")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()


def main():
    """Entry point for console script."""
    cli()
