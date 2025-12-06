"""
Simple Command Line Interface for VoiceGenHub.
"""

import asyncio
import json
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import click

from .content.effect import EffectGenerationError, StableAudioEffectGenerator
from .core.engine import VoiceGenHub
from .providers.base import AudioFormat
from .utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_TTS_PROVIDERS = [
    "edge",
    "piper",
    "melotts",
    "kokoro",
    "elevenlabs",
    "bark",
    "chatterbox",
]
DEFAULT_EFFECT_OUTPUT = "sound_effect.wav"


def _apply_tts_run_options(func):
    options = [
        click.option(
            "--voice",
            "-v",
            help="Voice ID (e.g., 'en-US-AriaNeural', 'kokoro-af_alloy', 'melotts-EN')",
        ),
        click.option("--language", "-l", help="Language code (e.g., 'en')"),
        click.option(
            "--output",
            "-o",
            type=click.Path(),
            help="Output file path (auto-numbered for multiple texts)",
        ),
        click.option(
            "--format",
            "-f",
            "audio_format",
            type=click.Choice(["mp3", "wav"]),
            default="wav",
            help="Audio format",
        ),
        click.option(
            "--rate",
            "-r",
            type=float,
            default=1.0,
            help="Speech rate (0.5-2.0, default 1.0)",
        ),
        click.option(
            "--pitch",
            type=float,
            default=1.0,
            help="Speech pitch (0.5-2.0, default 1.0)",
        ),
        click.option("--provider", "-p", help="TTS provider"),
        click.option(
            "--lowpass",
            type=int,
            help="Apply lowpass filter with cutoff frequency in Hz (e.g., 1200 for horror effect)",
        ),
        click.option(
            "--normalize",
            is_flag=True,
            help="Normalize audio loudness",
        ),
        click.option(
            "--distortion",
            type=float,
            help="Apply distortion/overdrive (gain 1-20, higher = more evil)",
        ),
        click.option(
            "--noise",
            type=float,
            help="Add white noise static (volume 0.0-1.0)",
        ),
        click.option(
            "--reverb",
            is_flag=True,
            help="Add reverb with delay effect",
        ),
        click.option(
            "--pitch-shift",
            type=int,
            help="Pitch shift in semitones (negative = lower/darker)",
        ),
    ]
    for option in reversed(options):
        func = option(func)
    return func


def _apply_tts_voice_options(func):
    options = [
        click.option("--language", "-l", help="Filter by language"),
        click.option(
            "--format",
            "-f",
            "output_format",
            type=click.Choice(["table", "json"]),
            default="table",
            help="Output format",
        ),
        click.option("--provider", "-p", help="TTS provider"),
    ]
    for option in reversed(options):
        func = option(func)
    return func


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

        output_path = Path(output) if output else Path(f"output.{audio_format}")
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


def _execute_tts_run(
    texts,
    voice,
    language,
    output,
    audio_format,
    rate,
    pitch,
    provider,
    lowpass,
    normalize,
    distortion,
    noise,
    reverb,
    pitch_shift,
):
    """Execute the shared TTS workflow for both primary and alias commands."""
    if provider and provider not in SUPPORTED_TTS_PROVIDERS:
        click.echo(
            f"Error: Unsupported provider '{provider}'. Supported providers: {', '.join(SUPPORTED_TTS_PROVIDERS)}",
            err=True,
        )
        sys.exit(1)

    if not language:
        click.echo("Error: --language (-l) is required", err=True)
        sys.exit(1)

    if not voice:
        click.echo("Error: --voice (-v) is required", err=True)
        sys.exit(1)

    all_texts = list(texts)
    if not all_texts:
        click.echo("Error: No text provided", err=True)
        sys.exit(1)

    effects_requested = any([lowpass, normalize, distortion, noise, reverb, pitch_shift])
    is_batch = len(all_texts) > 1

    if is_batch:
        _process_batch(
            texts=all_texts,
            provider=provider,
            voice=voice,
            language=language,
            output_base=output,
            audio_format=audio_format,
            speed=rate,
            pitch=pitch,
            effects_enabled=effects_requested,
            lowpass=lowpass,
            normalize=normalize,
            distortion=distortion,
            noise=noise,
            reverb=reverb,
            pitch_shift=pitch_shift,
        )
    else:
        _process_single(
            text=all_texts[0],
            provider=provider,
            voice=voice,
            language=language,
            output=output,
            audio_format=audio_format,
            speed=rate,
            pitch=pitch,
            lowpass=lowpass,
            normalize=normalize,
            distortion=distortion,
            noise=noise,
            reverb=reverb,
            pitch_shift=pitch_shift,
        )


def _list_tts_voices(language: Optional[str], output_format: str, provider: Optional[str]) -> None:
    """Shared voice listing logic for TTS commands."""
    if provider and provider not in SUPPORTED_TTS_PROVIDERS:
        click.echo(
            f"Error: Unsupported provider '{provider}'. Supported providers: {', '.join(SUPPORTED_TTS_PROVIDERS)}",
            err=True,
        )
        sys.exit(1)

    try:
        tts = VoiceGenHub(provider=provider)
        voices_data = asyncio.run(tts.get_voices(language=language))

        if output_format == "json":
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
            for voice in voices_data[:10]:
                click.echo(f"{voice['id']} - {voice['name']} ({voice['language']})")

            if len(voices_data) > 10:
                click.echo(f"... and {len(voices_data) - 10} more voices")
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@click.group()
def cli():
    """VoiceGenHub - multi-modal generation CLI."""


@cli.group()
def tts():
    """Text-to-Speech commands."""


@tts.command("run")
@_apply_tts_run_options
@click.argument("texts", nargs=-1, required=True)
def tts_run(
    texts,
    voice,
    language,
    output,
    audio_format,
    rate,
    pitch,
    provider,
    lowpass,
    normalize,
    distortion,
    noise,
    reverb,
    pitch_shift,
):
    """Generate speech from text(s)."""
    _execute_tts_run(
        texts,
        voice,
        language,
        output,
        audio_format,
        rate,
        pitch,
        provider,
        lowpass,
        normalize,
        distortion,
        noise,
        reverb,
        pitch_shift,
    )


@tts.command("voices")
@_apply_tts_voice_options
def tts_list_voices(language: Optional[str], output_format: str, provider: Optional[str]):
    """List available voices for the selected providers."""
    _list_tts_voices(language, output_format, provider)


@cli.command("synthesize")
@_apply_tts_run_options
@click.argument("texts", nargs=-1, required=True)
def synthesize(
    texts,
    voice,
    language,
    output,
    audio_format,
    rate,
    pitch,
    provider,
    lowpass,
    normalize,
    distortion,
    noise,
    reverb,
    pitch_shift,
):
    """Generate speech from text(s)."""
    _execute_tts_run(
        texts,
        voice,
        language,
        output,
        audio_format,
        rate,
        pitch,
        provider,
        lowpass,
        normalize,
        distortion,
        noise,
        reverb,
        pitch_shift,
    )


@cli.command("voices")
@_apply_tts_voice_options
def voices_alias(language: Optional[str], output_format: str, provider: Optional[str]):
    """List available voices."""
    _list_tts_voices(language, output_format, provider)


@cli.group()
def music():
    """Music generation commands (coming soon)."""


@music.command("run")
@click.option(
    "--style",
    "-s",
    default="cinematic",
    show_default=True,
    help="Target music style (placeholder).",
)
@click.option(
    "--duration",
    "-d",
    type=click.IntRange(10, 360),
    default=60,
    show_default=True,
    help="Duration in seconds (placeholder).",
)
@click.argument("keywords", nargs=-1)
def music_run(style: str, duration: int, keywords: tuple[str, ...]):
    """Placeholder music generator that logs the requested style."""
    keyword_summary = " ".join(keywords) if keywords else "no keywords"
    click.echo(
        f"Music generation will be available soon (style={style}, duration={duration}, keywords={keyword_summary})."
    )


@cli.group()
def effect():
    """Sound effect generation commands."""


@effect.command("run")
@click.option(
    "--prompt",
    "-p",
    required=True,
    help="Describe the sound effect you want to generate.",
)
@click.option(
    "--model",
    "-m",
    default="stabilityai/stable-audio-open-1.0",
    show_default=True,
    help="HuggingFace model ID to use for generation.",
)
@click.option(
    "--duration",
    "-d",
    type=click.IntRange(1, 60),
    default=30,
    show_default=True,
    help="Duration of the generated clip in seconds.",
)
@click.option(
    "--guidance-scale",
    type=click.FloatRange(0.1, 15.0),
    default=7.0,
    show_default=True,
    help="Guidance scale that balances prompt adherence and diversity.",
)
@click.option("--seed", type=int, help="Optional random seed for deterministic outputs.")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["wav", "mp3"]),
    default="wav",
    show_default=True,
    help="Output audio format.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True),
    default=DEFAULT_EFFECT_OUTPUT,
    help="Destination path for the generated sound effect.",
)
def effect_run(
    prompt: str,
    model: str,
    duration: int,
    guidance_scale: float,
    seed: Optional[int],
    output_format: str,
    output: str,
):
    """Generate a single sound effect clip using StabilityAI's models."""
    generator = StableAudioEffectGenerator(model_id=model)
    try:
        result = generator.generate(
            prompt=prompt,
            output_path=Path(output),
            duration=duration,
            output_format=output_format,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        click.echo(f"Sound effect saved to {result.path}")
    except EffectGenerationError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        generator.close()


if __name__ == "__main__":
    cli()


def main():
    """Entry point for console script."""
    cli()
