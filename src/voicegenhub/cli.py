"""
Simple Command Line Interface for VoiceGenHub.
"""

import os
import asyncio
import json
import sys
import tempfile
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import click

from .core.engine import VoiceGenHub
from .providers.base import AudioFormat
from .utils.logger import get_logger

# Force eager attention implementation to prevent SDPA warnings
os.environ['TRANSFORMERS_ATTENTION_IMPLEMENTATION'] = 'eager'

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
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    audio_prompt_path: Optional[str] = None,
    instruct: Optional[str] = None,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
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
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            audio_prompt_path=audio_prompt_path,
            instruct=instruct,
            ref_audio=ref_audio,
            ref_text=ref_text,
        ))

        output_path = Path(output).resolve() if output else Path(".") / f"voicegenhub_output.{audio_format}"
        logger.info(f"Target output path: {output_path}", path=str(output_path))
        effects_requested = any([lowpass, normalize, distortion, noise, reverb, pitch_shift])

        if effects_requested:
            # Always use a true temp file for effects
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_format}") as tmp:
                temp_path = Path(tmp.name)
            response.save(temp_path, log=False)
        else:
            temp_path = output_path
            response.save(temp_path)

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
                logger.info(f"SUCCESS: Audio saved to: {output_path.absolute()}", path=str(output_path.absolute()))
            except subprocess.CalledProcessError as e:
                logger.warning(f"Post-processing failed: {e.stderr.decode()}")
                logger.info(f"Original audio saved to: {temp_path.absolute()}", path=str(temp_path.absolute()))
            except FileNotFoundError:
                logger.warning("FFmpeg not found. Install FFmpeg for post-processing.")
                logger.info(f"Original audio saved to: {temp_path.absolute()}", path=str(temp_path.absolute()))
        else:
            logger.info(f"SUCCESS: Audio saved to: {output_path.absolute()}", path=str(output_path.absolute()))

    except Exception as e:
        logger.error(f"Error: {e}")
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
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    audio_prompt_path: Optional[str] = None,
    instruct: Optional[str] = None,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
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
        "kokoro": 1,  # Conservative: local model memory intensive
        "elevenlabs": 1,  # Conservative: cloud API with rate limiting
        "bark": 2,  # Cautiously increased to 2 (tested and verified safe)
        "chatterbox": 1,  # Conservative: heavy model (3.7GB), avoid OOM in multiprocessing
        "qwen": 1,  # New: heavy model (up to 1.7B), avoid OOM
    }

    # Determine max concurrent jobs (defaults to provider's conservative limit for safety)
    limit = provider_limits.get(provider, 1)
    max_concurrent = max(1, limit)

    logger.info(f"Processing batch of {len(texts)} texts", provider=provider, max_concurrent=max_concurrent)

    # Create shared provider instance (loaded once, reused across jobs)
    logger.info(f"Initializing {provider} provider...", provider=provider)

    async def init_provider():
        shared_tts = VoiceGenHub(provider=provider)
        await shared_tts.initialize()
        return shared_tts

    shared_tts = asyncio.run(init_provider())
    logger.info(f"Provider {provider} ready for batch processing")

    # Use threading for concurrent processing
    results = []
    lock = threading.Lock()

    if output_base is None:
        output_base = "voicegenhub_batch"
        logger.info("Batch output directory: Current directory", base_path=str(Path('.').absolute()))
    else:
        output_path_obj = Path(output_base).resolve()
        logger.info(f"Batch output directory: {output_path_obj.parent}", base_path=str(output_path_obj.parent))

    def process_item(index: int, text: str):
        """Process a single text item."""
        output_file = Path(f"{output_base}_{index + 1:02d}.{audio_format}").resolve()

        with lock:
            logger.info(f"Processing item {index + 1}/{len(texts)}", item_index=index + 1, total=len(texts), text_preview=text[:50])

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
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    audio_prompt_path=audio_prompt_path,
                    instruct=instruct,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
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
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    audio_prompt_path=audio_prompt_path,
                    instruct=instruct,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
            else:
                # Save output directly
                response.save(output_file)
                with lock:
                    logger.info(f"SUCCESS: Audio saved to: {output_file.absolute()}", path=str(output_file.absolute()))
            return True

        except Exception as e:
            with lock:
                logger.error(f"Item {index + 1} failed: {e}", item_index=index + 1, error=str(e))
            return False

    # Run jobs with controlled concurrency
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [executor.submit(process_item, i, text) for i, text in enumerate(texts)]

        for future in as_completed(futures):
            results.append(future.result())

    successful = sum(1 for r in results if r is True)
    failed = len(results) - successful

    logger.info("Batch processing complete", successful=successful, failed=failed, total=len(texts))

    if failed > 0:
        sys.exit(1)


@click.group()
def cli():
    """VoiceGenHub - Simple Text-to-Speech CLI."""


@cli.command()
@click.argument("texts", nargs=-1, required=True)
@click.option(
    "--voice",
    "-v",
    help="Voice ID (e.g., 'en-US-AriaNeural', 'kokoro-af_alloy')",
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
    "--gpu",
    type=click.Choice(["p100", "t4"]),
    help="Use remote Kaggle GPU for generation (currently Qwen3-TTS only)",
)
@click.option(
    "--cpu",
    is_flag=True,
    help="Use local CPU for generation (default)",
)
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
@click.option(
    "--exaggeration",
    type=float,
    default=0.5,
    help="Chatterbox: Emotion intensity (0.0-1.0, default 0.5)",
)
@click.option(
    "--cfg-weight",
    type=float,
    default=0.5,
    help="Chatterbox: Classifier-free guidance weight (0.0-1.0, default 0.5)",
)
@click.option(
    "--audio-prompt",
    type=click.Path(exists=True),
    help="Chatterbox: Path to audio file for voice cloning",
)
@click.option(
    "--turbo",
    is_flag=True,
    help="Chatterbox: Use turbo model (English only, faster)",
)
@click.option(
    "--multilingual",
    is_flag=True,
    help="Chatterbox: Use multilingual model",
)
@click.option(
    "--instruct",
    type=str,
    help="Qwen 3 TTS: Emotion, style, or voice design instruction",
)
@click.option(
    "--ref-audio",
    type=click.Path(exists=True),
    help="Qwen 3 TTS: Reference audio for voice cloning",
)
@click.option(
    "--ref-text",
    type=str,
    help="Qwen 3 TTS: Reference text for voice cloning",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default=None,
    help="Qwen 3 TTS: HuggingFace model ID (e.g. Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)",
)
@click.option(
    "--output-dir",
    type=str,
    default=None,
    help="Kaggle GPU: Local directory for the downloaded audio (default: YYYYMMDD_HHMMSS_<gpu>)",
)
@click.option(
    "--output-filename",
    type=str,
    default="qwen3_tts.wav",
    show_default=True,
    help="Kaggle GPU: Filename for the generated audio file",
)
@click.option(
    "--timeout",
    type=int,
    default=60,
    show_default=True,
    help="Kaggle GPU: Timeout in minutes to wait for the kernel",
)
@click.option(
    "--poll-interval",
    type=int,
    default=60,
    show_default=True,
    help="Kaggle GPU: Status polling interval in seconds",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Kaggle GPU: Random seed for reproducible generation",
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    show_default=True,
    help="Kaggle GPU: Sampling temperature (lower = more stable/neutral tone, higher = more expressive)",
)
def synthesize(
    texts, voice, language, output, format, rate, pitch, provider,
    gpu, cpu, lowpass, normalize, distortion, noise, reverb, pitch_shift,
    exaggeration, cfg_weight, audio_prompt, turbo, multilingual,
    instruct, ref_audio, ref_text,
    model, output_dir, output_filename, timeout, poll_interval, seed, temperature,
):
    """Generate speech from text(s). Use --gpu [p100|t4] for remote Kaggle GPU acceleration."""
    # Redirect to Kaggle pipeline if --gpu is specified
    if gpu:
        from .kaggle.pipeline import KaggleQwenPipeline
        pipeline = KaggleQwenPipeline(
            model_id=model or "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            timeout_minutes=timeout,
            poll_interval_seconds=poll_interval,
        )

        # Determine output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{gpu}"
        if output_dir:
            resolved_output_dir = output_dir
        elif output:
            output_path_obj = Path(output)
            resolved_output_dir = str(output_path_obj if not output_path_obj.suffix else output_path_obj.parent / f"{timestamp}{suffix}")
        else:
            resolved_output_dir = f"{timestamp}{suffix}"

        try:
            result_paths = pipeline.run(
                texts=list(texts),
                voice=voice or "Ryan",
                language=language or "en",
                output_dir=resolved_output_dir,
                gpu_type=gpu,
                seed=seed,
                temperature=temperature,
            )
            click.echo(f"SUCCESS: {len(result_paths)} audio file(s) in: {Path(resolved_output_dir).absolute()}")
            for p in result_paths:
                click.echo(f"  {p.name}")
            manifest = Path(resolved_output_dir) / "manifest.json"
            if manifest.exists():
                click.echo("  manifest.json  (prompt→file mapping)")
            return
        except Exception as e:
            click.echo(f"Error during remote generation: {e}", err=True)
            sys.exit(1)

    # For local CPU runs, ensure directory structure matches requested format
    if not output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{timestamp}_cpu"
        os.makedirs(output_dir, exist_ok=True)
        # For single text, we still want to respect the output_dir
        if len(texts) == 1:
            output = os.path.join(output_dir, "output.wav")
        else:
            output = os.path.join(output_dir, "batch")

    # Validate provider immediately
    supported_providers = [
        "edge", "kokoro", "elevenlabs", "bark", "chatterbox", "qwen"
    ]
    if provider and provider not in supported_providers:
        click.echo(
            f"Error: Unsupported provider '{provider}'. Supported providers: {', '.join(supported_providers)}",
            err=True,
        )
        sys.exit(1)

    # Chatterbox specific model flags validation
    if provider == "chatterbox":
        # Check mutual exclusivity
        if sum([bool(turbo), bool(multilingual), bool(voice)]) > 1:
            click.echo("Error: --turbo, --multilingual, and --voice are mutually exclusive for Chatterbox", err=True)
            sys.exit(1)

        if turbo:
            voice = "chatterbox-turbo"
        elif multilingual:
            lang_code = language or "en"
            voice = f"chatterbox-{lang_code}"
    elif turbo or multilingual:
        click.echo(f"Warning: --turbo and --multilingual are only supported by the 'chatterbox' provider, not '{provider}'")

    # Collect all texts
    all_texts = list(texts)

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
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            audio_prompt_path=audio_prompt,
            instruct=instruct,
            ref_audio=ref_audio,
            ref_text=ref_text,
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
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            audio_prompt_path=audio_prompt,
            instruct=instruct,
            ref_audio=ref_audio,
            ref_text=ref_text,
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
    supported_providers = ["edge", "kokoro", "elevenlabs", "bark", "chatterbox", "qwen"]
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
