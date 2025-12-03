"""
Simple Command Line Interface for VoiceGenHub.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click

from .core.engine import VoiceGenHub
from .providers.base import AudioFormat
from .utils.logger import get_logger

logger = get_logger(__name__)


@click.group()
def cli():
    """VoiceGenHub - Simple Text-to-Speech CLI."""
    pass


@cli.command()
@click.argument("text")
@click.option(
    "--voice",
    "-v",
    help="Voice ID (e.g., 'en-US-AriaNeural', 'kokoro-af_alloy', 'melotts-EN')",
)
@click.option("--language", "-l", help="Language code (e.g., 'en')")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
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
def synthesize(text, voice, language, output, format, rate, pitch, provider, lowpass, normalize, distortion, noise, reverb, pitch_shift):
    """Generate speech from text."""
    # Validate provider immediately
    supported_providers = ["edge", "google", "piper", "melotts", "kokoro", "elevenlabs"]
    if provider and provider not in supported_providers:
        click.echo(
            f"Error: Unsupported provider '{provider}'. Supported providers: {', '.join(supported_providers)}",
            err=True,
        )
        sys.exit(1)

    try:
        # Create engine (will auto-select provider if none specified)
        tts = VoiceGenHub(provider=provider)

        # Generate speech - engine handles all validation
        response = asyncio.run(
            tts.generate(
                text=text,
                voice=voice,
                language=language,
                audio_format=AudioFormat(format),
                speed=rate,
                pitch=pitch,
            )
        )

        # Save output

        import tempfile
        output_path = Path(output) if output else Path(f"speech.{format}")
        effects_requested = any([lowpass, normalize, distortion, noise, reverb, pitch_shift])
        if effects_requested:
            # Always use a true temp file for effects
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}") as tmp:
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
                # audio_filters.append(f"asetrate=44100*{2**(pitch_shift/12.0)},aresample=44100")
                # Use rubberband for pitch shifting without changing tempo
                # rubberband shifts pitch by semitones while maintaining duration
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
                noise_filter = f"anoisesrc=d=10:c=white:r=44100:a={noise}[noise];[0:a][noise]amix=inputs=2:duration=first"
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
    supported_providers = ["edge", "google", "piper", "melotts", "kokoro", "elevenlabs"]
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
