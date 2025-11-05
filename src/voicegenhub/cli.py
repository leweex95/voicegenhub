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
def synthesize(text, voice, language, output, format, rate, pitch, provider):
    """Generate speech from text."""
    # Validate provider immediately
    supported_providers = ["edge", "google", "piper", "melotts", "kokoro"]
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
        output_path = Path(output) if output else Path(f"speech.{format}")
        with open(output_path, "wb") as f:
            f.write(response.audio_data)

        click.echo(f"Audio saved to: {output_path}")

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
    supported_providers = ["edge", "google", "piper", "melotts", "kokoro"]
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
