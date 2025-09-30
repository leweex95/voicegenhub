"""
Simple Command Line Interface for VoiceGenHub.
"""

import asyncio
import click
from pathlib import Path
from typing import Optional

from .core.engine import VoiceGenHub
from .providers.base import AudioFormat


@click.group()
def cli():
    """VoiceGenHub - Simple Text-to-Speech CLI."""
    pass


@cli.command()
@click.argument("text")
@click.option("--voice", "-v", help="Voice ID")
@click.option("--language", "-l", help="Language code (e.g., 'en')")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--format", "-f", type=click.Choice(["mp3", "wav"]), default="mp3", help="Audio format")
def synthesize(text: str, voice: Optional[str], language: Optional[str], output: Optional[str], format: str):
    """Generate speech from text."""
    async def _synthesize():
        try:
            tts = VoiceGenHub()
            await tts.initialize()
            
            print("Generating speech...")
            
            response = await tts.generate(
                text=text,
                voice=voice,
                language=language,
                audio_format=AudioFormat(format)
            )
            
            output_path = Path(output) if output else Path(f"speech.{format}")
            
            with open(output_path, "wb") as f:
                f.write(response.audio_data)
            
            print(f"Audio saved to: {output_path}")
            
        except Exception as e:
            print(f"Error: {e}")
            raise click.Abort()
    
    asyncio.run(_synthesize())


@cli.command()
@click.option("--language", "-l", help="Filter by language")
def voices(language: Optional[str]):
    """List available voices."""
    async def _list_voices():
        try:
            tts = VoiceGenHub()
            await tts.initialize()
            
            voices_data = await tts.get_voices(language=language)
            
            print("Available Voices:")
            print("-" * 50)
            for voice in voices_data[:10]:  # Show first 10
                print(f"{voice['id']} - {voice['name']} ({voice['language']})")
            
            if len(voices_data) > 10:
                print(f"... and {len(voices_data) - 10} more voices")
        
        except Exception as e:
            print(f"Error: {e}")
            raise click.Abort()
    
    asyncio.run(_list_voices())


if __name__ == "__main__":
    cli()