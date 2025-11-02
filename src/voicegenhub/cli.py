"""
Simple Command Line Interface for VoiceGenHub.
"""

import asyncio
import click
import json
import sys
from pathlib import Path
from typing import Optional

from .core.engine import VoiceGenHub
from .providers.base import AudioFormat
from .providers.factory import provider_factory


async def get_available_providers():
    """Get list of available provider IDs."""
    await provider_factory.discover_and_register_providers()
    
    providers = []
    if provider_factory._edge_provider_class:
        providers.append("edge")
    if provider_factory._google_provider_class:
        providers.append("google")
    if provider_factory._piper_provider_class:
        providers.append("piper")
    if provider_factory._coqui_provider_class:
        providers.append("coqui")
    if provider_factory._melotts_provider_class:
        providers.append("melotts")
    if provider_factory._kokoro_provider_class:
        providers.append("kokoro")
    
    return providers


@click.group()
def cli():
    """VoiceGenHub - Simple Text-to-Speech CLI."""
    pass


@cli.command()
@click.argument("text")
@click.option("--voice", "-v", help="Voice ID (e.g., 'en-US-AriaNeural', 'kokoro-af_alloy', 'melotts-EN'). For MeloTTS use 'melotts-EN' (EN-US) for best quality. For Kokoro use specific voice names like 'kokoro-af_alloy'.")
@click.option("--language", "-l", help="Language code (e.g., 'en')")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--format", "-f", type=click.Choice(["mp3", "wav"]), default="wav", help="Audio format")
@click.option("--rate", "-r", type=float, default=1.0, help="Speech rate (0.5-2.0, default 1.0)")
@click.option("--provider", "-p", help="TTS provider. For MeloTTS use 'melotts' with voice 'melotts-EN'. For Kokoro use 'kokoro' with specific voices like 'kokoro-af_alloy'.")
def synthesize(text: str, voice: Optional[str], language: Optional[str], output: Optional[str], format: str, rate: float, provider: str):
    """Generate speech from text."""
    async def _synthesize():
        try:
            # Get available providers
            available_providers = await get_available_providers()
            
            # Validate provider
            if provider not in available_providers:
                print(f"Error: Provider '{provider}' not available. Available providers: {', '.join(available_providers)}", file=sys.stderr)
                sys.exit(1)
            
            # Validate rate parameter
            if rate < 0.5 or rate > 2.0:
                print("Error: Rate must be between 0.5 and 2.0", file=sys.stderr)
                sys.exit(1)
            
            tts = VoiceGenHub(provider=provider)
            await tts.initialize()
            
            print("Generating speech...")
            
            response = await tts.generate(
                text=text,
                voice=voice,
                language=language,
                audio_format=AudioFormat(format),
                speed=rate
            )
            
            output_path = Path(output) if output else Path(f"speech.{format}")
            
            with open(output_path, "wb") as f:
                f.write(response.audio_data)
            
            print(f"Audio saved to: {output_path}")
            
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    asyncio.run(_synthesize())


@cli.command()
@click.option("--language", "-l", help="Filter by language")
@click.option("--format", "-f", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.option("--provider", "-p", help="TTS provider")
def voices(language: Optional[str], format: str, provider: str):
    """List available voices."""
    async def _list_voices():
        try:
            # Get available providers
            available_providers = await get_available_providers()
            
            # Validate provider
            if provider not in available_providers:
                print(f"Error: Provider '{provider}' not available. Available providers: {', '.join(available_providers)}", file=sys.stderr)
                sys.exit(1)
            
            tts = VoiceGenHub(provider=provider)
            await tts.initialize()
            
            voices_data = await tts.get_voices(language=language)
            
            if format == "json":
                output = {
                    "voices": [
                        {
                            "id": voice["id"],
                            "language": voice["locale"],
                            "gender": voice["gender"]
                        }
                        for voice in voices_data
                    ]
                }
                print(json.dumps(output, indent=2))
            else:
                print("Available Voices:")
                print("-" * 50)
                for voice in voices_data[:10]:  # Show first 10
                    print(f"{voice['id']} - {voice['name']} ({voice['language']})")
                
                if len(voices_data) > 10:
                    print(f"... and {len(voices_data) - 10} more voices")
        
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    asyncio.run(_list_voices())


if __name__ == "__main__":
    cli()


def main():
    """Entry point for console script."""
    cli()