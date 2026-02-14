"""
Comparison script for Qwen 3 TTS vs Chatterbox TTS.
Generates 4 audio files: English and Hungarian with both providers.
"""

import asyncio
from pathlib import Path

from voicegenhub.providers.factory import provider_factory
from voicegenhub.providers.base import TTSRequest, AudioFormat


# Long-form test sentences
ENGLISH_TEXT = """
In the depths of ancient forests, where sunlight barely penetrates the thick canopy,
mysterious creatures have evolved remarkable adaptations to survive. Scientists have
discovered species that communicate through bioluminescence, creating mesmerizing light
shows in the darkness. These fascinating organisms demonstrate nature's incredible
capacity for innovation and survival against all odds.
"""

SPANISH_TEXT = """
En las profundidades de los bosques antiguos, donde la luz del sol apenas penetra el denso dosel,
las criaturas misteriosas han desarrollado adaptaciones notables para sobrevivir. Los científicos
han descubierto especies que se comunican a través de bioluminiscencia, creando fascinantes
espectáculos de luz en la oscuridad. Estos organismos fascinantes demuestran la increíble
capacidad de la naturaleza para la innovación y la supervivencia contra todo pronóstico.
"""


async def generate_audio(provider_name: str, text: str, language: str, output_path: str, config: dict = None, voice_override: str = None):
    """Generate audio with the specified provider."""
    print(f"\n🎙️  Generating {language} audio with {provider_name}...")
    print(f"   Text length: {len(text)} characters")

    try:
        # Discover and create provider
        await provider_factory.discover_provider(provider_name)
        provider = await provider_factory.create_provider(provider_name, config=config)

        # Get available voices
        voices = await provider.get_voices()
        if voices:
            # Use override if specified, otherwise use first available
            voice_id = voice_override or voices[0].id
            print("   Using voice: {}".format(voice_id))
        else:
            voice_id = voice_override or "default"
            print("   Using default voice")

        # Create request
        request = TTSRequest(
            text=text.strip(),
            voice_id=voice_id,
            language=language,
            audio_format=AudioFormat.WAV,
        )

        # Synthesize
        response = await provider.synthesize(request)

        # Save
        response.save(output_path, log=False)

        print(f"   ✅ Saved to: {output_path}")
        print(f"   Duration: {response.duration:.2f}s")
        print(f"   Sample rate: {response.sample_rate} Hz")

        # Cleanup
        if hasattr(provider, 'cleanup'):
            await provider.cleanup()

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


async def main():
    """Run comparison test."""
    print("=" * 80)
    print("Qwen 3 TTS vs Chatterbox TTS Comparison")
    print("=" * 80)

    output_dir = Path("comparison_outputs")
    output_dir.mkdir(exist_ok=True)

    results = []

    # Chatterbox configuration - using multilingual voice for Spanish
    chatterbox_config_en = {}
    chatterbox_config_es = {"voice": "chatterbox-es"}  # Use multilingual voice for Spanish

    # Qwen configuration - using CustomVoice mode
    qwen_config = {
        "model_name_or_path": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",  # Smaller model
        "generation_mode": "custom_voice",
        "device": "auto",
        "dtype": "float32",
        "speaker": None,  # Will use first available speaker
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 50,
    }

    # Generate all 4 audio files
    tasks = [
        ("chatterbox", ENGLISH_TEXT, "en", output_dir / "chatterbox_english.wav", chatterbox_config_en, "chatterbox-default"),
        ("chatterbox", SPANISH_TEXT, "es", output_dir / "chatterbox_spanish.wav", chatterbox_config_es, "chatterbox-es"),
        ("qwen", ENGLISH_TEXT, "en", output_dir / "qwen_english.wav", qwen_config, None),
        ("qwen", SPANISH_TEXT, "es", output_dir / "qwen_spanish.wav", qwen_config, None),
    ]

    for provider_name, text, language, output_path, config, voice_override in tasks:
        success = await generate_audio(provider_name, text, language, str(output_path), config, voice_override)
        results.append((provider_name, language, output_path, success))

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    for provider_name, language, output_path, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"{status} - {provider_name:12s} {language:2s} -> {output_path}")

    successful_files = [path for _, _, path, success in results if success]

    print("\n" + "=" * 80)
    print(f"Generated {len(successful_files)}/{len(results)} audio files successfully!")
    print("=" * 80)

    if successful_files:
        print("\n📁 Output files:")
        for path in successful_files:
            if path.exists():
                size_kb = path.stat().st_size / 1024
                print(f"   {path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    asyncio.run(main())
