"""
Generate 4 audio files for comparison: 2 English (one from each provider) + 2 Spanish/German (one from each provider).
Since Chatterbox multilingual is broken, we'll use Qwen for multilingual and create variations for comparison.
"""

import asyncio
from pathlib import Path

from voicegenhub.providers.factory import provider_factory
from voicegenhub.providers.base import TTSRequest, AudioFormat


# Test sentences
ENGLISH_LONG = """
In the depths of ancient forests, where sunlight barely penetrates the thick canopy,
mysterious creatures have evolved remarkable adaptations to survive. Scientists have
discovered species that communicate through bioluminescence, creating mesmerizing light
shows in the darkness. These fascinating organisms demonstrate nature's incredible
capacity for innovation and survival against all odds.
"""

ENGLISH_SHORT = """
The advancement of technology has transformed the way we communicate and interact with
the world around us. From artificial intelligence to renewable energy, innovations
continue to shape our future in unprecedented ways.
"""

SPANISH_LONG = """
En las profundidades de los bosques antiguos, donde la luz del sol apenas penetra el denso dosel,
las criaturas misteriosas han desarrollado adaptaciones notables para sobrevivir. Los científicos
han descubierto especies que se comunican a través de bioluminiscencia, creando fascinantes
espectáculos de luz en la oscuridad. Estos organismos fascinantes demuestran la increíble
capacidad de la naturaleza para la innovación y la supervivencia contra todo pronóstico.
"""


async def main():
    """Generate final comparison audio files."""
    output_dir = Path("comparison_outputs")
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Final Audio Generation: Chatterbox vs Qwen 3 TTS")
    print("=" * 80)

    # Configuration
    chatterbox_config = {}
    qwen_config = {
        "model_name_or_path": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "generation_mode": "custom_voice",
        "device": "auto",
        "dtype": "float32",
    }

    results = []

    # Task 1: Chatterbox English (long text)
    print("\n🎙️  [1/4] Generating English audio with Chatterbox (long)...")
    try:
        await provider_factory.discover_provider("chatterbox")
        provider = await provider_factory.create_provider("chatterbox", chatterbox_config)
        voice_id = "chatterbox-default"

        request = TTSRequest(
            text=ENGLISH_LONG.strip(),
            voice_id=voice_id,
            language="en",
            audio_format=AudioFormat.WAV,
        )

        response = await provider.synthesize(request)
        output_path = output_dir / "1_chatterbox_english_long.wav"
        response.save(str(output_path), log=False)

        print(f"   ✅ Duration: {response.duration:.2f}s | Size: {len(response.audio_data)/1024:.1f} KB")
        results.append(("Chatterbox", "English (long)", output_path, True))

        if hasattr(provider, 'cleanup'):
            await provider.cleanup()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results.append(("Chatterbox", "English (long)", None, False))

    # Task 2: Chatterbox English (short text  - for variation)
    print("\n🎙️  [2/4] Generating English audio with Chatterbox (short)...")
    try:
        await provider_factory.discover_provider("chatterbox")
        provider = await provider_factory.create_provider("chatterbox", chatterbox_config)
        voice_id = "chatterbox-default"

        request = TTSRequest(
            text=ENGLISH_SHORT.strip(),
            voice_id=voice_id,
            language="en",
            audio_format=AudioFormat.WAV,
        )

        response = await provider.synthesize(request)
        output_path = output_dir / "2_chatterbox_english_short.wav"
        response.save(str(output_path), log=False)

        print(f"   ✅ Duration: {response.duration:.2f}s | Size: {len(response.audio_data)/1024:.1f} KB")
        results.append(("Chatterbox", "English (short)", output_path, True))

        if hasattr(provider, 'cleanup'):
            await provider.cleanup()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results.append(("Chatterbox", "English (short)", None, False))

    # Task 3: Qwen English
    print("\n🎙️  [3/4] Generating English audio with Qwen 3 TTS...")
    try:
        await provider_factory.discover_provider("qwen")
        provider = await provider_factory.create_provider("qwen", qwen_config)

        request = TTSRequest(
            text=ENGLISH_LONG.strip(),
            voice_id="default",
            language="en",
            audio_format=AudioFormat.WAV,
        )

        response = await provider.synthesize(request)
        output_path = output_dir / "3_qwen_english.wav"
        response.save(str(output_path), log=False)

        print(f"   ✅ Duration: {response.duration:.2f}s | Size: {len(response.audio_data)/1024:.1f} KB")
        results.append(("Qwen", "English", output_path, True))

        if hasattr(provider, 'cleanup'):
            await provider.cleanup()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results.append(("Qwen", "English", None, False))

    # Task 4: Qwen Spanish
    print("\n🎙️  [4/4] Generating Spanish audio with Qwen 3 TTS...")
    try:
        await provider_factory.discover_provider("qwen")
        provider = await provider_factory.create_provider("qwen", qwen_config)

        request = TTSRequest(
            text=SPANISH_LONG.strip(),
            voice_id="default",
            language="es",
            audio_format=AudioFormat.WAV,
        )

        response = await provider.synthesize(request)
        output_path = output_dir / "4_qwen_spanish.wav"
        response.save(str(output_path), log=False)

        print(f"   ✅ Duration: {response.duration:.2f}s | Size: {len(response.audio_data)/1024:.1f} KB")
        results.append(("Qwen", "Spanish", output_path, True))

        if hasattr(provider, 'cleanup'):
            await provider.cleanup()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results.append(("Qwen", "Spanish", None, False))

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    success_count = sum(1 for _, _, _, success in results if success)

    for provider_name, description, path, success in results:
        status = "✅" if success else "❌"
        if path and path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"{status} {provider_name:12s} {description:20s} -> {path.name} ({size_kb:.1f} KB)")
        else:
            print(f"{status} {provider_name:12s} {description:20s} -> FAILED")

    print("\n" + "=" * 80)
    print(f"Generated {success_count}/4 audio files successfully!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
