"""Quick test of native English audio with Qwen 3 TTS."""
import asyncio
import os

from voicegenhub.providers.factory import provider_factory
from voicegenhub.providers.base import TTSRequest


async def main():
    os.makedirs("test_outputs", exist_ok=True)

    config = {
        "model_name_or_path": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "device": "cpu",
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "generation_mode": "custom_voice",
    }

    await provider_factory.discover_provider("qwen")
    provider = await provider_factory.create_provider("qwen", config=config)

    print("Test 1: Auto-selected native English speaker...")
    print("This will take a few moments on CPU...")

    request = TTSRequest(
        text="Hello! This is native English speech.",
        language="en",
        voice_id="default"
    )
    response = await provider.synthesize(request)

    with open("test_outputs/native_english_ryan.wav", "wb") as f:
        f.write(response.audio_data)
    print("✓ Generated: test_outputs/native_english_ryan.wav")
    print(f"✓ Duration: {response.duration:.2f}s")
    print(f"✓ Sample rate: {response.sample_rate}Hz")

    # Check metadata
    if 'language' in response.metadata:
        print(f"✓ Language: {response.metadata['language']}")

    print("\n✓ Test completed! Audio file should have native English pronunciation without Chinese accent.")


if __name__ == "__main__":
    asyncio.run(main())
