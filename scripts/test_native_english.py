"""Test native English audio generation with Qwen 3 TTS."""

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

    # Test 1: English with auto-selected native speaker (Ryan)
    print("Test 1: Native English with auto-selected speaker...")
    request1 = TTSRequest(
        text="Hello! This is a test of the Qwen 3 TTS system. The voice should sound natural and native, without any Chinese accent.",
        language="en",
        voice_id="default"
    )
    response1 = await provider.synthesize(request1)

    with open("test_outputs/1_native_english_auto.wav", "wb") as f:
        f.write(response1.audio_data)
    print("✓ Saved to test_outputs/1_native_english_auto.wav")

    # Test 2: English with explicit Ryan speaker
    print("\nTest 2: Native English with explicit Ryan speaker...")
    request2 = TTSRequest(
        text="The quick brown fox jumps over the lazy dog. This sentence demonstrates natural English speech patterns and pronunciation.",
        language="en",
        voice_id="default",
        extra_params={"speaker": "Ryan"}
    )
    response2 = await provider.synthesize(request2)

    with open("test_outputs/2_native_english_ryan.wav", "wb") as f:
        f.write(response2.audio_data)
    print("✓ Saved to test_outputs/2_native_english_ryan.wav")

    # Test 3: English with Aiden speaker
    print("\nTest 3: Native English with Aiden speaker (American voice)...")
    request3 = TTSRequest(
        text="Welcome to the presentation. Today we'll explore the capabilities of advanced text-to-speech systems and their practical applications.",
        language="en",
        voice_id="default",
        extra_params={"speaker": "Aiden"}
    )
    response3 = await provider.synthesize(request3)

    with open("test_outputs/3_native_english_aiden.wav", "wb") as f:
        f.write(response3.audio_data)
    print("✓ Saved to test_outputs/3_native_english_aiden.wav")

    # Test 4: English with emotion instruction
    print("\nTest 4: Native English with happy emotion...")
    request4 = TTSRequest(
        text="I'm so excited to announce that we've achieved incredible results! This is absolutely fantastic news for everyone involved!",
        language="en",
        voice_id="default",
        extra_params={"speaker": "Ryan", "instruct": "Speak with excitement and joy"}
    )
    response4 = await provider.synthesize(request4)

    with open("test_outputs/4_native_english_happy.wav", "wb") as f:
        f.write(response4.audio_data)
    print("✓ Saved to test_outputs/4_native_english_happy.wav")

    # Test 5: Long form English
    print("\nTest 5: Long form native English...")
    long_text = """
    Artificial intelligence has revolutionized the way we interact with technology.
    Text-to-speech systems have advanced significantly in recent years,
    enabling natural and expressive voice synthesis across multiple languages.
    These systems now support various emotions, speaking styles, and even voice cloning capabilities.
    The Qwen 3 TTS model represents a significant step forward in this field,
    offering high-quality speech generation with remarkable flexibility and control.
    """
    request5 = TTSRequest(
        text=long_text.strip(),
        language="en",
        voice_id="default",
        extra_params={"speaker": "Ryan"}
    )
    response5 = await provider.synthesize(request5)

    with open("test_outputs/5_native_english_long.wav", "wb") as f:
        f.write(response5.audio_data)
    print("✓ Saved to test_outputs/5_native_english_long.wav")

    print("\n✓ All tests completed successfully!")
    print("Audio files saved in test_outputs/ directory")


if __name__ == "__main__":
    asyncio.run(main())
