#!/usr/bin/env python3
"""
Test Coqui TTS generation and save audio files.
Demonstrates Coqui TTS functionality on Windows.
"""

import asyncio
import sys
import os

# Add src to path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from voicegenhub import VoiceGenHub


async def test_coqui_tts():
    """Test Coqui TTS generation."""
    print("Testing Coqui TTS provider...")
    print("Note: First run will download models (may take 1-2 minutes on first initialization)")

    try:
        # Initialize VoiceGenHub with Coqui provider
        tts = VoiceGenHub(provider='coqui')
        print("\nInitializing Coqui TTS (this may take a moment)...")
        await tts.initialize()

        print("✅ Coqui TTS initialized successfully!")

        # Test English voice
        print("\nGenerating English speech...")
        response = await tts.generate(
            text='Hello, this is a test of Coqui TTS in English.',
            voice='tacotron2-en'
        )

        # Save English audio
        with open('production_english_coqui.wav', 'wb') as f:
            f.write(response.audio_data)

        print(f"✅ English audio saved: production_english_coqui.wav ({len(response.audio_data)} bytes)")
        print(f"   Duration: {response.duration:.2f} seconds")
        print(f"   Sample rate: {response.sample_rate} Hz")
        print(f"   Format: {response.format}")

        # Test Russian voice
        print("\nGenerating Russian speech...")
        response = await tts.generate(
            text='Привет, это тест Coqui TTS на русском языке.',
            voice='glow-tts-ru'
        )

        # Save Russian audio
        with open('production_russian_coqui.wav', 'wb') as f:
            f.write(response.audio_data)

        print(f"✅ Russian audio saved: production_russian_coqui.wav ({len(response.audio_data)} bytes)")
        print(f"   Duration: {response.duration:.2f} seconds")
        print(f"   Sample rate: {response.sample_rate} Hz")
        print(f"   Format: {response.format}")

        print("\n✅✅✅ Coqui TTS test completed successfully! ✅✅✅")
        print("Audio files are ready to play!")

    except Exception as e:
        print(f"\n❌ Coqui TTS test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_coqui_tts())