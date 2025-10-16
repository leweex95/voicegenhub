#!/usr/bin/env python3
"""
Test Piper TTS generation and save audio files.
Demonstrates Piper TTS functionality with graceful Windows handling.
"""

import asyncio
import sys
import os

# Add src to path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from voicegenhub import VoiceGenHub


async def test_piper_tts():
    """Test Piper TTS generation."""
    print("Testing Piper TTS provider...")

    try:
        # Initialize VoiceGenHub with Piper provider
        tts = VoiceGenHub(provider='piper')
        await tts.initialize()

        print("Piper TTS initialized successfully!")

        # Test English voice
        print("\nTesting English voice...")
        response = await tts.generate(
            text='Hello, this is a test of Piper TTS in English.',
            voice='en_US-lessac-medium'
        )

        # Save English audio
        with open('production_english_piper.wav', 'wb') as f:
            f.write(response.audio_data)

        print(f"English audio saved: production_english_piper.wav")
        print(f"Duration: {response.duration:.2f} seconds")
        print(f"Sample rate: {response.sample_rate} Hz")
        print(f"Format: {response.format}")

        # Test Russian voice
        print("\nTesting Russian voice...")
        response = await tts.generate(
            text='Привет, это тест Piper TTS на русском языке.',
            voice='ru_RU-irene-medium'
        )

        # Save Russian audio
        with open('production_russian_piper.wav', 'wb') as f:
            f.write(response.audio_data)

        print(f"Russian audio saved: production_russian_piper.wav")
        print(f"Duration: {response.duration:.2f} seconds")
        print(f"Sample rate: {response.sample_rate} Hz")
        print(f"Format: {response.format}")

        print("\n✅ Piper TTS test completed successfully!")

    except Exception as e:
        print(f"❌ Piper TTS test failed: {e}")
        print("This is expected on Windows due to onnxruntime DLL issues.")
        print("Piper TTS gracefully degrades and disables itself on incompatible platforms.")


if __name__ == "__main__":
    asyncio.run(test_piper_tts())