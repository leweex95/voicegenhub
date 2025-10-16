"""
Production code test - using VoiceGenHub to generate audio files
"""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from voicegenhub import VoiceGenHub


async def test_production_audio():
    """Test VoiceGenHub production code to generate actual audio files"""

    print("=" * 60)
    print("Testing VoiceGenHub Production Code")
    print("=" * 60)

    # Test with Edge TTS (should work)
    print("\nTesting Edge TTS...")
    try:
        tts = VoiceGenHub(provider='edge')
        await tts.initialize()

        # English
        english_text = "Hello, this is VoiceGenHub with Edge TTS. This is the production code working correctly."
        response = await tts.generate(text=english_text, voice='en-US-AriaNeural')

        english_file = os.path.join(os.path.dirname(__file__), "production_english_edge.mp3")
        with open(english_file, "wb") as f:
            f.write(response.audio_data)

        print(f"✓ English audio saved: {english_file} ({len(response.audio_data)} bytes)")

        # Russian
        russian_text = "Привет, это VoiceGenHub с Edge TTS на русском языке."
        response_ru = await tts.generate(text=russian_text, voice='en-US-AriaNeural')  # Edge supports Russian voices too

        russian_file = os.path.join(os.path.dirname(__file__), "production_russian_edge.mp3")
        with open(russian_file, "wb") as f:
            f.write(response_ru.audio_data)

        print(f"✓ Russian audio saved: {russian_file} ({len(response_ru.audio_data)} bytes)")

    except Exception as e:
        print(f"✗ Edge TTS failed: {e}")

    # Test with Google TTS (if credentials available)
    print("\nTesting Google TTS...")
    try:
        # Check for credentials
        creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if not creds_path:
            config_creds = os.path.join(os.path.dirname(__file__), 'config', 'google-credentials.json')
            if os.path.exists(config_creds):
                creds_path = config_creds
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path

        if creds_path:
            tts_google = VoiceGenHub(provider='google')
            await tts_google.initialize()

            voices = await tts_google.get_voices(language='en')
            if voices:
                voice_id = voices[0]['id']

                response_google = await tts_google.generate(text=english_text, voice=voice_id)

                google_file = os.path.join(os.path.dirname(__file__), "production_english_google.mp3")
                with open(google_file, "wb") as f:
                    f.write(response_google.audio_data)

                print(f"✓ Google audio saved: {google_file} ({len(response_google.audio_data)} bytes)")
            else:
                print("✗ No Google voices available")
        else:
            print("✗ No Google credentials found")

    except Exception as e:
        print(f"✗ Google TTS failed: {e}")

    print("\n" + "=" * 60)
    print("Production audio files generated!")
    print("Listen to the MP3 files to verify TTS functionality:")
    print("  - production_english_edge.mp3")
    print("  - production_russian_edge.mp3")
    if os.path.exists(os.path.join(os.path.dirname(__file__), "production_english_google.mp3")):
        print("  - production_english_google.mp3")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_production_audio())