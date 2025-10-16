"""
Simple test to generate actual audio files using pyttsx3
This will create MP3 files you can listen to
"""
import pyttsx3
import os

def test_pyttsx3_audio():
    """Generate audio files using pyttsx3"""

    print("Testing pyttsx3 TTS (works on Windows)...")

    # Initialize engine
    engine = pyttsx3.init()

    # Get available voices
    voices = engine.getProperty('voices')
    print(f"Available voices: {len(voices)}")
    for i, voice in enumerate(voices):
        print(f"  {i}: {voice.name} - {voice.languages}")

    # Set properties
    engine.setProperty('rate', 150)  # Speed
    engine.setProperty('volume', 0.9)  # Volume

    # English test
    print("\nGenerating English audio...")
    english_text = "Hello, this is a test of pyttsx3 text to speech. This library works on Windows and generates actual audio files."

    # Save to file
    english_file = os.path.join(os.path.dirname(__file__), "test_english.mp3")
    engine.save_to_file(english_text, english_file)
    engine.runAndWait()

    print(f"✓ English audio saved to: {english_file}")

    # Russian test (if Russian voice available)
    russian_text = "Привет, это тест преобразования текста в речь на русском языке."

    # Try to find Russian voice
    russian_voice = None
    for voice in voices:
        if 'ru' in str(voice.languages).lower() or 'russian' in voice.name.lower():
            russian_voice = voice
            break

    if russian_voice:
        print(f"\nGenerating Russian audio with voice: {russian_voice.name}")
        engine.setProperty('voice', russian_voice.id)

        russian_file = os.path.join(os.path.dirname(__file__), "test_russian.mp3")
        engine.save_to_file(russian_text, russian_file)
        engine.runAndWait()

        print(f"✓ Russian audio saved to: {russian_file}")
    else:
        print("\nNo Russian voice found, generating with default voice...")
        engine.setProperty('voice', voices[0].id)  # Default voice

        russian_file = os.path.join(os.path.dirname(__file__), "test_russian_default.mp3")
        engine.save_to_file(russian_text, russian_file)
        engine.runAndWait()

        print(f"✓ Russian audio (default voice) saved to: {russian_file}")

    print("\n" + "="*60)
    print("Audio files generated! You can now listen to them:")
    print(f"  - English: {english_file}")
    print(f"  - Russian: {russian_file}")
    print("="*60)

if __name__ == "__main__":
    test_pyttsx3_audio()