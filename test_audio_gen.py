#!/usr/bin/env python3
"""
Simple test to generate audio with each TTS model.
Tests Kokoro, XTTS-v2, and Bark one at a time.
"""

import asyncio
import os
from pathlib import Path

# Ensure output directory exists
output_dir = Path("comparison_output")
output_dir.mkdir(exist_ok=True)

test_text = "Communication was fragile: intermittent phone signals, dropped calls, delayed messages, each carrying the weight of potential loss."

print("\n" + "=" * 80)
print("TTS AUDIO GENERATION TEST")
print("=" * 80 + "\n")


async def test_all():
    from voicegenhub.core.engine import VoiceGenHub
    from voicegenhub.providers.base import AudioFormat

    # Test 1: Kokoro
    print("[1/3] Testing KOKORO TTS...")
    try:
        engine = VoiceGenHub("kokoro")
        response = await engine.generate(text=test_text, voice="kokoro-af_alloy", audio_format=AudioFormat.WAV)

        output_file = output_dir / "01_KOKORO.wav"
        with open(output_file, "wb") as f:
            f.write(response.audio_data)
        file_size = os.path.getsize(output_file)
        print(f"[OK] Kokoro generated: {output_file} ({file_size} bytes)\n")
    except Exception as e:
        import traceback
        print(f"[FAIL] Kokoro: {type(e).__name__}: {e}")
        traceback.print_exc()
        print()

    # Test 2: XTTS-v2 - get first available voice
    print("[2/3] Testing XTTS-v2...")
    try:
        engine = VoiceGenHub("xtts_v2")
        await engine.initialize()

        # Get available voices
        voices = await engine._provider.get_voices(language="en")
        if voices:
            voice_name = voices[0].name
            print(f"  Using voice: {voice_name}")
            response = await engine.generate(text=test_text, voice=voice_name, audio_format=AudioFormat.WAV)

            output_file = output_dir / "02_XTTS_V2.wav"
            with open(output_file, "wb") as f:
                f.write(response.audio_data)
            file_size = os.path.getsize(output_file)
            print(f"[OK] XTTS-v2 generated: {output_file} ({file_size} bytes)\n")
        else:
            print("[FAIL] XTTS-v2: No voices available\n")
    except Exception as e:
        import traceback
        print(f"[FAIL] XTTS-v2: {type(e).__name__}: {e}")
        traceback.print_exc()
        print()

    # Test 3: Bark
    print("[3/3] Testing BARK...")
    try:
        engine = VoiceGenHub("bark")
        response = await engine.generate(text=test_text, voice="bark-en_speaker_0", audio_format=AudioFormat.WAV)

        output_file = output_dir / "03_BARK.wav"
        with open(output_file, "wb") as f:
            f.write(response.audio_data)
        file_size = os.path.getsize(output_file)
        print(f"[OK] Bark generated: {output_file} ({file_size} bytes)\n")
    except Exception as e:
        import traceback
        print(f"[FAIL] Bark: {type(e).__name__}: {e}")
        traceback.print_exc()
        print()

asyncio.run(test_all())

print("=" * 80)
print("Test complete. Check comparison_output/ for generated audio files.")
print("=" * 80 + "\n")
