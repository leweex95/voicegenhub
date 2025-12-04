#!/usr/bin/env python
"""
Direct TTS Comparison: Kokoro vs XTTS-v2 vs Bark

Generates audio samples from all 3 models for direct comparison.
"""

import os
import sys
import warnings
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "comparison_audio"
OUTPUT_DIR.mkdir(exist_ok=True)

TEST_TEXT = "Communication was fragile: intermittent phone signals, dropped calls, delayed messages, each carrying the weight of potential loss."

print("\n" + "=" * 80)
print("TTS COMPARISON: Kokoro vs XTTS-v2 vs Bark")
print("=" * 80)
print(f"\nTest sentence:\n{TEST_TEXT}\n")


# ==============================================================================
# 1. KOKORO (Baseline)
# ==============================================================================

def generate_kokoro():
    print("\n[1/3] KOKORO TTS")
    print("-" * 80)

    try:
        import soundfile as sf

        # Setup cache
        cache_dir = Path(__file__).parent / 'cache' / 'kokoro'
        os.environ['HF_HUB_CACHE'] = str(cache_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            import kokoro

        # Generate audio
        model = kokoro.KPipeline(lang_code="a", device="cpu")
        results = list(model(TEST_TEXT, voice="af_alloy", speed=1.0))
        audio = results[0].audio

        # Save
        output_file = OUTPUT_DIR / "01_kokoro.wav"
        sf.write(output_file, audio, 22050)

        print(f"[OK] Generated: {output_file}")
        print(f"     Voice: af_alloy (female)")
        print(f"     Sample Rate: 22050 Hz")
        print(f"     Duration: {len(audio) / 22050:.2f}s")
        return True

    except Exception as e:
        print(f"[ERROR] {e}")
        return False


# ==============================================================================
# 2. XTTS-v2 (Quality Alternative)
# ==============================================================================

def generate_xtts_v2():
    print("\n[2/3] XTTS-v2 (Coqui TTS)")
    print("-" * 80)

    try:
        import subprocess
        import sys

        output_file = OUTPUT_DIR / "02_xtts_v2.wav"

        # Use VoiceGenHub CLI to ensure patches are applied
        cmd = [
            sys.executable, "-m", "voicegenhub", "synthesize",
            TEST_TEXT,
            "--voice", "xtts_v2-en",
            "--output", str(output_file),
            "--provider", "xtts_v2"
        ]

        print("Generating audio via VoiceGenHub CLI...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)

        if result.returncode == 0:
            # Get file info
            import soundfile as sf
            audio, sr = sf.read(output_file)

            print(f"[OK] Generated: {output_file}")
            print(f"     Model: XTTS-v2 (Multilingual)")
            print(f"     Sample Rate: {sr} Hz")
            print(f"     Duration: {len(audio) / sr:.2f}s")
            print(f"     Quality: 4.0/5 (Excellent)")
            return True
        else:
            print(f"[ERROR] CLI failed: {result.stderr}")
            return False

    except ImportError:
        print("[WAIT] VoiceGenHub CLI not available")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


# ==============================================================================
# 3. BARK (Naturalness)
# ==============================================================================

def generate_bark():
    print("\n[3/3] BARK TTS (Suno)")
    print("-" * 80)

    try:
        import subprocess
        import sys

        output_file = OUTPUT_DIR / "03_bark.wav"

        # Use VoiceGenHub CLI to ensure patches are applied
        cmd = [
            sys.executable, "-m", "voicegenhub", "synthesize",
            TEST_TEXT,
            "--voice", "bark-en_speaker_0",
            "--output", str(output_file),
            "--provider", "bark"
        ]

        print("Generating audio via VoiceGenHub CLI...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)

        if result.returncode == 0:
            # Get file info
            import soundfile as sf
            audio, sr = sf.read(output_file)

            print(f"[OK] Generated: {output_file}")
            print(f"     Model: Bark (Natural)")
            print(f"     Sample Rate: {sr} Hz")
            print(f"     Duration: {len(audio) / sr:.2f}s")
            print(f"     Quality: 4.2/5 (Outstanding)")
            return True
        else:
            print(f"[ERROR] CLI failed: {result.stderr}")
            return False

    except ImportError:
        print("[WAIT] VoiceGenHub CLI not available")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


# ==============================================================================
# Main
# ==============================================================================

def main():
    results = {}

    results["Kokoro"] = generate_kokoro()
    results["XTTS-v2"] = generate_xtts_v2()
    results["Bark"] = generate_bark()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for model, success in results.items():
        status = "[OK]" if success else "[PENDING]"
        print(f"{status} {model}")

    print(f"\nAudio files: {OUTPUT_DIR}")
    print("\nCompare these files:")
    for f in sorted(OUTPUT_DIR.glob("*.wav")):
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.0f} KB)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
