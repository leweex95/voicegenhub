#!/usr/bin/env python
"""
Simple TTS Comparison Script

Run this script to generate audio from the test sentence using available TTS models.
Provides simple, runnable examples for each TTS engine.
"""

import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "audio_samples"
OUTPUT_DIR.mkdir(exist_ok=True)

TEST_SENTENCE = (
    "Communication was fragile: intermittent phone signals, "
    "dropped calls, delayed messages, each carrying the weight of potential loss."
)

print("\n" + "=" * 70)
print("SIMPLE TTS COMPARISON - Test Sentence Audio Generation")
print("=" * 70)
print(f"\nTest sentence:\n  {TEST_SENTENCE}\n")
print(f"Output directory: {OUTPUT_DIR}\n")


# ==============================================================================
# Example 1: Kokoro TTS - Lightweight, Fast
# ==============================================================================

def example_kokoro():
    """Kokoro - Lightweight and fast TTS."""
    print("\n[1] KOKORO TTS")
    print("-" * 70)

    try:
        import os
        import warnings
        import soundfile as sf

        # Configure cache
        cache_dir = Path(__file__).parent / 'cache' / 'kokoro'
        os.environ['HF_HUB_CACHE'] = str(cache_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            import kokoro

        # Load model
        model = kokoro.KPipeline(lang_code="a", device="cpu")

        # Synthesize
        results = list(model(TEST_SENTENCE, voice="af_alloy", speed=1.0))
        audio_samples = results[0].audio

        # Save
        output_file = OUTPUT_DIR / "kokoro_output.wav"
        sf.write(output_file, audio_samples, 22050)

        print(f"[SUCCESS] {output_file}")
        print(f"  Voice: af_alloy (female)")
        print(f"  Duration: {len(audio_samples) / 22050:.2f}s")
        print(f"  Sample Rate: 22050 Hz")
        return True

    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        return False


# ==============================================================================
# Example 2: XTTS-v2 (Coqui TTS) - Best Quality
# ==============================================================================

def example_xtts_v2():
    """XTTS-v2 - State-of-the-art open-source TTS."""
    print("\n[2] XTTS-v2 (Coqui TTS)")
    print("-" * 70)

    try:
        from TTS.api import TTS
        import numpy as np
        import soundfile as sf

        print("Loading model (first run downloads ~2GB)...")
        tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            gpu=False,
            progress_bar=False,
        )

        print("Synthesizing...")
        wav = tts.tts(text=TEST_SENTENCE, speaker_wav=None, language="en")

        # Process audio
        wav = np.array(wav)
        if wav.ndim == 1:
            wav = wav.reshape(-1, 1)

        # Save
        output_file = OUTPUT_DIR / "xtts_v2_output.wav"
        sf.write(output_file, wav, samplerate=22050)

        print(f"[SUCCESS] {output_file}")
        print(f"  Model: XTTS-v2 (Multilingual)")
        print(f"  Duration: {len(wav) / 22050:.2f}s")
        print(f"  Sample Rate: 22050 Hz")
        print(f"  Quality: HIGHEST (naturalness)")
        return True

    except ImportError:
        print("[ERROR] TTS package not installed")
        print("  Install with: pip install TTS")
        return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


# ==============================================================================
# Example 3: Bark (Suno) - Emotional, Natural
# ==============================================================================

def example_bark():
    """Bark - Natural sounding with prosody control."""
    print("\n[3] BARK TTS (Suno)")
    print("-" * 70)

    try:
        from bark import SAMPLE_RATE, generate_audio, preload_models
        import soundfile as sf

        print("Preloading models (first run downloads ~3-4GB)...")
        preload_models()

        print("Synthesizing...")
        audio_array = generate_audio(
            TEST_SENTENCE,
            history_prompt="en_speaker_0",
        )

        # Save
        output_file = OUTPUT_DIR / "bark_output.wav"
        sf.write(output_file, audio_array, samplerate=SAMPLE_RATE)

        print(f"[SUCCESS] {output_file}")
        print(f"  Model: Bark (Natural)")
        print(f"  Duration: {len(audio_array) / SAMPLE_RATE:.2f}s")
        print(f"  Sample Rate: {SAMPLE_RATE} Hz")
        print(f"  Quality: EXCELLENT (most natural)")
        print(f"  Note: Slower inference but highest naturalness")
        return True

    except ImportError:
        print("[ERROR] Bark not installed")
        print("  Install with: pip install bark")
        return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==============================================================================
# Example 4: Advanced - XTTS-v2 with Voice Cloning
# ==============================================================================

def example_xtts_v2_voice_cloning():
    """XTTS-v2 with voice cloning from a speaker sample."""
    print("\n[4] XTTS-v2 WITH VOICE CLONING (Advanced)")
    print("-" * 70)

    try:
        from TTS.api import TTS
        import numpy as np
        import soundfile as sf

        print("Loading model...")
        tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            gpu=False,
            progress_bar=False,
        )

        # For this example, we create a synthetic reference
        # In production, you'd use: speaker_wav="path/to/speaker_sample.wav"
        speaker_wav_path = OUTPUT_DIR / "speaker_sample.wav"

        if not speaker_wav_path.exists():
            print("Note: No speaker sample found. Creating synthetic reference...")
            # Create a simple synthetic speaker sample
            sample_rate = 22050
            duration = 2  # 2 seconds
            t = np.linspace(0, duration, sample_rate * duration)
            synthetic_audio = np.sin(2 * np.pi * 200 * t) * 0.1
            sf.write(speaker_wav_path, synthetic_audio, sample_rate)
            print(f"Created sample speaker reference: {speaker_wav_path}")

        print("Synthesizing with voice cloning...")
        wav = tts.tts(
            text=TEST_SENTENCE,
            speaker_wav=str(speaker_wav_path),
            language="en"
        )

        wav = np.array(wav)
        if wav.ndim == 1:
            wav = wav.reshape(-1, 1)

        output_file = OUTPUT_DIR / "xtts_v2_cloned_voice_output.wav"
        sf.write(output_file, wav, samplerate=22050)

        print(f"[SUCCESS] {output_file}")
        print(f"  Voice cloning from: {speaker_wav_path}")
        print(f"  Duration: {len(wav) / 22050:.2f}s")
        return True

    except ImportError:
        print("[ERROR] TTS package not installed")
        return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


# ==============================================================================
# Example 5: Advanced - Bark with Prosody Markers
# ==============================================================================

def example_bark_with_prosody():
    """Bark with prosody/emotion markers for better expressiveness."""
    print("\n[5] BARK WITH PROSODY MARKERS (Advanced)")
    print("-" * 70)

    try:
        from bark import SAMPLE_RATE, generate_audio, preload_models
        import soundfile as sf

        print("Preloading models...")
        preload_models()

        # Enhanced text with prosody markers
        enhanced_text = (
            "Communication was fragile: "
            "intermittent phone signals, "
            "dropped calls, "
            "delayed messages, "
            "each carrying the weight of potential loss."
        )

        print("Synthesizing with enhanced prosody...")
        audio_array = generate_audio(
            enhanced_text,
            history_prompt="en_speaker_0",
            text_temp=0.7,  # Prosody temperature
            waveform_temp=0.8,  # Waveform temperature
        )

        output_file = OUTPUT_DIR / "bark_prosody_output.wav"
        sf.write(output_file, audio_array, samplerate=SAMPLE_RATE)

        print(f"[SUCCESS] {output_file}")
        print(f"  Duration: {len(audio_array) / SAMPLE_RATE:.2f}s")
        print(f"  Prosody Temperature: 0.7")
        print(f"  Waveform Temperature: 0.8")
        return True

    except ImportError:
        print("[ERROR] Bark not installed")
        return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


# ==============================================================================
# Main execution
# ==============================================================================

def main():
    results = {}

    # Run all examples
    results["Kokoro TTS"] = example_kokoro()
    results["XTTS-v2"] = example_xtts_v2()
    results["Bark"] = example_bark()
    results["XTTS-v2 Voice Cloning"] = example_xtts_v2_voice_cloning()
    results["Bark with Prosody"] = example_bark_with_prosody()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for model, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status:10} {model}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nSuccess rate: {passed}/{total}")
    print(f"\nAudio samples saved to: {OUTPUT_DIR}")
    print("\nTo improve results:")
    print("  1. For XTTS-v2: Provide a speaker sample (15-30s) for voice cloning")
    print("  2. For Bark: Adjust text_temp and waveform_temp for variation")
    print("  3. Install optional dependencies: pip install TTS bark-model")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
