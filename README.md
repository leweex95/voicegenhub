[![Unit tests](https://github.com/leweex95/voicegenhub/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/leweex95/voicegenhub/actions/workflows/unit-tests.yml)
[![Daily regression test](https://github.com/leweex95/voicegenhub/actions/workflows/daily-regression-test.yml/badge.svg)](https://github.com/leweex95/voicegenhub/actions/workflows/daily-regression-test.yml)
[![codecov](https://codecov.io/gh/leweex95/voicegenhub/branch/master/graph/badge.svg)](https://codecov.io/gh/leweex95/voicegenhub)

# VoiceGenHub

Simple, user-friendly Text-to-Speech (TTS) library with CLI and Python API. Supports multiple free and commercial TTS providers.

## Optional Dependencies

- **Microsoft Edge TTS** (free, cloud-based)
- **Kokoro TTS** (Apache 2.0 licensed, self-hosted lightweight TTS)
- **Bark TTS** (MIT licensed, self-hosted high-naturalness TTS with prosody control)
- **Chatterbox TTS** (MIT licensed, multilingual with emotion control) - Works on CPU or GPU
- **ElevenLabs TTS** (commercial, high-quality voices)

## Usage

### Chatterbox TTS

```bash
poetry run voicegenhub synthesize "Hello, world!" --provider chatterbox --voice chatterbox-default --output hello.wav
```

**Chatterbox features:**
- **Model selection via voice**: Choose between standard, turbo, or multilingual models using the `--voice` flag
- Emotion/intensity control with `exaggeration` parameter (0.0-1.0)
- Zero-shot voice cloning from audio samples
- MIT License - fully commercial compatible
- State-of-the-art quality (competitive with ElevenLabs)
- Built-in Perth watermarking for responsible AI

**Chatterbox voices:**
- `chatterbox-default`: Standard English model with emotion control
- `chatterbox-turbo`: Turbo English model (faster generation, English only)
- `chatterbox-<lang>`: Multilingual model for specific languages (e.g., `chatterbox-es` for Spanish)

**Chatterbox parameters:**
- `--exaggeration`: Emotion intensity (0.0-1.0, default 0.5). Higher values = more dramatic/emotional.
- `--cfg-weight`: Classifier-free guidance weight (0.0-1.0, default 0.5). Controls the influence of the text prompt.
- `--audio-prompt`: Path to reference audio for voice cloning (optional).
- `temperature`, `max_new_tokens`, `repetition_penalty`, `min_p`, `top_p`: Advanced generation parameters (available in Python API).

**Multilingual Support:**
Chatterbox supports 23 languages. Use the appropriate voice for the target language:
```bash
poetry run voicegenhub synthesize "Hola, esto es una prueba de voz en español." --provider chatterbox --voice chatterbox-es --output spanish.wav
```

**Chatterbox supported languages:** ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh

### Bark

```bash
poetry run voicegenhub synthesize "Hello, world!" --provider bark --voice bark-en_speaker_0 --output hello.wav
```

**Bark features:**
- Highest naturalness among open-source TTS
- Prosody markers for emotional expression: `[laughs]`, `[sighs]`, `[pause]`, `[whisper]`
- 100+ speaker presets
- Sound effects generation

**Bark supported voices:** Use preset names like `bark-en_speaker_0`, `bark-en_speaker_1`, etc.

### Edge TTS

```bash
poetry run voicegenhub synthesize "Hello, world!" --provider edge --voice en-US-AriaNeural --output hello.mp3
```

**Edge TTS supported voices:** Check the list of supported voices [here](https://speech.microsoft.com/portal/voicegallery).

### Kokoro TTS

```bash
poetry run voicegenhub synthesize "Hello, world!" --provider kokoro --voice kokoro-af_alloy --output hello.wav
```

**Kokoro supported voices:** Check the list of supported voices [here](https://github.com/nazdridoy/kokoro-tts?tab=readme-ov-file#supported-voices).

### ElevenLabs

```bash
poetry run voicegenhub synthesize "Hello, world!" --provider elevenlabs --voice elevenlabs-EXAVITQu4vr4xnSDxMaL --output hello.mp3
```

Set your API key in `config/elevenlabs-api-key.json` (the key should be stored as the value for `"ELEVENLABS_API_KEY"` in the JSON file).

**ElevenLabs supported voices:** Check the list of supported voices [here](https://elevenlabs.io/docs/voices).

## Print all available voices per provider

```bash
poetry run voicegenhub voices --language en --provider chatterbox
poetry run voicegenhub voices --language en --provider bark
poetry run voicegenhub voices --language en --provider edge
poetry run voicegenhub voices --language en --provider kokoro
poetry run voicegenhub voices --language en --provider elevenlabs
```

## Batch Processing with Concurrency Control

Process multiple texts concurrently with automatic provider-specific resource management:

```bash
# Process multiple texts (auto-numbered output files)
poetry run voicegenhub synthesize "First text" "Second text" "Third text" --provider edge --output batch_output

# Control concurrency (auto-configured per provider if not specified)
poetry run voicegenhub synthesize "Text 1" "Text 2" --provider bark --max-concurrent 2 --output output
```

**Provider Concurrency Limits (automatic):**
- **Fast providers** (Edge, Kokoro, ElevenLabs): Use all CPU cores
- **Heavy providers** (Bark: 2 concurrent, Chatterbox: 1 concurrent)

**Benefits:**
- Model instances are shared across concurrent jobs (no reloading)
- Automatic resource management prevents system overload
- Progress tracking for each job
- Failed jobs don't stop the batch

## Voice Cloning with Kokoro and Chatterbox

VoiceGenHub supports zero-shot voice cloning by combining Kokoro's lightweight voices with Chatterbox's advanced cloning capabilities. This allows you to create custom voices that sound like Kokoro but with Chatterbox's superior quality and emotion control.

### Step-by-Step Guide

1. **Generate a Kokoro voice sample** (modify as desired or keep undistorted):
   ```bash
   # Undistorted voice
   poetry run voicegenhub synthesize "Sample text for cloning." --provider kokoro --voice kokoro-am_michael --output reference.wav --format wav

   # Or with effects (e.g., horror/distortion)
   poetry run voicegenhub synthesize "Sample text for cloning." --provider kokoro --voice kokoro-am_adam --output reference.wav --format wav --pitch-shift -2 --distortion 0.02 --lowpass 2000 --normalize
   ```

2. **Clone the voice with Chatterbox**:
   ```bash
   poetry run voicegenhub synthesize "Your longer text here." --provider chatterbox --voice chatterbox-default --output cloned_voice.wav --audio-prompt reference.wav
   ```

3. **Optional: Adjust emotion and style**:
   ```bash
   poetry run voicegenhub synthesize "Your text." --provider chatterbox --voice chatterbox-default --output cloned_voice.wav --audio-prompt reference.wav --exaggeration 0.8 --cfg-weight 0.7
   ```

**Tips:**
- Use short, clear reference audio (5-10 seconds) for best cloning results
- Combine multiple Kokoro samples with FFmpeg for richer voice profiles
- Experiment with Kokoro effects to create unique voice characteristics before cloning
- Chatterbox supports multilingual cloning from any language reference audio

## Concurrency and Memory Management

**Async Concurrency (Recommended):**
- Use the `synthesize` command with multiple texts for safe concurrent processing within a single process
- Models are loaded once and shared across concurrent jobs
- Prevents out-of-memory (OOM) errors from duplicate model loading
- Automatic provider-specific limits ensure stability

**Multiprocessing Risks:**
- Running multiple CLI processes simultaneously (e.g., via scripts or parallel jobs) loads separate model instances
- Heavy models like Chatterbox (3.7GB) and Bark (4GB) can cause OOM when duplicated across processes
- **Recommendation:** Use async batch processing instead of multiprocessing for heavy providers
- For light providers (Edge, Kokoro), multiprocessing is safer due to minimal memory footprint

## Performance Comparison: All TTS Providers

Here's how all providers compare in terms of speed and quality:

| Provider | Quality (MOS) | Startup Time | Sequential (per req) | Async (3x parallel) | Model Size | Commercial Licensed |
|----------|---------------|--------------|---------------------|-------------------|------------|----------------|
| **Edge TTS** | 3.8/5 | 4.9s | 3.2s | 2.5s | 0MB (cloud) | ✅ Free |
| **Kokoro** | 3.5/5 | 94s | 14.2s | 2.5s | 625MB | ✅ Apache 2.0 |
| **Bark** | 4.2/5 | 180s | 25-40s | 8-12s | 4GB | ✅ MIT |
| **Chatterbox** | 4.3/5 | 120s | 15-30s | 5-15s | 3.7GB | ✅ MIT |
| **ElevenLabs** | 4.5/5* | 2s | 3-5s | 2-3s | 0MB (cloud) | ⚠️ Paid API |

*ElevenLabs quality estimate based on provider reputation; not yet tested with API key.

**Key Findings:**
- **Chatterbox**: Excellent quality with emotion control and multilingual support; MIT licensed, works on CPU
- **Bark**: Highest naturalness for premium narration; MIT licensed (full commercial freedom)
- **Kokoro**: Best balance of quality vs speed for offline use; Apache 2.0 licensed
- **Edge TTS**: Best for real-time, low-latency applications; cloud-based (Microsoft)
- **ElevenLabs**: Highest quality but requires paid API and credit card
- **For commercial purposes:** Use Bark (MIT), Chatterbox (MIT), or Kokoro (Apache 2.0)

## Chatterbox Concurrency Analysis

**Memory Safety**: Chatterbox uses a **shared model instance** (3.6GB) across all threads - **no duplication**. Safe to use 2-8 concurrent threads without OOM risk.

**Performance**: ~2.8x speedup at 4 threads on CPU. Optimal thread count: **2-4 threads**.

**[View Interactive Performance Analysis](assets/concurrency_plot.html)** - Shows speedup curves, memory usage, and timing breakdowns.

## Commercial Licensing

### ✅ Commercially Safe Models:
- **Bark** (MIT License) - Unrestricted commercial use, no attribution required ⭐
- **Chatterbox** (MIT License) - Unrestricted commercial use, no attribution required
- **Kokoro** (Apache 2.0) - Commercial use allowed, attribution required
- **Edge TTS** (Microsoft) - Commercial use allowed
- **ElevenLabs** (Paid API) - Commercial use with valid subscription

## Provider Licenses

For transparency and compliance, here are direct links to the official license terms for each supported TTS provider:

- **Edge TTS (Microsoft)**: [Microsoft Terms of Use](https://www.microsoft.com/en-us/legal/terms-of-use)
- **Kokoro TTS**: [Apache License 2.0](https://github.com/hexgrad/kokoro/blob/main/LICENSE)
- **ElevenLabs TTS**: [ElevenLabs Terms of Service](https://elevenlabs.io/terms)
- **Bark TTS**: [MIT License](https://github.com/suno-ai/bark/blob/main/LICENSE)
- **Chatterbox TTS**: [MIT License](https://github.com/rsxdalv/chatterbox/blob/main/LICENSE)

## Optional Dependencies

Install optional TTS providers:

```bash
# Install Kokoro TTS (self-hosted lightweight TTS)
pip install voicegenhub[kokoro]

# Install Bark (self-hosted high-naturalness TTS)
pip install voicegenhub[bark]

# Install Chatterbox TTS (MIT licensed, multilingual with emotion control)
pip install chatterbox-tts

### Kokoro TTS Installation
Kokoro TTS requires Python 3.11 or higher.

#### Windows & Python 3.13+ Build Limitation

**Important:** On Windows with Python 3.13+, Kokoro TTS (via curated-tokenizers) may require compiling native code if pre-built wheels are not available. This requires Microsoft Visual C++ Build Tools.

If you see errors about missing C++ compilers or build failures when installing Kokoro, follow these steps:

1. Download and install [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
2. During installation, select "Desktop development with C++" workload.
3. After installation, restart your terminal and retry installation:
  ```bash
  poetry install --with kokoro
  # or
  pip install voicegenhub[kokoro]
  ```

If you still see build errors, check for available wheels for `curated-tokenizers` on [PyPI](https://pypi.org/project/curated-tokenizers/#files). If no wheel is available for your Python version, you must build from source (requires Visual C++).

**Recommendation:** For easiest installation, use Python 3.11 or 3.12 on Windows until wheels for Python 3.13+ are published.

#### Installation

```bash
# Using Poetry (recommended):
poetry add voicegenhub[kokoro]
# or:
poetry install --with kokoro
```
