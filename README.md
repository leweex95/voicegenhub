[![Unit tests](https://github.com/leweex95/voicegenhub/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/leweex95/voicegenhub/actions/workflows/unit-tests.yml)
[![Daily regression test](https://github.com/leweex95/voicegenhub/actions/workflows/daily-regression-test.yml/badge.svg)](https://github.com/leweex95/voicegenhub/actions/workflows/daily-regression-test.yml)
[![codecov](https://codecov.io/gh/leweex95/voicegenhub/branch/master/graph/badge.svg)](https://codecov.io/gh/leweex95/voicegenhub)

# VoiceGenHub

Simple, user-friendly Text-to-Speech (TTS) library with CLI and Python API.
Supports multiple free and commercial TTS providers.

## Supported Providers

- Microsoft Edge TTS (free, cloud-based)
- Piper TTS (free, offline neural TTS - Linux/macOS)
- Coqui TTS (free, offline neural TTS - Linux/macOS)
- MeloTTS (free, self-hosted neural TTS)
- Kokoro TTS (free, self-hosted lightweight TTS)
- XTTS-v2 (free, self-hosted multilingual TTS with voice cloning)
- Bark (free, self-hosted high-naturalness TTS with prosody control)
- ElevenLabs TTS (commercial, high-quality voices)

## Usage

For Edge TTS:

```bash
poetry run voicegenhub synthesize "Hello, world!" --provider edge --voice en-US-AriaNeural --output hello.mp3
```

**Edge TTS supported voices:** Check the list of supported voices [here](https://speech.microsoft.com/portal/voicegallery).

For Kokoro TTS:

```bash
poetry run voicegenhub synthesize "Hello, world!" --provider kokoro --voice kokoro-af_alloy --output hello.wav
```

**Kokoro supported voices:** Check the list of supported voices [here](https://github.com/nazdridoy/kokoro-tts?tab=readme-ov-file#supported-voices).

For MeloTTS:

```bash
poetry run voicegenhub synthesize "Hello, world!" --provider melotts --voice melotts-EN-US --output hello.wav
```

or for other types of English: `melotts-EN-BR` (Great Britain), `melotts-EN-AU` (Australia), `melotts-EN-INDIA`.

**MeloTTS supported voices/languages:** Check the list of supported voices [here](https://github.com/myshell-ai/MeloTTS?tab=readme-ov-file#introduction).

For ElevenLabs:

```bash
poetry run voicegenhub synthesize "Hello, world!" --provider elevenlabs --voice elevenlabs-EXAVITQu4vr4xnSDxMaL --output hello.mp3
```

Set your API key: `export ELEVENLABS_API_KEY=your_api_key_here`

**ElevenLabs supported voices:** Check the list of supported voices [here](https://elevenlabs.io/docs/voices).

For XTTS-v2 (Coqui TTS):

```bash
poetry run voicegenhub synthesize "Hello, world!" --provider xtts_v2 --voice xtts_v2-en --output hello.wav
```

**XTTS-v2 features:**
- 16 languages support (English, Spanish, French, German, Italian, Portuguese, Polish, Dutch, Russian, Chinese, Japanese, Korean, Turkish, Arabic, Hindi, Greek)
- Zero-shot voice cloning (provide a 15-30 second speaker sample)
- High-quality multilingual synthesis

**XTTS-v2 supported voices:** Use language codes like `xtts_v2-en`, `xtts_v2-es`, `xtts_v2-fr`, etc.

For Bark (Suno TTS):

```bash
poetry run voicegenhub synthesize "Hello, world!" --provider bark --voice bark-en_speaker_0 --output hello.wav
```

**Bark features:**
- Highest naturalness among open-source TTS
- Prosody markers for emotional expression: `[laughs]`, `[sighs]`, `[pause]`, `[whisper]`
- 100+ speaker presets
- Sound effects generation

**Bark supported voices:** Use preset names like `bark-en_speaker_0`, `bark-en_speaker_1`, etc.

# Print all available voices per provider

```bash
poetry run voicegenhub voices --language en --provider edge
poetry run voicegenhub voices --language en --provider melotts
poetry run voicegenhub voices --language en --provider kokoro
poetry run voicegenhub voices --language en --provider elevenlabs
poetry run voicegenhub voices --language en --provider xtts_v2
poetry run voicegenhub voices --language en --provider bark
```

## Performance Comparison: Edge TTS vs Kokoro TTS

Both providers support **full async/parallelized operations**. Here's how they compare:

| Metric | Edge TTS | Kokoro TTS | Notes |
|--------|----------|-----------|-------|
| **Startup** | 4.9s | 94s | Kokoro downloads & loads model on first use; cached afterwards |
| **Sequential** (single request) | 3.2s | 14.2s | Local inference adds overhead vs cloud API |
| **Async** (per request, 3x parallel) | 2.5s | 2.5s | Both highly parallelizable via async |

**Key Findings:**
- Kokoro achieves **parity with Edge TTS in async scenarios** (2.5s/req with full parallelization)
- Kokoro model caching is transparent: first synthesis ~95s, subsequent ~24s (logs show timing)
- Cold startup penalty is one-time; negligible for batched/continuous workloads
- Both providers use thread pool executors (`run_in_executor`) to maintain async efficiency
- For batch operations or long-running services, Kokoro's offline nature and strong async support make it competitive despite higher startup cost

**Cache Location:**
- **Kokoro models** are cached locally within the project: `cache/kokoro/` (~625MB)
- **XTTS-v2 models** are cached locally: `cache/` (~2GB for full model)
- **Bark models** are cached locally: `cache/` (~4GB for full model)
- **Edge TTS** uses Microsoft's cloud API (no local caching needed)
- Cache persists across sessions and survives repository operations

## Performance Comparison: All TTS Providers

Here's how all providers compare in terms of speed and quality:

| Provider | Quality (MOS) | Startup Time | Sequential (per req) | Async (3x parallel) | Model Size | Commercial Use |
|----------|---------------|--------------|---------------------|-------------------|------------|----------------|
| **Edge TTS** | 3.8/5 | 4.9s | 3.2s | 2.5s | 0MB (cloud) | ✅ Free |
| **Kokoro** | 3.5/5 | 94s | 14.2s | 2.5s | 625MB | ✅ Apache 2.0 |
| **XTTS-v2** | 4.0/5 | 120s | 18-25s | 6-8s | 2GB | ✅ MPL-2.0 |
| **Bark** | 4.2/5 | 180s | 25-40s | 8-12s | 4GB | ✅ MIT |
| **ElevenLabs** | 4.5/5 | 2s | 3-5s | 2-3s | 0MB (cloud) | ⚠️ Paid API |

**Key Findings:**
- **Edge TTS**: Best for real-time, low-latency applications
- **Kokoro**: Best balance of quality vs speed for offline use
- **XTTS-v2**: Best for multilingual content and voice cloning
- **Bark**: Highest naturalness, best for premium narration
- All local providers have high initial startup cost but excellent async performance
- Commercial licensing: All open-source providers are free for commercial use

## Requirements

- Python 3.11+

## Optional Dependencies

Install optional TTS providers:

```bash
# Install Piper TTS (offline neural TTS, Linux/macOS only)
pip install voicegenhub[piper]

# Install Coqui TTS (offline neural TTS, Linux/macOS only)
pip install voicegenhub[coqui]

# Install MeloTTS (self-hosted neural TTS)
pip install voicegenhub[melotts]


# Install Kokoro TTS (self-hosted lightweight TTS)
pip install voicegenhub[kokoro]

# Install XTTS-v2 (self-hosted multilingual TTS with voice cloning)
pip install voicegenhub[xtts_v2]

# Install Bark (self-hosted high-naturalness TTS)
pip install voicegenhub[bark]

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

# Or if already installed:
poetry install --with kokoro

# Using pip:
pip install voicegenhub[kokoro]
```

#### Supported Python Versions

- Python 3.11 ✅
- Python 3.12 ✅
- Python 3.13 ✅
- Python 3.14+ (untested, likely works)

#### Troubleshooting

If you get `ImportError: No module named 'kokoro'`:

1. Verify your Python version: `python --version` (must be 3.11+)
2. Reinstall with extras: `pip install --force-reinstall voicegenhub[kokoro]`
3. Check cache cleanup: `rm -rf ~/.cache/huggingface/` and retry

# Install all optional providers
pip install voicegenhub[piper,coqui,melotts,kokoro,xtts_v2,bark]
```
