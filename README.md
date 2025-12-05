[![Unit tests](https://github.com/leweex95/voicegenhub/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/leweex95/voicegenhub/actions/workflows/unit-tests.yml)
[![Daily regression test](https://github.com/leweex95/voicegenhub/actions/workflows/daily-regression-test.yml/badge.svg)](https://github.com/leweex95/voicegenhub/actions/workflows/daily-regression-test.yml)
[![codecov](https://codecov.io/gh/leweex95/voicegenhub/branch/master/graph/badge.svg)](https://codecov.io/gh/leweex95/voicegenhub)

# VoiceGenHub

Simple, user-friendly Text-to-Speech (TTS) library with CLI and Python API.
Supports multiple free and commercial TTS providers.

## Supported Providers

- **Microsoft Edge TTS** (free, cloud-based)
- **Kokoro TTS** (Apache 2.0 licensed, self-hosted lightweight TTS)
- **Bark TTS** (MIT licensed, self-hosted high-naturalness TTS with prosody control) ⭐ Recommended for commercial use
- **ElevenLabs TTS** (commercial, high-quality voices)

**For YouTube & Monetized Content:** Use Bark (MIT License) or Kokoro (Apache 2.0) - both are fully commercial-friendly. See [Commercial Licensing](#commercial-licensing) section below for details.

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

For ElevenLabs:

```bash
poetry run voicegenhub synthesize "Hello, world!" --provider elevenlabs --voice elevenlabs-EXAVITQu4vr4xnSDxMaL --output hello.mp3
```

Set your API key: `export ELEVENLABS_API_KEY=your_api_key_here`

**ElevenLabs supported voices:** Check the list of supported voices [here](https://elevenlabs.io/docs/voices).

### ⚠️ XTTS-v2 (Deprecated)

**⚠️ IMPORTANT:** XTTS-v2 uses CPML (Coqui Public Model License) which **does NOT allow commercial/monetized use** without purchasing a commercial license from Coqui.

- ❌ **NOT suitable for YouTube, monetized content, or commercial applications**
- ❌ Requires commercial license purchase from Coqui AI
- ✅ Still supported for non-commercial use only

**Use Bark instead** for commercial projects (see Bark section below).

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

| Provider | Quality (MOS) | Startup Time | Sequential (per req) | Async (3x parallel) | Model Size | Commercial Licensed |
|----------|---------------|--------------|---------------------|-------------------|------------|----------------|
| **Edge TTS** | 3.8/5 | 4.9s | 3.2s | 2.5s | 0MB (cloud) | ✅ Free |
| **Kokoro** | 3.5/5 | 94s | 14.2s | 2.5s | 625MB | ✅ Apache 2.0 |
| **Bark** | 4.2/5 | 180s | 25-40s | 8-12s | 4GB | ✅ MIT |
| **ElevenLabs** | 4.5/5 | 2s | 3-5s | 2-3s | 0MB (cloud) | ⚠️ Paid API |

**Key Findings:**
- **Edge TTS**: Best for real-time, low-latency applications; cloud-based (Microsoft)
- **Kokoro**: Best balance of quality vs speed for offline use; Apache 2.0 licensed
- **Bark**: Highest naturalness for premium narration; MIT licensed (full commercial freedom)
- **ElevenLabs**: Highest quality but requires paid API and credit card
- All locally-hosted providers require GPU or CPU inference (slow on first run, fast after caching)
- **For YouTube & Monetized Content:** Use Bark (MIT) or Kokoro (Apache 2.0) - both fully commercial-friendly

## Commercial Licensing

If you're monetizing your content (YouTube, podcasts, etc.), ensure your TTS provider is commercially licensed:

### ✅ Commercially Safe Models:
- **Bark** (MIT License) - Unrestricted commercial use, no attribution required ⭐
- **Kokoro** (Apache 2.0) - Commercial use allowed, attribution required
- **Edge TTS** (Microsoft) - Commercial use allowed
- **ElevenLabs** (Paid API) - Commercial use with valid subscription

### ❌ NON-Commercial (Do Not Use):
- **XTTS-v2** (CPML License) - Requires commercial license purchase from Coqui AI
- Other CC-BY-NC-ND or custom non-commercial licenses

For full commercial licensing details, see [commercial_models_summary.txt](./commercial_models_summary.txt).

## Requirements

- Python 3.11+

## Optional Dependencies

Install optional TTS providers:

```bash
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
pip install voicegenhub[kokoro,xtts_v2,bark]
```
