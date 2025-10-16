[![Unit tests](https://github.com/leweex95/voicegenhub/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/leweex95/voicegenhub/actions/workflows/unit-tests.yml)
[![Daily regression test](https://github.com/leweex95/voicegenhub/actions/workflows/daily-regression-test.yml/badge.svg)](https://github.com/leweex95/voicegenhub/actions/workflows/daily-regression-test.yml)
[![codecov](https://codecov.io/gh/leweex95/voicegenhub/branch/master/graph/badge.svg)](https://codecov.io/gh/leweex95/voicegenhub)

# VoiceGenHub

Simple, user-friendly Text-to-Speech (TTS) library with CLI and Python API.
Supports multiple free and commercial TTS providers.

## Features

- **Multiple Providers**: 
  - Microsoft Edge TTS (free, cloud-based)
  - Google Cloud TTS (commercial, requires credentials)
  - Piper TTS (free, offline neural TTS - Linux/macOS)
  - Coqui TTS (free, offline neural TTS - Linux/macOS)
- **Resilient**: Built-in retry logic and graceful degradation for transient service issues
- **Easy to Use**: Simple CLI and Python API
- **Rich Voice Selection**: Access to hundreds of voices in multiple languages
- **Platform Support**: Works on Windows, macOS, and Linux (with provider-specific limitations)

## Usage

### CLI

#### Generate speech

For Edge TTS:

```bash
voicegenhub synthesize "Hello, world!" \
  --voice en-US-AriaNeural \
  --provider edge \
  --output hello.mp3
```

or for Google TTS:

```bash
voicegenhub synthesize "Hello, world!" \
  --voice en-US-Wavenet-D \
  --provider google \
  --output hello_google.mp3
```

or for Piper TTS (offline, Linux/macOS):

```bash
voicegenhub synthesize "Hello, world!" \
  --voice en_US-lessac-medium \
  --provider piper \
  --output hello_piper.wav
```

or for Coqui TTS (offline, Linux/macOS):

```bash
voicegenhub synthesize "Hello, world!" \
  --voice tacotron2-en \
  --provider coqui \
  --output hello_coqui.wav
```

# List available voices

```bash
voicegenhub voices --language en --provider edge
voicegenhub voices --language en --provider google
```

### Python API

```python
import asyncio
from voicegenhub import VoiceGenHub

async def main():
    tts = VoiceGenHub()
    await tts.initialize(provider="edge")  # or "google"

    response = await tts.generate(
        text="Hello, world!",
        voice="en-US-AriaNeural"  # Edge voice
        # voice="en-US-Wavenet-D"  # Google voice
    )

    with open("speech.mp3", "wb") as f:
        f.write(response.audio_data)

asyncio.run(main())
```

### Help

```bash
voicegenhub --help
```

# CLI Reference
voicegenhub --help

# Commands
`synthesize` – Generate speech from text
`voices` – List available voices

Options for synthesize:
`-v`, `--voice TEXT` – Voice ID (e.g., en-US-AriaNeural, en-US-Wavenet-D, en_US-lessac-medium, tacotron2-en)
`-l`, `--language TEXT` – Language code (e.g., en)
`-o`, `--output PATH` – Output file path
`-f`, `--format [mp3|wav]` – Audio format
`-r`, `--rate FLOAT` – Speech rate (0.5-2.0, default 1.0)
`-p`, `--provider [edge|google|piper|coqui]` – Choose TTS provider

## Reliability

VoiceGenHub is designed to handle transient service issues gracefully:

- **Automatic Retries**: Failed API calls are automatically retried with exponential backoff
- **Lazy Initialization**: Provider initialization doesn't fail your application if the service is temporarily unavailable
- **Graceful Degradation**: Transient errors (like Microsoft API 401/403) are handled to prevent downstream project outages
- **Clock Skew Correction**: Automatically adjusts for time differences between client and server to resolve 401 Unauthorized errors (see [edge-tts#416](https://github.com/rany2/edge-tts/issues/416))

Configuration options (in provider config):
- `max_retries`: Number of retry attempts (default: 3)
- `retry_delay`: Initial delay between retries in seconds (default: 1.0)
- `rate_limit_delay`: Delay after successful requests (default: 0.1)

## Requirements

- Python 3.11+
- For Google TTS: Google Cloud credentials (set via `GOOGLE_APPLICATION_CREDENTIALS` or `GOOGLE_APPLICATION_CREDENTIALS_JSON`)

## Optional Dependencies

Install optional TTS providers:

```bash
# Install Piper TTS (offline neural TTS, Linux/macOS only)
pip install voicegenhub[piper]

# Install Coqui TTS (offline neural TTS, Linux/macOS only)
pip install voicegenhub[coqui]

# Install both
pip install voicegenhub[piper,coqui]
```

### Platform Support

| Provider | Windows | macOS | Linux | Notes |
|----------|---------|-------|-------|-------|
| Edge TTS | ✅ | ✅ | ✅ | Cloud-based, requires internet |
| Google TTS | ✅ | ✅ | ✅ | Requires Google Cloud credentials |
| Piper TTS | ⚠️ | ✅ | ✅ | Offline neural TTS; gracefully disables on Windows |
| Coqui TTS | ⚠️ | ✅ | ✅ | Offline neural TTS; requires Visual C++ Build Tools on Windows |

**Legend:**
- ✅ Full support
- ⚠️ Graceful degradation (logs warning, provider disables automatically)

