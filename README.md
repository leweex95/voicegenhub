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
  - MeloTTS (free, self-hosted neural TTS)
  - Kokoro TTS (free, self-hosted lightweight TTS)
- **Resilient**: Built-in retry logic and graceful degradation for transient service issues
- **Easy to Use**: Simple CLI and Python API
- **Rich Voice Selection**: Access to hundreds of voices in multiple languages
- **Platform Support**: Works on Windows, macOS, and Linux (with provider-specific limitations)

## Usage

For Kokoro TTS:

```bash
voicegenhub synthesize "Hello, world!" --provider kokoro --voice kokoro-af_alloy --output hello.wav
```

**Kokoro supported voices:** Check the list of supported voices [here](https://github.com/nazdridoy/kokoro-tts?tab=readme-ov-file#supported-voices).

For MeloTTS:

```bash
voicegenhub synthesize "Hello, world!" --provider melotts --voice melotts-EN-US --output hello.wav
```

or for other types of English: `melotts-EN-BR` (Great Britain), `melotts-EN-AU` (Australia), `melotts-EN-INDIA`.

**MeloTTS supported voices/languages:** Check the list of supported voices [here](https://github.com/myshell-ai/MeloTTS?tab=readme-ov-file#introduction).

# Print all available voices per provider

```bash
voicegenhub voices --language en --provider edge
voicegenhub voices --language en --provider google
voicegenhub voices --language en --provider melotts
voicegenhub voices --language en --provider kokoro
```

### Python API

```python
import asyncio
from voicegenhub import VoiceGenHub

async def main():
    # Specify provider in constructor
    tts = VoiceGenHub(provider="edge")  # or "google", "melotts", "kokoro"
    await tts.initialize()

    response = await tts.generate(
        text="Hello, world!",
        voice="en-US-AriaNeural"  # Edge voice
        # voice="en-US-Wavenet-D"  # Google voice
        # voice="melotts-EN-US"    # MeloTTS American English
        # voice="kokoro-af_alloy"  # Kokoro voice
    )

    with open("speech.mp3", "wb") as f:
        f.write(response.audio_data)

asyncio.run(main())
```

## Reliability (applicable for Edge TTS)

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

# Install MeloTTS (self-hosted neural TTS)
pip install voicegenhub[melotts]

# Install Kokoro TTS (self-hosted lightweight TTS)
pip install voicegenhub[kokoro]

# Install all optional providers
pip install voicegenhub[piper,coqui,melotts,kokoro]
```
