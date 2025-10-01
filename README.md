# VoiceGenHub

Simple, user-friendly Text-to-Speech (TTS) library with CLI and Python API.
Supports Microsoft Edge and Google Cloud voices.

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
`-v`, `--voice TEXT` – Voice ID (e.g., en-US-AriaNeural, en-US-Wavenet-D)
`-l`, `--language TEXT` – Language code (e.g., en)
`-o`, `--output PATH` – Output file path
`-f`, `--format [mp3|wav]` – Audio format
`-r`, `--rate FLOAT` – Speech rate (0.5-2.0, default 1.0)
`-p`, `--provider [edge|google]` – Choose TTS provider
