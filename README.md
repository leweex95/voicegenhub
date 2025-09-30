# VoiceGenHub

Simple Text-to-Speech library using Edge TTS.

## Installation

```bash
pip install voicegenhub
```

## Usage

### Python API

```python
import asyncio
from voicegenhub import VoiceGenHub

async def main():
    tts = VoiceGenHub()
    await tts.initialize()
    
    response = await tts.generate(
        text="Hello, world!",
        voice="en-US-AriaNeural"
    )
    
    with open("speech.mp3", "wb") as f:
        f.write(response.audio_data)

asyncio.run(main())
```

### CLI

```bash
# Generate speech
voicegenhub synthesize "Hello, world!" --voice en-US-AriaNeural --output hello.mp3

# List available voices
voicegenhub voices --language en
```

## Requirements

- Python 3.11+
- edge-tts

## License

MIT License