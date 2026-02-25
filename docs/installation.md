# Installation and Requirements

Detailed installation guide for various TTS providers and optional features.

## Basic Installation

```bash
pip install voicegenhub
```

## Optional Provider Dependencies

To use certain providers, you need to install their respective dependencies:

```bash
# Kokoro TTS (Lightweight, self-hosted)
pip install voicegenhub[kokoro]

# Bark TTS (High Quality, MIT)
pip install voicegenhub[bark]

# Chatterbox TTS (High Quality, MIT)
pip install chatterbox-tts

# Qwen 3 TTS (State-of-the-Art, Apache 2.0)
pip install voicegenhub[qwen]

# ElevenLabs TTS (Commercial)
pip install elevenlabs
```

---

## 2. Dependencies

### Voice Cloning Requirements (Chatterbox)

For voice cloning features with Chatterbox TTS:

```bash
pip install voicegenhub[voice-cloning]
```

**System Requirements:**
- **FFmpeg**: Required when `torchcodec` is installed for voice cloning.
- **PyTorch**: Required for local model execution.

**Windows Installations**: Download the "full-shared" FFmpeg build from [ffmpeg.org](https://ffmpeg.org/download.html#build-windows) and add the `bin` directory to your system PATH.

---

## Technical Note: CUDA and CPU Execution

- VoiceGenHub automatically detects if a GPU is available.
- For **Chatterbox** and **Bark**, if no GPU is found, the library will fall back to **CPU execution**.
- For **Qwen 3 TTS**, high-quality models (1.7B) are recommended for **GPU acceleration** (remote or local).

---

## Windows & Python 3.13+ (Kokoro)

On Windows with Python 3.13+, **Kokoro TTS** may require Microsoft Visual C++ Build Tools for compilation if pre-built wheels are not available.

1.  Download [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
2.  Select "Desktop development with C++" workload.
3.  Restart terminal and retry installation.
