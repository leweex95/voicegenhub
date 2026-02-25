# Voice Cloning and Design

VoiceGenHub supports both zero-shot voice cloning (from audio samples) and voice design (from textual descriptions).

## 1. Voice Cloning with [Chatterbox](https://github.com/rsxdalv/chatterbox)

### Steps

1.  **Generate a Reference Audio** (or use an existing sample):
    ```bash
    voicegenhub synthesize "Sample text for cloning." \
        --provider kokoro \
        --voice kokoro-am_michael \
        --output reference.wav
    ```

2.  **Clone the Voice**:
    ```bash
    voicegenhub synthesize "Your text to be synthesized in the cloned voice." \
        --provider chatterbox \
        --audio-prompt reference.wav \
        --output cloned_voice.wav
    ```

3.  **Adjust Emotion and Style**:
    ```bash
    voicegenhub synthesize "Your text." \
        --provider chatterbox \
        --audio-prompt reference.wav \
        --exaggeration 0.8 \
        --cfg-weight 0.7
    ```

### Tips for Better Quality
-   Use clear, noise-free reference audio (5-10 seconds recommended).
-   Chatterbox supports **multilingual cloning** (clone any language, synthesize in any other language).

## 2. Voice Design with [Qwen 3 TTS](https://github.com/QwenLM/Qwen3-TTS)

*Requires `Qwen3-TTS-VoiceDesign` model for full control, available via Python API or remote GPU.*

### Qwen 3 TTS Voice Design Features

-   **Natural Language Instruction**: Design custom voices using descriptions.
-   **Example Voice Design**:
    -   `"Female, 25 years old, cheerful and energetic, slightly high-pitched with playful intonation"`
    -   `"Male, 17 years old, gaining confidence, deeper breath support, vowels tighten when nervous"`
    -   `"Elderly male, 70 years old, wise and gentle, slightly raspy with warm timbre"`

---
*For more details on Qwen 3 TTS design modes, see the [Qwen 3 TTS documentation](https://github.com/QwenLM/Qwen3-TTS).*
