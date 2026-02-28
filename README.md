[![Unit tests](https://github.com/leweex95/voicegenhub/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/leweex95/voicegenhub/actions/workflows/unit-tests.yml)
[![Daily regression test](https://github.com/leweex95/voicegenhub/actions/workflows/daily-regression-test.yml/badge.svg)](https://github.com/leweex95/voicegenhub/actions/workflows/daily-regression-test.yml)
[![codecov](https://codecov.io/gh/leweex95/voicegenhub/branch/master/graph/badge.svg)](https://codecov.io/gh/leweex95/voicegenhub)

# VoiceGenHub

Simple CLI-first Text-to-Speech library supporting multiple free and commercial providers â€” including free Kaggle GPU inference for state-of-the-art models.

---

## Install

```bash
poetry add voicegenhub
```

---

## Providers

| Provider | License | Local / Cloud | Notes |
|---|---|---|---|
| **Edge TTS** | Free (Microsoft) | Cloud | Fastest, zero setup |
| **Kokoro** | Apache 2.0 | Local | Lightweight, high quality |
| **Bark** | MIT | Local | Prosody markers, 100+ voices |
| **Chatterbox** | MIT | Local | Emotion control, voice cloning |
| **Qwen 3 TTS** | Apache 2.0 | Local / Kaggle GPU | State-of-the-art multilingual |
| **ElevenLabs** | Paid API | Cloud | Commercial-grade voices |

---

## Synthesize

### Edge TTS (fastest, no setup)

```bash
poetry run voicegenhub synthesize "Hello, world!" --provider edge --voice en-US-AriaNeural --output hello.mp3
```

### Kokoro

```bash
poetry run voicegenhub synthesize "Hello, world!" --provider kokoro --voice kokoro-af_alloy --output hello.wav
```

### Bark

```bash
poetry run voicegenhub synthesize "Hello, world!" --provider bark --voice bark-en_speaker_0 --output hello.wav
```

Bark supports prosody markers: `[laughs]`, `[sighs]`, `[pause]`, `[whisper]`.

### Chatterbox (emotion control + voice cloning)

```bash
# Basic
poetry run voicegenhub synthesize "Hello, world!" --provider chatterbox --voice chatterbox-default --output hello.wav

# With emotion intensity
poetry run voicegenhub synthesize "This is incredible!" --provider chatterbox --voice chatterbox-default --exaggeration 0.8 --output excited.wav

# Voice cloning from a reference file
poetry run voicegenhub synthesize "Hello, cloned voice." --provider chatterbox --voice chatterbox-default --audio-prompt reference.wav --output cloned.wav
```

### ElevenLabs

```bash
poetry run voicegenhub synthesize "Hello, world!" --provider elevenlabs --voice elevenlabs-EXAVITQu4vr4xnSDxMaL --output hello.mp3
```

Store your API key in `config/elevenlabs-api-key.json` as `{"ELEVENLABS_API_KEY": "..."}`.

---

## Qwen 3 TTS

```bash
poetry run voicegenhub synthesize "Hello from the GPU!" --provider qwen --voice Ryan --language en --gpu p100

# Batch multiple sentences in one GPU job, saved as audio_001.wav â€¦ audio_007.wav + manifest.json
poetry run voicegenhub synthesize \
  "The quick brown fox jumps over the lazy dog." \
  "Technology is changing the world at an unprecedented pace." \
  "The sunset painted the sky in shades of orange and pink." \
  --provider qwen --voice Ryan --language en --gpu p100

# Chinese with native speaker
poetry run voicegenhub synthesize "ä˝ ĺĄ˝ďĽŚčż™ćŻä¸€ä¸Şćµ‹čŻ•ă€‚" --provider qwen --voice Serena --language zh --gpu p100

# Use T4 GPUs instead of P100
poetry run voicegenhub synthesize "Hello!" --provider qwen --gpu t4

# Custom model, output directory, polling options
poetry run voicegenhub synthesize "Hello!" \
  --provider qwen \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --output-dir my_output \
  --gpu p100 \
  --timeout 90 \
  --poll-interval 30
```

### Voice Cloning (Qwen3-TTS, Kaggle GPU)

Clone your own voice onto arbitrary text using the Qwen3-TTS Base model and a reference WAV:

```bash
poetry run voicegenhub synthesize "this is my speech using my own voice" \
  --provider qwen \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --audio-prompt "<path to reference audio>" \
  --ref-text "<exact transcript of the reference audio>" \
  --gpu p100
```

**Tips:**
- Use a reference WAV of at least 20 seconds, with clear speech and a matching transcript for best results.
- The `--ref-text` should be the exact transcript of the reference audio (no ellipsis or truncation).
- For batch synthesis, pass multiple texts in quotes.

See [docs/cloning_and_design.md](docs/cloning_and_design.md) for advanced usage and troubleshooting.
```

Batch output lands in a timestamped folder (e.g. `20260227_123130_p100/`) with:
- `audio_001.wav`, `audio_002.wav`, â€¦ (one per input sentence)
- `manifest.json` â€” maps each filename to its source text and duration

---

## Batch Processing (local providers)

Pass multiple texts to any provider â€” processed concurrently with shared model instances:

```bash
poetry run voicegenhub synthesize "First." "Second." "Third." --provider edge --output batch_output
```

---

## Voices

```bash
poetry run voicegenhub voices --language en --provider edge
poetry run voicegenhub voices --language en --provider kokoro
poetry run voicegenhub voices --language en --provider bark
poetry run voicegenhub voices --language en --provider chatterbox
poetry run voicegenhub voices --language en --provider elevenlabs
poetry run voicegenhub voices --provider qwen
```

### Qwen speakers

| Speaker | Gender | Native language | Notes |
|---|---|---|---|
| `Ryan` | Male | English | Dynamic, rhythmic — works for all languages |
| `Aiden` | Male | English | Sunny American voice |
| `Vivian` | Female | Chinese | Bright, slightly edgy |
| `Serena` | Female | Chinese | Warm, gentle |
| `Uncle_Fu` | Male | Chinese | Low, mellow timbre |
| `Dylan` | Male | Chinese (Beijing) | Natural, youthful |
| `Eric` | Male | Chinese (Sichuan) | Slightly husky |
| `Ono_Anna` | Female | Japanese | Playful, nimble |
| `Sohee` | Female | Korean | Warm, emotional |

---

## Key `synthesize` options

| Flag | Description |
|---|---|
| `TEXT ...` | One or more texts to synthesize |
| `--provider` | TTS provider (`edge`, `kokoro`, `bark`, `chatterbox`, `qwen`, `elevenlabs`) |
| `--voice`, `-v` | Voice / speaker name |
| `--language`, `-l` | Language code (`en`, `zh`, `fr`, ...) |
| `--output`, `-o` | Output file or directory |
| `--gpu [p100|t4]` | Run on free Kaggle GPU (Qwen) |
| `--model`, `-m` | HuggingFace model ID override |
| `--output-dir` | Local directory for Kaggle batch output |
| `--timeout` | Kaggle polling timeout in minutes (default 60) |
| `--poll-interval` | Kaggle polling interval in seconds (default 60) |
| `--seed` | Random seed for reproducible generation (default 42) |
| `--temperature` | Sampling temperature (lower = more neutral, higher = more expressive; default 0.7) |
| `--exaggeration` | Chatterbox emotion intensity 0–1 |
| `--audio-prompt` | Reference audio for voice cloning (Chatterbox) |

---

## Docs

- [Installation & optional dependencies](docs/installation.md)
- [Provider details & voice lists](docs/providers.md)
- [Kaggle GPU setup](docs/kaggle_gpu.md)
- [Voice cloning & design](docs/cloning_and_design.md)
- [Benchmarks & performance](docs/benchmarks_and_performance.md)
- [Licensing](docs/licensing.md)
