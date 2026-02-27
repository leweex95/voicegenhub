[![Unit tests](https://github.com/leweex95/voicegenhub/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/leweex95/voicegenhub/actions/workflows/unit-tests.yml)
[![Daily regression test](https://github.com/leweex95/voicegenhub/actions/workflows/daily-regression-test.yml/badge.svg)](https://github.com/leweex95/voicegenhub/actions/workflows/daily-regression-test.yml)
[![codecov](https://codecov.io/gh/leweex95/voicegenhub/branch/master/graph/badge.svg)](https://codecov.io/gh/leweex95/voicegenhub)

# VoiceGenHub

Simple, user-friendly Text-to-Speech (TTS) library with CLI and Python API. Supports multiple free and commercial TTS providers.

### Optional Dependencies

- **Microsoft Edge TTS** (free, cloud-based)
- **Kokoro TTS** (Apache 2.0 licensed, self-hosted lightweight TTS)
- **Bark TTS** (MIT licensed, self-hosted high-naturalness TTS with prosody control)
- **Chatterbox TTS** (MIT licensed, multilingual with emotion control) - Works on CPU or GPU
- **Qwen 3 TTS** (Apache 2.0 licensed, multilingual with voice design and cloning) - State-of-the-art quality
- **ElevenLabs TTS** (commercial, high-quality voices)

### Voice Cloning Support

For voice cloning features with Chatterbox TTS:

```bash
pip install voicegenhub[voice-cloning]
# or
poetry install -E voice-cloning
```

**Voice cloning requirements:**
- FFmpeg (manual installation required)
- PyTorch (standard version)

**On Windows:** Download the "full-shared" FFmpeg build from [ffmpeg.org](https://ffmpeg.org/download.html#build-windows) and add the `bin` directory to your system PATH.

**Note:** VoiceGenHub includes a compatibility layer to ensure stable execution on CPU-only systems and prevents common import-time crashes related to experimental dependencies like TorchCodec. Standard TTS and voice cloning mechanisms will automatically fall back to supported audio loaders if needed.

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

**Chatterbox Installation Requirements:**
- **TorchCodec** (optional): Required for voice cloning features. Install with `pip install torchcodec` or `poetry install -E voice-cloning`.
- **FFmpeg**: Required when TorchCodec is installed for voice cloning. On Windows, install the "full-shared" build from [ffmpeg.org](https://ffmpeg.org/download.html#build-windows) and ensure FFmpeg's `bin` directory is in your system PATH.
- **PyTorch Compatibility**: TorchCodec 0.9.1 requires PyTorch ≤ 2.4.x. If you have a newer PyTorch version, voice cloning will be automatically disabled with a fallback to standard TTS.
- Without TorchCodec/FFmpeg, basic TTS will work but voice cloning (`--audio-prompt`) will gracefully fall back to standard TTS without cloning.

### Qwen 3 TTS

```bash
poetry run voicegenhub synthesize "Hello, world!" --provider qwen --voice Ryan --output hello.wav
```

**Qwen 3 TTS features:**
- **Three generation modes**: CustomVoice (predefined speakers), VoiceDesign (natural language voice description), VoiceClone (reference audio-based)
- **10 languages**: Chinese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish
- **Native speakers**: Automatic selection of native speakers per language for natural, accent-free speech
- **Voice control via natural language**: Use `instruct` parameter to control emotion, tone, speaking rate, and style
- **Ultra-low latency**: Streaming generation with <100ms first-token latency
- **Apache 2.0 License**: Fully commercial compatible
- **State-of-the-art quality**: Competitive with ElevenLabs, developed by Alibaba's Qwen team

#### Mode 1: CustomVoice (Predefined Speakers)

Use predefined premium speakers with optional emotion/style control:

```bash
# Basic usage with auto-selected native speaker
poetry run voicegenhub synthesize "Hello, this is a test." --provider qwen --language en --output output.wav

# Explicit speaker selection
poetry run voicegenhub synthesize "Hello, this is a test." --provider qwen --language en --voice Ryan --output output.wav

# With emotion instruction
poetry run voicegenhub synthesize "I'm so excited about this news!" --provider qwen --language en --voice Ryan --instruct "Speak with excitement and joy" --output happy.wav
```

**Available speakers and their native languages:**

| Speaker | Description | Native Language | Best For |
|---------|-------------|----------------|----------|
| **Ryan** | Dynamic male voice with strong rhythmic drive | English | English content, presentations |
| **Aiden** | Sunny American male voice with clear midrange | English | English content, narration |
| **Vivian** | Bright, slightly edgy young female voice | Chinese | Mandarin content, audiobooks |
| **Serena** | Warm, gentle young female voice | Chinese | Mandarin content, customer service |
| **Uncle_Fu** | Seasoned male voice with low, mellow timbre | Chinese | Mandarin narration, mature content |
| **Dylan** | Youthful Beijing male voice, natural timbre | Chinese (Beijing) | Beijing dialect content |
| **Eric** | Lively Chengdu male voice, slightly husky | Chinese (Sichuan) | Sichuan dialect content |
| **Ono_Anna** | Playful Japanese female, light and nimble | Japanese | Japanese content, anime |
| **Sohee** | Warm Korean female with rich emotion | Korean | Korean content, storytelling |

**Auto-speaker selection:** If no speaker is specified, Qwen 3 TTS automatically selects a native speaker based on the target language (e.g., Ryan for English, Serena for Chinese).

**Emotion and style control:** Use the `--instruct` parameter with natural language to control voice characteristics:
- `"Speak with excitement and joy"`
- `"Very angry tone"`
- `"Whisper gently"`
- `"Speak slowly and calmly"`
- `"Energetic and enthusiastic"`

#### Mode 2: VoiceDesign (Natural Language Voice Description)

Design custom voices using natural language instructions (requires `Qwen3-TTS-VoiceDesign` model):

```python
from voicegenhub.providers.factory import provider_factory
from voicegenhub.providers.base import TTSRequest

config = {
    "model_name_or_path": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "generation_mode": "voice_design",
}

await provider_factory.discover_provider("qwen")
provider = await provider_factory.create_provider("qwen", config=config)

request = TTSRequest(
    text="Welcome to our demonstration.",
    language="en",
    voice_id="default",
    extra_params={
        "instruct": "Male, 30 years old, confident and professional tone, deep voice with clear articulation"
    }
)
response = await provider.synthesize(request)
```

**VoiceDesign instruction examples:**
- `"Female, 25 years old, cheerful and energetic, slightly high-pitched with playful intonation"`
- `"Male, 17 years old, gaining confidence, deeper breath support, vowels tighten when nervous"`
- `"Elderly male, 70 years old, wise and gentle, slightly raspy with warm timbre"`

#### Mode 3: VoiceClone (Reference Audio-Based)

Clone voices from 3-second audio samples (requires `Qwen3-TTS-Base` model):

```python
from voicegenhub.providers.factory import provider_factory
from voicegenhub.providers.base import TTSRequest

config = {
    "model_name_or_path": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "generation_mode": "voice_clone",
}

await provider_factory.discover_provider("qwen")
provider = await provider_factory.create_provider("qwen", config=config)

request = TTSRequest(
    text="This is synthesized using the cloned voice.",
    language="en",
    voice_id="default",
    extra_params={
        "ref_audio": "path/to/reference.wav",  # Can be local path, URL, or numpy array
        "ref_text": "Transcript of the reference audio",  # Required for best quality
        "x_vector_only_mode": False  # Set True to skip ref_text (lower quality)
    }
)
response = await provider.synthesize(request)
```

**Voice cloning tips:**
- Use clear, noise-free reference audio (3-10 seconds)
- Provide accurate transcript (`ref_text`) for best cloning quality
- Supports multilingual cloning (clone any language, synthesize in any language)
- Combine with VoiceDesign to create reusable custom voices

#### Word Emphasis and Pause Control

**Note:** Qwen 3 TTS does not support explicit word-level emphasis markup (like SSML tags) or pause control. Instead, the model intelligently interprets text and applies natural prosody based on:

1. **Context understanding**: The model reads the entire sentence and applies appropriate emphasis to important words automatically
2. **Natural language instructions**: Use the `instruct` parameter to guide overall tone and pacing:
   - `"Speak slowly with emphasis on key words"`
   - `"Pause dramatically between sentences"`
   - `"Fast-paced and energetic delivery"`
3. **Punctuation**: The model respects punctuation for natural pauses (commas, periods, ellipses, em-dashes)

**Example:**
```bash
# The model will naturally emphasize "incredible results" due to context
poetry run voicegenhub synthesize "We achieved incredible results!" --provider qwen --voice Ryan --instruct "Speak with excitement and emphasis" --output emphasized.wav
```

#### Model Selection

Qwen 3 TTS offers multiple models optimized for different use cases:

| Model | Size | Best For | Streaming | GPU Recommended |Supports |
|-------|------|----------|-----------|-----------------|---------|
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 600M | Default, fast generation, predefined speakers | ✅ | Optional | CustomVoice |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 1.7B | Higher quality, predefined speakers | ✅ | Yes | CustomVoice |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 1.7B | Custom voice design via natural language | ✅ | Yes | VoiceDesign |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 1.7B | Voice cloning from audio samples | ✅ | Yes | VoiceClone |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 600M | Voice cloning, faster generation | ✅ | Optional | VoiceClone |

**Installation:**
```bash
pip install voicegenhub[qwen]
# or
poetry install --with qwen
```

**Qwen 3 TTS parameters (Python API):**
- `model_name_or_path`: Model to use (see table above)
- `device`: "cuda", "cpu", or "auto" (default: auto)
- `dtype`: "float32", "float16", "bfloat16" (default: bfloat16)
- `attn_implementation`: "eager", "sdpa", "flash_attention_2" (default: eager)
- `generation_mode`: "custom_voice", "voice_design", "voice_clone"
- `speaker`: Speaker name for CustomVoice mode
- `instruct`: Emotion/style instruction (for CustomVoice) or voice description (for VoiceDesign)
- `temperature`, `top_p`, `top_k`, `repetition_penalty`, `max_new_tokens`: Advanced sampling parameters

### Qwen3-TTS on Kaggle P100 GPU

Run the full Qwen3-TTS pipeline on a **free Kaggle P100 GPU**. VoiceGenHub automatically pushes a notebook to Kaggle, runs it with GPU acceleration, polls for completion, and downloads the audio to a local timestamped folder — no Kaggle web UI interaction required.

#### Prerequisites

1. **Install the Kaggle CLI:**
   ```bash
   pip install kaggle
   ```

2. **Set up Kaggle API credentials** (`~/.kaggle/kaggle.json`):
   - Go to https://www.kaggle.com/settings → API → Create New Token
   - Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
   - On Windows: `%USERPROFILE%\.kaggle\kaggle.json`

3. **Enable internet on Kaggle notebooks** (required for `pip install`):
   - Kaggle by default allows internet access from notebooks (no action needed).

#### Usage


```bash
# Basic usage — outputs to a timestamped folder (YYYYMMDD_HHMMSS_p100)
poetry run voicegenhub synthesize "Hello from the Kaggle GPU!" --provider qwen --gpu p100

# To use dual T4 GPUs, use --gpu t4. To force CPU, use --cpu (or omit both flags for default CPU mode).

# Specify voice and language
poetry run voicegenhub synthesize "This is a test." --provider qwen --voice Ryan --language en --gpu p100

# Chinese with native speaker
poetry run voicegenhub synthesize "你好，这是一个测试。" --provider qwen --voice Serena --language zh --gpu p100

# Explicit output directory and filename
poetry run voicegenhub synthesize "Big model test." \
   --provider qwen \
   --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
   --output-dir 20260225_153045 \
   --output-filename my_audio.wav \
   --gpu p100

# Adjust polling timeout (default 60 min)
poetry run voicegenhub synthesize "Long text..." --provider qwen --gpu p100 --timeout 90 --poll-interval 30
```


#### All `synthesize` flags for Kaggle GPU

| Flag | Default | Description |
|------|---------|-------------|
| `TEXT` | *(required)* | Text to synthesize |
| `--provider` | *(required)* | TTS provider: `qwen`, `chatterbox`, etc. |
| `--voice`, `-v` | `Ryan` | Speaker name: `Ryan`, `Serena`, etc. |
| `--language`, `-l` | `en` | Language code: `en`, `zh`, `fr`, etc. |
| `--model`, `-m` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | HuggingFace model ID |
| `--output-dir` | `YYYYMMDD_HHMMSS_<gpu>` (current datetime) | Local folder for the downloaded audio |
| `--output-filename` | `qwen3_tts.wav` | Filename for the generated audio |
| `--gpu [p100|t4]` | *(optional)* | Run remotely on Kaggle GPU (specify `p100` or `t4`) |
| `--cpu` | *(optional)* | Force CPU mode (default if neither flag is set) |
| `--timeout` | `60` | Timeout in minutes to wait for the kernel |
| `--poll-interval` | `60` | Status polling interval in seconds |

#### How it works

1. **Build** — VoiceGenHub generates a Jupyter notebook with your text, voice, and model parameters.
2. **Push** — The notebook is pushed to Kaggle with `enable_gpu: true` (P100).
3. **Run** — Kaggle executes the notebook: installs `qwen-tts`, loads the model on the GPU, generates audio.
4. **Poll** — VoiceGenHub polls `kaggle kernels status` every 60 seconds until completion.
5. **Download** — The `.wav` file is fetched with `kaggle kernels output` and placed in your local output directory.




**Note:** If you do not specify `--gpu` or `--cpu`, VoiceGenHub will run on CPU by default. For Qwen3-TTS and Chatterbox, running on CPU will print a **BIG VISIBLE WARNING** and may be extremely slow or fail. Use `--gpu p100` or `--gpu t4` for remote GPU. Use `--cpu` to force CPU mode explicitly.

**The output directory defaults to the current datetime plus GPU type** (e.g. `20260225_153045_p100/qwen3_tts.wav`).

---

## ⚠️ IMPORTANT: GPU Requirement for Qwen3/Chatterbox

**Qwen3-TTS and Chatterbox require a GPU for practical generation speed.**

- If you run these providers **without** `--gpu` (or on a CPU-only machine), you will see a **BIG WARNING** and generation will be extremely slow or may fail.
- Always use `--gpu` for Qwen3 and Chatterbox unless you are on a local machine with a powerful GPU.

**Example warning:**

```
WARNING: Qwen3-TTS and Chatterbox require a GPU for fast generation. Use --gpu (and optionally --gpu-type) to run on Kaggle or your local GPU. CPU-only runs are not recommended and may fail.
```

#### Available Qwen3-TTS Models on Kaggle GPU

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 600M | Fast | Quick iterations |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 1.7B | Normal | **Best quality** ✅ |

> **Tip:** The 1.7B model is recommended for production quality. A P100 has 16 GB VRAM — more than enough for the 1.7B model at float16.

---

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
- **Qwen 3 TTS** (Apache 2.0) - Commercial use allowed, attribution required
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
- **Qwen 3 TTS**: [Apache License 2.0](https://github.com/QwenLM/Qwen3-TTS/blob/main/LICENSE)

## Optional Dependencies

Install optional TTS providers:

```bash
# Install Kokoro TTS (self-hosted lightweight TTS)
pip install voicegenhub[kokoro]

# Install Bark (self-hosted high-naturalness TTS)
pip install voicegenhub[bark]

# Install Chatterbox TTS (MIT licensed, multilingual with emotion control)
pip install chatterbox-tts

# Install Qwen 3 TTS (Apache 2.0 licensed, state-of-the-art multilingual TTS)
pip install voicegenhub[qwen]
```

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
