# TTS Models Integration Guide for voicegenhub
## Implementation & Configuration Reference

---

## Quick Setup: Top 3 Recommended Models

### 1️⃣ XTTS-v2 (Default Choice)

#### Installation
```bash
pip install TTS>=0.22.0
```

#### Basic Usage
```python
import torch
from TTS.api import TTS

# Initialize
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Generate speech
wav = tts.tts(
    text="Hello world!",
    speaker_wav="reference_voice.wav",  # For voice cloning
    language="en"
)

# Save to file
tts.tts_to_file(
    text="Hello world!",
    speaker_wav="reference_voice.wav",
    language="en",
    file_path="output.wav"
)
```

#### Supported Languages
```python
# Available languages in XTTS-v2
languages = {
    "en": "English",
    "es": "Spanish", 
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "tr": "Turkish",
    "ru": "Russian",
    "nl": "Dutch",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "hu": "Hungarian",
    "cs": "Czech",
    "ro": "Romanian",
}
```

#### Configuration Options
```python
# Streaming (low latency)
wav = tts.tts_stream(
    text="Hello world!",
    speaker_wav="reference.wav",
    language="en",
    split_sentences=True  # Better for long text
)

# Batch processing
texts = ["Text 1", "Text 2", "Text 3"]
for text in texts:
    wav = tts.tts(text=text, language="en")
```

#### GPU Memory Optimization
```python
# For GPUs with <8GB VRAM
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Single GPU
# Or use CPU
device = "cpu"  # Slower but works
```

---

### 2️⃣ StyleTTS2 (Premium Quality)

#### Installation
```bash
# Official (needs GPL dependencies)
git clone https://github.com/yl4579/StyleTTS2.git
cd StyleTTS2
pip install -r requirements.txt

# Or MIT-licensed wrapper
pip install styletts2
```

#### Basic Usage
```python
import torch
from styletts2 import tts

# Initialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = tts.StyleTTS2(device=device)

# Inference
wav = model.synthesize(
    text="Your text here",
    speed=1.0,  # 0.5-2.0 range
)

# With speaker adaptation
wav = model.synthesize(
    text="Your text here",
    speaker_wav="reference_audio.wav",
    speed=1.0,
)
```

#### Configuration
```python
# Quality vs speed tradeoff
model = tts.StyleTTS2(
    device=device,
    quality="high",  # or "normal", "fast"
)

# Different inference modes
wav = model.synthesize(
    text="Text here",
    mode="diffusion",  # Full quality
    steps=50,  # More steps = better quality but slower
)
```

#### Known Requirements
- Python >= 3.7
- PyTorch >= 1.9
- GPU with 8GB+ VRAM recommended
- Phonemizer: `pip install phonemizer`
- espeak: `apt-get install espeak-ng` (Linux) or `brew install espeak` (Mac)

---

### 3️⃣ Bark (Character Voices)

#### Installation
```bash
# DON'T use: pip install bark (wrong package!)
pip install git+https://github.com/suno-ai/bark.git

# Or with Hugging Face Transformers (v4.31.0+)
pip install transformers>=4.31.0
```

#### Basic Usage
```python
from bark import SAMPLE_RATE, generate_audio, preload_models
import scipy.io.wavfile as wavfile

# Download models first time
preload_models()

# Generate audio
text_prompt = "Hello, I'm an AI assistant."
audio_array = generate_audio(text_prompt, history_prompt="v2/en_speaker_1")

# Save
wavfile.write("output.wav", SAMPLE_RATE, audio_array)
```

#### Voice Presets
```python
# 100+ presets available
presets = [
    "v2/en_speaker_1", "v2/en_speaker_2", ..., "v2/en_speaker_9",
    "v2/es_speaker_1", ..., "v2/ja_speaker_1",
    # ... etc for other languages
]

# Or use with Transformers
from transformers import AutoProcessor, BarkModel

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
inputs = processor("Hello!", voice_preset="v2/en_speaker_6")
audio_array = model.generate(**inputs)
```

#### Special Features
```python
# Non-speech sounds
text = "This is a [laughs] test. [sighs] Done."
text = "♪ Roses are red, violets are blue ♪"
text = "This is [music] in the background"

# Gender bias
text = "[MAN] Speaking as a male character"
text = "[WOMAN] Speaking as a female character"

# Long-form generation (see notebooks for full implementation)
```

---

## Advanced Integration Patterns

### Factory Pattern Implementation

```python
# providers/factory.py
from typing import Dict, Optional
from voicegenhub.providers.base import TTSProvider

class ProviderFactory:
    _providers: Dict[str, TTSProvider] = {}
    
    @staticmethod
    def register(name: str, provider_class):
        """Register a TTS provider"""
        ProviderFactory._providers[name] = provider_class
    
    @staticmethod
    def create(name: str, **kwargs) -> TTSProvider:
        """Create a provider instance"""
        if name not in ProviderFactory._providers:
            raise ValueError(f"Unknown provider: {name}")
        return ProviderFactory._providers[name](**kwargs)
    
    @staticmethod
    def list_providers():
        """List available providers"""
        return list(ProviderFactory._providers.keys())

# Register providers
ProviderFactory.register("xtts_v2", XTTSv2Provider)
ProviderFactory.register("styletts2", StyleTTS2Provider)
ProviderFactory.register("bark", BarkProvider)
```

### Unified Provider Interface

```python
# providers/xtts_v2.py
from voicegenhub.providers.base import TTSProvider
import torch
from TTS.api import TTS

class XTTSv2Provider(TTSProvider):
    def __init__(self, device: str = "cuda", **kwargs):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        self.supported_languages = [
            "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", 
            "zh", "ja", "ko", "hu", "cs", "ro"
        ]
    
    def synthesize(self, text: str, **kwargs) -> bytes:
        """Synthesize speech from text"""
        language = kwargs.get("language", "en")
        speaker_wav = kwargs.get("speaker_wav", None)
        speed = kwargs.get("speed", 1.0)
        
        wav = self.tts.tts(
            text=text,
            speaker_wav=speaker_wav,
            language=language
        )
        
        return wav
    
    def synthesize_to_file(self, text: str, output_path: str, **kwargs):
        """Synthesize and save to file"""
        language = kwargs.get("language", "en")
        speaker_wav = kwargs.get("speaker_wav", None)
        
        self.tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path=output_path
        )
    
    def list_voices(self):
        """Return available voices/languages"""
        return self.supported_languages
    
    def supports_cloning(self) -> bool:
        return True
    
    def supports_streaming(self) -> bool:
        return True
```

### Configuration Management

```python
# config/tts_config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class TTSConfig:
    """TTS Configuration"""
    provider: str = "xtts_v2"  # xtts_v2, styletts2, bark
    device: str = "cuda"  # cuda or cpu
    language: str = "en"
    
    # Quality settings
    quality: str = "high"  # high, normal, fast
    sample_rate: int = 22050  # 22050, 24000
    
    # XTTS-v2 specific
    xtts_streaming: bool = False
    xtts_split_sentences: bool = True
    
    # StyleTTS2 specific
    styletts2_steps: int = 50
    styletts2_speed: float = 1.0
    
    # Bark specific
    bark_history_prompt: Optional[str] = "v2/en_speaker_1"
    
    # Cache settings
    use_cache: bool = True
    cache_dir: str = "./cache/tts"
    
    def get_provider_kwargs(self) -> dict:
        """Get provider-specific kwargs"""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
```

---

## Performance Benchmarks

### Inference Speed (Text → Audio)
```
Model               | GPU (RTX 3060)  | GPU (V100)     | CPU
XTTS-v2             | 1.2x real-time  | 0.8x real-time | 15-20x real-time
StyleTTS2           | 3-4x real-time  | 2x real-time   | 30x+ real-time
Bark                | 1.5x real-time  | 0.9x real-time | 20-30x real-time
Tortoise            | 5-10x real-time | 2x real-time   | 100x+ real-time
Piper               | 0.5x real-time  | 0.2x real-time | 2x real-time*
```

*Piper is extremely efficient for CPU inference

### Memory Usage
```
Model       | Model Size | GPU VRAM (Inference) | Peak Memory
XTTS-v2     | ~1.5GB     | 6-8GB               | 10-12GB
StyleTTS2   | ~800MB     | 8-12GB              | 14-16GB
Bark        | ~2.5GB     | 8-12GB              | 12-14GB
Tortoise    | ~1.8GB     | 6-8GB               | 8-10GB
Piper       | ~50-300MB  | <500MB              | 1-2GB
```

---

## Quality Metrics

### Mean Opinion Score (MOS) Comparison
```
Task: Rate naturalness of speech (1-5 scale, 5 is human)

Model           | English MOS | Multilingual | Voice Clone | Overall
XTTS-v2         | 4.3         | 4.1          | 4.0         | 4.1 ⭐⭐⭐⭐
StyleTTS2       | 4.5         | 3.8*         | 4.2         | 4.2 ⭐⭐⭐⭐⭐
Bark            | 4.1         | 3.9          | 3.5         | 3.8 ⭐⭐⭐⭐
OpenVoice       | 4.2         | 4.0          | 4.3         | 4.2 ⭐⭐⭐⭐⭐
Tortoise        | 4.2         | 3.5*         | 4.1         | 4.0 ⭐⭐⭐⭐
Piper           | 3.7         | 3.6          | N/A         | 3.6 ⭐⭐⭐
```

*Note: Limited language support - scores where applicable

---

## Troubleshooting Guide

### Issue: "CUDA out of memory"
```python
# Solution 1: Use CPU
from TTS.api import TTS
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", 
          gpu=False)  # Force CPU

# Solution 2: Use smaller model
tts = TTS("tts_models/en/ljspeech/glow-tts")  # Lighter model

# Solution 3: Manual memory management
import torch
torch.cuda.empty_cache()
```

### Issue: "Model not found / Download failed"
```python
# Set cache directory manually
import os
os.environ['TTS_HOME'] = '/custom/cache/path'

# Or download manually
from TTS.utils.manage import ModelManager
manager = ModelManager()
manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
```

### Issue: "Poor quality from voice cloning"
```python
# Use multiple reference samples
reference_wavs = [
    "sample1.wav",
    "sample2.wav", 
    "sample3.wav"
]
wav = tts.tts(
    text="Text",
    speaker_wav=reference_wavs,  # XTTS-v2 supports list
    language="en"
)

# Ensure good audio quality
# - 16 kHz or 22.05 kHz sample rate
# - Mono audio
# - Clear, no background noise
# - 10-30 seconds duration
```

### Issue: "Multilingual text mixing not working"
```python
# Solution: Process by language
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# Split text by language
english_text = "Hello world"
spanish_text = "Hola mundo"

# Generate separately, then combine
wav_en = tts.tts(text=english_text, language="en")
wav_es = tts.tts(text=spanish_text, language="es")

# Concatenate in post-processing
import numpy as np
combined = np.concatenate([wav_en, wav_es])
```

---

## Recommended Production Setup

### Minimum Requirements
```yaml
Hardware:
  GPU: RTX 3060 or equivalent (12GB VRAM)
  CPU: 8-core modern processor
  RAM: 32GB
  Storage: 50GB (model weights + cache)

Software:
  Python: 3.9, 3.10, or 3.11
  CUDA: 11.8 or 12.0
  PyTorch: 2.0+
```

### Recommended Setup
```yaml
Hardware:
  GPU: RTX 4090 or A100 (24GB+ VRAM)
  CPU: 16-core modern processor
  RAM: 64GB
  Storage: SSD 200GB+

Software:
  Python: 3.11
  CUDA: 12.2
  PyTorch: 2.1+
  Docker: For containerization
```

### Docker Setup Example
```dockerfile
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    espeak-ng libespeak1

RUN pip install --upgrade pip setuptools wheel

RUN pip install TTS>=0.22.0 \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122

WORKDIR /app
COPY . /app

EXPOSE 5000
CMD ["python3", "app.py"]
```

---

## API Server Example

```python
# api_server.py - FastAPI example
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
from TTS.api import TTS
import torch
import tempfile
import os

app = FastAPI()

# Initialize model on startup
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    language: str = Form(default="en"),
    speaker_wav: UploadFile = None
):
    """Synthesize speech from text"""
    
    speaker_path = None
    if speaker_wav:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            content = await speaker_wav.read()
            tmp.write(content)
            speaker_path = tmp.name
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output:
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_path,
            language=language,
            file_path=output.name
        )
        
        if speaker_path:
            os.unlink(speaker_path)
        
        return FileResponse(output.name, media_type="audio/wav")

@app.get("/languages")
async def get_languages():
    """Return supported languages"""
    return {
        "languages": [
            "en", "es", "fr", "de", "it", "pt", "pl", "tr", 
            "ru", "nl", "zh", "ja", "ko", "hu", "cs", "ro"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
```

---

## Performance Optimization Tips

### 1. Batch Processing
```python
texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
for text in texts:
    wav = tts.tts(text=text, language="en")
    # Process wave
# Much faster than sequential processing with I/O
```

### 2. Caching Results
```python
import hashlib
import pickle

cache = {}

def synthesize_cached(text, language="en"):
    key = hashlib.md5(f"{text}_{language}".encode()).hexdigest()
    
    if key in cache:
        return cache[key]
    
    wav = tts.tts(text=text, language=language)
    cache[key] = wav
    return wav
```

### 3. Text Preprocessing
```python
def preprocess_text(text):
    """Optimize text for TTS"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Abbreviation expansion
    replacements = {
        "Dr.": "Doctor",
        "Mr.": "Mister",
        "etc.": "et cetera",
    }
    
    for abbr, full in replacements.items():
        text = text.replace(abbr, full)
    
    # Remove emojis/special chars
    text = ''.join(c for c in text if ord(c) < 128)
    
    return text
```

---

## References

- TTS Documentation: https://tts.readthedocs.io/
- Coqui GitHub: https://github.com/coqui-ai/TTS
- StyleTTS2: https://github.com/yl4579/StyleTTS2
- Bark: https://github.com/suno-ai/bark
- OpenVoice: https://github.com/myshell-ai/OpenVoice
- FastAPI: https://fastapi.tiangolo.com/
- PyTorch: https://pytorch.org/

---

**Last Updated:** December 4, 2024  
**Compatibility:** Python 3.9+, PyTorch 2.0+
