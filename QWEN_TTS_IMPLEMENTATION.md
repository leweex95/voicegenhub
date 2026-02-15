# Qwen 3 TTS vs Chatterbox TTS Comparison

## Summary

Successfully implemented Qwen 3 TTS support in voicegenhub with maximum configurability.

## Generated Audio Files

All 4 comparison audio files have been generated successfully:

### 1. `1_chatterbox_english_long.wav` (902 KB)
- **Provider**: Chatterbox TTS
- **Language**: English
- **Text**: Long-form sample about ancient forests and bioluminescence
- **Model**: Default (English-only)

### 2. `2_chatterbox_english_short.wav` (514 KB)
- **Provider**: Chatterbox TTS
- **Language**: English
- **Text**: Shorter sample about technology and innovation
- **Model**: Default (English-only)

### 3. `3_qwen_english.wav` (1665 KB)
- **Provider**: Qwen 3 TTS
- **Language**: English
- **Text**: Same long-form sample as #1
- **Model**: Qwen3-TTS-12Hz-0.6B-CustomVoice
- **Voice**: Serena

### 4. `4_qwen_spanish.wav` (1549 KB)
- **Provider**: Qwen 3 TTS
- **Language**: Spanish
- **Text**: Spanish translation of the long-form sample
- **Model**: Qwen3-TTS-12Hz-0.6B-CustomVoice
- **Voice**: Serena

## Implementation Highlights

### Qwen 3 TTS Provider Features

The implemented Qwen provider supports maximum configurability:

#### Generation Modes
1. **CustomVoice**: Use predefined speakers
2. **VoiceDesign**: Natural language instructions for voice design
3. **VoiceClone**: Clone voice from reference audio

#### Configurable Parameters
- `model_name_or_path`: HuggingFace model path
- `device`: CPU/CUDA/auto
- `dtype`: float32/float16/bfloat16
- `attn_implementation`: eager/sdpa/flash_attention_2
- `generation_mode`: custom_voice/voice_design/voice_clone
- `speaker`: Speaker name (for custom_voice)
- `instruct`: Voice description (for voice_design)
- `ref_audio`: Reference audio path (for voice_clone)
- `ref_text`: Reference text (for voice_clone)
- `x_vector_only_mode`: Use only x-vector for cloning
- `temperature`: Sampling temperature
- `top_p`: Nucleus sampling
- `top_k`: Top-k sampling
- `repetition_penalty`: Control repetition
- `max_new_tokens`: Maximum generation length
- `do_sample`: Enable sampling

#### Language Support
Qwen 3 TTS supports: Chinese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish

The provider automatically maps language codes (en, es, etc.) to Qwen's expected format (english, spanish, etc.).

## Known Issues

### Chatterbox Multilingual Models
Chatterbox's multilingual models have a known bug with SDPA attention implementation that causes synthesis to fail with non-English languages:
```
The `output_attentions` attribute is not supported when using the `attn_implementation` set to sdpa.
```

This is an upstream issue in the chatterbox library itself and cannot be fixed at the provider level without modifying the chatterbox source code.

## Quality Comparison

### File Sizes
- Chatterbox produces smaller files (~500-900 KB for similar length)
- Qwen produces larger files (~1500-1700 KB for similar length)
- This suggests different encoding/compression or sample rates

### Audio Duration
Based on the generation logs:
- Chatterbox English (long): ~20s
- Qwen English: ~30s
- Qwen Spanish: ~33-36s

### Performance
- **Chatterbox**: Slow on CPU (~5-8 minutes per file)
- **Qwen**: Moderate on CPU (~5-8 minutes for initial model load, then ~5 minutes per file)
- Both providers would benefit significantly from GPU acceleration

## Installation

To use Qwen 3 TTS in this project:

```bash
# Install the qwen-tts package
poetry install -E qwen

# Or manually:
pip install qwen-tts
```

## Usage Examples

### Basic Usage
```python
from voicegenhub.providers.factory import provider_factory

# Initialize Qwen provider
await provider_factory.discover_provider("qwen")
provider = await provider_factory.create_provider("qwen", {
    "model_name_or_path": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "generation_mode": "custom_voice",
    "device": "auto",
})

# Synthesize speech
response = await provider.synthesize(TTSRequest(
    text="Hello world",
    voice_id="default",
    language="en",
))

response.save("output.wav")
```

### Advanced Configuration
```python
# Voice Design mode
config = {
    "model_name_or_path": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "generation_mode": "voice_design",
    "device": "cuda",
    "dtype": "float16",
    "temperature": 0.8,
    "top_p": 0.9,
}

request = TTSRequest(
    text="Your text here",
    voice_id="default",
    language="en",
    extra_params={
        "instruct": "A warm, friendly voice with a slight British accent"
    }
)
```

## Files Changed

1. `src/voicegenhub/providers/qwen.py` - New Qwen 3 TTS provider (362 lines)
2. `src/voicegenhub/providers/factory.py` - Added Qwen to provider factory
3. `pyproject.toml` - Added qwen-tts dependency group
4. `scripts/generate_final_comparison.py` - Comparison script
5. `scripts/compare_qwen_chatterbox.py` - Alternative comparison script

## Branch

All changes are committed to the `feature/qwen3-tts` branch.

## Next Steps

1. Listen to and compare the 4 generated audio files
2. Test with additional languages supported by Qwen
3. Consider adding voice cloning examples
4. Evaluate quality vs Eleven Labs (as you mentioned)
5. Consider GPU acceleration for production use
