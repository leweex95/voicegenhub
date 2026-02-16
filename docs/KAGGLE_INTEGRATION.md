# Kaggle GPU Integration

## Overview

This module enables VoiceGenHub to deploy voice generation tasks to Kaggle's free GPU instances. Heavy providers (qwen, chatterbox) can now run on Kaggle's T4 GPUs instead of local hardware.

## Architecture

```
voicegenhub/kaggle/
├── __init__.py              # Module exports
├── config_loader.py         # Configuration management
├── pipeline.py              # Main orchestration
├── core/
│   ├── deploy.py           # Kernel deployment
│   ├── poll_status.py      # Status monitoring
│   └── download.py         # Output retrieval
├── config/
│   └── kaggle_settings.json # Configuration
└── notebooks/
    ├── kernel-metadata.json # Kernel configuration
    ├── kaggle-qwen.ipynb   # Qwen TTS notebook
    └── kaggle-chatterbox.ipynb # Chatterbox TTS notebook
```

## Features

- **Automatic parameter injection**: Text, voice, and output settings are dynamically injected into Kaggle notebooks
- **Provider support**: qwen and chatterbox (heavy GPU providers)
- **Status polling**: Automatic monitoring until kernel completion
- **Selective download**: Only downloads audio files, skips build artifacts
- **CLI integration**: Simple `--gpu` flag to enable

## Usage

### Basic Usage

```bash
poetry run voicegenhub synthesize "Hello world" \
    --provider qwen \
    --voice Ryan \
    --gpu \
    --output generated.wav
```

### Chatterbox Example

```bash
poetry run voicegenhub synthesize "Testing Chatterbox on GPU" \
    --provider chatterbox \
    --gpu \
    --output chatterbox_gpu.wav
```

### Unsupported Providers

For providers without Kaggle support (edge, kokoro, elevenlabs, bark), the `--gpu` flag will show a warning and fall back to local CPU execution:

```bash
poetry run voicegenhub synthesize "test" --provider edge --gpu
# Warning: Kaggle GPU deployment is only supported for qwen, chatterbox providers.
# Provider 'edge' will run locally.
```

## Prerequisites

1. **Kaggle API credentials**: Place your `kaggle.json` in `~/.kaggle/`
2. **Kaggle kernel**: Create a kernel at https://www.kaggle.com/code
   - Title: "VoiceGenHub GPU"
   - Enable GPU accelerator
   - Make private
   - Note the kernel ID (format: `username/kernel-slug`)
3. **Update kernel ID**: Edit `src/voicegenhub/kaggle/config/kaggle_settings.json`:
   ```json
   {
     "kernel_id": "your-username/your-kernel-slug"
   }
   ```

## Configuration

### Kaggle Settings

File: `src/voicegenhub/kaggle/config/kaggle_settings.json`

```json
{
  "deployment_timeout_minutes": 30,
  "polling_interval_seconds": 60,
  "retry_interval_seconds": 60,
  "kernel_id": "leventecsibi/voicegenhub-gpu"
}
```

- `deployment_timeout_minutes`: Maximum wait time for GPU availability
- `polling_interval_seconds`: How often to check kernel status
- `retry_interval_seconds`: Retry interval on errors
- `kernel_id`: Your Kaggle kernel identifier

### Kernel Metadata

File: `src/voicegenhub/kaggle/notebooks/kernel-metadata.json`

```json
{
  "id": "leventecsibi/voicegenhub-gpu",
  "title": "VoiceGenHub GPU",
  "code_file": "kaggle-qwen.ipynb",
  "enable_gpu": "true",
  "enable_internet": "true"
}
```

## How It Works

### Pipeline Flow

1. **Deployment**:
   - Load appropriate notebook (qwen or chatterbox)
   - Inject text, voice, and output parameters
   - Update kernel metadata
   - Push to Kaggle via API

2. **Polling**:
   - Monitor kernel status every 60 seconds
   - Wait for status: `queued` → `running` → `complete`
   - Handle errors and timeouts

3. **Download**:
   - Start download via Kaggle CLI
   - Monitor local directory for audio files
   - Terminate download once audio is stable (prevents downloading build artifacts)

4. **Verification**:
   - Verify audio file exists
   - Return path for local access

### Notebooks

Each notebook follows this structure:

1. Environment setup (disable validation warnings)
2. Install provider dependencies
3. Configure parameters (TEXT, VOICE_ID, OUTPUT_FILE)
4. Load model
5. Generate audio
6. Save to file
7. Cleanup GPU memory

Parameters are dynamically replaced by the deploy module before push.

## Security

- **Credentials**: Kaggle credentials should never be committed
  - Added to `.gitignore`: `.env_from_imggenhub`, `kaggle.json`
  - Store in `~/.kaggle/kaggle.json` (standard location)

- **Private kernels**: All kernels should be private to protect API usage

## Limitations

1. **Network required**: Full pipeline requires internet connectivity
2. **Kaggle quotas**: Subject to Kaggle's GPU quota limits (30 hours/week free)
3. **Cold start**: First run downloads models (~1-5 minutes)
4. **Sequential processing**: Multiple texts processed one at a time
5. **Provider support**: Only qwen and chatterbox

## Troubleshooting

### Network Errors

```
Error: Please check your firewall rules and network connection
```

**Solution**: Ensure internet connectivity and Kaggle API access

### Kernel Not Found

```
Error: Kernel ID not found
```

**Solution**: Create kernel manually at https://www.kaggle.com/code and update `kernel_id` in config

### GPU Quota Exceeded

**Solution**: Wait for quota reset (weekly) or upgrade Kaggle account

### Parameter Injection Failed

**Solution**: Verify notebook structure matches expected cell format (see notebooks/)

## Testing

### Offline Validation

```python
# Test module imports and structure
poetry run python -c "
from voicegenhub.kaggle.config_loader import load_kaggle_config
from voicegenhub.kaggle.core.deploy import _update_param
from voicegenhub.kaggle.pipeline import run_kaggle_pipeline
print('All imports successful')
"
```

### Full Test (requires network)

```bash
poetry run voicegenhub synthesize "Integration test" \
    --provider qwen \
    --voice Ryan \
    --gpu \
    --output test_kaggle_output.wav
```

## Implementation Notes

### Methodology from imggenhub

This implementation follows the same methodology as `imggenhub`:

1. **Notebook templates**: Jupyter notebooks stored in repo
2. **Parameter injection**: Dynamic replacement via JSON manipulation
3. **Metadata management**: kernel-metadata.json for Kaggle API
4. **Selective download**: Monitor directory, terminate early when outputs found
5. **Poetry integration**: Use `poetry run python -m kaggle.cli` for consistency

### Differences from imggenhub

- **Audio focus**: Downloads `.wav` files instead of images
- **Simpler models**: No multi-model support (FLUX, SD, etc.)
- **No parallel deployment**: Single kernel approach (can be extended)
- **Provider-specific notebooks**: Separate notebooks per TTS provider

## Future Enhancements

1. **Parallel deployment**: Support multiple kernels for batch processing
2. **More providers**: Add support for other GPU-intensive providers
3. **Voice cloning**: Support audio prompt upload to Kaggle datasets
4. **Result caching**: Cache generated audio to avoid re-generation
5. **Cost tracking**: Monitor Kaggle GPU quota usage

## Credits

Based on the Kaggle integration methodology from imggenhub by leweex95.
