# Kaggle Remote GPU Generation

Generate high-quality Qwen3-TTS audio using remote Kaggle GPUs (P100 or T4x2). This is useful for high-quality 1.7B models when you don't have a local GPU.

## Prerequisites

1.  **Kaggle API Credentials**:
    -   Go to [Kaggle Settings](https://www.kaggle.com/settings) → API → Create New Token.
    -   Save the `kaggle.json` to `~/.kaggle/kaggle.json` (on Windows: `%USERPROFILE%\.kaggle\kaggle.json`).
2.  **Kaggle CLI**:
    ```bash
    pip install kaggle
    ```
3.  **Kaggle Internet Access**:
    -   Ensure your Kaggle account has phone verification completed (allows internet access in kernels).

## Usage

Use the `--gpu` flag with the `synthesize` command to trigger remote generation.

### P100 GPU (default)

```bash
voicegenhub synthesize "Hello from the remote P100!" --gpu
```

### T4 x 2 GPU

```bash
voicegenhub synthesize "Hello from the remote T4!" --gpu --gpu-type t4
```

### Advanced Usage

```bash
voicegenhub synthesize "Chinese test." \
    --gpu \
    --gpu-type p100 \
    --voice Serena \
    --language zh \
    --output ./remote_output/serena.wav
```

## How It Works

1.  **Automation**: VoiceGenHub generates a Jupyter notebook cell-by-cell.
2.  **Deployment**: It pushes the notebook to Kaggle using the specified accelerator (`nvidia-p100-1` or `nvidia-t4-2`).
3.  **Execution**: On Kaggle, the notebook installs necessary dependencies (`transformers`, `qwen-tts`), loads the model onto the GPU, and generates the audio.
4.  **Syncing**: The CLI polls for completion and automatically downloads the generated `.wav` file into a local timestamped directory (or your specified output path).

---
*Note: Remote generation takes approximately 2-4 minutes due to environment setup on Kaggle's side.*
