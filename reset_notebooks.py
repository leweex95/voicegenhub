import json
import os

def create_clean_notebook(path, code):
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(True)
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Created clean notebook at {path}")

qwen_code = r"""import os, sys, subprocess, gc, traceback, torch
from pathlib import Path
import soundfile as sf
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# 1. Setup
log("Installing dependencies...")
# Qwen needs newer transformers, so we let it use the default Kaggle numpy 2.x which torch 2.6+ also needs
subprocess.run("pip install -q --no-cache-dir qwen-tts==0.1.1 soundfile", shell=True)
from qwen_tts import Qwen3TTSModel
log("Setup complete.")

# 2. Parameters (will be updated by JobManager)
text = "Default text"
voice = "Ryan"
output_file = "output.wav"

# 3. Generation
log(f"Loading model and generating: {text[:50]}...")
try:
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    output = model.generate_custom_voice(
        text=text,
        speaker=voice,
        do_sample=False,
        subtalker_dosample=False,
    )
    
    sr = 24000
    if isinstance(output, (tuple, list)):
        audio_data = output[0]
        sr = output[1] if len(output) > 1 else 24000
    else:
        audio_data = output

    # Handle list of chunks (common in Qwen)
    if isinstance(audio_data, list):
        log(f"Concatenating {len(audio_data)} audio chunks...")
        if len(audio_data) > 0:
            if isinstance(audio_data[0], torch.Tensor):
                audio_data = torch.cat(audio_data, dim=-1)
            else:
                audio_data = np.concatenate(audio_data)
        else:
            audio_data = None

    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.detach().cpu().float().numpy()
    
    if audio_data is not None:
        if hasattr(audio_data, 'ndim') and audio_data.ndim > 1:
            audio_data = audio_data.squeeze()
        
        sf.write(output_file, audio_data, int(sr))
        log(f"\u2713 Audio saved to: {output_file}")
    else:
        log("ERROR: No audio data generated")
        sys.exit(1)

except Exception as e:
    log(f"CRITICAL ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

# 4. Cleanup
finally:
    if 'model' in locals():
        del model
    torch.cuda.empty_cache()
    gc.collect()
    log("Job complete.")
"""

chatterbox_code = r"""import os, sys, subprocess, gc, traceback, torch
from pathlib import Path
import soundfile as sf
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# 1. Setup
log("Installing dependencies (this takes 1-2 mins)...")
subprocess.run("apt-get update && apt-get install -y espeak-ng libsndfile1 > /dev/null 2>&1", shell=True)
log("Downgrading to numpy < 2 and torch 2.4.1...")
subprocess.run("pip install --no-cache-dir --only-binary=:all: 'numpy==1.26.4' 'torch==2.4.1' 'torchaudio==2.4.1' 'torchvision==0.19.1' 'transformers==4.44.2' soundfile num2words phonemizer", shell=True)
log("Installing Chatterbox...")
subprocess.run("pip install --no-cache-dir --prefer-binary chatterbox-tts misaki[en]", shell=True)
from chatterbox import ChatterboxTTS
log("Setup complete.")

# 2. Parameters (will be updated by JobManager)
text = "Default text"
voice = "af_sky"
output_file = "output.wav"

# 3. Generation
log(f"Loading model and generating: {text[:50]}...")
try:
    model = ChatterboxTTS.from_pretrained(
        "hexgrad/Kokoro-82M", 
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    output = model.synthesize(
        text=text,
        voice=voice,
        speed=1.0
    )
    
    sr = 24000
    if hasattr(output, 'audio'):
        audio_data = output.audio
        sr = getattr(output, 'samples_per_second', 24000)
    elif isinstance(output, (tuple, list)):
        audio_data = output[0]
        if len(output) > 1: sr = output[1]
    else:
        audio_data = output

    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.detach().cpu().numpy()
    if audio_data is not None and audio_data.ndim > 1:
        audio_data = audio_data.squeeze()

    if audio_data is not None:
        sf.write(output_file, audio_data, int(sr))
        log(f"\u2713 Audio saved to: {output_file}")
    else:
        log("ERROR: No audio data generated")
        sys.exit(1)

except Exception as e:
    log(f"CRITICAL ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

# 4. Cleanup
finally:
    if 'model' in locals():
        del model
    torch.cuda.empty_cache()
    gc.collect()
    log("Job complete.")
"""

create_clean_notebook('src/voicegenhub/kaggle/notebooks/kaggle-qwen.ipynb', qwen_code)
create_clean_notebook('src/voicegenhub/kaggle/notebooks/kaggle-chatterbox.ipynb', chatterbox_code)
