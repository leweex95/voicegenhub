import json
from pathlib import Path

def write_nb(path, code):
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_1",
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code.splitlines()]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Restored {path}")

qwen_code = r"""import os, sys, subprocess, gc, traceback, torch
from pathlib import Path
import soundfile as sf
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Setup logging to file so kaggle-gpu-connector can fetch it in real-time
def log(msg): 
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open("kaggle_remote.log", "a") as f:
        f.write(line + "\n")

# 1. Setup - Targeted FAST installation
log("Checking environment...")
try:
    if subprocess.run("sox --version", shell=True, capture_output=True).returncode != 0:
        log("Installing system sox and upgrading transformers...")
        subprocess.run("apt-get update -q && apt-get install -y -q sox libsox-fmt-all", shell=True)
        subprocess.run("pip install -q -U sox qwen-tts==0.1.1 'transformers>=4.48.0'", shell=True)
    else:
        log("Sox already present. Ensuring python packages...")
        subprocess.run("pip install -q -U qwen-tts==0.1.1 'transformers>=4.48.0'", shell=True)
    
    from qwen_tts import Qwen3TTSModel
    log("Setup complete (Imports done).")
except Exception as e:
    log(f"Setup warning: {e}. Attempting recovery full install...")
    subprocess.run("pip install -q -U qwen-tts==0.1.1 sox 'transformers>=4.48.0' --no-cache-dir", shell=True)
    from qwen_tts import Qwen3TTSModel
    log("Setup complete (Recovery Imports done).")

# Parameters
text = "The GPU is faster. No more excuses. Successfully comparing CPU and GPU."
voice = "Ryan"
output_file = "qwen.wav"

# 3. Model Loading & Verification
log(f"Loading Qwen model. Text: {text[:50]}...")
try:
    # Use float32 for T4 stability (avoiding NaNs in multinomial)
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        torch_dtype=torch.float32,
        device_map="auto" 
    )
except Exception as e:
    log(f"Load failed: {e}. Retrying with float32 explicitly...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        torch_dtype=torch.float32,
        device_map="auto"
    )

# 4. Prompt Logic
log(f"Generating audio for voice: {voice}")
choices = {
    "Ryan": ("Ryan", "Ryan is a male voice..."),
}
ref_text, ref_prompt = choices.get(voice, ("Ryan", "Ryan voice"))

# 5. Synthesis block
try:
    log("Synthesizing...")
    # ENSURE BOTH do_sample and subtalker_dosample are False 
    # to avoid internal multinomial calls on T4 (prone to NaNs)
    with torch.inference_mode():
        audio_array, sample_rate = model.generate_custom_voice(
            text=text,
            speaker=ref_text,
            language="english",
            do_sample=False,
            subtalker_dosample=False
        )
    
    # Check if we got something
    if audio_array is None or len(audio_array) == 0:
        raise ValueError("Empty audio_array returned from model.generate")
        
    log(f"Synthesis done. Array length: {len(audio_array)}")
    
    # 6. Saving
    sf.write(output_file, audio_array, sample_rate)
    log(f"SUCCESS: Audio saved to {output_file}. Array shape: {audio_array.shape}")
except Exception as e:
    log(f"ERROR: Synthesis failed: {str(e)}")
    traceback.print_exc()

# 7. Cleanup
del model
torch.cuda.empty_cache()
gc.collect()
log("Job complete.")
"""

write_nb("src/voicegenhub/kaggle/notebooks/kaggle-qwen.ipynb", qwen_code)
