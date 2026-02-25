import json
import os
from pathlib import Path

def write_notebook(path, code):
    nb = {
        'cells': [{
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [line for line in code.splitlines(True)]
        }],
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'codemirror_mode': {
                    'name': 'ipython',
                    'version': 3
                },
                'file_extension': '.py',
                'mimetype': 'text/x-python',
                'name': 'python',
                'nbconvert_exporter': 'python',
                'pygments_lexer': 'ipython3',
                'version': '3.10.12'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Restored {path}")

# --- Qwen Code ---
qwen_code = """import os, sys, subprocess, gc, traceback, torch
from pathlib import Path
import soundfile as sf
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# 1. Setup - Targeted FAST installation
log("Checking environment...")
try:
    # Check if qwen-tts is already installed
    try:
        from qwen_tts import Qwen3TTSModel
        log("Qwen-TTS already present.")
    except ImportError:
        log("Installing qwen-tts (minimal)...")
        # Install without deps to avoid re-downloading core libs like torch
        subprocess.run("pip install -q qwen-tts==0.1.1 --no-deps", shell=True)

    from qwen_tts import Qwen3TTSModel
    log("Setup complete (Imports done).")
except Exception as e:
    log(f"Setup warning: {e}. Trying full install...")
    subprocess.run("pip install -q qwen-tts==0.1.1 --no-cache-dir", shell=True)
    from qwen_tts import Qwen3TTSModel

# 2. Parameters (Injected by JobManager)
text = "Default text"
voice = "Ryan"
output_file = "output.wav"

# 3. Model Loading & Verification
log(f"Loading Qwen model. Text: {text[:50]}...")
try:
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" 
    )

    log("Generating audio...")
    output = model.generate_custom_voice(
        text=text,
        speaker=voice,
        do_sample=False,
    )
    
    sr = 24000
    if isinstance(output, (tuple, list)):
        audio_data = output[0]
        sr = output[1] if len(output) > 1 else 24000
    else:
        audio_data = output

    if isinstance(audio_data, list):
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
        log(f"SUCCESS: Audio saved to {output_file}")
    else:
        log("ERROR: No audio data")
        sys.exit(1)

except Exception as e:
    log(f"CRITICAL ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
finally:
    if 'model' in locals(): del model
    torch.cuda.empty_cache()
    gc.collect()
    log("Job finished.")"""

# --- Chatterbox Code ---
chatterbox_code = """import os, sys, subprocess, gc, traceback, torch
from pathlib import Path
import numpy as np
import soundfile as sf
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# 1. Setup - Targeted FAST installation
log("Checking environment...")
try:
    # Check for espeak-ng
    if subprocess.run("espeak-ng --version", shell=True, capture_output=True).returncode != 0:
        log("Installing espeak-ng...")
        subprocess.run("apt-get update -q; apt-get install -y -q espeak-ng", shell=True)
    
    # Fast minimal installs
    log("Installing chatterbox missing deps...")
    # Skip huge ones (torch, transformers are already on Kaggle)
    # phonemizer, num2words are small
    subprocess.run("pip install -q chatterbox-tts phonemizer num2words --no-deps", shell=True)

    from chatterbox import Chatterbox
    log("Setup complete (Imports done).")
except Exception as e:
    log(f"Setup warning: {e}. Trying full install...")
    subprocess.run("pip install -q chatterbox-tts phonemizer num2words --no-cache-dir", shell=True)
    from chatterbox import Chatterbox

# 2. Parameters (Injected by JobManager)
text = "Default text"
voice = "af_sky"
output_file = "output.wav"

# 3. Model Loading & Verification
log(f"Loading Chatterbox model. Text: {text[:50]}...")
try:
    # Model is Kokoro-82M (very fast, <100MB)
    model = Chatterbox()
    
    log("Generating audio...")
    audio, sr = model.generate(
        text=text,
        voice=voice,
        speed=1.0,
        lang_code="a" # American English
    )
    
    if audio is not None:
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().float().numpy()
        
        if hasattr(audio, 'ndim') and audio.ndim > 1:
            audio = audio.squeeze()
        
        sf.write(output_file, audio, int(sr))
        log(f"SUCCESS: Audio saved to {output_file}")
    else:
        log("ERROR: No audio data")
        sys.exit(1)

except Exception as e:
    log(f"CRITICAL ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
finally:
    if 'model' in locals(): del model
    torch.cuda.empty_cache()
    gc.collect()
    log("Job finished.")"""

workspace_root = r"c:\Users\csibi\Desktop\voicegenhub"
write_notebook(os.path.join(workspace_root, 'src', 'voicegenhub', 'kaggle', 'notebooks', 'kaggle-qwen.ipynb'), qwen_code)
write_notebook(os.path.join(workspace_root, 'src', 'voicegenhub', 'kaggle', 'notebooks', 'kaggle-chatterbox.ipynb'), chatterbox_code)
