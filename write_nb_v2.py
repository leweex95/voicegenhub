import json
from pathlib import Path

nb = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "cell_1",
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, sys, subprocess, gc, traceback, torch\n",
                "from pathlib import Path\n",
                "import soundfile as sf\n",
                "import numpy as np\n",
                "from datetime import datetime\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "def log(msg):\n",
                "    ts = datetime.now().strftime('%H:%M:%S')\n",
                "    line = f'[{ts}] {msg}'\n",
                "    print(line, flush=True)\n",
                "    with open('kaggle_remote.log', 'a') as f:\n",
                "        f.write(line + '\\n')\n",
                "\n",
                "log('Checking environment...')\n",
                "try:\n",
                "    if subprocess.run('sox --version', shell=True, capture_output=True).returncode != 0:\n",
                "        log('Installing system sox and upgrading transformers...')\n",
                "        subprocess.run('apt-get update -q && apt-get install -y -q sox libsox-fmt-all', shell=True)\n",
                "        subprocess.run('pip install -q -U sox qwen-tts==0.1.1 \"transformers>=4.48.0\"', shell=True)\n",
                "    else:\n",
                "        log('Sox already present. Ensuring python packages...')\n",
                "        subprocess.run('pip install -q -U qwen-tts==0.1.1 \"transformers>=4.48.0\"', shell=True)\n",
                "    \n",
                "    from qwen_tts import Qwen3TTSModel\n",
                "    log('Setup complete (Imports done).')\n",
                "except Exception as e:\n",
                "    log(f'Setup warning: {e}. Attempting recovery full install...')\n",
                "    subprocess.run('pip install -q -U qwen-tts==0.1.1 sox \"transformers>=4.48.0\" --no-cache-dir', shell=True)\n",
                "    from qwen_tts import Qwen3TTSModel\n",
                "    log('Setup complete (Recovery Imports done).')\n",
                "\n",
                "# Parameters\n",
                "text = 'The GPU is faster. No more excuses. Successfully comparing CPU and GPU.'\n",
                "voice = 'Ryan'\n",
                "output_file = 'qwen.wav'\n",
                "\n",
                "log(f'Loading Qwen model...')\n",
                "try:\n",
                "    # T4 GPU works better with float32 to avoid NaNs in greedy decoding\n",
                "    model = Qwen3TTSModel.from_pretrained(\n",
                "        'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice', \n",
                "        torch_dtype=torch.float32, \n",
                "        device_map='auto'\n",
                "    )\n",
                "    \n",
                "    log(f'Synthesizing: {text[:50]}...')\n",
                "    # Synthesis block\n",
                "    with torch.inference_mode():\n",
                "        audio_array, sample_rate = model.generate_custom_voice(\n",
                "            text=text,\n",
                "            speaker=voice,\n",
                "            language='english',\n",
                "            do_sample=False,\n",
                "            subtalker_dosample=False\n",
                "        )\n",
                "    \n",
                "    if audio_array is not None and len(audio_array) > 0:\n",
                "        sf.write(output_file, audio_array, int(sample_rate))\n",
                "        log(f'SUCCESS: Audio saved to {output_file}')\n",
                "    else:\n",
                "        log('ERROR: Empty result')\n",
                "except Exception as e:\n",
                "    log(f'ERROR: {str(e)}')\n",
                "    traceback.print_exc()\n",
                "\n",
                "del model\n",
                "torch.cuda.empty_cache()\n",
                "gc.collect()\n",
                "log('Cleanup done.')\n"
            ]
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

target = Path("src/voicegenhub/kaggle/notebooks/kaggle-qwen.ipynb")
with open(target, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Successfully wrote clean notebook to {target}")
