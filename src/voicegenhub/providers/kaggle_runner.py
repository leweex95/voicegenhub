"""Small self-contained script uploaded with a VoiceGenHub Kaggle job."""

import json
import subprocess
import sys
from pathlib import Path

# Kaggle's base image has CUDA-enabled PyTorch; install the same Chatterbox fork
# used by VoiceGenHub without replacing the platform's CUDA PyTorch build.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "git+https://github.com/rsxdalv/chatterbox.git@004a0214c308f544f4276e74ccf516c96c271f5b",
])

import torch
import soundfile as sf
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

request = json.loads(Path("request.json").read_text(encoding="utf-8"))
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
audio = model.generate(
    request["text"],
    language_id=(request.get("language") or "en").split("-")[0],
    exaggeration=request["exaggeration"],
    cfg_weight=request["cfg_weight"],
)
audio = audio.squeeze().detach().cpu().numpy()
sf.write("voicegenhub_output.wav", audio, model.sr)
print(f"Wrote WAV on {torch.cuda.get_device_name(0)}")
