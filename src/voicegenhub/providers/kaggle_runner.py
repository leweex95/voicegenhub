"""Small self-contained script uploaded with a VoiceGenHub Kaggle job."""

import json
import subprocess
import sys

# Kaggle's base image has CUDA-enabled PyTorch; install the same Chatterbox fork
# used by VoiceGenHub without replacing the platform's CUDA PyTorch build.
print("[voicegenhub] Installing Chatterbox", flush=True)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "git+https://github.com/rsxdalv/chatterbox.git@004a0214c308f544f4276e74ccf516c96c271f5b",
])
print("[voicegenhub] Chatterbox install finished; importing runtime", flush=True)

import torch
import soundfile as sf
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

REQUEST_JSON = None  # Replaced with the per-request payload before Kaggle upload.
request = json.loads(REQUEST_JSON)
print(
    f"[voicegenhub] Runtime ready: torch={torch.__version__}, "
    f"cuda_build={torch.version.cuda}, cuda_available={torch.cuda.is_available()}, "
    f"gpu_count={torch.cuda.device_count()}",
    flush=True,
)
if not torch.cuda.is_available():
    raise RuntimeError("Kaggle runner cannot access CUDA; GPU audio generation is unavailable")

print("[voicegenhub] Loading Chatterbox model on CUDA", flush=True)
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
print("[voicegenhub] Model loaded; generating audio", flush=True)
audio = model.generate(
    request["text"],
    language_id=(request.get("language") or "en").split("-")[0],
    exaggeration=request["exaggeration"],
    cfg_weight=request["cfg_weight"],
)
audio = audio.squeeze().detach().cpu().numpy()
sf.write("voicegenhub_output.wav", audio, model.sr)
print(f"[voicegenhub] Wrote WAV on {torch.cuda.get_device_name(0)}", flush=True)
