import asyncio
import traceback
from pathlib import Path

try:
    from voicegenhub.core.engine import VoiceGenHub
    from voicegenhub.providers.base import AudioFormat

    engine = VoiceGenHub(provider='chatterbox')
    asyncio.run(engine.initialize())
    print('Initialized, now generating...')
    resp = asyncio.run(engine.generate('This is a quick test of Chatterbox audio output.', voice='chatterbox-default', audio_format=AudioFormat('wav')))
    out = Path('test_chatterbox_output.wav')
    resp.save(out)
    print(f'Saved output to: {out.resolve()} ({out.stat().st_size} bytes)')
except Exception:
    traceback.print_exc()
