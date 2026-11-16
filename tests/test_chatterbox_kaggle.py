import asyncio
import ast
import io
import json
import wave

import pytest

from voicegenhub.providers.base import TTSRequest, TTSError
from voicegenhub.providers.chatterbox_kaggle import (
    KaggleChatterboxProvider,
    _embed_request_payload,
)


def test_kaggle_provider_returns_remote_audio_without_local_model(monkeypatch):
    provider = KaggleChatterboxProvider()
    provider._initialized = True

    def fake_run_remote(request):
        assert request.text == "hello"
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(24000)
            audio.writeframes(b"\0\0" * 2400)
        return buffer.getvalue(), 24000

    monkeypatch.setattr(provider, "_run_remote", fake_run_remote)
    response = asyncio.run(provider.synthesize(TTSRequest(text="hello", voice_id="chatterbox-en")))

    assert response.audio_data.startswith(b"RIFF")
    assert response.sample_rate == 24000
    assert response.duration == pytest.approx(0.1)
    assert response.metadata["execution"] == "kaggle"


def test_kaggle_provider_rejects_unsupported_language_before_submission(monkeypatch):
    provider = KaggleChatterboxProvider()
    provider._initialized = True
    monkeypatch.setattr(provider, "_run_remote", lambda request: pytest.fail("must not submit"))

    with pytest.raises(TTSError, match="does not support language"):
        asyncio.run(provider.synthesize(
            TTSRequest(text="hello", voice_id="chatterbox-hu", language="hu")
        ))


def test_kaggle_provider_rejects_unimplemented_voice_cloning(monkeypatch):
    provider = KaggleChatterboxProvider()
    provider._initialized = True
    monkeypatch.setattr(provider, "_run_remote", lambda request: pytest.fail("must not submit"))

    with pytest.raises(TTSError, match="not yet supported on Kaggle"):
        asyncio.run(provider.synthesize(
            TTSRequest(
                text="hello", voice_id="chatterbox-en",
                extra_params={"audio_prompt_path": "voice.wav"},
            )
        ))


def test_kaggle_runner_embeds_request_payload_for_upload():
    runner_code = 'import json\nREQUEST_JSON = None\nrequest = json.loads(REQUEST_JSON)\n'
    embedded = _embed_request_payload(runner_code, {"text": "hello", "cfg_weight": 0.5})
    tree = ast.parse(embedded)
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "REQUEST_JSON" for target in node.targets)
    )

    assert json.loads(ast.literal_eval(assignment.value)) == {
        "text": "hello",
        "cfg_weight": 0.5,
    }
