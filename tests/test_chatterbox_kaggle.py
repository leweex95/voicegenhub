import asyncio
import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from voicegenhub.providers.base import TTSRequest, TTSError
from voicegenhub.providers.chatterbox_kaggle import (
    KaggleChatterboxProvider,
    _embed_request_payload,
    _kaggle_cli_env,
    _select_available_kernel,
    _wait_for_kernel,
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


def test_kaggle_kernel_pool_skips_queued_notebook():
    statuses = {
        "user/first": "kernelworkerstatus.queued",
        "user/second": "kernelworkerstatus.complete",
    }

    selected = _select_available_kernel(
        ["user/first", "user/second"], statuses.__getitem__
    )

    assert selected == "user/second"


def test_kaggle_kernel_pool_rejects_when_both_notebooks_are_active():
    statuses = {
        "user/first": "kernelworkerstatus.queued",
        "user/second": "kernelworkerstatus.running",
    }

    with pytest.raises(TTSError, match="All configured Kaggle.*no new run was submitted"):
        _select_available_kernel(["user/first", "user/second"], statuses.__getitem__)


def test_kaggle_kernel_pool_fails_closed_when_status_is_unknown():
    with pytest.raises(TTSError, match="user/first: unknown"):
        _select_available_kernel(["user/first"], lambda _kernel_id: "unknown")


def test_kaggle_waits_through_long_queue_until_kernel_completes():
    # The kernel stays queued for 615 seconds, beyond the previous 10-minute timeout.
    statuses = ["kernelworkerstatus.queued"] * 41 + ["kernelworkerstatus.complete"]
    elapsed = [0]

    class Job:
        def get_status(self):
            return statuses.pop(0)

    def sleep(seconds):
        elapsed[0] += seconds

    status = _wait_for_kernel(
        Job(), 15, sleep=sleep, clock=lambda: elapsed[0]
    )

    assert status == "kernelworkerstatus.complete"
    assert elapsed[0] == 41 * 15
    assert statuses == []


def test_kaggle_wait_stops_on_unknown_status():
    class Job:
        def get_status(self):
            return "kernelworkerstatus.unexpected"

    with pytest.raises(TTSError, match="unknown status"):
        _wait_for_kernel(Job(), 15, sleep=lambda _seconds: None, clock=lambda: 0)


def test_kaggle_wait_stops_when_kernel_is_cancelled():
    class Job:
        def get_status(self):
            return "kernelworkerstatus.cancel_acknowledged"

    with pytest.raises(TTSError, match="was cancelled"):
        _wait_for_kernel(Job(), 15, sleep=lambda _seconds: None, clock=lambda: 0)


def test_kaggle_runner_switches_sdpa_to_eager_before_inference():
    runner_path = Path(__file__).parents[1] / "src/voicegenhub/providers/kaggle_runner.py"
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    transformer_config = SimpleNamespace(_attn_implementation="sdpa")

    switch = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and any(
            isinstance(value, ast.Constant) and value.value == "sdpa"
            for value in node.test.comparators
        )
    )
    namespace = {"transformer_config": transformer_config, "getattr": getattr}
    exec(compile(ast.Module(body=[switch], type_ignores=[]), str(runner_path), "exec"), namespace)

    assert transformer_config._attn_implementation == "eager"


def test_kaggle_cli_subprocesses_force_utf8():
    env = _kaggle_cli_env()

    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
