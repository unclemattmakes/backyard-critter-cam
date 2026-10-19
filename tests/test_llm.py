"""llm.py may only ever fail quietly: off, busy, unreachable and useless all come back as
(None, note), and none of them may touch the GPU. Ollama is never contacted -- urlopen is a fake
that records the request -- and heavyio's lock dir lives in tmp_path."""
from __future__ import annotations

import io
import json
import time
import urllib.error
from types import SimpleNamespace

import pytest

import heavyio
import llm

REAL_FREE_VRAM = llm.free_vram_mb   # kept before the autouse fixture stubs it out


@pytest.fixture(autouse=True)
def _isolated_lockdir(tmp_path, monkeypatch):
    monkeypatch.setattr(heavyio, "LOCK_DIR", tmp_path / ".locks")
    monkeypatch.setattr(llm, "free_vram_mb", lambda: 7000)     # never shell out to nvidia-smi
    yield


def mkcfg(**over):
    cfg = SimpleNamespace(llm_enabled=True, llm_url="http://127.0.0.1:11434/",
                          llm_model="qwen3.5:9b", llm_timeout_s=5.0, llm_keep_alive=0,
                          llm_num_ctx=8192, llm_num_thread=4, llm_min_free_vram_mb=6500)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


class FakeUrlopen:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.requests = reply, exc, []

    def __call__(self, req, timeout=None):
        self.requests.append((req, timeout))
        if self.exc:
            raise self.exc
        return io.BytesIO(json.dumps(self.reply).encode("utf-8"))


def test_off_means_no_request():
    fake = FakeUrlopen(reply={})
    text, note = llm.ask(mkcfg(llm_enabled=False), "s", "p", _urlopen=fake)
    assert text is None and "off" in note
    assert fake.requests == []


def test_a_held_heavyio_lock_keeps_the_model_off_the_gpu(monkeypatch):
    monkeypatch.setattr(heavyio, "_alive", lambda pid: True)
    heavyio.LOCK_DIR.mkdir(parents=True)
    (heavyio.LOCK_DIR / "batch.lock").write_text(json.dumps(
        {"name": "batch", "pid": 4242, "started_epoch": time.time()}), encoding="utf-8")
    fake = FakeUrlopen(reply={})
    text, note = llm.ask(mkcfg(), "s", "p", _urlopen=fake)
    assert text is None and "batch" in note
    assert fake.requests == []


def test_too_little_free_vram_means_no_partial_offload():
    fake = FakeUrlopen(reply={})
    text, note = llm.ask(mkcfg(), "s", "p", _urlopen=fake, _free_vram=lambda: 3000)
    assert text is None and "3000 MiB" in note
    assert fake.requests == []
    # No NVIDIA card to ask: nothing to guard, so the call goes ahead.
    fake2 = FakeUrlopen(reply={"message": {"content": "ok"}})
    assert llm.ask(mkcfg(), "s", "p", _urlopen=fake2, _free_vram=lambda: None)[0] == "ok"


def test_free_vram_reads_nvidia_smi_and_fails_open():
    assert REAL_FREE_VRAM(lambda *a, **k: SimpleNamespace(stdout="812, 8192\n")) == 7380

    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    assert REAL_FREE_VRAM(missing) is None
    assert REAL_FREE_VRAM(lambda *a, **k: SimpleNamespace(stdout="")) is None


def test_request_carries_the_shared_gpu_terms():
    fake = FakeUrlopen(reply={"message": {"content": "  Hello there.  "}, "eval_count": 4})
    text, note = llm.ask(mkcfg(), "sys", "prompt", max_tokens=50, _urlopen=fake)
    assert text == "Hello there." and "ok" in note
    req, timeout = fake.requests[0]
    assert req.full_url == "http://127.0.0.1:11434/api/chat" and timeout == 5.0
    body = json.loads(req.data)
    assert body["keep_alive"] == 0 and body["think"] is False and body["stream"] is False
    assert body["options"]["num_thread"] == 4 and body["options"]["num_ctx"] == 8192
    assert body["options"]["num_predict"] == 50
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


@pytest.mark.parametrize("exc, word", [
    (urllib.error.HTTPError("u", 404, "not found", {}, None), "HTTP 404"),
    (urllib.error.URLError("refused"), "unreachable"),
    (TimeoutError("slow"), "unreachable"),
])
def test_failures_are_notes_not_exceptions(exc, word):
    text, note = llm.ask(mkcfg(), "s", "p", _urlopen=FakeUrlopen(exc=exc))
    assert text is None and word in note


def test_an_empty_answer_is_no_answer():
    text, note = llm.ask(mkcfg(), "s", "p", _urlopen=FakeUrlopen(reply={"message": {"content": " "}}))
    assert text is None and "nothing" in note
