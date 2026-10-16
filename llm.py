"""
A local language model, asked politely: one stdlib POST to an Ollama server on this machine.

The rig's own models (MegaDetector, BioCLIP, MegaDescriptor) decide what is true. This one only
writes prose over facts the caller hands it, and every caller must keep working when it says
nothing -- so ask() never raises: it returns (text, note), where text is None whenever the model
was off, busy, unreachable or useless, and note says which, for the caller's log line.

It shares an 8 GB card with the rig, and this box hard-hangs under sustained CPU+GPU load (docs/
host-instability-2026-08.md). So: off unless cfg.llm_enabled; skipped while ANY heavyio lock is
held (the nightly batch grows its VRAM and silently falls back to CPU if it can't get it);
skipped unless the whole model fits in free VRAM (a partial CPU offload is the hang-prone load);
and keep_alive 0 unloads the model as soon as it answers. The server side of the same agreement
(4 threads, no AVX-512, no partial offload) lives in the server's own config, not here.

    python llm.py "Say hello in five words."     # one-off check against cfg.llm_*
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request

import heavyio

# Ollama's chat endpoint. "think": False because these are short writing jobs: Qwen3.5's
# reasoning pass costs several hundred tokens and changes nothing in a three-sentence lede.
_CHAT = "/api/chat"


def busy_reason() -> str | None:
    """Who holds the box, or None if nobody does. Reading the locks also reaps stale ones, which
    is what every heavyio caller does by design."""
    try:
        holders = heavyio.read_holders()
    except Exception as e:                      # can't tell -> don't take the GPU
        return f"heavy-io lock unreadable ({e.__class__.__name__})"
    if not holders:
        return None
    return "heavy-io lock held by " + ", ".join(str(h.get("name", "?")) for h in holders)


def free_vram_mb(_run=subprocess.run) -> int | None:
    """Free VRAM on the first NVIDIA GPU, or None when nvidia-smi can't say (no NVIDIA card:
    Ollama then runs on the CPU or another backend, and this guard has nothing to guard)."""
    try:
        r = _run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        used, total = (int(x) for x in r.stdout.splitlines()[0].split(","))
        return total - used
    except Exception:
        return None


def ask(cfg, system: str, prompt: str, *, max_tokens: int = 300,
        temperature: float = 0.6, _urlopen=urllib.request.urlopen,
        _free_vram=None) -> tuple[str | None, str]:
    """One chat turn. Returns (text, note); text is None on every failure, never an exception."""
    if not getattr(cfg, "llm_enabled", False):
        return None, "llm off (cfg.llm_enabled)"
    why = busy_reason()
    if why:
        return None, f"llm skipped: {why}"
    free, need = (_free_vram or free_vram_mb)(), int(getattr(cfg, "llm_min_free_vram_mb", 0) or 0)
    if free is not None and free < need:
        return None, f"llm skipped: {free} MiB VRAM free, {cfg.llm_model} needs {need}"
    body = {
        "model": cfg.llm_model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "keep_alive": cfg.llm_keep_alive,
        "options": {"num_predict": int(max_tokens), "temperature": float(temperature),
                    "num_ctx": int(cfg.llm_num_ctx), "num_thread": int(cfg.llm_num_thread)},
    }
    req = urllib.request.Request(cfg.llm_url.rstrip("/") + _CHAT,
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with _urlopen(req, timeout=cfg.llm_timeout_s) as resp:
            reply = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:         # e.g. 404 = model not pulled
        return None, f"llm error: HTTP {e.code} from {cfg.llm_url} ({cfg.llm_model})"
    except Exception as e:                      # refused, timed out, bad JSON
        return None, f"llm unreachable: {e.__class__.__name__}: {e}"
    text = ((reply.get("message") or {}).get("content") or "").strip()
    if not text:
        return None, "llm returned nothing"
    return text, f"llm ok ({cfg.llm_model}, {reply.get('eval_count', '?')} tokens)"


def main(argv=None) -> int:
    import config
    cfg = config.CONFIG
    prompt = " ".join((argv if argv is not None else sys.argv[1:])) or "Say hello in five words."
    cfg.llm_enabled = True                      # a by-hand check means "try it"
    text, note = ask(cfg, "You are terse.", prompt)
    print(note)
    if text:
        print(text)
    return 0 if text else 1


if __name__ == "__main__":
    sys.exit(main())
