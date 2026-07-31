# Contributing

This is one person's backyard rig that got out of hand and then got published. There is no
team, no roadmap commitment, and no response-time promise — if a pull request sits for two
weeks it is because a raccoon-related side quest happened, not because it was rejected. That
said, patches are genuinely welcome, and the codebase was written to be read.

## Getting it running

Follow the [README's Setup section](README.md#setup). The short version: run `setup.bat`
(Windows) or `bash setup.sh` (Linux/macOS) rather than a bare
`pip install -r requirements.txt` — those scripts pick the torch build that matches your
hardware first, and `requirements.txt` deliberately does not pin torch because the right wheel
differs per machine (a stock CUDA 12.6 torch reports `cuda.is_available() == True` and then
dies at the first real op on a Blackwell GPU).

Two things the setup scripts do not install:

- **`ffmpeg` and `ffprobe` on your `PATH`**, needed by `clips.py` (H.264 clip recording via an
  ffmpeg pipe) and `reel.py` (the stitched highlight reel). Without them the rig still runs and
  still detects; clips fall back to a cv2 mp4v writer and reels don't build.
- **A camera.** Most of the code doesn't need one. All of the tests don't.

Local settings live in `config_local.py`, which is gitignored — copy
`config_local.example.py` and edit that. Never commit a `config_local.py`; it holds your
coordinates, camera indices, and ignore zones.

## Tests

```
python -m pytest tests/ -q
```

328 tests at the time of writing, and they should stay green. They are pure logic: no GPU, no
camera, no model download, no network. A test that needs any of those four is a test that will
be skipped forever, which is worse than no test.

Two rules that are load-bearing rather than stylistic:

- **Every test that touches a database uses `tmp_path`.** The `conn` / `db_path` fixtures in
  `tests/conftest.py` build a throwaway SQLite file per test, with the real schema, inside
  pytest's temp dir. The live `backyard.db` is 810 MB of irreplaceable raccoon history and no
  test may open it, ever.
- **Drive behaviour through the real insert helpers** (`db.insert_detection`, `visits.refresh`,
  …) with controlled timestamps, rather than hand-writing rows. That is what makes the
  gap-boundary and dominant-label assertions deterministic instead of lucky.

## House style

Read a neighbouring file before writing a new one — the conventions are consistent and mostly
visible from any two hundred lines. The parts worth spelling out:

- **Flat modules at the repo root.** `db.py`, `visits.py`, `clips.py`, `web.py` and friends sit
  next to each other and import each other by bare name; `tests/conftest.py` puts the root on
  `sys.path` instead of making this installable. That is a deliberate choice for a project run
  as scripts, not a stage on the way to a package. Please don't reorganise it into
  `src/backyard/` as a drive-by.
- **Comments explain WHY, with a number where a number exists.** Not `# resize the frame` but
  the reason the constant is what it is: the crop gate is 0.8 because that keeps ~1,600 of the
  trustworthy raccoon crops; `clip_scale` needs even dimensions because `0.667 × 1920 = 1281`
  and libx264 silently refused odd widths for a whole night. If you had to learn something the
  hard way to write the patch, the comment is where that goes.
- **Module docstrings carry the CLI.** Every script-shaped module opens with what it does, why
  it does it that way rather than the obvious alternative, and its actual invocations. New
  modules should too.
- **Boring and robust over clever.** Stdlib first — the dashboard is `http.server` and vanilla
  JS on purpose, and the web import path stays numpy-free so it starts instantly. No new
  dependency without a reason that couldn't be met by fifty lines of stdlib.
- **Schema changes are additive.** `db._migrate` grows columns and tables; it does not rewrite
  them. Deletion is soft where anything downstream might still reference the row (see
  `pruned_at` on clips).

## What's actually wanted

Useful, in roughly descending order:

- Bug fixes, especially anything that makes the rig survive a condition this yard doesn't have
  — a different camera, a non-Windows capture path, a timezone that isn't US Pacific.
- Portability fixes generally. Plenty of assumptions were written when this only ever ran on
  one machine.
- Tests for untested logic. Several modules are thinner on coverage than `visits.py`.
- Documentation that corrects something wrong, rather than restating something right.
- Detector/classifier tuning backed by numbers from `eval.py`. "This threshold is better" is
  interesting with a metric attached and unfalsifiable without one.

Out of scope, and likely to be declined however good the code is:

- Multi-user accounts, authentication, or anything that implies this should be internet-facing.
  See [SECURITY.md](SECURITY.md) — the threat model is "LAN only, deliberately", and adding a
  login page would advertise a safety it doesn't have.
- Cloud services, hosted inference, or telemetry of any kind. The whole point is that the
  animals stay on your own disk.
- Repackaging, build systems, containers, or a framework rewrite of the dashboard.
- Retraining or fine-tuning pipelines. Zero-shot labelling is a feature: editing the species
  list in `classify.py` and re-running costs nothing.

If a change is larger than a bug fix, open an issue first and describe what you want to do.
It's cheaper for both of us than a rejected 800-line diff.

## Licensing

The project is **AGPL-3.0** (because Ultralytics is — see [NOTICE.md](NOTICE.md)). By opening a
pull request you agree your contribution is licensed under AGPL-3.0 as well. There is no CLA.
Don't paste in code from a differently-licensed project, and don't add a dependency whose
license is incompatible with AGPL-3.0.
