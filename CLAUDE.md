# Notes for AI assistants working in this repo

A live backyard wildlife camera rig: capture + MegaDetector, BioCLIP species naming, re-ID,
behaviour, a stdlib web dashboard. One maintainer. **Public repo, AGPL-3.0.** Start with the
[README](README.md); [CONTRIBUTING.md](CONTRIBUTING.md) has the house style in full.

## The rig runs out of a checkout. Don't disturb it.

- The live rig, its watchdog and the nightly batch all run from the maintainer's main checkout.
  **Do dev work in a git worktree** (`git worktree add ../<name> -b claude/<topic> origin/main`).
  Never `checkout`/`switch`/`stash`/`reset`/`rebase` in the rig's folder: a branch switch rewrites
  files under a running process. A checkout once left `run_clipmotion.bat` hung for two days.
  This matters most while the nightly batch runs, roughly **16:20–18:00** local (it tracks
  sunset; `sunsched.py --show` prints the exact time).
- **Never write the live `backyard.db`** (several GB, irreplaceable). Don't open it for writing,
  don't run migrations against it by hand. Read-only queries only, and only when asked. Tests use
  `tmp_path` via the `conn`/`db_path` fixtures. Schema changes go through `db._migrate`,
  additive only (new columns/tables, never rewrites).
- **The host hard-hangs under sustained heavy CPU/GPU load** (see
  [docs/host-instability-2026-08.md](docs/host-instability-2026-08.md)). Don't run `embed.py`,
  `eval.py`, `clipmotion.py`, `classify.py`/naming, any GPU job, big directory walks over
  `crops/`/`clips/`, or the full test suite repeatedly, without asking first. Run targeted test
  files. To see or stop the rig: `python rigwatch.py --status`, `--hold` (keeps it off across
  reboots), `--release`.

## Ground truth

- **Cameras:** the `cameras` table in the DB is authoritative. `config_local.py` only seeds it
  once per source (`cameras.load_specs`). `glass_door_cam` is a Raspberry Pi serving MJPEG over
  HTTP, not a USB webcam.
- **Config:** every knob is in `config.py`; machine-specific overrides in `config_local.py`
  (gitignored). Secrets (API keys, `operator_token`, `mqtt_password`, `heartbeat_url`, camera
  URLs with passwords) live in `~/.critter-cam/secrets.json` (or `$CRITTER_CAM_SECRETS`), read
  via `config.secret()` / `SECRET_FIELDS`.
- **The MQTT broker** the rig publishes to is managed in a separate repo, not here.

## Tests

```
.venv\Scripts\python.exe -m pytest tests -q                    # whole suite
.venv\Scripts\python.exe -m pytest tests/test_backup.py -q     # prefer one file
```

Pure logic: no GPU, camera, network or model download. Keep it that way. CI runs Python 3.12;
the rig runs 3.14; the floor is 3.10.

## Public repo hygiene

Never commit `config_local.py`, the secrets file, `backyard.db`, crops, clips or reports. Never
write real LAN IPs (use `192.168.1.x` placeholders), email addresses, Windows usernames, home
coordinates, keys or tokens, or the names of people other than the maintainer, in code, docs,
commit messages or PR bodies.

## Workflow and style

- One `claude/<topic>` branch per PR, from `origin/main`. Delete the branch after merge.
- Commit titles are plain-English sentences that state the change and why ("The nightly batch
  now runs from a copy of itself, so a checkout mid-run cannot hang it"). The body carries the
  rationale, the numbers and the history.
- Flat modules at the repo root, imported by bare name; no package, no `src/`. Stdlib first; no
  new dependency without a strong reason. The dashboard is `http.server` + vanilla JS.
- Short comments that say *why*, with the number where there is one. History, incidents and
  "used to" belong in commit bodies, not in code.

## Pointers

- [docs/deferred-work.md](docs/deferred-work.md) — the ranked backlog.
- [docs/host-instability-2026-08.md](docs/host-instability-2026-08.md) — why the box hangs.
- [docs/README.md](docs/README.md) — which docs are current and which are dated snapshots.
- [SECURITY.md](SECURITY.md) — threat model (LAN only, no login) and where secrets live.
