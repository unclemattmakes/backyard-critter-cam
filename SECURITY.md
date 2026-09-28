# Security

## What this software is, threat-model-wise

A single-user tool that watches a camera and serves a dashboard on **localhost**. It has
**no accounts and no login.** That is a design decision, not an oversight: it is the family
raccoon dashboard, and a login page would advertise a safety it does not have.

The consequence, stated plainly and as the default: **anyone who can reach the dashboard's port
has full control.** They can rename individuals, reassign or reject identity suggestions, correct
or overwrite species labels, change camera settings live, and delete sightings and clips. There
is no audit trail of who did what, and no undo for a delete.

So the boundary is the network, and — unless you set the one knob below — only the network.

### The one knob: `operator_token`

Set `cfg.operator_token = "some phrase"` in `config_local.py` and the dashboard splits in two.
Devices that have not entered that phrase (dashboard footer, once per browser) become **viewers**:
they read and play everything and can log a "who's here" sighting as reviewable testimony, but
every label, name, camera and settings write is refused by the server, at the same choke point as
the cross-site guard — so endpoints nobody has written yet are covered too. Loopback is always the
operator; you are at the rig. A wrong token is simply a viewer, never an error.

It is off by default, and off means every device on your Wi-Fi is an operator. That is fine for a
household alone with its own devices and **wrong the moment a houseguest's phone joins the
network**, because "delete this clip" and "rename Notch" are one tap away with nothing in between.
If you run the LAN launcher at all, set it.

What it is not: it is a phrase in cleartext over plain HTTP, so it keeps honest devices in their
lane on a network you already trust. It is not authentication and it does not make the dashboard
safe to expose. The network is still the real boundary.

## Camera credentials — the one exception

Since 2026-08-22 the dashboard can manage the camera list, and a networked camera's login is a
real secret: an RTSP URL *is* `rtsp://user:pass@host/path`. That is the only credential this
software stores, and it gets narrower rules than everything else above.

- **Setting one requires loopback.** Every other edit in this dashboard is available to anyone
  who can reach the port; typing a camera password is not. It must come from the machine running
  the rig. The dashboard is plain HTTP with no login, so a password entered from a phone would
  cross the network in the clear, to a server that would have accepted it from anyone else on
  that network. Requiring loopback removes that exposure rather than mitigating it.
- **There is no read path.** The password column is never selected by the query that lists
  cameras, so no API response, log line, or error message can carry it. What the dashboard shows
  is `has_password: true`. The edit form is blank even when a password is set, and submitting it
  blank leaves the stored one alone.
- **It is stored in cleartext** in `backyard.db`, alongside the same secret that has always sat
  in cleartext in `config_local.py`. If you back the database up, you are backing that up too —
  see "What a backup carries off the machine" below, because the default backup destination is a
  cloud folder. Encrypting it would not change who can read it: anyone who can read the database
  file can read the rig's config file next to it.
- **It must not reach a log.** A camera's `src` *is* its password, and every line that prints one
  goes to `logs/backyard_cam.log`, which `backup.py` sweeps into the meta zip and off the machine.
  `cameras.safe_src()` masks it, and every human-facing print of a src goes through it. That was
  not always true: **logs written before 2026-08-22 contain RTSP passwords in cleartext**, in
  `logs/backyard_cam.log.*` and in every meta zip built from them. If you ran a networked camera
  before that date, rotate its password and delete those log copies — the fix protects new lines,
  not the ones already written.
- **Deleting the camera does not delete the password.** Removing a camera from the dashboard is a
  soft delete: the row is tombstoned so a config-listed camera can't come back on the next start,
  and so re-adding the name is a one-click undelete — which only works because the stored password
  is still there. The credential therefore outlives the camera, in `backyard.db` and in every
  backup of it. To actually destroy one: re-add the camera, clear its password *at the rig*, then
  remove it again. Or rotate it on the camera, which is the better answer anyway.
- **Give the camera its own account.** Make a dedicated user on the camera rather than reusing
  its admin login — most cameras support this, and it means the credential on disk cannot also
  reconfigure the camera.

## How the network boundary is enforced

By default `web_host` is `127.0.0.1` — the dashboard is reachable only from the machine running
the rig, and nothing else on your LAN can see it.

LAN mode is a deliberate, separate opt-in: `start_critter_cam_lan.bat` (or `--host 0.0.0.0`),
which exists so you can confirm a raccoon from the sofa. Every request — GET and POST alike —
goes through `_lan_guard` in `web.py` before it is routed, and that runs three checks:

1. **Origin and Content-Type, on state-changing requests** (`_csrf_refusal`, new in this
   release): a `POST` must carry an `Origin` naming this dashboard *and* declare
   `Content-Type: application/json`. A missing `Origin` is refused too — browsers always send
   one on `fetch`/XHR, so only hand-rolled tooling lacks it. Without this, any page you happened
   to have open in another tab could `fetch` `/api/individual` and blank months of confirmed
   re-ID labels: no preflight, no confirmation, no undo, and both checks below would pass,
   because your own browser makes the request. It runs before the `lan_only` shortcut on
   purpose, so a rig deliberately opened past the LAN still refuses other people's pages.
2. **Peer IP** (`_is_lan_client`): the client address must be loopback, RFC1918 private,
   link-local, or an IPv4-mapped IPv6 form of those. A public address gets a 403. This is what
   stops a forwarded port from serving the internet.
3. **Host header** (`_is_allowed_host`): the `Host` must be `localhost`, a
   loopback/private/link-local IP literal, or your configured `web_host`. Without this, a
   malicious website could point *its own* hostname at your rig's LAN IP and drive the dashboard
   from your own browser — the peer IP would look local because it *is* local (DNS rebinding).

The practical cost of check 1: a script that POSTs to the API needs
`-H 'Origin: http://127.0.0.1:8000' -H 'Content-Type: application/json'`. That is the intended
trade — the dashboard's own JS sends both for free.

POST bodies are capped (`_MAX_POST_BYTES`) so a huge body can't be used to exhaust memory, and
every file the dashboard serves out of `crops/`, `clips/`, and `clips_web/` is confined to those
directories by an explicit containment check, so a crafted `?path=` cannot walk up into the rest
of your disk.

**None of this makes it safe to expose to the internet.** Do not port-forward it, do not put it
behind a naked reverse proxy, do not `--host 0.0.0.0` on a machine with a public IP, and do not
run it on a network you share with people you would not hand the delete button to. If you want
remote access, terminate it in something that actually does authentication — a VPN or an
authenticating tunnel — and leave this bound to loopback behind it.

## What a backup carries off the machine

`backup.py` is documented as writing "into a cloud-synced folder" (Google Drive, Dropbox,
OneDrive), and `config_local.example.py` suggests exactly that for `backup_dest`. So the
question is not who can read your disk — it is who can read that folder, and what is in it.

Every `snapshots/meta-<date>.zip` contains **`config_local.py`**, unencrypted. That is your
latitude and longitude, your camera list, your `email_resend_api_key`, and your
`operator_token`. Every `snapshots/backyard-db-<date>.zip` contains the database, and therefore
your camera passwords. Both are there on purpose: a restore without them is a rig with no
location, no cameras and no mail, which is not a restore. `backup.py` names them on every run
rather than leaving it to be discovered.

What follows from that:

- **The cloud account holding that folder holds those secrets.** Whoever can read it — a shared
  Drive, a family account, anyone you ever gave the folder to, and the provider — can send mail
  as you and log into your cameras. If you share it, share a subfolder that is not `snapshots/`.
- **Rotate rather than assume.** If that folder was ever shared, public, or on an account you no
  longer control, rotate the Resend key, the operator token and the camera passwords. They are
  in every daily zip going back to the day you started, not just the newest one.
- **Deleting the newest copy is not enough.** These zips are per-day and they accumulate. A
  secret leaked once is in every zip written since.
- **`weights-archive.zip` and the media zips hold no credentials** — but the media is timestamped
  video of your property, and possibly of people walking past it.

## Other things worth knowing

- **Model weights are executable.** A YOLO `.pt` is a pickle, and Ultralytics loads it with
  `weights_only=False`, so a tampered weight file is remote code execution on your machine.
  `detector.py` therefore verifies every downloaded *and cached* MDv6 weight against a pinned
  SHA-256 and refuses to load on a mismatch — all five variants are pinned. If you add a new
  variant, pin it. The other three models are a different story: BioCLIP 2 (species names),
  MegaDescriptor (re-ID), and the OpenCLIP gate are fetched from Hugging Face at runtime by
  their libraries (pybioclip / timm / open_clip) with **no revision and no hash** — this repo
  does not verify them, and they change whenever upstream does. You are trusting Hugging Face
  and the publishing orgs on first download. If that bothers you, pre-download the weights
  once, verify them yourself, and let the libraries' caches serve your copy from then on.
- **The database and the media are plain files.** `backyard.db`, `crops/`, and `clips/` are
  unencrypted and readable by anything running as your user. They contain timestamped video of
  your own property, and possibly of people walking past it — MegaDetector has a `person`
  class for a reason. Treat backups accordingly.
- **`config_local.py` holds your coordinates** (for the sun/day-night profiles) and is
  gitignored. Keep it that way; don't paste it into an issue.

## Reporting a vulnerability

Please **open a private security advisory** on this repository:
[Security → Advisories → Report a vulnerability](https://github.com/unclemattmakes/backyard-critter-cam/security/advisories/new).
That keeps the report non-public until there's a fix. Do not open a normal public issue for
something exploitable.

Include what you'd want to receive: the version or commit, what an attacker needs (LAN access?
a link the operator clicks?), and the smallest reproduction you have.

**Realistic expectations.** This is maintained by one hobbyist in evenings. Expect an
acknowledgement within a couple of weeks, not a couple of hours, and no guaranteed patch
window. A vulnerability that requires internet exposure will most likely be answered with
"don't do that" plus a documentation fix, because internet exposure is outside the supported
configuration. Something exploitable *within* the documented LAN configuration — a CSRF path,
a directory-traversal escape, a rebinding bypass — is a real bug and will be treated as one.

There is no release process to speak of: there is one branch, `main`, and fixes land there.
There are no supported older versions.
