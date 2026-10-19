"""
A second opinion from Claude on the crops BioCLIP was unsure of -- in SHADOW MODE.

Graded against the human labels, BioCLIP's species call is right 0.906 of the time at
confidence >= 0.8, 0.225 at 0.5-0.8 and 0.026 below 0.5 (docs/deferred-work.md, item 9). This
module asks Claude, independently, to name the animal in those uncertain crops, then grades both
models against the same human verdicts. The question "should Claude ever drive a label?" gets a
number before anyone answers it.

What shadow mode means here:
  * backyard.db is opened READ-ONLY. Answers go to a sidecar, second_opinion.db, beside it, and
    nothing in the rig, the dashboard or the email reads that file.
  * Claude is never shown BioCLIP's guess, so when the two agree it is two looks, not one echo.

Privacy: this is the only part of the project that sends pixels off the LAN. Only
detection_class = 'animal' crops are sent (MegaDetector's person and vehicle boxes never are),
downscaled to IMAGE_MAX_SIDE, along with the capture time and nothing else. See SECURITY.md.

  python secondopinion.py --graded --dry-run      # how many crops, and roughly what it costs
  python secondopinion.py --graded --limit 200    # ask about human-labelled uncertain crops
                                                  # (at most --per-visit crops from any one visit)
  python secondopinion.py --grade                 # Claude vs BioCLIP on those crops
  python secondopinion.py --graded --model claude-haiku-5-5   # the same crops, a cheaper model
  python secondopinion.py --recent 24             # unreviewed uncertain crops from the last day
  python secondopinion.py --nightly               # the batch's step: --recent on Haiku, only if
                                                  # second_opinion_nightly is on in config_local.py

Needs `pip install anthropic` (only this module imports it, so the rig runs without it) and an
anthropic_api_key in the secrets file, or ANTHROPIC_API_KEY in the environment.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import config
import db
from classify import SPECIES_LABELS
from clipfilter import NONANIMAL_LABEL

# Claude's answer when it sees an animal that is not on the yard's list. BioCLIP has no such
# escape: it must pick a listed species, which is one way it ends up confidently wrong.
OTHER_LABEL = "other animal"

# Human labels that always mean a person. A crop carrying one, and every crop in its visit, is
# never sent; config's second_opinion_never_send adds names to these.
PERSON_LABELS = ("person", "human", "people", "man", "woman", "child")

# A crop's long side is downscaled to this before sending. Image input costs about w*h/750
# tokens, so 768 px caps a crop near 790 tokens, and most detector boxes are smaller anyway.
IMAGE_MAX_SIDE = 768

# Thinking is always on for Opus 5.5 and counts toward this; at low effort a one-label answer
# uses a few hundred tokens, so this is headroom, not a target.
MAX_TOKENS = 4000

# Retry a refused request on Anthropic's recommended fallback model. Haiku has no fallback.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Network-bound, not CPU-bound: four in flight is a few seconds per crop with no load on the host.
WORKERS = 4

# Every crop is the same request shape, so these statuses (bad request, key, permission, model
# name) would fail all of them the same way. Stop the run instead of logging 300 errors.
FATAL_STATUS = {400, 401, 403, 404}

# USD per million tokens (input, output), 2026-10. Only for --dry-run estimates and the cost line
# in the grade report; the token counts themselves come from the API's own usage figures.
PRICE_PER_MTOK = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

# Per-crop overhead for --dry-run, measured on the first 200-crop graded run (2026-10-17): about
# 1,100 input tokens of system prompt, label list and schema besides the image, and 67 (Opus) to
# 79 (Haiku) output tokens of low-effort thinking plus the JSON answer.
EST_PROMPT_TOKENS = 1100
EST_OUTPUT_TOKENS = 80

# A graded sample takes at most this many crops from any one visit. A visit labelled as a whole
# gives every crop in it the same label, and an uncapped random draw once took 100 of 200 crops
# from a single visit, most of them unreadable smudges.
PER_VISIT = 3

# How far back the nightly run looks. A little over a day, because the batch start moves with
# sunset; crops already answered are never sent twice, so the overlap costs nothing.
NIGHTLY_HOURS = 26

SYSTEM = (
    "You identify animals in crops from a backyard wildlife camera. Each image is the box an "
    "animal detector drew, so it is often small, blurred, partly out of frame, or taken at night, "
    "when the camera sees in infrared and the image is greyscale, often with bright eyeshine. The "
    "detector also fires on things that are not animals: food, dishes, garden furniture, shadows "
    "and empty ground.\n\n"
    "Name the animal with exactly one label from the allowed list. Answer "
    f"'{NONANIMAL_LABEL}' if no animal is visible, and '{OTHER_LABEL}' if an animal is visible "
    "but is not on the list. Give 'low' certainty whenever the crop is too dark, blurred or "
    "partial to be sure, rather than guessing confidently. Keep the reason to one short sentence "
    "about what you can see.")

SIDECAR_SCHEMA = """
CREATE TABLE IF NOT EXISTS opinions (
    detection_id  INTEGER NOT NULL,   -- detections.id in backyard.db
    model         TEXT    NOT NULL,   -- the model asked
    served_by     TEXT,               -- the model that answered; differs after a refusal fallback
    species       TEXT,               -- NULL unless outcome = 'answered'
    certainty     TEXT,               -- 'high' | 'medium' | 'low'
    reason        TEXT,
    outcome       TEXT    NOT NULL,   -- 'answered' | 'refusal' | 'max_tokens' | 'unreadable' | 'error'
    input_tokens  INTEGER,
    output_tokens INTEGER,
    asked_at      TEXT    NOT NULL,
    PRIMARY KEY (detection_id, model)
);
"""


def allowed_labels(species_labels=None) -> list[str]:
    """The yard's species list plus the two escape labels, de-duplicated, order kept."""
    labels = list(species_labels if species_labels is not None else SPECIES_LABELS)
    return list(dict.fromkeys(labels + [NONANIMAL_LABEL, OTHER_LABEL]))


# ---------------------------------------------------------------------------
# Which crops to ask about
# ---------------------------------------------------------------------------

_CANDIDATE_COLS = ("id, crop_path, timestamp, source, visit_id, species, species_confidence, "
                   "species_verified, species_source, model_species, model_species_confidence")


def never_send_labels(cfg) -> list[str]:
    """PERSON_LABELS plus the owner's own list, lower-cased, de-duplicated."""
    extra = getattr(cfg, "second_opinion_never_send", None) or []
    return list(dict.fromkeys(str(x).strip().lower() for x in (*PERSON_LABELS, *extra)))


def candidates(conn, *, graded: bool, max_conf: float, recent_hours: float | None = None,
               now: datetime | None = None, never_send=None) -> list[dict]:
    """Animal crops BioCLIP named below max_conf, oldest first.

    graded=True keeps only crops a human has ruled on, which is the eval set. Otherwise it keeps
    only crops nobody has reviewed, optionally from the last recent_hours. Crops a human corrected
    before model_species existed are left out: BioCLIP's confidence on them is gone, so they
    cannot be placed in a band.

    Crops a human labelled with a never_send label (default never_send_labels(config.CONFIG)) are
    left out, and so is every other crop in the same visit: a person mistaken for an animal is
    usually in more than the one box somebody labelled."""
    never = never_send_labels(config.CONFIG) if never_send is None else         [str(x).strip().lower() for x in never_send]
    # BioCLIP's own confidence, even where a human has since overwritten species.
    conf = "COALESCE(model_species_confidence, species_confidence)"
    where = ["detection_class = 'animal'",
             "COALESCE(model_species_source, species_source) = 'bioclip'",
             f"{conf} IS NOT NULL", f"{conf} < ?"]
    params: list = [float(max_conf)]
    if graded:
        where.append("species_verified IS NOT NULL")
    else:
        where.append("species_verified IS NULL AND species_source = 'bioclip'")
        if recent_hours is not None:
            cutoff = (now or datetime.now().astimezone()) - timedelta(hours=recent_hours)
            where.append("timestamp >= ?")
            params.append(cutoff.isoformat())
    if never:
        marks = ",".join("?" * len(never))
        person = f"species_source = 'human' AND LOWER(species) IN ({marks})"
        where.append(f"NOT ({person})")
        where.append(f"(visit_id IS NULL OR visit_id NOT IN (SELECT visit_id FROM detections "
                     f"WHERE visit_id IS NOT NULL AND {person}))")
        params += never + never
    rows = conn.execute(
        f"SELECT {_CANDIDATE_COLS} FROM detections WHERE {' AND '.join(where)} ORDER BY id",
        params).fetchall()
    return [dict(r) for r in rows]


def cap_per_visit(rows: list[dict], per_visit: int) -> list[dict]:
    """The first `per_visit` rows of each visit, order kept. A crop not yet stamped with a visit
    counts as its own visit."""
    seen: Counter = Counter()
    out = []
    for r in rows:
        key = r.get("visit_id")
        key = ("crop", r["id"]) if key is None else key
        if seen[key] < per_visit:
            seen[key] += 1
            out.append(r)
    return out


def pick(rows: list[dict], *, limit: int | None, already: set, graded: bool, seed: int = 0,
         per_visit: int | None = PER_VISIT) -> list[dict]:
    """The crops this run will send. Graded runs shuffle the WHOLE pool with a fixed seed, keep at
    most per_visit crops of each visit, take the first `limit`, and only then drop crops already
    answered: a second model asked with the same seed gets the same crops, and a bigger --limit
    extends the sample rather than redrawing it. Recent runs take the newest first."""
    if graded:
        pool = list(rows)
        random.Random(seed).shuffle(pool)
        if per_visit:
            pool = cap_per_visit(pool, per_visit)
        pool = pool[:limit]
    else:
        pool = list(reversed(rows))[:limit]
    return [r for r in pool if r["id"] not in already]


# ---------------------------------------------------------------------------
# The sidecar DB (never backyard.db)
# ---------------------------------------------------------------------------

def sidecar_path(cfg) -> Path:
    return Path(cfg.db_path).with_name("second_opinion.db")


def open_sidecar(path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SIDECAR_SCHEMA)
    return conn


def already_asked(sidecar, model: str) -> set:
    """Detection ids this model has a final answer for. Errors are left out so they retry."""
    return {r[0] for r in sidecar.execute(
        "SELECT detection_id FROM opinions WHERE model = ? AND outcome != 'error'", (model,))}


def record(sidecar, detection_id: int, model: str, result: dict) -> None:
    sidecar.execute(
        "INSERT OR REPLACE INTO opinions (detection_id, model, served_by, species, certainty, "
        "reason, outcome, input_tokens, output_tokens, asked_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (int(detection_id), model, result.get("served_by"), result.get("species"),
         result.get("certainty"), result.get("reason"), result["outcome"],
         result.get("input_tokens"), result.get("output_tokens"), db.now_local_iso()))


# ---------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------

def encode_crop(path) -> str | None:
    """The crop as base64 JPEG, long side at most IMAGE_MAX_SIDE, or None if it will not decode
    (a host hang can leave NUL-filled crops; see classify._decodes)."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=90)
    except Exception:       # noqa: BLE001 -- any failure to decode means "do not send this"
        return None
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def answer_schema(labels: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "species": {"type": "string", "enum": labels},
            "certainty": {"type": "string", "enum": ["high", "medium", "low"]},
            "reason": {"type": "string"},
        },
        "required": ["species", "certainty", "reason"],
        "additionalProperties": False,
    }


def _clock(timestamp: str) -> str:
    try:
        return datetime.fromisoformat(timestamp).strftime("%H:%M")
    except (TypeError, ValueError):
        return "an unknown time"


def build_request(model: str, labels: list[str], image_b64: str, timestamp: str) -> dict:
    """Keyword arguments for client.beta.messages.create. BioCLIP's guess is deliberately absent."""
    text = (f"Allowed labels: {', '.join(labels)}.\n"
            f"Captured at {_clock(timestamp)} local time.")
    req = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM,
        # Low effort: naming one animal from a list does not repay deep thinking.
        "output_config": {"effort": "low",
                          "format": {"type": "json_schema", "schema": answer_schema(labels)}},
        "messages": [{"role": "user", "content": [
            {"type": "image",
             "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
            {"type": "text", "text": text},
        ]}],
    }
    if not model.startswith("claude-haiku"):
        req["betas"] = [FALLBACK_BETA]
        req["fallbacks"] = "default"
    return req


def parse_response(resp, labels: list[str]) -> dict:
    """One API response as a sidecar row. Checks stop_reason before reading the content."""
    usage = getattr(resp, "usage", None)
    out = {"served_by": getattr(resp, "model", None),
           "input_tokens": getattr(usage, "input_tokens", None),
           "output_tokens": getattr(usage, "output_tokens", None)}
    if resp.stop_reason in ("refusal", "max_tokens"):
        out["outcome"] = resp.stop_reason
        return out
    text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return {**out, "outcome": "error", "reason": "unparseable answer"}
    if not isinstance(data, dict) or data.get("species") not in labels:
        return {**out, "outcome": "error", "reason": f"label off the list: {str(data)[:200]}"}
    return {**out, "outcome": "answered", "species": data["species"],
            "certainty": data.get("certainty"), "reason": str(data.get("reason", ""))[:300]}


def _resolve(crop_path: str, root: Path) -> Path:
    p = Path(crop_path)
    return p if p.is_absolute() else root / p


def ask_all(rows: list[dict], *, client, model: str, labels: list[str], sidecar,
            root: Path = config.ROOT, workers: int = WORKERS, log=print) -> Counter:
    """Ask about each row and write every answer to the sidecar. Returns a tally of outcomes."""
    def one(r):
        image = encode_crop(_resolve(r["crop_path"], root))
        if image is None:
            return r["id"], {"outcome": "unreadable"}
        try:
            resp = client.beta.messages.create(**build_request(model, labels, image, r["timestamp"]))
        except Exception as e:      # noqa: BLE001 -- one bad crop must not end the run
            if getattr(e, "status_code", None) in FATAL_STATUS:
                raise
            return r["id"], {"outcome": "error", "reason": f"{type(e).__name__}: {e}"[:300]}
        return r["id"], parse_response(resp, labels)

    tally: Counter = Counter()
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        for i, (det_id, result) in enumerate(pool.map(one, rows), 1):
            record(sidecar, det_id, model, result)
            tally[result["outcome"]] += 1
            if i % 25 == 0:
                sidecar.commit()
                log(f"[second-opinion] {i}/{len(rows)} {dict(tally)}")
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        sidecar.commit()
        raise
    pool.shutdown()
    sidecar.commit()
    return tally


def make_client(cfg):
    try:
        import anthropic
    except ImportError:
        raise SystemExit("secondopinion.py needs the Anthropic SDK: pip install anthropic")
    key = getattr(cfg, "anthropic_api_key", None)
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()


def estimate_cost(rows: list[dict], model: str,
                  root: Path = config.ROOT) -> tuple[int, int, float | None, int]:
    """(input tokens, output tokens, USD or None, unreadable crops) for sending these rows. Reads
    image headers only. Unreadable crops are skipped at send time and cost nothing, but a run
    where most are unreadable is a wrong root, not a damaged corpus."""
    from PIL import Image
    tokens_in = unreadable = 0
    for r in rows:
        try:
            with Image.open(_resolve(r["crop_path"], root)) as im:
                w, h = im.size
        except Exception:   # noqa: BLE001 -- counted, and skipped at send time
            unreadable += 1
            continue
        scale = min(1.0, IMAGE_MAX_SIDE / max(w, h))
        tokens_in += int(w * scale * h * scale / 750) + EST_PROMPT_TOKENS
    tokens_out = EST_OUTPUT_TOKENS * (len(rows) - unreadable)
    return tokens_in, tokens_out, _usd(model, tokens_in, tokens_out), unreadable


def _usd(model: str, tokens_in: int, tokens_out: int) -> float | None:
    price = PRICE_PER_MTOK.get(model)
    if price is None:
        return None
    return (tokens_in * price[0] + tokens_out * price[1]) / 1e6


# ---------------------------------------------------------------------------
# Grading against the human labels
# ---------------------------------------------------------------------------

def band(conf: float) -> str:
    if conf < 0.5:
        return "<0.5"
    if conf < 0.8:
        return "0.5-0.8"
    return ">=0.8"


def _claude_right(claude: str, truth: str, labels: list[str]) -> bool:
    """Exact match, or OTHER_LABEL for an animal the human named that is not on the list."""
    if claude == truth:
        return True
    return claude == OTHER_LABEL and truth not in labels and truth != NONANIMAL_LABEL


def grade_rows(det_rows: list[dict], opinions: dict, labels: list[str], period_fn) -> list[dict]:
    """One graded record per crop with both a human verdict and a Claude answer.

    Truth follows eval.py: a human correction or a confirmation gives the real species. A
    rejection only says BioCLIP was wrong, so Claude is wrong there if it agreed with BioCLIP and
    unknown (None) if it said something else."""
    out = []
    for r in det_rows:
        op = opinions.get(r["id"])
        if op is None:
            continue
        human_set = r["species_source"] == "human"
        bio = r["model_species"] if human_set else r["species"]
        conf = r["model_species_confidence"] if human_set else r["species_confidence"]
        if bio is None or conf is None:
            continue
        if human_set or r["species_verified"] == 1:
            truth = r["species"]
            bio_right = bio == truth
            claude_right = _claude_right(op["species"], truth, labels)
        else:
            truth = None
            bio_right = False
            claude_right = False if op["species"] == bio else None
        try:
            period = period_fn(datetime.fromisoformat(r["timestamp"]))
        except (TypeError, ValueError):
            period = "unknown"
        out.append({"id": r["id"], "band": band(float(conf)), "period": period, "truth": truth,
                    "bioclip": bio, "claude": op["species"], "certainty": op["certainty"],
                    "bio_right": bio_right, "claude_right": claude_right})
    return out


def summarize(graded: list[dict]) -> dict:
    """Accuracy per slice, on the crops whose truth is known."""
    def stats(rows):
        known = [g for g in rows if g["truth"] is not None]
        high = [g for g in known if g["certainty"] == "high"]
        agree = [g for g in known if g["claude"] == g["bioclip"]]
        rejected = [g for g in rows if g["truth"] is None]

        def acc(xs, key):
            return round(sum(1 for g in xs if g[key]) / len(xs), 3) if xs else None
        return {
            "n": len(known),
            "bioclip_acc": acc(known, "bio_right"),
            "claude_acc": acc(known, "claude_right"),
            "claude_high_n": len(high), "claude_high_acc": acc(high, "claude_right"),
            # The number a future gate would hang on: when the two models agree, how often right?
            "agree_n": len(agree), "agree_acc": acc(agree, "claude_right"),
            "rejected_n": len(rejected),
            "rejected_claude_echoed": sum(1 for g in rejected if g["claude_right"] is False),
        }

    by: dict = defaultdict(list)
    for g in graded:
        by[("all", "all")].append(g)
        by[("band", g["band"])].append(g)
        by[("period", g["period"])].append(g)
    misses = Counter((g["truth"], g["claude"]) for g in graded
                     if g["truth"] is not None and not g["claude_right"])
    return {"slices": {f"{k}:{v}": stats(rows) for (k, v), rows in sorted(by.items())},
            "claude_misses": misses.most_common(10)}


def grade(conn, sidecar, model: str, labels: list[str], period_fn, sample_ids=None) -> dict:
    """Grade this model's answers. With sample_ids, only those crops: the sample pick() defines,
    so answers left over from an earlier, differently drawn sample do not leak into the grade."""
    opinions = {r["detection_id"]: dict(r) for r in sidecar.execute(
        "SELECT * FROM opinions WHERE model = ? AND outcome = 'answered'", (model,))}
    if sample_ids is not None:
        opinions = {k: v for k, v in opinions.items() if k in sample_ids}
    ids = sorted(opinions)
    det_rows = []
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        det_rows += [dict(r) for r in conn.execute(
            f"SELECT {_CANDIDATE_COLS} FROM detections WHERE species_verified IS NOT NULL "
            f"AND id IN ({','.join('?' * len(chunk))})", chunk)]
    report = summarize(grade_rows(det_rows, opinions, labels, period_fn))
    t_in, t_out, n_all = sidecar.execute(
        "SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), COUNT(*) "
        "FROM opinions WHERE model = ?", (model,)).fetchone()
    report.update(model=model, asked=n_all, graded_answers=len(opinions),
                  visits=len({r["visit_id"] for r in det_rows if r["visit_id"] is not None}),
                  input_tokens=t_in, output_tokens=t_out,
                  usd=_usd(model, t_in, t_out),
                  outcomes=dict(sidecar.execute(
                      "SELECT outcome, COUNT(*) FROM opinions WHERE model = ? GROUP BY outcome",
                      (model,)).fetchall()))
    return report


def print_report(rep: dict) -> None:
    usd = f"${rep['usd']:.2f}" if rep["usd"] is not None else "unknown price"
    print(f"Second opinion: {rep['model']}  ({rep['asked']} asked, outcomes {rep['outcomes']}, "
          f"{rep['input_tokens']:,} in / {rep['output_tokens']:,} out tokens, about {usd})")
    print(f"Grading {rep['graded_answers']} answers in the sample, from {rep['visits']} visits.\n")

    def f(x):
        return "  -  " if x is None else f"{x:.3f}"
    print(f"{'slice':<16}{'n':>5}{'BioCLIP':>9}{'Claude':>8}{'high n':>8}{'high':>7}"
          f"{'agree n':>9}{'agree':>7}{'rejected':>10}{'echoed':>8}")
    for name, s in rep["slices"].items():
        print(f"{name:<16}{s['n']:>5}{f(s['bioclip_acc']):>9}{f(s['claude_acc']):>8}"
              f"{s['claude_high_n']:>8}{f(s['claude_high_acc']):>7}{s['agree_n']:>9}"
              f"{f(s['agree_acc']):>7}{s['rejected_n']:>10}{s['rejected_claude_echoed']:>8}")
    print("\nn, accuracies: crops whose true species is known. high: Claude said 'high' certainty.\n"
          "agree: Claude and BioCLIP said the same thing. rejected: a human said BioCLIP was wrong\n"
          "without naming the animal; echoed: Claude repeated BioCLIP's wrong label on one of them.")
    if rep["claude_misses"]:
        print("\nClaude's commonest misses (truth -> Claude):")
        for (truth, said), n in rep["claude_misses"]:
            print(f"  {n:>4}  {truth} -> {said}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    cfg = config.CONFIG
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--graded", action="store_true",
                      help="ask about uncertain crops a human has already labelled (the eval set)")
    mode.add_argument("--recent", type=float, metavar="HOURS",
                      help="ask about unreviewed uncertain crops from the last HOURS")
    mode.add_argument("--grade", action="store_true", help="report Claude vs BioCLIP")
    mode.add_argument("--nightly", action="store_true",
                      help="the nightly batch's shadow run; does nothing unless "
                           "second_opinion_nightly is on")
    p.add_argument("--model", default=cfg.second_opinion_model)
    p.add_argument("--limit", type=int, default=None,
                   help="most crops in the sample (asking: default second_opinion_max_crops; "
                        "grading: the whole capped pool)")
    p.add_argument("--per-visit", type=int, default=PER_VISIT,
                   help="most crops from any one visit in a graded sample (0 = no cap)")
    p.add_argument("--max-conf", type=float, default=cfg.second_opinion_max_conf,
                   help="only crops BioCLIP named below this confidence")
    p.add_argument("--seed", type=int, default=0, help="--graded sample seed")
    p.add_argument("--dry-run", action="store_true", help="count and estimate cost; send nothing")
    p.add_argument("--db", default=str(cfg.db_path))
    args = p.parse_args(argv)
    if args.nightly:
        if not getattr(cfg, "second_opinion_nightly", False):
            print("[second-opinion] nightly run is off (second_opinion_nightly); nothing sent.")
            return 0
        # 26 h, not 24: the batch start moves with sunset, and answered crops are never re-sent.
        args.recent = NIGHTLY_HOURS
        args.model = cfg.second_opinion_nightly_model
        if args.limit is None:
            args.limit = cfg.second_opinion_nightly_max_crops

    conn = db.connect_readonly(args.db)
    if conn is None:
        print(f"No database at {args.db}.", file=sys.stderr)
        return 1
    conn.execute("PRAGMA query_only = ON")
    side_path = Path(args.db).with_name("second_opinion.db")
    # crop_path is relative to the project root, which is where backyard.db lives; resolving from
    # the database rather than this file keeps a run from a worktree pointed at the rig's crops.
    root = Path(args.db).resolve().parent
    labels = allowed_labels()
    # A dry run writes nothing, not even an empty sidecar beside the live database.
    if args.dry_run and not side_path.exists():
        sidecar = sqlite3.connect(":memory:")
        sidecar.executescript(SIDECAR_SCHEMA)
    else:
        sidecar = open_sidecar(side_path)

    if args.grade:
        from eval import _daynight_fn
        period_fn, _ = _daynight_fn(cfg)
        pool = candidates(conn, graded=True, max_conf=args.max_conf)
        sample = {r["id"] for r in pick(pool, limit=args.limit, already=set(), graded=True,
                                         seed=args.seed, per_visit=args.per_visit)}
        print_report(grade(conn, sidecar, args.model, labels, period_fn, sample_ids=sample))
        return 0

    limit = cfg.second_opinion_max_crops if args.limit is None else args.limit
    rows = candidates(conn, graded=args.graded, max_conf=args.max_conf, recent_hours=args.recent)
    todo = pick(rows, limit=limit, already=already_asked(sidecar, args.model),
                graded=args.graded, seed=args.seed, per_visit=args.per_visit)
    t_in, t_out, usd, unreadable = estimate_cost(todo, args.model, root)
    cost = f"about ${usd:.2f}" if usd is not None else "price unknown for this model"
    print(f"[second-opinion] {len(rows)} candidate crops below {args.max_conf}; sending {len(todo)} "
          f"to {args.model} (~{t_in:,} in / ~{t_out:,} out tokens, {cost}).")
    if unreadable:
        print(f"[second-opinion] {unreadable} of those crops will not open under {root}.")
    if todo and unreadable == len(todo):
        print("[second-opinion] none of the crops open; is --db in the rig's folder?",
              file=sys.stderr)
        return 1
    if args.dry_run or not todo:
        return 0
    tally = ask_all(todo, client=make_client(cfg), model=args.model, labels=labels,
                    sidecar=sidecar, root=root)
    print(f"[second-opinion] done: {dict(tally)}. Answers are in {side_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
