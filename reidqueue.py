"""
The re-ID review queue as data: which visits the Individuals tab offers for naming, the evidence
on each card, the confirmed cast, the funnel, and the one-visit dossier.

Pure functions over an open read-only connection -- no HTTP, no caching. web.py opens the
connection, caches per endpoint and serves the JSON. individuals (numpy) is imported inside the
functions that need it, so importing this module costs no more than importing web.
"""
from __future__ import annotations

from datetime import timedelta

import db
import stats

# ---- review-queue modes (the queue is the only source of templates) --------------------
# `recent` is what the dashboard has always shown and stays the default; the other three are
# opt-in filters over the WHOLE species pool, because a recent-only window reaches a few dozen
# visits out of hundreds and human confirmations are the only thing that grows the template set.
# All four are computed from stored columns + the appearance prototypes, so nothing here is
# calibrated to a camera position and a camera move can't silently stale them out.
QUEUE_MODES = ("recent", "unreviewed_auto", "ambiguous", "stale")

#: How far back the review queue reaches by DEFAULT. Two nights is what a person can actually
#: still remember seeing; the whole corpus (500+ visits) is a wall, not a work list.
DEFAULT_QUEUE_WINDOW_H = 48

#: How far either side of a visit to look for the SAME MOMENT on another camera. Ten minutes,
#: because the trail cam's clock is set by hand each cycle and drifts a few minutes against the
#: rig's, and because a raccoon that leaves the door frame reaches the far camera within that.
CROSS_CAMERA_PAD_S = 600
#: Cap on how many crops one dossier ships. The whole point of the one-at-a-time flow is that the
#: eye gets EVERYTHING, but a 2,000-crop visit would hang the tab; sharpest-first means the cap
#: only ever removes the crops least worth looking at.
DOSSIER_MAX_CROPS = 60

ADJACENT_CONTEXT_S = 3600.0   # the measured window; display only, never a ranking input

_IN_CHUNK = 500   # ids per IN (...) query: under SQLite's pre-3.32 cap of 999 bound parameters


def _rows_in(conn, sql: str, ids) -> list:
    """`sql` (one `{}` where the IN list goes) run over the distinct non-None `ids`, chunked."""
    ids = list(dict.fromkeys(i for i in ids if i is not None))
    out = []
    for k in range(0, len(ids), _IN_CHUNK):
        part = ids[k:k + _IN_CHUNK]
        out += conn.execute(sql.format(",".join("?" * len(part))), part).fetchall()
    return out


def _rep_crops(conn, det_ids) -> dict:
    """{detection id: crop path, forward slashes} for each id that has a crop. One query per 500
    ids instead of one per visit; an id with no row or an empty path is simply absent."""
    return {r["id"]: r["crop_path"].replace("\\", "/")
            for r in _rows_in(conn, "SELECT id, crop_path FROM detections WHERE id IN ({})",
                              det_ids)
            if r["crop_path"]}


def appearance_rank(matcher, vid):
    """(name, similarity, lead) for one visit from the APPEARANCE templates only -- the same two
    numbers auto_assign gates on, so a mode built on them shows exactly what the auto tier sees.

    Two things it deliberately does NOT do. It takes no behaviour or adjacency input: the queue's
    filters must not become a second, sloppier ranker (two-axis principle). And it ranks through
    `templates_for`, the SOURCE-GUARDED pool -- a probe is only compared with templates from its
    own camera, so a visit from an untemplated camera ranks against nothing and drops out of the
    filters rather than being sorted by noise."""
    import individuals
    if vid not in matcher.protos:
        return None
    temps = (matcher.templates_for(vid) if hasattr(matcher, "templates_for")
             else matcher.templates())
    if not temps:
        return None
    ranked = individuals.rank_templates(matcher.protos[vid], temps)
    if not ranked:
        return None
    name, sim, _via = ranked[0]
    return name, sim, sim - (ranked[1][1] if len(ranked) > 1 else 0.0)


def queue_filter(cfg, matcher, rows, mode, freshness):
    """Keep the visits `mode` addresses, newest first (the caller's row order is preserved).

      recent           -- everything; today's behaviour, unchanged.
      unreviewed_auto  -- machine-named and neither kept nor rejected by a human. These wear a
                          name nobody confirmed; each one is a single click from being a template
                          or a tombstone.
      ambiguous        -- the appearance match clears the similarity bar but its lead over the
                          runner-up INDIVIDUAL is under the margin. This is precisely what
                          auto_assign refuses (its `ambiguous` skip bucket), which makes it the
                          highest-information click on offer: the machine cannot resolve it and
                          says so; the eye usually can.
      stale            -- unreviewed visits whose top candidate is an individual whose newest
                          human template has aged past reid_queue_stale_days. Confirming one
                          refreshes the template the next week of matching stands on.
    """
    if mode not in QUEUE_MODES:
        mode = "recent"
    if mode == "recent":
        return mode, list(rows)
    # 'ambiguous' mirrors the auto tier's OWN two bars when the tier is live, so the mode shows
    # exactly the visits it refuses on the margin (its `ambiguous` skip bucket). While the tier is
    # disabled -- reid_auto_threshold 0.0, the shipped default, and the state this mode is most
    # useful in -- both bars go inert, so fall back to the novelty cut and the queue's own margin.
    auto_on = (cfg.reid_auto_threshold or 0) > 0
    clears = max(cfg.reid_auto_threshold or 0.0, cfg.reid_novel_threshold)
    margin = (cfg.reid_auto_margin if auto_on and (cfg.reid_auto_margin or 0) > 0
              else cfg.reid_queue_ambiguous_margin)
    out = []
    for v in rows:
        vid = v["id"]
        if mode == "unreviewed_auto":
            if vid in matcher.auto and vid not in matcher.confirmed and vid not in matcher.rejected:
                out.append(v)
            continue
        # ambiguous / stale both look at an UNREVIEWED visit's appearance ranking.
        if vid in matcher.confirmed or vid in matcher.rejected or matcher.is_multi(vid):
            continue
        r = appearance_rank(matcher, vid)
        if r is None:
            continue
        name, sim, lead = r
        if mode == "ambiguous":
            if sim >= clears and lead < margin:
                out.append(v)
        elif mode == "stale":
            days = (freshness.get(name) or {}).get("days_since_template")
            if sim >= cfg.reid_novel_threshold and days is not None \
                    and days >= cfg.reid_queue_stale_days:
                out.append(v)
    return mode, out


def adjacency_context(rows_by_source, matcher, v):
    """DISPLAY-ONLY: the nearest same-source visit within an hour that already carries a human
    name ("started 6 min after the visit you named Stan").

    Measured on this corpus, adjacent same-source visits inside 60 min are the same individual
    67-77% of the time against a 28% base rate -- genuinely worth showing a human. It is NOT
    worth scoring: as a ranking input under session blocking it fired 3 times and was wrong 3
    times, and adjacency + a same-night template are the same confound. So this rides in its own
    payload field, is never consulted by appearance_rank, and never touches the sort order."""
    lst = rows_by_source.get(v["source"]) or []
    try:
        i = lst.index(v["id"])
    except ValueError:
        return None
    best = None
    # `lst` is newest-first, so i+1 is the OLDER neighbour -- this visit started AFTER it.
    for j, direction in ((i + 1, "after"), (i - 1, "before")):
        if j < 0 or j >= len(lst):
            continue
        other = lst[j]
        name = matcher.confirmed.get(other)
        if not name:
            continue
        a, b = matcher.visit_started.get(v["id"]), matcher.visit_started.get(other)
        ta, tb = db.parse_local(a) if a else None, db.parse_local(b) if b else None
        if ta is None or tb is None:
            continue
        gap = abs((ta - tb).total_seconds())
        if gap <= ADJACENT_CONTEXT_S and (best is None or gap < best["gap_s"]):
            best = {"name": name, "visit_id": other, "gap_s": int(gap), "direction": direction}
    return best


def funnel(matcher, pool, source_of, template_sources) -> dict:
    """Where this species' visits go, counted live: total -> has a prototype -> human-confirmed
    -> usable (solo) template, plus what the auto tier is looking at.

    Published on the panel because "automation contributes nothing" is useless as a feeling and
    actionable as a number. `addressable` is the auto tier's own arithmetic -- visits with a
    prototype that are not confirmed, not multi-animal and not tombstoned -- so the gap between
    it and `auto_named` is exactly the automation's shortfall. The per-source split matters for
    the same reason: a source with no templates can never be named from templates, and folding
    it into one denominator quietly understates every coverage figure."""
    protos = set(matcher.protos)
    confirmed = set(matcher.confirmed)
    multi = {v for v in protos if v not in confirmed and matcher.is_multi(v)}
    addressable = {v for v in protos
                   if v not in confirmed and v not in matcher.rejected and v not in multi}
    by_source: dict = {}
    for v in pool:
        s = by_source.setdefault(v["source"], {"source": v["source"], "visits": 0, "confirmed": 0,
                                               "auto": 0, "addressable": 0, "templated": False})
        s["visits"] += 1
        vid = v["id"]
        s["confirmed"] += int(vid in confirmed)
        s["auto"] += int(vid in matcher.auto)
        s["addressable"] += int(vid in addressable)
    for s in by_source.values():
        s["templated"] = s["source"] in template_sources
    return {
        "visits": len(pool),
        "with_prototype": len(protos),
        "confirmed": len(confirmed),
        "templates": len(matcher.templates()),
        "multi_animal": len(multi),
        "addressable": len(addressable),
        "auto_named": len(matcher.auto),
        "rejected": len(matcher.rejected),
        "by_source": sorted(by_source.values(), key=lambda s: -s["visits"]),
    }


def dossier(conn, cfg, visit_id: int, species: str = "raccoon") -> dict:
    """EVERYTHING known about ONE visit -- the payload behind the one-at-a-time review flow.

    The queue card is deliberately small: it has to render 30 of them. This is the opposite --
    one visit, every crop, every clip, and the same moment as seen by any OTHER camera. That last
    part is the reason this exists rather than the card just growing:

      a trail-cam visit CANNOT be appearance-matched (measured: trail-cam prototypes score a
      median 0.249 against every glass-door template, and trail-cam-to-trail-cam similarity is
      flat -- there is no identity structure to threshold). So the only evidence that can ever
      name one is a human noticing that the glass door saw the same animal in the same minutes.
      109 of 521 glass-door raccoon visits have a trail-cam visit within CROSS_CAMERA_PAD_S, so
      that pairing is available for a fifth of the corpus and is currently shown nowhere.

    Returns {} for an unknown visit."""
    import individuals

    v = conn.execute(
        """SELECT id, source, species, started_at, ended_at, detection_count,
                  representative_detection_id, individual_id
           FROM visits WHERE id = ?""", (int(visit_id),)).fetchone()
    if v is None:
        return {}
    species = v["species"] or species
    matcher = individuals.VisitMatcher(conn, species, cfg)
    all_clips = stats.load_clips(conn)

    def _crop(det_id):
        if det_id is None:
            return None
        r = conn.execute("SELECT crop_path FROM detections WHERE id = ?",
                         (det_id,)).fetchone()
        return r["crop_path"].replace("\\", "/") if r and r["crop_path"] else None

    def _evidence(row):
        """Crops + clips for one visit, sharpest crop first."""
        crops = [{"path": r["crop_path"].replace("\\", "/"),
                  "at": r["timestamp"], "conf": r["confidence"]}
                 for r in conn.execute(
                     """SELECT crop_path, timestamp, confidence FROM detections
                        WHERE visit_id = ? AND crop_path IS NOT NULL
                        ORDER BY crop_quality DESC, confidence DESC LIMIT ?""",
                     (row["id"], DOSSIER_MAX_CROPS)).fetchall()]
        clips = [stats._clip_out(c) for c in stats.clips_overlapping(
            all_clips, row["source"], db.parse_local(row["started_at"]),
            db.parse_local(row["ended_at"]))]
        return crops, clips

    crops, clips = _evidence(v)
    s = matcher.suggest(v["id"])

    # THE SAME MOMENT ON ANOTHER CAMERA. Compared as instants via db.parse_local, never as
    # ISO strings: the two sources' rows can carry different UTC offsets (the trail cam's
    # timestamps are reconstructed at import) and a string compare would silently mis-order
    # them across a DST boundary.
    start, end = db.parse_local(v["started_at"]), db.parse_local(v["ended_at"])
    neighbours = []
    if start and end:
        pad = timedelta(seconds=CROSS_CAMERA_PAD_S)
        lo, hi = start - pad, end + pad
        day = (lo - timedelta(days=1)).strftime("%Y-%m-%d")
        for n in conn.execute(
                """SELECT id, source, species, started_at, ended_at, detection_count,
                          representative_detection_id, individual_id
                   FROM visits WHERE id != ? AND source != ? AND started_at >= ?
                   ORDER BY started_at""",
                (v["id"], v["source"], day)).fetchall():
            ns, ne = db.parse_local(n["started_at"]), db.parse_local(n["ended_at"])
            if not ns or not ne or ns > hi or ne < lo:
                continue
            ncrops, nclips = _evidence(n)
            neighbours.append({
                "visit_id": n["id"], "source": n["source"], "species": n["species"],
                "started_at": n["started_at"], "ended_at": n["ended_at"],
                "n_crops": n["detection_count"], "individual_id": n["individual_id"],
                "rep_crop": _crop(n["representative_detection_id"]),
                "crops": ncrops, "clips": nclips,
                "offset_s": int((ns - start).total_seconds()),
            })

    # The answer buttons: who the human could plausibly say. Confirmed cast, busiest first,
    # each carrying how stale its template is so "who is this?" and "who needs a fresh
    # template?" are the same glance.
    freshness = individuals.template_freshness(matcher)
    counts: dict = {}
    for vid, name in matcher.confirmed.items():
        counts[name] = counts.get(name, 0) + 1
    statuses = db.individual_statuses(conn)
    by_key = {str(k).strip().casefold(): st for k, st in statuses.items()}
    cast = []
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        f = freshness.get(name) or {}
        st = by_key.get(str(name).strip().casefold()) or {}
        cast.append({"name": name, "n_visits": n,
                     "n_templates": f.get("n_templates", 0),
                     "days_since_template": f.get("days_since_template"),
                     "status": st.get("status") or "resident"})

    return {
        "visit_id": v["id"], "source": v["source"], "species": species,
        "started_at": v["started_at"], "ended_at": v["ended_at"],
        "n_crops": v["detection_count"],
        "rep_crop": _crop(v["representative_detection_id"]),
        "crops": crops, "clips": clips,
        "confirmed_as": s["confirmed_as"], "auto_as": s["auto_as"],
        "rejected": v["id"] in matcher.rejected,
        "candidates": s["candidates"], "clip_candidates": s["clip_candidates"],
        "novel": s["novel"], "multi": s["multi"],
        "co_present_frames": s["co_present_frames"],
        "co_present_clips": s["co_present_clips"],
        "cross_source": s.get("cross_source", False),
        "n_embedded": s["n_embedded"], "note": s["note"],
        "neighbours": neighbours, "cast": cast,
    }


def _dwell(a, b):
    try:
        from datetime import datetime as _dt
        return int((_dt.fromisoformat(b) - _dt.fromisoformat(a)).total_seconds())
    except (ValueError, TypeError):
        return 0


def queue(conn, cfg, species: str = "raccoon", limit: int = 30, offset: int = 0,
          mode: str = "recent", since_h: int = DEFAULT_QUEUE_WINDOW_H) -> dict:
    """The Individuals tab's review queue: visits of `species` with a WHO suggestion each
    (nearest confirmed visit / novelty flag / 2+-animals badge), the confirmed cast, and -- while
    nothing is confirmed yet -- the cold-start visit-groups to name first. The heavy lifting
    (prototype matching over the embedding matrix) lives in individuals.VisitMatcher and is
    rebuilt per call, which is fine for a tab that loads on demand.

    `mode` picks WHICH visits (see QUEUE_MODES); `offset`/`limit` PAGINATE within that mode.
    Pagination rather than a bigger limit on purpose: the per-visit work below (clip overlap +
    crop strip) and the matcher rebuild are what cost, so a page stays cheap however deep the
    filter reaches."""
    import individuals

    matcher = individuals.VisitMatcher(conn, species, cfg)

    cards = []
    all_clips = stats.load_clips(conn)   # once; overlap-match each visit's footage in memory
    # The WHOLE species pool, newest first. The mode filters it and offset/limit page it --
    # the expensive per-visit work below runs on the page only.
    pool = conn.execute(
        """SELECT id, source, started_at, ended_at, detection_count, representative_detection_id
           FROM visits WHERE species = ? ORDER BY started_at DESC""", (species,)).fetchall()
    # THE WINDOW, applied before the mode filter so "12 ambiguous" means 12 in the window the
    # reader is looking at, not 12 somewhere in two months. since_h <= 0 means "all of it",
    # which is what the UI's "load older" walks out to.
    n_all = len(pool)
    cutoff = None
    if since_h and since_h > 0:
        newest = max((db.parse_local(v["started_at"]) for v in pool), default=None)
        if newest is not None:
            # Anchored on the NEWEST VISIT, not on wall-clock now: after a quiet spell (or a
            # rig that was down, which happens here) a now-anchored window shows an empty
            # queue and reads as "nothing to review" when the truth is "nothing last night".
            cutoff = newest - timedelta(hours=float(since_h))
            pool = [v for v in pool
                    if (db.parse_local(v["started_at"]) or newest) >= cutoff]
    freshness = individuals.template_freshness(matcher)
    mode, matched = queue_filter(cfg, matcher, pool, mode, freshness)
    rows = matched[offset:offset + limit]
    # Per-source visit order (newest first, same as `pool`), for the display-only temporal
    # context chip. Built from the whole pool so a page boundary can't hide a neighbour.
    by_source: dict = {}
    source_of: dict = {}
    for v in pool:
        by_source.setdefault(v["source"], []).append(v["id"])
        source_of[v["id"]] = v["source"]
    # Which sources the appearance templates actually come from. A visit from any OTHER source
    # cannot be matched -- not "is probably a new animal", CANNOT BE MATCHED -- and the card
    # says so instead of showing a meaningless top-1. Derived from the data, never a source
    # name in code, so it stays true when a camera is added, moved or retired.
    template_sources = {source_of.get(tvid) for _n, tvid, _p in matcher.templates()}
    template_sources.discard(None)
    reps = _rep_crops(conn, [v["representative_detection_id"] for v in rows])
    for v in rows:
        s = matcher.suggest(v["id"])
        # Evidence for the human doing the naming: the clips that rolled during this visit
        # (busiest first, click to play) + a strip of its sharpest crops for extra angles.
        vclips = [stats._clip_out(c) for c in stats.clips_overlapping(
            all_clips, v["source"],
            db.parse_local(v["started_at"]), db.parse_local(v["ended_at"]))]
        rep = reps.get(v["representative_detection_id"])
        # Still one query per card: each strip is its own time range, and a page is <= 100.
        vcrops = [r["crop_path"] for r in conn.execute(
            "SELECT crop_path FROM detections WHERE source = ? AND species = ? "
            "AND timestamp >= ? AND timestamp <= ? AND crop_path IS NOT NULL "
            "ORDER BY crop_quality DESC LIMIT 7",
            (v["source"], species, v["started_at"], v["ended_at"])).fetchall()]
        vcrops = [c for c in vcrops if c != rep][:6]   # "other" crops -> drop the hero thumb
        # A visit from a source no template comes from is structurally unmatchable. Measured:
        # every trail-cam raccoon prototype scores a median 0.249 / max 0.363 against every
        # glass-door template, and trail-cam-to-trail-cam similarity is flat (0.510 near in
        # time vs 0.514 far) -- there is no identity structure to threshold. Saying "possibly
        # someone new" there states something about the ANIMAL when the truth is about the
        # CAMERA, so the flag rides here and the card swaps the wording.
        cross_source = bool(template_sources) and v["source"] not in template_sources
        cards.append({
            "visit_id": v["id"], "started_at": v["started_at"], "source": v["source"],
            "dwell_s": _dwell(v["started_at"], v["ended_at"]),
            "n_crops": v["detection_count"],
            "rep_crop": rep,
            "clips": vclips, "crops": vcrops,
            "confirmed_as": s["confirmed_as"], "auto_as": s["auto_as"],
            "rejected": v["id"] in matcher.rejected,
            # Every candidate carries the LAPSE state of the name it proposes. A suggestion is
            # only as good as the freshest template behind it, and a two-week-old one is at or
            # below "just say the commonest name" -- so the card can stop presenting those two
            # cases as though they were the same offer. It is a LABEL, never a filter: gating
            # on template age was measured and rejected (coverage fell, wrong names rose).
            "candidates": [dict(c, lapse=(freshness.get(c.get("name")) or {}).get("lapse"))
                           for c in s["candidates"]],
            "clip_candidates": s["clip_candidates"],
            "novel": s["novel"], "multi": s["multi"],
            "cross_source": cross_source,
            "co_present_frames": s["co_present_frames"],
            "co_present_clips": s["co_present_clips"],
            # DISPLAY ONLY -- see adjacency_context. Never read by any ranking path, and
            # only offered where there is still a call to make (an already-confirmed visit
            # doesn't need a hint, it needs to stay out of the way).
            "context": None if s["confirmed_as"] else adjacency_context(by_source, matcher, v),
            "n_embedded": s["n_embedded"], "note": s["note"], "species": species,
        })

    # The confirmed cast, with how much template material backs each name -- plus how many
    # visits the nightly pass has auto-named to it (pending the human's glance).
    cast: dict = {}
    for vid, name in matcher.confirmed.items():
        c = cast.setdefault(name, {"name": name, "n_visits": 0, "n_auto": 0, "last_seen": None})
        c["n_visits"] += 1
        started = matcher.visit_started.get(vid)
        if started and (c["last_seen"] is None or started > c["last_seen"]):
            c["last_seen"] = started
    for vid, name in matcher.auto.items():
        if name in cast:               # auto only ever assigns confirmed names, but be safe
            cast[name]["n_auto"] += 1
    # TEMPLATE FRESHNESS: how old this individual's newest USABLE (confirmed solo) template is.
    # n_visits counts confirmations; a confirmation on a 2+-animal visit is NOT a template, so
    # the two numbers differ and the difference is the point -- an individual can look busy and
    # still be unrecognisable. See individuals.template_freshness for the decay curve.
    # THE ROSTER: what the human has said about who still lives here. A departed individual is
    # shown, ranked and suggested exactly as before -- the flag only tells the reader why the
    # stale-template warning next to it is not a to-do, and gives the auto tier the one fact it
    # can never infer (individuals.VisitMatcher.is_departed).
    statuses = db.individual_statuses(conn)
    by_key = {str(k).strip().casefold(): v for k, v in statuses.items()}
    for name, c in cast.items():
        f = freshness.get(name) or {}
        c["n_templates"] = f.get("n_templates", 0)
        c["newest_template"] = f.get("newest_template")
        c["days_since_template"] = f.get("days_since_template")
        c["lapse"] = f.get("lapse") or individuals.identity_lapse(None, 0, cfg=cfg)
        st = by_key.get(str(name).strip().casefold()) or {}
        c["status"] = st.get("status") or "resident"
        c["departed_on"] = st.get("effective_date")
        c["status_note"] = st.get("note")

    def _group_crops(visit_ids):
        if not visit_ids:
            return []
        reps = conn.execute(
            f"""SELECT representative_detection_id FROM visits
                WHERE id IN ({','.join('?' * len(visit_ids))})""", visit_ids).fetchall()
        crops = _rep_crops(conn, [r["representative_detection_id"] for r in reps])
        return [c for c in (crops.get(r["representative_detection_id"]) for r in reps) if c]

    has_templates = bool(matcher.templates())

    # Cold start only: until something is confirmed, naming visit-GROUPS beats naming visits.
    bootstrap = []
    if not has_templates:
        for g in matcher.bootstrap_groups():
            bootstrap.append({**g, "crops": _group_crops(g["visits"])})

    # Once there's a cast, RE-FIT: sort the unconfirmed remainder into "looks like <name>"
    # buckets (for bulk-confirm) + candidate-new-individual groups, and flag any confirmed
    # individual that has no clean solo template yet.
    refit = None
    if has_templates:
        r = matcher.refit()
        # The visit may have been renumbered/removed by a rebuild since refit() listed it, so a
        # missing row is a None crop, not an error.
        rep_of = {row["id"]: row["representative_detection_id"] for row in _rows_in(
            conn, "SELECT id, representative_detection_id FROM visits WHERE id IN ({})",
            [x["visit_id"] for lst in r["fits"].values() for x in lst])}
        fit_crops = _rep_crops(conn, rep_of.values())
        fits = {name: {"visits": [{**x, "rep_crop": fit_crops.get(rep_of.get(x["visit_id"]))}
                                  for x in lst]}
                for name, lst in r["fits"].items()}
        refit = {
            "fits": fits,
            "novel_groups": [{**g, "crops": _group_crops(g["visits"])}
                             for g in r["novel_groups"]],
            "untemplated": r["untemplated"], "n_fit": r["n_fit"], "n_novel": r["n_novel"]}

    # Heads-up when suggestions are running blind: high-conf crops still missing vectors.
    backlog = conn.execute(
        """SELECT COUNT(*) FROM detections d
           WHERE d.species = ? AND d.confidence >= ? AND NOT EXISTS
             (SELECT 1 FROM detection_embeddings e
              WHERE e.detection_id = d.id AND e.model = ?)""",
        (species, cfg.reid_suggest_min_conf, individuals.EMBED_MODEL)).fetchone()[0]

    return {"species": species, "queue": cards,
            "cast": sorted(cast.values(), key=lambda c: -c["n_visits"]),
            "bootstrap": bootstrap, "refit": refit, "unembedded": backlog,
            "novel_threshold": cfg.reid_novel_threshold,
            "mode": mode, "modes": list(QUEUE_MODES),
            "offset": offset, "limit": limit, "n_matched": len(matched),
            # The window, so the UI can say what it is hiding rather than just hiding it.
            "since_h": since_h, "n_in_window": len(pool), "n_all": n_all,
            "window_from": cutoff.isoformat() if cutoff else None,
            "stale_days": cfg.reid_queue_stale_days,
            "funnel": funnel(matcher, pool, source_of, template_sources)}
