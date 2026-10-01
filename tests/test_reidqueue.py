"""
reidqueue.py -- the re-ID review queue as data, tested without a server.

The filters and the funnel take a matcher, so most of these use a tiny stand-in with the four
attributes they read; the end-to-end ones build a throwaway DB from the `conn` fixture with
hand-built 3-D embeddings (as test_web.py does), never the live backyard.db.
"""
from __future__ import annotations

import json
import threading
import urllib.request
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np

import config
import db
import individuals
import reidqueue
import web

NOW = datetime.now().astimezone()


def _at(days_ago: float, minutes: float = 0.0) -> str:
    return (NOW - timedelta(days=days_ago) + timedelta(minutes=minutes)).isoformat()


def _unit(*xs) -> np.ndarray:
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def _cfg(db_path, **kw):
    return replace(config.Config(), db_path=db_path, **kw)


def _visit(conn, *, vec, days_ago, source=db.SOURCE_GLASS_DOOR_CAM, n=3, minutes=0.0,
           rep=True, backslash=False):
    ids = []
    for i in range(n):
        sep = "\\" if backslash else "/"
        ids.append(db.insert_detection(
            conn, timestamp=_at(days_ago, minutes + i), source=source, detection_class="animal",
            confidence=0.9, bbox=(0, 0, 10, 10), frame_w=100, frame_h=100,
            crop_path=f"crops{sep}{source}-{days_ago}-{minutes}-{i}.jpg", species="raccoon",
            crop_quality=10.0))
    vid = db.insert_visit(
        conn, source=source, species="raccoon", individual_id=None,
        started_at=_at(days_ago, minutes), ended_at=_at(days_ago, minutes + n),
        detection_count=n, max_confidence=0.9, representative_detection_id=ids[0] if rep else None)
    db.assign_visit(conn, ids, vid)
    for d in ids:
        db.insert_embedding(conn, d, individuals.EMBED_MODEL, len(vec),
                            np.asarray(vec, dtype=np.float32).tobytes())
    conn.commit()
    return vid


class _Matcher:
    """Just what queue_filter / funnel / adjacency_context read."""

    def __init__(self, *, protos=(), confirmed=None, auto=None, rejected=(), multi=(),
                 started=None, templates=()):
        self.protos = {v: None for v in protos}
        self.confirmed = dict(confirmed or {})
        self.auto = dict(auto or {})
        self.rejected = set(rejected)
        self._multi = set(multi)
        self.visit_started = dict(started or {})
        self._templates = list(templates)

    def is_multi(self, vid):
        return vid in self._multi

    def templates(self):
        return self._templates


# ---- queue_filter --------------------------------------------------------------------
def test_queue_filter_keeps_the_callers_order_and_falls_back_to_recent():
    rows = [{"id": 3}, {"id": 1}, {"id": 2}]
    m = _Matcher()
    assert reidqueue.queue_filter(config.Config(), m, rows, "recent", {}) == ("recent", rows)
    assert reidqueue.queue_filter(config.Config(), m, rows, "nonsense", {}) == ("recent", rows)


def test_unreviewed_auto_skips_what_a_human_already_judged():
    rows = [{"id": i} for i in (5, 4, 3, 2, 1)]
    m = _Matcher(auto={5: "Stan", 4: "Stan", 3: "Notch", 2: "Notch"},
                 confirmed={4: "Stan"}, rejected={3})
    mode, out = reidqueue.queue_filter(config.Config(), m, rows, "unreviewed_auto", {})
    assert mode == "unreviewed_auto"
    assert [v["id"] for v in out] == [5, 2]


# ---- adjacency_context ---------------------------------------------------------------
def test_adjacency_context_takes_the_nearer_named_neighbour_within_the_hour():
    m = _Matcher(confirmed={1: "Stan", 3: "Notch"},
                 started={1: _at(1, 0), 2: _at(1, 10), 3: _at(1, 15), 4: _at(1, 200)})
    by_source = {"door": [4, 3, 2, 1]}            # newest first
    ctx = reidqueue.adjacency_context(by_source, m, {"id": 2, "source": "door"})
    assert ctx == {"name": "Notch", "visit_id": 3, "gap_s": 300, "direction": "before"}
    # 4 is 185 min after Notch's visit: outside the window, so nothing.
    assert reidqueue.adjacency_context(by_source, m, {"id": 4, "source": "door"}) is None
    assert reidqueue.adjacency_context({}, m, {"id": 2, "source": "door"}) is None


# ---- funnel --------------------------------------------------------------------------
def test_funnel_counts_per_source_and_the_addressable_gap():
    pool = [{"id": i, "source": "door"} for i in (1, 2, 3, 4, 5)] + [{"id": 6, "source": "trail"}]
    m = _Matcher(protos=(1, 2, 3, 4, 5, 6), confirmed={1: "Stan"}, auto={2: "Stan"},
                 rejected={3}, multi={4}, templates=[("Stan", 1, None)])
    f = reidqueue.funnel(m, pool, {}, {"door"})
    assert (f["visits"], f["with_prototype"], f["confirmed"], f["templates"]) == (6, 6, 1, 1)
    assert (f["multi_animal"], f["rejected"], f["auto_named"]) == (1, 1, 1)
    assert f["addressable"] == 3                   # 2, 5, 6: not confirmed, tombstoned or multi
    assert [s["source"] for s in f["by_source"]] == ["door", "trail"]   # busiest first
    door, trail = f["by_source"]
    assert (door["visits"], door["confirmed"], door["auto"], door["addressable"]) == (5, 1, 1, 2)
    assert door["templated"] is True and trail["templated"] is False


# ---- queue, end to end on a throwaway DB ---------------------------------------------
def _small_corpus(conn):
    ids = {"stan_t": _visit(conn, vec=_unit(1, 0, 0), days_ago=5),
           "notch_t": _visit(conn, vec=_unit(0, 1, 0), days_ago=1, backslash=True)}
    db.label_visit(conn, ids["stan_t"], "Stan")
    db.label_visit(conn, ids["notch_t"], "Notch")
    ids["probe"] = _visit(conn, vec=_unit(1, 0.05, 0), days_ago=0.5)
    ids["norep"] = _visit(conn, vec=_unit(0, 0.1, 1), days_ago=0.4, rep=False)
    ids["auto"] = _visit(conn, vec=_unit(1, 0, 0), days_ago=0.3, backslash=True)
    db.label_visit(conn, ids["auto"], "Stan", source="auto")
    return ids


def test_queue_is_newest_first_and_mode_filters_page_within_the_window(conn, db_path):
    ids = _small_corpus(conn)
    cfg = _cfg(db_path)
    out = reidqueue.queue(conn, cfg, since_h=0)
    starts = [v["started_at"] for v in out["queue"]]
    assert starts == sorted(starts, reverse=True) and len(starts) == len(ids)
    assert [v["visit_id"] for v in reidqueue.queue(conn, cfg, mode="unreviewed_auto")["queue"]] \
        == [ids["auto"]]
    page = reidqueue.queue(conn, cfg, since_h=0, limit=2, offset=2)
    assert [v["visit_id"] for v in page["queue"]] == [v["visit_id"] for v in out["queue"][2:4]]
    assert page["n_matched"] == len(ids)
    assert out["funnel"]["confirmed"] == 2 and out["funnel"]["auto_named"] == 1


# ---- web.py only wraps: each endpoint serves the pure function's JSON, byte for byte ------
def test_the_queue_endpoint_serves_reidqueue_output_byte_for_byte(conn, db_path):
    ids = _small_corpus(conn)
    cfg = _cfg(db_path, web_host="127.0.0.1", web_port=0)
    web.clear_api_cache()
    server = web.make_server(cfg, {cfg.source: web.FrameBuffer()},
                             {cfg.source: web.CameraControlBridge()})
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        port = server.server_address[1]
        for path, kw in (("/api/reid/queue?since_h=0", {"since_h": 0}),
                         ("/api/reid/queue?mode=unreviewed_auto", {"mode": "unreviewed_auto"}),
                         ("/api/reid/queue?since_h=0&limit=2&offset=1",
                          {"since_h": 0, "limit": 2, "offset": 1})):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as r:
                body = r.read()
            ro = db.connect_readonly(db_path)
            try:
                want = reidqueue.queue(ro, cfg, species=cfg.reid_species, **kw)
            finally:
                ro.close()
            assert body == json.dumps(want).encode("utf-8"), path
        vid = ids["probe"]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/reid/dossier?visit_id={vid}",
                                    timeout=30) as r:
            body = r.read()
        ro = db.connect_readonly(db_path)
        try:
            assert body == json.dumps(reidqueue.dossier(ro, cfg, vid)).encode("utf-8")
        finally:
            ro.close()
    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=5)
        web.clear_api_cache()


def test_no_database_is_an_empty_queue_not_an_error(tmp_path):
    cfg = _cfg(tmp_path / "missing.db")
    out = web._reid_queue(cfg)
    assert out["queue"] == [] and out["funnel"] == {}
    assert web._reid_dossier(cfg, 1) == {}
