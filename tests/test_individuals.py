"""
Tests for individuals.py -- the suggest-confirm loop -- and db.label_visit / individual_source.

Pure logic, no models: embeddings are tiny hand-built unit vectors, so who-matches-whom is
exact by construction. The geometry here mirrors the real finding the module encodes: crops
from one visit cluster tight; a different animal points a different way; a prototype is the
quality-weighted average of a visit's best crops.

Conventions match the suite: throwaway DB via the conn fixture, synthetic detections at
controlled timestamps, columns read by name (sqlite3.Row).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

import config
import db
import individuals
from individuals import (VisitMatcher, co_present_frames, iou, prototype, rank_templates)

BASE = datetime(2026, 6, 10, 21, 0, 0, tzinfo=datetime.now().astimezone().tzinfo)


def _ts(minutes: float = 0.0) -> str:
    return (BASE + timedelta(minutes=minutes)).isoformat()


def _unit(*xs) -> np.ndarray:
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Pure helpers.
# ---------------------------------------------------------------------------

def test_iou_disjoint_identical_and_partial():
    a = (0, 0, 10, 10)
    assert iou(a, (20, 20, 30, 30)) == 0.0
    assert iou(a, a) == pytest.approx(1.0)
    # Partial overlap: inter 50, union 100 + 150 - 50 = 200 -> 1/4.
    assert iou(a, (5, 0, 20, 10)) == pytest.approx(50 / 200)


def test_prototype_uses_top_quality_crops_and_renormalizes():
    # Three vectors; the two HIGH-quality ones agree, the junk crop points elsewhere.
    X = np.stack([_unit(0, 1), _unit(1, 0), _unit(1, 0)])
    qualities = [0.1, 0.9, 0.8]
    p = prototype(X, qualities, top_k=2)          # only the two quality crops vote
    assert p @ _unit(1, 0) == pytest.approx(1.0)
    assert np.linalg.norm(p) == pytest.approx(1.0)


def test_prototype_handles_none_quality():
    X = np.stack([_unit(1, 0), _unit(0, 1)])
    p = prototype(X, [None, 0.5], top_k=1)        # None ranks as 0 -> the scored crop wins
    assert p @ _unit(0, 1) == pytest.approx(1.0)


def test_rank_templates_is_nearest_visit_per_individual():
    q = _unit(1, 0)
    temps = [
        ("Stan", 101, _unit(1, 0.2)),     # Stan's best template
        ("Stan", 102, _unit(0, 1)),       # Stan's other look-mode (far) -- must not drag him down
        ("Notch", 201, _unit(0.5, 1)),
    ]
    ranked = rank_templates(q, temps)
    assert [r[0] for r in ranked] == ["Stan", "Notch"]
    name, sim, via = ranked[0]
    assert via == 101                      # matched via the NEAREST visit, not a centroid
    assert sim > ranked[1][1]


def test_co_present_frames_requires_separated_boxes():
    near = (0, 0, 10, 10)
    double_box = (1, 1, 11, 11)            # ~IoU 0.68 with near: one animal boxed twice
    far = (50, 50, 60, 60)
    rows = [
        ("t1", near), ("t1", far),         # two bodies            -> counts
        ("t2", near), ("t2", double_box),  # detector double-box   -> doesn't
        ("t3", near),                      # solo                  -> doesn't
    ]
    assert co_present_frames(rows) == 1


# ---------------------------------------------------------------------------
# DB round-trip: label_visit, individual_source, confirmed_visit_labels.
# ---------------------------------------------------------------------------

def _det(conn, *, minutes, species="raccoon", confidence=0.9, bbox=(0, 0, 10, 10)):
    return db.insert_detection(
        conn, timestamp=_ts(minutes), source=db.SOURCE_GLASS_DOOR_CAM,
        detection_class="animal", confidence=confidence, bbox=bbox,
        frame_w=100, frame_h=100, crop_path=f"crops/{minutes}.jpg", species=species,
        crop_quality=10.0,
    )


def _visit(conn, det_ids, *, species="raccoon", start_min=0.0, end_min=1.0):
    vid = db.insert_visit(
        conn, source=db.SOURCE_GLASS_DOOR_CAM, species=species, individual_id=None,
        started_at=_ts(start_min), ended_at=_ts(end_min), detection_count=len(det_ids),
        max_confidence=0.9, representative_detection_id=det_ids[0])
    db.assign_visit(conn, det_ids, vid)
    conn.commit()
    return vid


def _embed(conn, det_id, vec):
    db.insert_embedding(conn, det_id, individuals.EMBED_MODEL, len(vec),
                        np.asarray(vec, dtype=np.float32).tobytes())
    conn.commit()


def test_label_visit_stamps_species_matching_crops_only(conn):
    d1 = _det(conn, minutes=0)
    d2 = _det(conn, minutes=0.5)
    crow = _det(conn, minutes=0.7, species="American crow")
    vid = _visit(conn, [d1, d2, crow])
    n = db.label_visit(conn, vid, "Stan")
    assert n == 2                          # the stray crow keeps its own identity
    rows = {r["id"]: r for r in conn.execute(
        "SELECT id, individual_id, individual_source FROM detections")}
    assert rows[d1]["individual_id"] == "Stan" and rows[d1]["individual_source"] == "human"
    assert rows[crow]["individual_id"] is None
    assert conn.execute("SELECT individual_id FROM visits WHERE id=?", (vid,)).fetchone()[0] == "Stan"
    # Clearing wipes both the name and the source.
    db.label_visit(conn, vid, None)
    r = conn.execute("SELECT individual_id, individual_source FROM detections WHERE id=?",
                     (d1,)).fetchone()
    assert r["individual_id"] is None and r["individual_source"] is None


def test_apply_visit_label_species_and_name_together(conn):
    d1, d2 = _det(conn, minutes=0), _det(conn, minutes=0.5)
    vid = _visit(conn, [d1, d2])
    r = db.apply_visit_label(conn, visit_id=vid, name="Stan", species="raccoon")
    assert r["detections"] == 2
    rows = {x["id"]: x for x in conn.execute(
        "SELECT id, species, species_verified, species_source, individual_id, individual_source "
        "FROM detections")}
    for d in (d1, d2):
        assert rows[d]["species"] == "raccoon" and rows[d]["species_verified"] == 1
        assert rows[d]["species_source"] == "human"
        assert rows[d]["individual_id"] == "Stan" and rows[d]["individual_source"] == "human"
    v = conn.execute("SELECT species, individual_id FROM visits WHERE id=?", (vid,)).fetchone()
    assert v["species"] == "raccoon" and v["individual_id"] == "Stan"


def test_apply_visit_label_verify_only_keeps_species(conn):
    d = _det(conn, minutes=0, species="raccoon")
    vid = _visit(conn, [d])
    db.apply_visit_label(conn, visit_id=vid, verify=True)
    row = conn.execute("SELECT species, species_verified FROM detections WHERE id=?", (d,)).fetchone()
    assert row["species"] == "raccoon" and row["species_verified"] == 1


def test_apply_visit_label_name_scopes_to_dominant_species(conn):
    r1, r2 = _det(conn, minutes=0), _det(conn, minutes=0.4)
    crow = _det(conn, minutes=0.6, species="American crow")
    vid = _visit(conn, [r1, r2, crow])
    db.apply_visit_label(conn, visit_id=vid, name="Stan")        # name only, no species change
    ind = {x["id"]: x["individual_id"] for x in conn.execute("SELECT id, individual_id FROM detections")}
    assert ind[r1] == "Stan" and ind[r2] == "Stan" and ind[crow] is None


def test_apply_visit_label_by_time_range(conn):
    d1, d2 = _det(conn, minutes=1), _det(conn, minutes=2)
    outside = _det(conn, minutes=100)
    r = db.apply_visit_label(conn, source=db.SOURCE_GLASS_DOOR_CAM, start=_ts(0), end=_ts(10),
                             name="Notch", species="raccoon")
    assert r["detections"] == 2
    ind = {x["id"]: x["individual_id"] for x in conn.execute("SELECT id, individual_id FROM detections")}
    assert ind[d1] == "Notch" and ind[d2] == "Notch" and ind[outside] is None


def test_apply_visit_label_clear_name(conn):
    d = _det(conn, minutes=0)
    vid = _visit(conn, [d])
    db.apply_visit_label(conn, visit_id=vid, name="Stan")
    db.apply_visit_label(conn, visit_id=vid, name=None)
    row = conn.execute("SELECT individual_id, individual_source FROM detections WHERE id=?", (d,)).fetchone()
    assert row["individual_id"] is None and row["individual_source"] is None


def test_confirmed_visit_labels_reads_only_human_labels(conn):
    d1, d2 = _det(conn, minutes=0), _det(conn, minutes=10)
    v1 = _visit(conn, [d1], start_min=0)
    v2 = _visit(conn, [d2], start_min=10)
    db.label_visit(conn, v1, "Stan")
    db.set_individual_bulk(conn, [d2], "raccoon_c01", source="cluster")  # placeholder, not human
    labels = db.confirmed_visit_labels(conn, "raccoon")
    assert labels == {v1: "Stan"}
    assert v2 not in labels


def test_set_individual_bulk_records_source(conn):
    d = _det(conn, minutes=0)
    db.set_individual_bulk(conn, [d], "raccoon_c03", source="cluster")
    r = conn.execute("SELECT individual_source FROM detections WHERE id=?", (d,)).fetchone()
    assert r["individual_source"] == "cluster"
    db.rename_individual(conn, "raccoon_c03", "Notch")   # naming a cluster is a human act
    r = conn.execute("SELECT individual_id, individual_source FROM detections WHERE id=?",
                     (d,)).fetchone()
    assert r["individual_id"] == "Notch" and r["individual_source"] == "human"


# ---------------------------------------------------------------------------
# VisitMatcher end-to-end on synthetic vectors.
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    """A test config with a low min-crops gate so 3-crop synthetic visits form prototypes."""
    c = config.Config()
    c.reid_proto_min_crops = 3
    return c


def _three_crop_visit(conn, vec, *, start_min, jitter=0.01):
    """A visit of three near-identical crops pointing along `vec` (one animal lingering)."""
    base = np.asarray(vec, dtype=np.float32)
    ids = []
    for k in range(3):
        d = _det(conn, minutes=start_min + k * 0.2)
        v = base + jitter * np.array([k, -k, k], dtype=np.float32)[: len(base)]
        _embed(conn, d, v / np.linalg.norm(v))
        ids.append(d)
    return _visit(conn, ids, start_min=start_min, end_min=start_min + 1)


def test_suggest_ranks_confirmed_individual_and_flags_novelty(conn, cfg):
    stan_a = _three_crop_visit(conn, [1, 0, 0], start_min=0)
    stan_b = _three_crop_visit(conn, [1, 0.05, 0], start_min=60)     # Stan again, next session
    stranger = _three_crop_visit(conn, [0, 0, 1], start_min=120)     # orthogonal = someone new
    db.label_visit(conn, stan_a, "Stan")

    m = VisitMatcher(conn, "raccoon", cfg)
    s = m.suggest(stan_b)
    assert s["candidates"][0]["name"] == "Stan"
    assert s["candidates"][0]["via_visit"] == stan_a
    assert s["candidates"][0]["similarity"] > 0.9
    assert not s["novel"]

    s2 = m.suggest(stranger)
    assert s2["novel"]                     # far from every template -> possibly someone new

    s3 = m.suggest(stan_a)
    assert s3["confirmed_as"] == "Stan"    # already-confirmed visits read back as confirmed


def test_suggest_degrades_honestly_without_templates_or_embeddings(conn, cfg):
    lone = _three_crop_visit(conn, [1, 0, 0], start_min=0)
    m = VisitMatcher(conn, "raccoon", cfg)
    s = m.suggest(lone)
    assert s["candidates"] == [] and "no confirmed individuals" in s["note"]

    bare = _visit(conn, [_det(conn, minutes=300)], start_min=300)    # no embeddings at all
    m2 = VisitMatcher(conn, "raccoon", cfg)
    assert "embed" in (m2.suggest(bare)["note"] or "")


def test_multi_animal_visit_is_flagged_and_excluded_from_templates(conn, cfg):
    cfg.reid_co_presence_min = 1
    # One visit where every frame holds TWO separated raccoons (the pair at the dish).
    ids = []
    for k in range(3):
        a = _det(conn, minutes=k * 0.2, bbox=(0, 0, 10, 10))
        b = _det(conn, minutes=k * 0.2, bbox=(50, 50, 60, 60))
        for d, vec in ((a, [1, 0, 0]), (b, [0, 1, 0])):
            v = np.asarray(vec, dtype=np.float32)
            _embed(conn, d, v / np.linalg.norm(v))
        ids += [a, b]
    pair_visit = _visit(conn, ids, start_min=0, end_min=1)
    db.label_visit(conn, pair_visit, "Notch")        # naming it is allowed...

    m = VisitMatcher(conn, "raccoon", cfg)
    assert m.is_multi(pair_visit)
    assert m.suggest(pair_visit)["multi"]
    assert m.templates() == []                       # ...but its blended prototype never teaches


def _clip_with_tracks(conn, *, start_min, end_min, n_sustained, n_fragments=0,
                      source=db.SOURCE_GLASS_DOOR_CAM):
    """A clip spanning [start_min, end_min] with `n_sustained` tracklets of 40 boxes (>= the
    SUSTAINED_HITS gate) plus `n_fragments` short 6-box tracklets (dropout noise)."""
    cid = db.insert_clip(conn, source=source, clip_path=f"clips/{start_min}.mp4",
                         started_at=_ts(start_min), ended_at=_ts(end_min), fps=10.0,
                         width=1280, height=720, frame_count=300, detection_count=20,
                         max_confidence=0.9)
    tracklets = [{"track_json": "[]", "n_hits": 40, "features": {}} for _ in range(n_sustained)]
    tracklets += [{"track_json": "[]", "n_hits": 6, "features": {}} for _ in range(n_fragments)]
    db.insert_clip_tracks(conn, clip_id=cid, model="m", n_samples=300, tracklets=tracklets)
    return cid


def test_clip_co_presence_counts_only_sustained_pairs(conn, cfg):
    import individuals
    v = _visit(conn, [_det(conn, minutes=0)], start_min=0, end_min=10)
    _clip_with_tracks(conn, start_min=1, end_min=2, n_sustained=2)               # real pair
    _clip_with_tracks(conn, start_min=3, end_min=4, n_sustained=2, n_fragments=3)  # pair + noise
    _clip_with_tracks(conn, start_min=5, end_min=6, n_sustained=1, n_fragments=4)  # one animal, fragmented
    counts = individuals.clip_co_presence_by_visit(conn, "raccoon")
    assert counts.get(v) == 2          # two clips had 2+ sustained tracks; the fragmented solo didn't


def test_is_multi_via_clip_evidence_needs_corroboration(conn, cfg):
    # One clip with a pair is NOT enough (could be a fragmentation artifact); two clips are.
    v1 = _visit(conn, [_det(conn, minutes=0)], start_min=0, end_min=5)
    _clip_with_tracks(conn, start_min=1, end_min=2, n_sustained=2)
    v2 = _visit(conn, [_det(conn, minutes=20)], start_min=20, end_min=25)
    _clip_with_tracks(conn, start_min=21, end_min=22, n_sustained=2)
    _clip_with_tracks(conn, start_min=23, end_min=24, n_sustained=2)

    m = VisitMatcher(conn, "raccoon", cfg)
    assert m.clip_co_presence.get(v1) == 1 and not m.is_multi(v1)   # single clip -> not flagged
    assert m.clip_co_presence.get(v2) == 2 and m.is_multi(v2)       # corroborated -> flagged
    assert m.suggest(v2)["co_present_clips"] == 2


def test_clip_co_presence_excluded_from_templates(conn, cfg):
    # A confirmed visit that clips reveal as a pair must not become an appearance template.
    vid = _three_crop_visit(conn, [1, 0, 0], start_min=0)
    db.label_visit(conn, vid, "Stan")
    _clip_with_tracks(conn, start_min=0, end_min=1, n_sustained=2)
    _clip_with_tracks(conn, start_min=0.3, end_min=0.6, n_sustained=2)
    m = VisitMatcher(conn, "raccoon", cfg)
    assert m.is_multi(vid)
    assert m.templates() == []         # blended-pair visit excluded despite being confirmed


def test_pose_groups_split_distinct_postures(conn, cfg):
    # Stan's crops fall into two embedding directions = two characteristic poses.
    ids = []
    for k in range(6):
        d = _det(conn, minutes=k * 0.2)
        vec = [1, 0.02 * k, 0] if k < 3 else [0, 1, 0.02 * (k - 3)]   # pose A vs pose B
        v = np.asarray(vec, dtype=np.float32)
        _embed(conn, d, v / np.linalg.norm(v))
        ids.append(d)
    db.set_individual_bulk(conn, ids, "Stan", source="human")
    groups = individuals.pose_groups(conn, "Stan", distance=0.35, min_group=3)
    assert len(groups) == 2
    assert sorted(g["n"] for g in groups) == [3, 3]
    assert all(g["rep_crops"] for g in groups)


def test_pose_groups_empty_for_unknown_individual(conn, cfg):
    assert individuals.pose_groups(conn, "Nobody") == []


def test_clips_for_individual_attributes_overlapping_clips(conn, cfg):
    # A labelled Stan visit with a clip inside its window -> the clip is attributed to Stan.
    vid = _visit(conn, [_det(conn, minutes=0)], start_min=0, end_min=5)
    db.label_visit(conn, vid, "Stan")
    cid = db.insert_clip(conn, source=db.SOURCE_GLASS_DOOR_CAM, clip_path="clips/s.mp4",
                         started_at=_ts(1), ended_at=_ts(2), fps=10.0, width=1280, height=720,
                         frame_count=300, detection_count=12, max_confidence=0.9)
    # A clip well outside any Stan visit must NOT be attributed.
    db.insert_clip(conn, source=db.SOURCE_GLASS_DOOR_CAM, clip_path="clips/other.mp4",
                   started_at=_ts(500), ended_at=_ts(501), fps=10.0, width=1280, height=720,
                   frame_count=300, detection_count=12, max_confidence=0.9)
    clips = individuals.clips_for_individual(conn, "Stan")
    assert [c["clip_id"] for c in clips] == [cid]
    assert clips[0]["clip_path"] == "clips/s.mp4" and clips[0]["duration_s"] == 60.0
    assert individuals.clips_for_individual(conn, "Nobody") == []


def test_refit_sorts_unconfirmed_into_fits_and_novel(conn, cfg):
    stan_a = _three_crop_visit(conn, [1, 0, 0], start_min=0)
    _three_crop_visit(conn, [1, 0.05, 0], start_min=60)      # Stan again -> should fit Stan
    _three_crop_visit(conn, [0.98, 0.1, 0], start_min=120)   # Stan-ish -> should fit Stan
    _three_crop_visit(conn, [0, 1, 0], start_min=180)        # someone else -> novel
    _three_crop_visit(conn, [0, 0.98, 0.1], start_min=240)   # same someone else -> novel, same group
    db.label_visit(conn, stan_a, "Stan")

    m = VisitMatcher(conn, "raccoon", cfg)
    r = m.refit(distance=0.45)
    assert r["untemplated"] == []
    assert r["fits"]["Stan"] and r["n_fit"] == 2             # the two Stan-like unconfirmed visits
    assert all(x["similarity"] >= cfg.reid_novel_threshold for x in r["fits"]["Stan"])
    assert r["n_novel"] == 2
    # the two "someone else" visits land together as one candidate-new-individual group
    assert any(len(g["visits"]) == 2 for g in r["novel_groups"])


def test_refit_flags_individual_confirmed_only_on_pair_visit(conn, cfg):
    cfg.reid_co_presence_min = 1
    # Stan gets a clean solo visit; Elliot is named only on a two-animal visit.
    stan = _three_crop_visit(conn, [1, 0, 0], start_min=0)
    db.label_visit(conn, stan, "Stan")
    ids = []
    for k in range(3):
        a = _det(conn, minutes=60 + k * 0.2, bbox=(0, 0, 10, 10))
        b = _det(conn, minutes=60 + k * 0.2, bbox=(50, 50, 60, 60))
        for d, vec in ((a, [0, 1, 0]), (b, [0, 0, 1])):
            v = np.asarray(vec, dtype=np.float32)
            _embed(conn, d, v / np.linalg.norm(v))
        ids += [a, b]
    pair = _visit(conn, ids, start_min=60, end_min=61)
    db.label_visit(conn, pair, "Elliot")

    m = VisitMatcher(conn, "raccoon", cfg)
    r = m.refit()
    assert "Elliot" in r["untemplated"]      # named only on a pair visit -> no clean template
    assert "Elliot" not in r["fits"]         # ...so nothing can be matched to Elliot
    assert "Stan" not in r["untemplated"]    # Stan has a clean solo template


def _clip_tracklets_in_visit(conn, vid, vecs, *, start_min, end_min):
    """A clip inside visit `vid`'s window with one sustained, embedded tracklet per vector."""
    cid = db.insert_clip(conn, source=db.SOURCE_GLASS_DOOR_CAM, clip_path=f"clips/{start_min}.mp4",
                         started_at=_ts(start_min), ended_at=_ts(end_min), fps=10.0, width=1280,
                         height=720, frame_count=300, detection_count=12, max_confidence=0.9)
    db.insert_clip_tracks(conn, clip_id=cid, model="MDV6", n_samples=300,
                          tracklets=[{"track_json": "[]", "n_hits": 40, "features": {}} for _ in vecs])
    tids = [r["id"] for r in conn.execute(
        "SELECT id FROM clip_tracks WHERE clip_id=? ORDER BY track_idx", (cid,))]
    for tid, vec in zip(tids, vecs):
        v = np.asarray(vec, dtype=np.float32)
        db.insert_clip_track_embedding(conn, track_id=tid, model=individuals.EMBED_MODEL,
                                       dim=len(vec), embedding=(v / np.linalg.norm(v)).tobytes(),
                                       n_frames=8, rep_crop=f"clip_crops/track_{tid}.jpg")
    return cid, tids


def test_unblend_visit_separates_two_animals(conn, cfg):
    vid = _visit(conn, [_det(conn, minutes=0)], start_min=0, end_min=10)
    _clip_tracklets_in_visit(conn, vid, [[1, 0, 0], [1, .05, 0], [.98, .1, 0], [0, 1, 0], [0, .97, .1]],
                             start_min=1, end_min=2)
    r = individuals.unblend_visit(conn, vid, distance=0.45)
    assert r["n_tracklets"] == 5
    assert sorted((g["n"] for g in r["groups"]), reverse=True) == [3, 2]
    big = r["groups"][0]
    assert len(big["track_ids"]) == 3 and len(big["rep_crops"]) == 3 and big["label"] is None
    # Naming the cluster lands on the tracklets and reads back as the group's label.
    db.set_clip_track_individual(conn, big["track_ids"], "Notch")
    r2 = individuals.unblend_visit(conn, vid, distance=0.45)
    assert [g["label"] for g in r2["groups"] if g["n"] == 3][0] == "Notch"


def test_unblend_visit_handles_no_tracklets(conn, cfg):
    vid = _visit(conn, [_det(conn, minutes=0)], start_min=0, end_min=10)
    r = individuals.unblend_visit(conn, vid)
    assert r["groups"] == [] and "tracklet" in (r["note"] or "")


def test_set_clip_track_individual_round_trip(conn, cfg):
    vid = _visit(conn, [_det(conn, minutes=0)], start_min=0, end_min=10)
    _, tids = _clip_tracklets_in_visit(conn, vid, [[1, 0, 0], [0, 1, 0]], start_min=1, end_min=2)
    assert db.set_clip_track_individual(conn, tids, "Elliot") == 2
    rows = conn.execute(f"SELECT individual_id, individual_source FROM clip_tracks "
                        f"WHERE id IN ({','.join('?' * len(tids))})", tids).fetchall()
    assert all(r["individual_id"] == "Elliot" and r["individual_source"] == "human" for r in rows)
    db.set_clip_track_individual(conn, tids, None)
    rows = conn.execute(f"SELECT individual_id, individual_source FROM clip_tracks "
                        f"WHERE id IN ({','.join('?' * len(tids))})", tids).fetchall()
    assert all(r["individual_id"] is None and r["individual_source"] is None for r in rows)


def test_bootstrap_groups_split_two_animals(conn, cfg):
    a1 = _three_crop_visit(conn, [1, 0, 0], start_min=0)
    a2 = _three_crop_visit(conn, [1, 0.05, 0], start_min=60)
    b1 = _three_crop_visit(conn, [0, 0, 1], start_min=120)
    m = VisitMatcher(conn, "raccoon", cfg)
    groups = m.bootstrap_groups(distance=0.45)
    sets = sorted([sorted(g["visits"]) for g in groups], key=len, reverse=True)
    assert sets == [sorted([a1, a2]), [b1]]
    assert groups[0]["cohesion"] > 0.9
