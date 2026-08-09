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

import json
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

def _det(conn, *, minutes, species="raccoon", confidence=0.9, bbox=(0, 0, 10, 10),
         source=db.SOURCE_GLASS_DOOR_CAM):
    return db.insert_detection(
        conn, timestamp=_ts(minutes), source=source,
        detection_class="animal", confidence=confidence, bbox=bbox,
        frame_w=100, frame_h=100, crop_path=f"crops/{source}_{minutes}.jpg", species=species,
        crop_quality=10.0,
    )


def _visit(conn, det_ids, *, species="raccoon", start_min=0.0, end_min=1.0,
           source=db.SOURCE_GLASS_DOOR_CAM):
    vid = db.insert_visit(
        conn, source=source, species=species, individual_id=None,
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
# Live sightings: naming the visit AS IT HAPPENS (the dashboard Live tab).
# ---------------------------------------------------------------------------

def test_record_live_sighting_solo_stamps_the_span(conn):
    """One name = a live solo confirm: it stamps the span's crops (feeding the templates)."""
    d1, d2 = _det(conn, minutes=0), _det(conn, minutes=0.5)
    r = db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Stan"],
                                span_start=_ts(0), span_end=_ts(1))
    assert r["multi"] is False and r["stamped"] == 2 and r["names"] == ["Stan"]
    ind = {x["id"]: (x["individual_id"], x["individual_source"])
           for x in conn.execute("SELECT id, individual_id, individual_source FROM detections")}
    assert ind[d1] == ("Stan", "human") and ind[d2] == ("Stan", "human")
    # The sighting itself is logged.
    assert db.recent_live_sightings(conn)[0]["names"] == ["Stan"]


def test_record_live_sighting_pair_does_not_stamp(conn):
    """Two+ names = co-presence: the names are logged, but NO single id is stamped on both
    animals (the documented pair gotcha that contaminates a template)."""
    d1, d2 = _det(conn, minutes=0), _det(conn, minutes=0.5)
    r = db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Notch", "Elliot"],
                                span_start=_ts(0), span_end=_ts(1))
    assert r["multi"] is True and r["stamped"] == 0 and r["names"] == ["Notch", "Elliot"]
    ids = [x["individual_id"] for x in conn.execute("SELECT individual_id FROM detections")]
    assert ids == [None, None]
    rec = db.recent_live_sightings(conn)[0]
    assert rec["names"] == ["Notch", "Elliot"] and rec["stamped"] == 0


def test_record_live_sighting_dedupes_case_insensitively(conn):
    _det(conn, minutes=0)
    r = db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM,
                                names=["Notch", "notch", " Notch "], span_start=_ts(0), span_end=_ts(1))
    # Collapses to one name -> treated as solo (and so it stamps).
    assert r["names"] == ["Notch"] and r["multi"] is False and r["stamped"] == 1


def test_record_live_sighting_requires_a_name(conn):
    r = db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["", "  "])
    assert r.get("error") and r["sighting_id"] is None


def test_record_live_sighting_without_span_logs_but_does_not_stamp(conn):
    """No active visit (no span) -> the note is still recorded, just nothing to stamp."""
    _det(conn, minutes=0)
    r = db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Stan"])
    assert r["stamped"] == 0 and r["sighting_id"] is not None
    assert [x["individual_id"] for x in conn.execute("SELECT individual_id FROM detections")] == [None]


def test_recent_live_sightings_newest_first_and_limited(conn):
    for i in range(3):
        db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=[f"A{i}"])
    out = db.recent_live_sightings(conn, limit=2)
    assert [s["names"][0] for s in out] == ["A2", "A1"]   # newest first, capped at the limit


def test_co_present_sighting_names_unions_overlapping(conn):
    src = db.SOURCE_GLASS_DOOR_CAM
    db.record_live_sighting(conn, source=src, names=["Notch", "Elliot"],
                            span_start=_ts(0), span_end=_ts(10))
    db.record_live_sighting(conn, source=src, names=["Stan"],
                            span_start=_ts(100), span_end=_ts(110))     # a different window
    # A visit window inside the pair's span picks up the logged pair...
    co = db.co_present_sighting_names(conn, src, _ts(2), _ts(8))
    assert co["names"] == ["Notch", "Elliot"] and co["n"] == 2 and co["observed_at"]
    # ...the Stan window picks up only Stan...
    assert db.co_present_sighting_names(conn, src, _ts(101), _ts(109))["names"] == ["Stan"]
    # ...and a window overlapping neither is empty.
    assert db.co_present_sighting_names(conn, src, _ts(50), _ts(60))["names"] == []


# ---------------------------------------------------------------------------
# VisitMatcher end-to-end on synthetic vectors.
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    """A test config with a low min-crops gate so 3-crop synthetic visits form prototypes."""
    c = config.Config()
    c.reid_proto_min_crops = 3
    return c


def _three_crop_visit(conn, vec, *, start_min, jitter=0.01, source=db.SOURCE_GLASS_DOOR_CAM):
    """A visit of three near-identical crops pointing along `vec` (one animal lingering)."""
    base = np.asarray(vec, dtype=np.float32)
    ids = []
    for k in range(3):
        d = _det(conn, minutes=start_min + k * 0.2, source=source)
        v = base + jitter * np.array([k, -k, k], dtype=np.float32)[: len(base)]
        _embed(conn, d, v / np.linalg.norm(v))
        ids.append(d)
    return _visit(conn, ids, start_min=start_min, end_min=start_min + 1, source=source)


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


# ---------------------------------------------------------------------------
# Group labels ("Stan + Kits"): one archive name over several bodies. The stamp is wanted; the
# template is poison. 2026-08-08: before this plumbing, a family night logged as one group string
# read as SOLO everywhere -- 4 of the live DB's 9 group visits carried 0-1 co-present frames and
# were one nightly embed away from becoming blended pseudo-individual templates.
# ---------------------------------------------------------------------------

def test_is_group_label_convention():
    assert db.is_group_label("Stan + Kits")
    assert db.is_group_label("CutiePie + Kits")
    assert not db.is_group_label("Stan")
    assert not db.is_group_label("The Dude")        # spaces alone are not the marker
    assert not db.is_group_label("Stan+Kits")       # the separator is " + ", as typed
    assert not db.is_group_label(None)
    assert not db.is_group_label("")


def test_record_live_sighting_group_string_stamps_and_counts_multi(conn):
    """A single group-string name does BOTH: stamps the span (the family convention -- the
    archive label is wanted) and reports multi (several bodies -- must feed is_multi)."""
    d1, d2 = _det(conn, minutes=0), _det(conn, minutes=0.5)
    r = db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Stan + Kits"],
                                span_start=_ts(0), span_end=_ts(1))
    assert r["stamped"] == 2                        # the stamp happened...
    assert r["multi"] is True and r["group"] is True   # ...and it counts as several animals
    ind = {x["id"]: x["individual_id"]
           for x in conn.execute("SELECT id, individual_id FROM detections")}
    assert ind[d1] == "Stan + Kits" and ind[d2] == "Stan + Kits"
    # The sighting registers as a multi-animal span for every consumer of the sighting arm.
    spans = individuals.multi_name_sighting_spans(conn)
    assert len(spans) == 1 and spans[0][0] == db.SOURCE_GLASS_DOOR_CAM


def test_viewer_sighting_records_testimony_without_stamping(conn):
    """stamp=False (the viewer tier): the sighting lands attributed in live_sightings and no
    crop is written -- a guest's enthusiasm can never contaminate a template."""
    d1 = _det(conn, minutes=0)
    r = db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Stan"],
                                span_start=_ts(0), span_end=_ts(1),
                                stamp=False, labeled_by="niece")
    assert r["stamped"] == 0 and r["sighting_id"] is not None
    ind = conn.execute("SELECT individual_id FROM detections WHERE id = ?", (d1,)).fetchone()[0]
    assert ind is None                                   # nothing stamped
    row = conn.execute("SELECT names, labeled_by, stamped FROM live_sightings").fetchone()
    assert json.loads(row[0]) == ["Stan"] and row[1] == "niece" and row[2] == 0


def test_operator_stamp_carries_attribution_onto_crops(conn):
    d1 = _det(conn, minutes=0)
    db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Stan"],
                            span_start=_ts(0), span_end=_ts(1), labeled_by="matt")
    row = conn.execute("SELECT individual_id, labeled_by FROM detections WHERE id = ?",
                       (d1,)).fetchone()
    assert row[0] == "Stan" and row[1] == "matt"


def test_plain_solo_sighting_is_not_a_multi_span(conn):
    _det(conn, minutes=0)
    db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Stan"],
                            span_start=_ts(0), span_end=_ts(1))
    assert individuals.multi_name_sighting_spans(conn) == []


def test_group_labelled_visit_never_templates_even_without_co_presence(conn, cfg):
    """The dangerous case measured on the live DB: a family span whose sparse stills caught one
    body at a time (zero co-present frames, so is_multi's detector arms stay silent) but whose
    NAME says several animals. The name alone must keep it out of the template pool."""
    family = _three_crop_visit(conn, [1, 0, 0], start_min=0)         # looks solo to the stills
    probe = _three_crop_visit(conn, [1, 0.05, 0], start_min=60)
    db.label_visit(conn, family, "Pedro + Kits")

    m = VisitMatcher(conn, "raccoon", cfg)
    assert m.templates() == []                       # the blend teaches nothing
    assert m.suggest(probe)["candidates"] == []      # and nothing ranks against it
    # The identity still exists for every human surface -- the label was not touched.
    assert m.confirmed[family] == "Pedro + Kits"


def test_group_sighting_flags_overlapping_visit_multi(conn, cfg):
    """A group live-sighting over a visit trips is_multi via the sighting arm, exactly as a
    two-name sighting does -- so the queue badge and embed --co-present both see the family."""
    family = _three_crop_visit(conn, [1, 0, 0], start_min=0)
    db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["CutiePie + Kits"],
                            span_start=_ts(0), span_end=_ts(1))
    m = VisitMatcher(conn, "raccoon", cfg)
    assert m.sighting_multi(family) and m.is_multi(family)
    assert family in individuals.co_present_visit_ids(conn)


def test_auto_assign_never_writes_a_group_name(conn, cfg):
    """With only a group-labelled 'template' on file, the auto tier has nothing to rank against
    -- a family name can never be auto-written onto a solo visit."""
    cfg.reid_auto_threshold, cfg.reid_auto_margin, cfg.reid_auto_min_templates = 0.5, 0.0, 1
    family = _three_crop_visit(conn, [1, 0, 0], start_min=0)
    lookalike = _three_crop_visit(conn, [1, 0.02, 0], start_min=60)  # would match, if allowed
    db.label_visit(conn, family, "Stan + Kits")

    m = VisitMatcher(conn, "raccoon", cfg)
    out = m.auto_assign(conn, dry_run=True)
    assert out["assigned"] == []
    assert out["skipped"].get("no_templates") == len(m.protos)


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


def _u(*xs):
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def _embedded_tracklet(conn, *, start_min, end_min, vec, n_hits=40, individual=None):
    """A clip with one sustained tracklet carrying a clip-space appearance vector (clipembed
    output), optionally pre-labelled with an individual (the un-blend output)."""
    cid = db.insert_clip(conn, source=db.SOURCE_GLASS_DOOR_CAM, clip_path=f"clips/c{start_min}.mp4",
                         started_at=_ts(start_min), ended_at=_ts(end_min), fps=10.0, width=1280,
                         height=720, frame_count=300, detection_count=10, max_confidence=0.9)
    db.insert_clip_tracks(conn, clip_id=cid, model="MDV6", n_samples=300,
                          tracklets=[{"track_json": "[]", "n_hits": n_hits, "features": {}}])
    tid = conn.execute("SELECT id FROM clip_tracks WHERE clip_id=?", (cid,)).fetchone()[0]
    v = _u(*vec)
    db.insert_clip_track_embedding(conn, track_id=tid, model=individuals.EMBED_MODEL,
                                   dim=len(v), embedding=v.tobytes(), n_frames=8)
    if individual:
        db.set_clip_track_individual(conn, [tid], individual)
    return tid


def test_clip_match_keeps_only_above_threshold():
    templates = {"Stan": (_u(1, 0, 0), 5), "Notch": (_u(0, 1, 0), 3)}
    r = individuals.clip_match(_u(0.95, 0.1, 0), templates, 0.4)
    assert r[0][0] == "Stan" and r[0][2] == 5 and r[0][1] > 0.9
    assert all(s >= 0.4 for _, s, _ in r)
    assert individuals.clip_match(_u(0, 0, 1), templates, 0.4) == []   # nothing close -> empty


def test_clip_templates_from_explicit_labels(conn, cfg):
    _embedded_tracklet(conn, start_min=0, end_min=1, vec=[1, 0, 0], individual="Elliot")
    _embedded_tracklet(conn, start_min=2, end_min=3, vec=[0.9, 0.1, 0], individual="Elliot")
    t = individuals.clip_templates(conn, {}, cfg=cfg)          # explicit-only (empty solo map)
    assert set(t) == {"Elliot"} and t["Elliot"][1] == 2
    assert t["Elliot"][0] @ _u(1, 0, 0) > 0.95


def test_clip_templates_solo_attribution(conn, cfg):
    d = _det(conn, minutes=0)
    vid = _visit(conn, [d], start_min=0, end_min=5)
    _embedded_tracklet(conn, start_min=1, end_min=2, vec=[0, 1, 0])   # overlaps the visit, unlabelled
    t = individuals.clip_templates(conn, {vid: "Stan"}, cfg=cfg)
    assert "Stan" in t and t["Stan"][1] == 1                          # attributed via the solo visit
    assert individuals.clip_templates(conn, {}, cfg=cfg) == {}        # explicit-only ignores it


def test_unblend_suggests_from_clip_templates(conn, cfg):
    cfg.reid_co_presence_min = 1
    vid = _visit(conn, [_det(conn, minutes=0)], start_min=0, end_min=5)
    for k in range(3):
        _embedded_tracklet(conn, start_min=0.2 * k, end_min=0.2 * k + 0.1, vec=[1, 0, 0.02 * k])
    r = individuals.unblend_visit(conn, vid, templates={"Stan": (_u(1, 0, 0), 10)}, cfg=cfg)
    assert r["groups"] and r["groups"][0]["suggestion"][0]["name"] == "Stan"
    # No templates -> no suggestions (the honest cold start).
    r0 = individuals.unblend_visit(conn, vid, cfg=cfg)
    assert all(g["suggestion"] == [] for g in r0["groups"])


def test_suggest_surfaces_clip_candidate_for_never_solo_individual(conn, cfg):
    # Elliot has only a CLIP template (explicit un-blend label), no still template. A new still
    # visit whose appearance matches Elliot's clip centroid must surface him as a clip_candidate
    # -- the whole point: the never-solo pair member becomes findable.
    _embedded_tracklet(conn, start_min=100, end_min=101, vec=[1, 0, 0], individual="Elliot")
    qv = _three_crop_visit(conn, [1, 0, 0], start_min=0)
    m = VisitMatcher(conn, "raccoon", cfg)
    s = m.suggest(qv)
    assert any(c["name"] == "Elliot" for c in s["clip_candidates"])
    assert s["candidates"] == []                                     # no STILL template for Elliot


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


def test_unblend_seeds_logged_pair_as_quick_picks_cold_start(conn, cfg):
    """A logged co-presence pair pre-fills both clusters for one-tap assignment, even with no
    template yet (the honest cold start: it can't say WHICH is which, so no elimination)."""
    vid = _visit(conn, [_det(conn, minutes=0)], start_min=0, end_min=10)
    _clip_tracklets_in_visit(conn, vid, [[1, 0, 0], [1, .05, 0], [.98, .1, 0], [0, 1, 0], [0, .97, .1]],
                             start_min=1, end_min=2)
    db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Notch", "Elliot"],
                            span_start=_ts(0), span_end=_ts(10))
    r = individuals.unblend_visit(conn, vid, distance=0.45)            # no templates -> cold start
    assert r["co_present"]["names"] == ["Notch", "Elliot"]
    top2 = r["groups"][:2]
    assert all(g.get("co_names") == ["Notch", "Elliot"] for g in top2)   # both pre-filled for one tap
    assert all("co_elim" not in g for g in top2)                         # nothing to disambiguate yet


def test_unblend_names_never_solo_member_by_elimination(conn, cfg):
    """The cold-start UNLOCK: one member has a template + the human logged the pair => the OTHER
    cluster is named by ELIMINATION, though that member has no template of his own."""
    vid = _visit(conn, [_det(conn, minutes=0)], start_min=0, end_min=10)
    _clip_tracklets_in_visit(conn, vid, [[1, 0, 0], [1, .05, 0], [.98, .1, 0], [0, 1, 0], [0, .97, .1]],
                             start_min=1, end_min=2)
    db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Notch", "Elliot"],
                            span_start=_ts(0), span_end=_ts(10))
    # Only Notch has a (clip) template; Elliot has none.
    r = individuals.unblend_visit(conn, vid, distance=0.45, templates={},
                                  elim_templates={"Notch": (_u(1, 0, 0), 5)})
    big, small = r["groups"][0], r["groups"][1]      # big cluster ([1,0,0]) = Notch; the other = Elliot
    assert big["n"] == 3 and big["co_elim"] == "Notch"
    assert small["n"] == 2 and small["co_elim"] == "Elliot"


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


# ---------------------------------------------------------------------------
# Auto-assign: the nightly "review by exception" pass (and its db plumbing).
# ---------------------------------------------------------------------------

def test_label_visit_reject_leaves_human_tombstone(conn):
    vid = _visit(conn, [_det(conn, minutes=0)], start_min=0)
    db.label_visit(conn, vid, "Stan", source="auto")
    assert db.visit_labels_by_source(conn, "auto") == {vid: "Stan"}
    assert db.confirmed_visit_labels(conn) == {}       # an auto name is NOT a confirmation

    # The human's "not them, leave unnamed": id clears, but source='human' stays as the tombstone.
    db.label_visit(conn, vid, None, reject=True)
    row = conn.execute("SELECT individual_id, individual_source FROM detections "
                       "WHERE visit_id = ?", (vid,)).fetchone()
    assert row["individual_id"] is None and row["individual_source"] == "human"
    assert db.rejected_visit_ids(conn) == {vid}
    assert db.visit_labels_by_source(conn, "auto") == {}
    assert db.confirmed_visit_labels(conn) == {}       # a rejection is not a confirmation either

    # A plain clear (no reject) wipes the source too -- back to fully unlabelled.
    db.label_visit(conn, vid, None)
    row = conn.execute("SELECT individual_id, individual_source FROM detections "
                       "WHERE visit_id = ?", (vid,)).fetchone()
    assert row["individual_id"] is None and row["individual_source"] is None
    assert db.rejected_visit_ids(conn) == set()


def test_auto_assign_names_only_the_unambiguous(conn, cfg):
    stan_a = _three_crop_visit(conn, [1, 0, 0], start_min=0)
    notch_a = _three_crop_visit(conn, [0, 1, 0], start_min=30)
    db.label_visit(conn, stan_a, "Stan")
    db.label_visit(conn, notch_a, "Notch")
    clear = _three_crop_visit(conn, [1, 0.05, 0], start_min=60)      # unmistakably Stan
    # [1, 0.75, 0]: ~0.80 to Stan but ~0.60 to Notch -- above the similarity bar, but the lead
    # (~0.20) is under the 0.25 margin: a confident-looking near-tie the pass must NOT call.
    tie = _three_crop_visit(conn, [1, 0.75, 0], start_min=90)
    weak = _three_crop_visit(conn, [0, 0, 1], start_min=120)         # looks like nobody

    m = VisitMatcher(conn, "raccoon", cfg)
    # min_templates=1: this test is about the similarity/margin bars, so the per-individual
    # template floor (its own test below) is opened right up.
    r = m.auto_assign(conn, threshold=0.75, margin=0.25, min_templates=1)
    assert r["enabled"] and [a["visit_id"] for a in r["assigned"]] == [clear]
    assert r["assigned"][0]["name"] == "Stan"
    assert r["skipped"]["ambiguous"] == 1                            # the tie
    assert r["skipped"]["below_threshold"] == 1                      # the stranger
    assert db.visit_labels_by_source(conn, "auto", "raccoon") == {clear: "Stan"}

    # The auto name shows on suggest() but never becomes a template, and refit skips the visit.
    m2 = VisitMatcher(conn, "raccoon", cfg)
    assert m2.suggest(clear)["auto_as"] == "Stan"
    assert m2.suggest(clear)["confirmed_as"] is None
    assert clear not in {v for _, v, _ in m2.templates()}
    assert all(x["visit_id"] != clear
               for lst in m2.refit()["fits"].values() for x in lst)

    # Idempotent: the next nightly run (same bars) leaves it alone.
    r2 = m2.auto_assign(conn, threshold=0.75, margin=0.25, min_templates=1)
    assert r2["assigned"] == [] and r2["skipped"]["already_auto"] == 1

    # Promotion: a human ✓ turns the same name into a real, template-feeding confirmation.
    db.label_visit(conn, clear, "Stan")
    m3 = VisitMatcher(conn, "raccoon", cfg)
    assert clear in {v for _, v, _ in m3.templates()}


def test_auto_assign_respects_rejection_multi_and_dry_run(conn, cfg):
    stan_a = _three_crop_visit(conn, [1, 0, 0], start_min=0)
    db.label_visit(conn, stan_a, "Stan")

    # Human already looked at this one and said "leave it": the tombstone must hold nightly.
    vetoed = _three_crop_visit(conn, [1, 0.02, 0], start_min=60)
    db.label_visit(conn, vetoed, None, reject=True)

    # A pair visit (>= reid_co_presence_min separated-box frames) with a Stan-like blend: solo-only.
    ids = []
    for k in range(3):
        a = _det(conn, minutes=200 + k, bbox=(0, 0, 10, 10))
        b = _det(conn, minutes=200 + k, bbox=(50, 50, 60, 60))
        _embed(conn, a, _unit(1, 0.01, 0))
        ids += [a, b]
    pair = _visit(conn, ids, start_min=200, end_min=203)

    m = VisitMatcher(conn, "raccoon", cfg)
    r = m.auto_assign(conn, threshold=0.8, margin=0.1, min_templates=1)   # floor: tested below
    assert r["assigned"] == []
    assert r["skipped"]["human_rejected"] == 1
    assert r["skipped"]["multi_animal"] == 1

    # threshold 0.0 disables the pass entirely (the pre-eval state of a fresh install).
    assert m.auto_assign(conn, threshold=0.0)["enabled"] is False

    # Dry-run reports but writes nothing.
    fresh = _three_crop_visit(conn, [1, 0.03, 0], start_min=300)
    m2 = VisitMatcher(conn, "raccoon", cfg)
    r2 = m2.auto_assign(conn, threshold=0.8, margin=0.1, min_templates=1, dry_run=True)
    assert [a["visit_id"] for a in r2["assigned"]] == [fresh]
    assert db.visit_labels_by_source(conn, "auto", "raccoon") == {}


# ---------------------------------------------------------------------------
# Auto-assign guardrails: the per-individual TEMPLATE FLOOR and the SOURCE GUARD.
# Both are refusals that protect the human label set (docs/identity-eval-2026-08-05.md,
# phases A2 and C1). Neither may change what the matcher believes -- only what it writes.
# ---------------------------------------------------------------------------

def _named_visits(conn, name, vecs, *, first_min, source=db.SOURCE_GLASS_DOOR_CAM):
    """Confirm one individual on `len(vecs)` solo visits -> that many usable templates."""
    out = []
    for k, vec in enumerate(vecs):
        v = _three_crop_visit(conn, vec, start_min=first_min + 10 * k, source=source)
        db.label_visit(conn, v, name)
        out.append(v)
    return out


def test_auto_assign_template_floor_blocks_a_one_template_name(conn, cfg):
    """The tier has already machine-named a CutiePie and a The Dude visit off a SINGLE confirmed
    template. Nothing swept on a one-visit individual is a measurement, so the floor refuses to
    spend the label set on it -- while a many-template name goes through untouched."""
    _named_visits(conn, "Stan", [[1, 0, 0], [1, 0.03, 0], [1, -0.02, 0]], first_min=0)
    _named_visits(conn, "The Dude", [[0, 1, 0]], first_min=100)      # exactly one template
    dude_like = _three_crop_visit(conn, [0, 1, 0.03], start_min=200)  # unmistakably The Dude
    stan_like = _three_crop_visit(conn, [1, 0.04, 0], start_min=260)  # unmistakably Stan

    m = VisitMatcher(conn, "raccoon", cfg)
    assert sorted(n for n, _v, _p in m.templates()) == ["Stan"] * 3 + ["The Dude"]

    # Floor of 1 = today's behaviour: BOTH get named, including the thin one.
    loose = m.auto_assign(conn, threshold=0.75, margin=0.25, min_templates=1, dry_run=True)
    assert sorted(a["visit_id"] for a in loose["assigned"]) == sorted([dude_like, stan_like])

    # Floor of 3: the thin name is refused, and refused with its OWN reason -- it did not fail
    # the similarity bar (that is the whole point: it looked certain).
    r = m.auto_assign(conn, threshold=0.75, margin=0.25, min_templates=3, dry_run=True)
    assert [a["visit_id"] for a in r["assigned"]] == [stan_like]
    assert r["skipped"]["thin_templates"] == 1
    assert "below_threshold" not in r["skipped"] and "ambiguous" not in r["skipped"]
    assert r["min_templates"] == 3

    # The thin individual still COMPETES: it can still block an assignment as the runner-up.
    near_dude = _three_crop_visit(conn, [0.75, 1, 0], start_min=320)
    m2 = VisitMatcher(conn, "raccoon", cfg)
    r2 = m2.auto_assign(conn, threshold=0.5, margin=0.25, min_templates=3, dry_run=True)
    assert near_dude not in [a["visit_id"] for a in r2["assigned"]]


def test_auto_assign_floor_default_comes_from_config(conn, cfg):
    """The floor is a config setting with a safe (cautious) shipped default, not a constant."""
    assert config.Config().reid_auto_min_templates >= 2
    cfg.reid_auto_min_templates = 99                      # nobody can clear this
    _named_visits(conn, "Stan", [[1, 0, 0], [1, 0.03, 0]], first_min=0)
    probe = _three_crop_visit(conn, [1, 0.04, 0], start_min=100)
    r = VisitMatcher(conn, "raccoon", cfg).auto_assign(conn, threshold=0.75, margin=0.1,
                                                       dry_run=True)
    assert r["assigned"] == [] and r["skipped"]["thin_templates"] == 1
    assert db.visit_labels_by_source(conn, "auto", "raccoon") == {}
    assert probe                                          # (the probe was the only candidate)


# ---------------------------------------------------------------------------
# THE ROSTER: an individual the human has marked DEPARTED.
# Notch's last labelled crop is 2026-06-30 and Matt confirms the animal simply stopped coming --
# but the 46 templates stayed, and at the recommended operating point the tier lined up to write
# "Notch" onto two visits on 2026-07-03. This error class is invisible to every metric here (LOO
# scores against labels; a departed animal's labels stop, so no probe can ever exhibit it), which
# is why it takes a fact from the human instead of a threshold. NOT a recency gate on template
# age -- that was measured and made things worse. A DATE test, and only over what is WRITTEN.
# ---------------------------------------------------------------------------

_ONE_DAY = 1440.0     # minutes; _ts() counts from BASE = 2026-06-10 21:00 local


def test_is_departed_is_a_date_comparison_not_a_blanket_exclusion(conn, cfg):
    db.set_individual_status(conn, "Notch", status="departed", effective_date="2026-06-30")
    m = VisitMatcher(conn, "raccoon", cfg)
    assert m.is_departed("Notch", "2026-06-29T23:59:00-07:00") is False   # while resident
    assert m.is_departed("Notch", "2026-06-30T23:59:00-07:00") is False   # the last day counts
    assert m.is_departed("Notch", "2026-07-01T00:01:00-07:00") is True    # after
    assert m.is_departed("notch", "2026-07-03T00:13:00-07:00") is True    # names are free text
    assert m.is_departed("Pedro", "2026-07-03T00:13:00-07:00") is False   # nobody said anything
    assert m.is_departed("Notch", None) is True                           # no start -> fail closed


def test_auto_assign_will_not_write_a_departed_name_after_the_departure_date(conn, cfg):
    """The two 2026-07-03 assignments, in miniature: same animal-looking prototype, one visit
    before the departure and one after. Only the later one is refused, and refused with its own
    reason -- it did not fail the similarity bar, which is exactly the danger."""
    _named_visits(conn, "Notch", [[1, 0, 0], [1, 0.03, 0], [1, -0.02, 0]], first_min=0)
    before = _three_crop_visit(conn, [1, 0.04, 0], start_min=200)                # 2026-06-11
    after = _three_crop_visit(conn, [1, 0.04, 0], start_min=5 * _ONE_DAY)        # 2026-06-15

    # Without the roster the tier names BOTH -- this is the bug, reproduced.
    naive = VisitMatcher(conn, "raccoon", cfg).auto_assign(
        conn, threshold=0.75, margin=0.02, min_templates=1, dry_run=True)
    assert sorted(a["visit_id"] for a in naive["assigned"]) == sorted([before, after])

    # Matt records what he knows: Notch was last here on the 12th.
    db.set_individual_status(conn, "Notch", status="departed", effective_date="2026-06-12",
                             note="stopped coming")
    m = VisitMatcher(conn, "raccoon", cfg)
    r = m.auto_assign(conn, threshold=0.75, margin=0.02, min_templates=1, dry_run=True)
    assert [a["visit_id"] for a in r["assigned"]] == [before]      # June visits survive
    assert r["skipped"]["departed"] == 1
    assert "below_threshold" not in r["skipped"] and "ambiguous" not in r["skipped"]

    # And it holds on a real (non-dry) run: nothing is written for the later visit.
    m.auto_assign(conn, threshold=0.75, margin=0.02, min_templates=1)
    assert db.visit_labels_by_source(conn, "auto", "raccoon") == {before: "Notch"}


def test_a_departed_individual_is_still_ranked_suggested_and_templated(conn, cfg):
    """The guard governs the MACHINE's pen, nothing else. Matt may well look at an unreviewed
    visit and recognise a departed animal -- 231 unconfirmed July visits and 46 unconfirmed June
    ones are exactly the pile this is for -- so the name must stay on every human surface."""
    _named_visits(conn, "Notch", [[1, 0, 0], [1, 0.03, 0]], first_min=0)
    late = _three_crop_visit(conn, [1, 0.04, 0], start_min=5 * _ONE_DAY)
    db.set_individual_status(conn, "Notch", status="departed", effective_date="2026-06-12")

    m = VisitMatcher(conn, "raccoon", cfg)
    assert [n for n, _v, _p in m.templates()] == ["Notch"] * 2     # still a template set
    s = m.suggest(late)
    assert s["candidates"][0]["name"] == "Notch"                   # still ranked and suggested
    assert s["candidates"][0]["similarity"] > 0.9
    assert late in [x["visit_id"] for x in m.refit()["fits"]["Notch"]]
    # A human confirmation is untouched by the roster -- the tool augments the judgement.
    db.label_visit(conn, late, "Notch")
    assert db.confirmed_visit_labels(conn, "raccoon")[late] == "Notch"


def test_departed_with_no_date_blocks_every_auto_write_for_that_name(conn, cfg):
    """No date means residency can't be established for any visit, so the name is never written.
    Fail closed -- but only for that name; the rest of the cast is unaffected."""
    _named_visits(conn, "Notch", [[1, 0, 0], [1, 0.03, 0]], first_min=0)
    _named_visits(conn, "Pedro", [[0, 1, 0], [0.03, 1, 0]], first_min=100)
    notchy = _three_crop_visit(conn, [1, 0.04, 0], start_min=200)
    pedroy = _three_crop_visit(conn, [0.04, 1, 0], start_min=260)
    db.set_individual_status(conn, "Notch", status="departed")     # no effective_date

    r = VisitMatcher(conn, "raccoon", cfg).auto_assign(
        conn, threshold=0.75, margin=0.02, min_templates=1, dry_run=True)
    assert [a["visit_id"] for a in r["assigned"]] == [pedroy]
    assert r["skipped"]["departed"] == 1
    assert notchy not in [a["visit_id"] for a in r["assigned"]]


def test_marking_an_individual_resident_again_reopens_the_auto_tier(conn, cfg):
    """A departure is a claim about the world, and Matt can be wrong (or the animal comes back)."""
    _named_visits(conn, "Notch", [[1, 0, 0], [1, 0.03, 0]], first_min=0)
    late = _three_crop_visit(conn, [1, 0.04, 0], start_min=5 * _ONE_DAY)
    db.set_individual_status(conn, "Notch", status="departed", effective_date="2026-06-12")
    assert VisitMatcher(conn, "raccoon", cfg).auto_assign(
        conn, threshold=0.75, margin=0.02, min_templates=1, dry_run=True)["assigned"] == []
    db.set_individual_status(conn, "Notch", status="resident")
    r = VisitMatcher(conn, "raccoon", cfg).auto_assign(
        conn, threshold=0.75, margin=0.02, min_templates=1, dry_run=True)
    assert [a["visit_id"] for a in r["assigned"]] == [late]


def test_the_departed_guard_does_not_hand_the_name_down_the_ranking(conn, cfg):
    """When the winner is departed the VISIT is skipped, not re-awarded to the runner-up: the
    margin was measured against the departed individual, so promoting second place would write a
    name that never cleared a bar."""
    _named_visits(conn, "Notch", [[1, 0, 0], [1, 0.02, 0]], first_min=0)
    _named_visits(conn, "Pedro", [[0.8, 0.6, 0], [0.82, 0.58, 0]], first_min=100)
    probe = _three_crop_visit(conn, [1, 0.05, 0], start_min=5 * _ONE_DAY)   # Notch first, Pedro 2nd
    db.set_individual_status(conn, "Notch", status="departed", effective_date="2026-06-12")
    r = VisitMatcher(conn, "raccoon", cfg).auto_assign(
        conn, threshold=0.5, margin=0.0, min_templates=1, dry_run=True)
    assert r["assigned"] == [] and r["skipped"]["departed"] == 1
    assert probe not in [a["visit_id"] for a in r["assigned"]]


def test_source_guard_blocks_a_cross_camera_match_that_would_otherwise_rank(conn, cfg):
    """Two visits with the SAME prototype, one per camera, against one glass-door template: the
    same-camera visit matches at ~1.0 and the trail-cam visit does not match at all. Identical
    numbers, opposite outcomes -- so the rule is source equality, not a threshold. Measured
    justification: the best cross-source similarity in the whole 397x93 matrix is 0.363, while
    83.7% of trail-cam visit PAIRS clear the novelty cut, so one cross-camera confirmation would
    let refit() propose ~83 more visits under that single name."""
    stan = _three_crop_visit(conn, [1, 0, 0], start_min=0)
    db.label_visit(conn, stan, "Stan")
    twin = _three_crop_visit(conn, [1, 0.02, 0], start_min=60)
    trail = _three_crop_visit(conn, [1, 0.02, 0], start_min=120, source=db.SOURCE_TRAIL_CAM_SD)

    m = VisitMatcher(conn, "raccoon", cfg)
    assert m.suggest(twin)["candidates"][0]["name"] == "Stan"        # same camera: ranks
    s = m.suggest(trail)
    assert s["source"] == db.SOURCE_TRAIL_CAM_SD
    assert s["candidates"] == [] and not s["novel"]                  # other camera: never ranks
    assert "across cameras" in (s["note"] or "")                     # and says why, honestly

    # The pools themselves: scoped for ranking, corpus-wide for "does any template exist?".
    assert m.templates(db.SOURCE_TRAIL_CAM_SD) == []
    assert [n for n, _v, _p in m.templates(db.SOURCE_GLASS_DOOR_CAM)] == ["Stan"]
    assert len(m.templates()) == 1

    # auto_assign: names the twin, refuses the trail-cam visit with its own reason.
    r = m.auto_assign(conn, threshold=0.75, margin=0.1, min_templates=1, dry_run=True)
    assert [a["visit_id"] for a in r["assigned"]] == [twin]
    assert r["skipped"]["no_same_source_template"] == 1
    # refit: the twin fits Stan, the trail-cam visit falls to the novel residual instead.
    fit = m.refit()
    assert [x["visit_id"] for x in fit["fits"]["Stan"]] == [twin]
    assert trail in {v for g in fit["novel_groups"] for v in g["visits"]}


def test_source_guard_survives_a_camera_move(conn, cfg):
    """The guard may not encode WHERE a camera points: Matt repositions them on purpose. Same
    animal, same camera, boxes on the opposite side of a re-aimed frame -> still one pool."""
    a = _three_crop_visit(conn, [1, 0, 0], start_min=0)
    db.label_visit(conn, a, "Stan")
    moved = []
    for k in range(3):                                   # after the move: far-away geometry
        d = _det(conn, minutes=60 + k * 0.2, bbox=(900, 700, 990, 790))
        _embed(conn, d, _unit(1, 0.02, 0))
        moved.append(d)
    v = _visit(conn, moved, start_min=60, end_min=61)
    m = VisitMatcher(conn, "raccoon", cfg)
    assert m.suggest(v)["candidates"][0]["name"] == "Stan"


def test_refit_novel_groups_never_span_cameras(conn, cfg):
    """A candidate-new-individual group spanning two cameras invites one name onto both -- the
    same mass-mislabel through a different door."""
    anchor = _three_crop_visit(conn, [1, 0, 0], start_min=0)
    db.label_visit(conn, anchor, "Stan")
    g = _three_crop_visit(conn, [0, 1, 0], start_min=60)
    t = _three_crop_visit(conn, [0, 1, 0], start_min=120, source=db.SOURCE_TRAIL_CAM_SD)
    groups = VisitMatcher(conn, "raccoon", cfg).refit()["novel_groups"]
    assert sorted(len(x["visits"]) for x in groups) == [1, 1]        # identical vectors, NOT merged
    assert {v for x in groups for v in x["visits"]} == {g, t}


def test_source_guard_holds_for_clip_templates_and_stills_unblend(conn, cfg):
    """The guard covers every path that ranks a prototype: the cross-space clip candidates and
    the stills un-blend suggestions, not just the still-template ranking."""
    # A glass-door clip tracklet labelled Elliot; a trail-cam visit that looks exactly like it.
    _embedded_tracklet(conn, start_min=0, end_min=1, vec=[1, 0, 0], individual="Elliot")
    same_cam = _three_crop_visit(conn, [1, 0, 0], start_min=60)
    other_cam = _three_crop_visit(conn, [1, 0, 0], start_min=120, source=db.SOURCE_TRAIL_CAM_SD)
    m = VisitMatcher(conn, "raccoon", cfg)
    assert [c["name"] for c in m.suggest(same_cam)["clip_candidates"]] == ["Elliot"]
    assert m.suggest(other_cam)["clip_candidates"] == []
    assert "Elliot" in m.clip_templates                   # the corpus-wide view is unchanged

    # Stills un-blend: a two-animal TRAIL-CAM visit offered a glass-door template.
    u = _unit(1, 0)
    left, right = [], []
    for i in range(3):
        l = _det(conn, minutes=200 + i * 0.05, bbox=(0, 0, 60, 60), source=db.SOURCE_TRAIL_CAM_SD)
        r = _det(conn, minutes=200 + i * 0.05, bbox=(400, 0, 460, 60),
                 source=db.SOURCE_TRAIL_CAM_SD)
        _embed(conn, l, u)
        _embed(conn, r, _unit(0, 1))
        left.append(l)
        right.append(r)
    pair = _visit(conn, left + right, start_min=200, end_min=200.2,
                  source=db.SOURCE_TRAIL_CAM_SD)
    glass_template = _visit(conn, [_det(conn, minutes=300)], start_min=300, end_min=300.1)
    out = individuals.unblend_visit_stills(conn, pair,
                                           templates=[("Stan", glass_template, u)])
    assert len(out["groups"]) == 2
    assert all(g["suggestion"] == [] for g in out["groups"])   # no cross-camera suggestion


# ---------------------------------------------------------------------------
# Still tracklets + the stills un-blend basis (the multi-animal splitter).
# ---------------------------------------------------------------------------

def test_still_tracklets_two_parallel_animals_and_cannot_link():
    left = lambda i: (0 + i * 5, 100, 60 + i * 5, 160)     # ambling right, 3 s per frame
    right = (400, 100, 460, 160)                            # parked
    rows = []
    for i in range(3):
        rows.append((10 + i, _ts(i * 0.05), left(i)))
        rows.append((20 + i, _ts(i * 0.05), right))
    tracks, cannot = individuals.still_tracklets(rows)
    assert sorted(map(sorted, tracks)) == [[10, 11, 12], [20, 21, 22]]
    assert cannot == {(0, 1)}                               # same-frame = two bodies, pinned apart


def test_still_tracklets_merges_detector_double_box():
    a = (100, 100, 200, 200)
    a_jit = (105, 105, 205, 205)                            # IoU ~0.82: one animal boxed twice
    rows = [(1, _ts(0), a), (2, _ts(0), a_jit), (3, _ts(0.05), (110, 110, 210, 210))]
    tracks, cannot = individuals.still_tracklets(rows)
    assert len(tracks) == 1 and sorted(tracks[0]) == [1, 2, 3]
    assert cannot == set()                                  # a double-box is NOT co-presence


def test_still_tracklets_splits_on_position_jump():
    # The 2026-07-31 case: a coherent ground track, then a far-away wall box 9 s later. The
    # distance gate (~1.6 body lengths here) refuses the link -- that box was a kit, not Stan.
    ground = [(1, _ts(0.0), (800, 800, 1060, 950)), (2, _ts(0.10), (820, 810, 1090, 955))]
    wall = [(3, _ts(0.25), (1150, 420, 1380, 600))]
    tracks, cannot = individuals.still_tracklets(ground + wall)
    assert sorted(map(sorted, tracks)) == [[1, 2], [3]]
    assert cannot == set()                                  # never shared a frame


def test_still_tracklets_fragments_across_a_long_gap():
    # 30 s between sightings of the same spot: beyond link_gap_s, so two fragments (they
    # re-merge by appearance in the clustering step -- fragmenting is the safe direction).
    rows = [(1, _ts(0.0), (100, 100, 200, 200)), (2, _ts(0.5), (100, 100, 200, 200))]
    tracks, _cannot = individuals.still_tracklets(rows)
    assert sorted(map(sorted, tracks)) == [[1], [2]]


def test_unblend_visit_stills_separates_and_seeds(conn):
    # Two animals sharing every frame: a Stan-lookalike on the left, an unknown kit on the
    # right (low-conf, as second animals are). Appearance separates them, the sighting log
    # seeds the names: Stan by template match, Kit 1 by elimination.
    u_stan, u_kit = _unit(1, 0), _unit(0, 1)
    left, right = [], []
    for i in range(3):
        l = _det(conn, minutes=i * 0.05, bbox=(0, 0, 60, 60))
        r = _det(conn, minutes=i * 0.05, bbox=(400, 0, 460, 60), confidence=0.3)
        _embed(conn, l, u_stan)
        _embed(conn, r, u_kit)
        left.append(l)
        right.append(r)
    vid = _visit(conn, left + right, start_min=0.0, end_min=0.2)
    db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Stan", "Kit 1"],
                            span_start=_ts(0.0), span_end=_ts(0.2))
    # The template's visit must be a REAL visit on the same camera -- the source guard drops any
    # template it cannot resolve to this visit's source (see the cross-source test below).
    tvid = _visit(conn, [_det(conn, minutes=500)], start_min=500, end_min=500.1)
    out = individuals.unblend_visit_stills(conn, vid, templates=[("Stan", tvid, u_stan)])
    assert out["basis"] == "stills" and out["n_tracklets"] == 2
    assert len(out["groups"]) == 2
    by_ids = {tuple(sorted(g["detection_ids"])): g for g in out["groups"]}
    assert set(by_ids) == {tuple(sorted(left)), tuple(sorted(right))}
    gl, gr = by_ids[tuple(sorted(left))], by_ids[tuple(sorted(right))]
    assert gl["suggestion"] and gl["suggestion"][0]["name"] == "Stan"
    assert not gr["suggestion"]                             # orthogonal: below the threshold
    assert gl["co_elim"] == "Stan"                          # appearance resolves Stan's side...
    assert gr["co_elim"] == "Kit 1"                         # ...and the kit closes by elimination
    # Quick-picks exclude names claimed by the OTHER group -- fully resolved, so one name each.
    assert gl["co_names"] == ["Stan"] and gr["co_names"] == ["Kit 1"]


def test_unblend_stills_labelled_group_consumes_its_logged_name(conn):
    # Stan's live-stamped track arrives as an already-LABELLED group; the log says Stan + Kit 1.
    # The kit group must not be offered "+ Stan", and -- one name, one group left -- the kit
    # closes by elimination with NO templates at all (label + log, the cold-start case).
    u = _unit(1, 1)                                         # identical appearance: no help there
    a = [_det(conn, minutes=i * 0.05, bbox=(0, 0, 60, 60)) for i in range(2)]
    b = [_det(conn, minutes=i * 0.05, bbox=(400, 0, 460, 60)) for i in range(2)]
    for d in a + b:
        _embed(conn, d, u)
    db.set_individual_bulk(conn, a, "Stan")
    vid = _visit(conn, a + b, start_min=0.0, end_min=0.1)
    db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Stan", "Kit 1"],
                            span_start=_ts(0.0), span_end=_ts(0.1))
    out = individuals.unblend_visit_stills(conn, vid)       # templates: none
    by_label = {g["label"]: g for g in out["groups"]}
    assert set(by_label) == {"Stan", None}
    assert by_label["Stan"].get("co_elim") is None          # labelled: the chip already shows
    assert by_label["Stan"]["co_names"] == ["Stan"]
    assert by_label[None]["co_names"] == ["Kit 1"]          # Stan is claimed, not offered
    assert by_label[None]["co_elim"] == "Kit 1"             # closed by elimination, cold start


def test_unblend_visit_stills_cannot_link_beats_lookalike_appearance(conn):
    # Littermate kits: identical embeddings, but co-present in every frame. The same-frame
    # cannot-link must hold the clusters apart where appearance can't.
    u = _unit(1, 1)
    a = [_det(conn, minutes=i * 0.05, bbox=(0, 0, 60, 60)) for i in range(2)]
    b = [_det(conn, minutes=i * 0.05, bbox=(400, 0, 460, 60)) for i in range(2)]
    for d in a + b:
        _embed(conn, d, u)
    vid = _visit(conn, a + b, start_min=0.0, end_min=0.1)
    out = individuals.unblend_visit_stills(conn, vid)
    assert len(out["groups"]) == 2                          # NOT merged into one lookalike blob


def test_visitmatcher_sighting_multi_flags_visit_without_co_frames(conn):
    # Kits arrive one at a time: zero same-instant frames, yet the human logged 2+ names --
    # the third multi signal (2026-07-31, the Stan + 3 kits span).
    dets = [_det(conn, minutes=i * 0.2) for i in range(4)]
    for d in dets:
        _embed(conn, d, _unit(1, 0))
    vid = _visit(conn, dets, start_min=0.0, end_min=0.8)
    matcher = VisitMatcher(conn, "raccoon")
    assert matcher.is_multi(vid) is False                   # no detector-side signal
    db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM,
                            names=["Stan", "Kit 1", "Kit 2"],
                            span_start=_ts(0.0), span_end=_ts(0.8))
    matcher = VisitMatcher(conn, "raccoon")                 # sightings load at init
    assert matcher.is_multi(vid) is True
    assert matcher.suggest(vid)["co_present_sighting"] is True


def test_co_present_visit_ids_frames_and_sightings(conn):
    # Visit A holds a same-frame separated pair -> in. Visit B is solo -> out, until a
    # multi-name sighting overlaps it (the human saw what the stills didn't).
    a1 = _det(conn, minutes=0, bbox=(0, 0, 10, 10))
    a2 = _det(conn, minutes=0, bbox=(50, 50, 60, 60))
    va = _visit(conn, [a1, a2], start_min=0.0, end_min=0.1)
    b1 = _det(conn, minutes=5)
    vb = _visit(conn, [b1], start_min=5.0, end_min=5.1)
    assert individuals.co_present_visit_ids(conn) == {va}
    db.record_live_sighting(conn, source=db.SOURCE_GLASS_DOOR_CAM, names=["Stan", "Kit 1"],
                            span_start=_ts(5.0), span_end=_ts(5.1))
    assert individuals.co_present_visit_ids(conn) == {va, vb}


# ---------------------------------------------------------------------------
# THE TWO-AXIS PRINCIPLE, as a test: behaviour never touches the appearance ranking.
# ---------------------------------------------------------------------------

# clipmotion's per-tracklet gait/motion features, as stored on clip_tracks.
GAIT_FEATURES = ("duration_s", "path_len", "net_disp", "straightness", "avg_speed", "peak_speed",
                 "moving_frac", "area_trend", "stride_hz", "stride_strength", "walk_s")

# Two contradictory behaviour signatures: a beeline trot, and the sit-back-and-handle-it animal
# that barely moves. (Eating style -- "mouth-first at the dish" vs "grabs it and sits back" -- is
# the next behaviour feature Matt wants; it lands on these same rows and is bound by this test.)
_TROTS = {"duration_s": 12.0, "path_len": 9.0, "net_disp": 8.5, "straightness": 0.94,
          "avg_speed": 0.80, "peak_speed": 1.9, "moving_frac": 0.95, "area_trend": 2.4,
          "stride_hz": 2.6, "stride_strength": 0.90, "walk_s": 11.0}
_SITS_BACK = {"duration_s": 90.0, "path_len": 0.4, "net_disp": 0.05, "straightness": 0.02,
              "avg_speed": 0.01, "peak_speed": 0.05, "moving_frac": 0.05, "area_trend": 0.3,
              "stride_hz": 0.4, "stride_strength": 0.10, "walk_s": 2.0}


def _set_gait(conn, clip_id, feats):
    """Rewrite one clip's tracklet with a given behaviour signature (clipmotion's own writer)."""
    db.insert_clip_tracks(conn, clip_id=clip_id, model="m", n_samples=300,
                          tracklets=[{"track_json": "[]", "n_hits": 40, "features": feats}])
    got = conn.execute("SELECT straightness, avg_speed, stride_hz, n_hits FROM clip_tracks "
                       "WHERE clip_id = ?", (clip_id,)).fetchone()
    assert got["straightness"] == feats["straightness"]      # the features really landed...
    assert got["n_hits"] == 40                               # ...and the co-presence signal didn't move
    return got


def test_match_ordering_is_invariant_to_behaviour_features(conn, cfg):
    """THE TWO-AXIS PRINCIPLE (docs/plan.md, README): appearance and behaviour stay on separate
    axes. Behaviour may raise a flag and may be shown to the human beside a suggestion, but it
    must NEVER change which individual a visit looks like, or in what order. Measured on this
    corpus and recorded in docs/identity-eval-2026-08-05.md: nine motion features give LOO
    balanced accuracy 0.422 vs 0.333 chance (p=0.107), depth-normalized size separates the three
    adults at p=0.59, and stride_hz exists on 27 of 21,860 tracks with its values piled at the
    band edges. So behaviour is not merely forbidden here, it is also empty -- but the guarantee
    is structural, not statistical, which is why it is asserted rather than measured.

    Written to bind the FUTURE: eating style is the next behaviour signal wanted, it lands on
    these same clip_tracks rows, and this test is what stops it leaking into the ranker."""
    stan = _three_crop_visit(conn, [1, 0, 0], start_min=0)
    notch = _three_crop_visit(conn, [0, 1, 0], start_min=30)
    db.label_visit(conn, stan, "Stan")
    db.label_visit(conn, notch, "Notch")
    clear = _three_crop_visit(conn, [1, 0.05, 0], start_min=60)      # unmistakably Stan
    tie = _three_crop_visit(conn, [1, 0.75, 0], start_min=90)        # a near-tie, correctly refused
    # One clip per visit, ONE sustained tracklet each: enough to carry gait, never enough to
    # trip clip co-presence (which is a multi-ANIMAL signal, not a behaviour feature).
    clips = {v: _clip_with_tracks(conn, start_min=s + 0.1, end_min=s + 0.5, n_sustained=1)
             for v, s in ((stan, 0), (notch, 30), (clear, 60), (tie, 90))}

    def ranking():
        m = VisitMatcher(conn, "raccoon", cfg)
        return {
            "candidates": {v: [(c["name"], c["similarity"]) for c in m.suggest(v)["candidates"]]
                           for v in (clear, tie)},
            "novel": {v: m.suggest(v)["novel"] for v in (clear, tie)},
            "fits": {n: [(x["visit_id"], x["similarity"]) for x in lst]
                     for n, lst in sorted(m.refit()["fits"].items())},
            "auto": [(a["visit_id"], a["name"], a["similarity"], a["margin"])
                     for a in m.auto_assign(conn, threshold=0.75, margin=0.25, min_templates=1,
                                            dry_run=True)["assigned"]],
        }

    baseline = ranking()
    assert baseline["candidates"][clear][0][0] == "Stan"             # the ranking is real...
    assert [a[0] for a in baseline["auto"]] == [clear]               # ...and it does assign

    # Now hand the two Stan-side visits one signature and the Notch-side visit the other, with
    # the near-tie matching Stan's exactly: if any gait column reached the ranker, `tie` would
    # move off its refusal and `clear`'s order would firm up.
    for v, feats in ((stan, _SITS_BACK), (clear, _SITS_BACK), (notch, _TROTS), (tie, _SITS_BACK)):
        _set_gait(conn, clips[v], feats)
    assert ranking() == baseline

    # Swap every signature end-for-end (the near-tie now moves like Notch): still nothing moves.
    for v, feats in ((stan, _TROTS), (clear, _TROTS), (notch, _SITS_BACK), (tie, _TROTS)):
        _set_gait(conn, clips[v], feats)
    assert ranking() == baseline


def test_fetch_for_embedding_visit_ids_restriction(conn):
    # The --co-present widening pass: low-conf crops come back ONLY for the named visits.
    d1 = _det(conn, minutes=0, confidence=0.3)
    d2 = _det(conn, minutes=5, confidence=0.3)
    v1 = _visit(conn, [d1])
    _visit(conn, [d2], start_min=5.0, end_min=5.1)
    rows = db.fetch_for_embedding(conn, individuals.EMBED_MODEL, species=None,
                                  min_confidence=0.25, redo=False, visit_ids={v1})
    assert [r[0] for r in rows] == [d1]


# ---------------------------------------------------------------------------
# LAPSED IDENTITY -- the decay curve as a first-class per-individual state.
# ---------------------------------------------------------------------------
# The project's single most consequential measurement is that appearance identity decays in about
# a week. Until this shipped, one operator-only panel said so and every other surface presented a
# 40-day-old template as an equal to yesterday's. These pin the boundaries to the numbers they
# came from, so a future reader who moves one has to move a measurement with it.

def test_expected_top1_interpolates_the_measured_curve():
    assert individuals.expected_top1(0) == 0.741
    assert individuals.expected_top1(7) == 0.482
    assert 0.482 > individuals.expected_top1(8.5) > 0.403      # between the 7- and 10-day points
    assert individuals.expected_top1(None) is None


def test_expected_top1_goes_flat_past_the_last_measured_point():
    # Nine points on 139 probes do not support extrapolation, so it must not invent one.
    assert individuals.expected_top1(21) == individuals.expected_top1(400) == 0.122


def test_identity_lapse_crosses_where_the_matcher_stops_beating_a_guess():
    # Measured: 0.403 top-1 at 10 days (above the 0.345 majority baseline), 0.259 at 14 (below).
    assert individuals.identity_lapse(10, 5, stale_days=14)["state"] == "fading"
    assert individuals.identity_lapse(10, 5, stale_days=14)["beats_guessing"] is True
    lapsed = individuals.identity_lapse(14, 5, stale_days=14)
    assert lapsed["state"] == "lapsed" and lapsed["beats_guessing"] is False


def test_identity_lapse_fresh_and_none():
    assert individuals.identity_lapse(0.4, 5)["state"] == "fresh"
    # No usable template is NOT the same as a stale one, and it is the louder of the two.
    for n, d in ((0, 3.0), (5, None)):
        assert individuals.identity_lapse(d, n)["state"] == "none"


def test_identity_lapse_explains_the_fix_rather_than_the_failure():
    why = individuals.identity_lapse(30, 5)["why"]
    assert "re-anchor" in why and "%" in why


def test_template_freshness_reports_the_newest_solo_confirmation(conn):
    class _M:
        visit_started = {1: "2026-06-01T21:00:00-07:00", 2: "2026-06-20T21:00:00-07:00"}
        def templates(self):
            return [("Stan", 1, None), ("Stan", 2, None), ("Notch", 1, None)]
    now = datetime.fromisoformat("2026-06-22T21:00:00-07:00")
    f = individuals.template_freshness(_M(), now=now)
    assert f["Stan"]["n_templates"] == 2
    assert f["Stan"]["newest_template"] == "2026-06-20T21:00:00-07:00"
    assert f["Stan"]["days_since_template"] == pytest.approx(2.0)
    assert f["Stan"]["lapse"]["state"] == "fresh"
    assert f["Notch"]["lapse"]["state"] == "lapsed"      # its only template is 21 days old


def test_usable_template_visits_matches_what_the_matcher_would_use(conn):
    """The DB-side template set must be the matcher's set, or the surfaces disagree -- which they
    did, by six weeks, in the first cut of this. Three visits, three different reasons to differ.
    """
    # 1. a clean solo visit with enough embedded crops: a template.
    good = [_det(conn, minutes=m) for m in (0, 0.2, 0.4)]
    v_good = _visit(conn, good, start_min=0, end_min=1)
    for d in good:
        _embed(conn, d, _unit(1, 0))
    db.label_visit(conn, v_good, "Stan", source="human")
    # 2. confirmed and NEWER, but the embed pass has not reached it: NOT a template. This is the
    #    exact shape that made a profile page say "confirmed 0.7 days ago" about an animal the
    #    matcher had lost six weeks earlier.
    fresh = [_det(conn, minutes=m) for m in (100, 100.2, 100.4)]
    v_fresh = _visit(conn, fresh, start_min=100, end_min=101)
    db.label_visit(conn, v_fresh, "Stan", source="human")
    # 3. confirmed, embedded, but two separated boxes in the same instant: a blended prototype.
    pair = [_det(conn, minutes=200, bbox=(0, 0, 10, 10)), _det(conn, minutes=200, bbox=(50, 50, 60, 60)),
            _det(conn, minutes=200.2, bbox=(0, 0, 10, 10)), _det(conn, minutes=200.2, bbox=(50, 50, 60, 60)),
            _det(conn, minutes=200.4, bbox=(0, 0, 10, 10)), _det(conn, minutes=200.4, bbox=(50, 50, 60, 60))]
    v_pair = _visit(conn, pair, start_min=200, end_min=201)
    for d in pair:
        _embed(conn, d, _unit(1, 0))
    db.label_visit(conn, v_pair, "Stan", source="human")

    usable = individuals.usable_template_visits(conn, "raccoon")
    assert set(usable) == {v_good}
    assert usable[v_good][0] == "Stan"

    # And the lapse state is computed off THAT visit, not off the newer unembedded confirmation.
    lapses = individuals.lapse_by_name(conn, "raccoon")
    assert lapses["stan"]["state"] == "lapsed"       # the only template is well over a fortnight old
    assert individuals.lapse_by_name(conn, "raccoon", names=["nobody"]) == {}
