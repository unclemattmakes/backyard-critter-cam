"""
Tests for behavior.py -- the BEHAVIOUR axis (profiles off the collapsed `visits`).

Focus areas:
  * typical_window(hour_counts, coverage) -- the tightest CIRCULAR arrival window. The load-bearing
    case is a crepuscular animal spanning midnight: {22,23,0,1} must collapse to ONE window that
    wraps midnight (width ~4), not a 24-hour span.
  * species_profile(conn, species) -- the shape of the per-species dict (n_visits, active_days,
    arrival_hours, dwell, typical_window, ...).
  * co_occurrence(conn) -- species that share a visit (read off detections.visit_id + species),
    with the _NON_CRITTER denylist applied.

behavior.py imports `from stats import _NON_CRITTER`; stats imports only db + stdlib, so this is
cheap (no cv2/torch). Visit/detection rows are inserted into a throwaway DB via db.*.
"""
from __future__ import annotations

import behavior
import db


# --- typical_window: circular / midnight-spanning ---------------------------------------

def test_typical_window_midnight_span_wraps():
    """{22:5, 23:5, 0:5, 1:5} -> a single window that wraps midnight, width 4 -- NOT a 24h span.
    With coverage 0.8 of 20 visits (target 16), the tightest circular cover is 22h..01h."""
    win = behavior.typical_window({22: 5, 23: 5, 0: 5, 1: 5}, coverage=0.8)
    assert win is not None
    assert win["width_hours"] == 4
    assert win["start_hour"] == 22
    assert win["end_hour"] == 1
    # The window genuinely wraps midnight: start > end.
    assert win["start_hour"] > win["end_hour"]


def test_typical_window_normal_daytime_block():
    """A contiguous daytime cluster yields a non-wrapping window (start <= end)."""
    win = behavior.typical_window({9: 4, 10: 6, 11: 5, 12: 5}, coverage=0.8)
    assert win is not None
    assert win["start_hour"] <= win["end_hour"]
    # 0.8 * 20 = 16; the tightest cover is 10..12h (6+5+5 = 16 exactly), width 3 -- skipping the
    # lighter 9h hour. The key point for this test is that it does NOT wrap midnight.
    assert win["width_hours"] == 3
    assert win["start_hour"] == 10
    assert win["end_hour"] == 12


def test_typical_window_single_hour():
    """All visits in one hour -> a degenerate width-1 window on that hour."""
    win = behavior.typical_window({3: 10}, coverage=0.8)
    assert win == {"start_hour": 3, "end_hour": 3, "width_hours": 1}


def test_typical_window_empty_is_none():
    assert behavior.typical_window({}, coverage=0.8) is None
    assert behavior.typical_window({5: 0}, coverage=0.8) is None


def test_typical_window_full_coverage_can_span_all():
    """coverage 1.0 over hours on opposite sides of the circle needs the whole populated arc."""
    win = behavior.typical_window({0: 1, 12: 1}, coverage=1.0)
    assert win is not None
    # Tightest arc covering both 0h and 12h is 12 wide either way (12->0 or 0->12).
    assert win["width_hours"] == 13


# --- species_profile shape --------------------------------------------------------------

def _visit(conn, *, species, started_at, ended_at, detection_count=3, max_conf=0.9,
           individual_id=None, source=db.SOURCE_GLASS_DOOR_CAM):
    return db.insert_visit(
        conn, source=source, species=species, individual_id=individual_id,
        started_at=started_at, ended_at=ended_at, detection_count=detection_count,
        max_confidence=max_conf, representative_detection_id=None,
    )


def test_species_profile_shape_and_values(conn):
    """species_profile returns the expected keys; counts/derived fields are correct on a small set
    of three raccoon visits across two days at 20h, 21h, 22h with ~60s dwell each."""
    _visit(conn, species="raccoon", started_at="2026-06-07T20:00:00-07:00",
           ended_at="2026-06-07T20:01:00-07:00", detection_count=4)
    _visit(conn, species="raccoon", started_at="2026-06-07T21:00:00-07:00",
           ended_at="2026-06-07T21:01:00-07:00", detection_count=2)
    _visit(conn, species="raccoon", started_at="2026-06-08T22:00:00-07:00",
           ended_at="2026-06-08T22:01:00-07:00", detection_count=6)
    conn.commit()

    p = behavior.species_profile(conn, "raccoon")
    assert p is not None
    # shape
    for key in ("label", "n_visits", "active_days", "visits_per_day", "first_seen", "last_seen",
                "arrival_hours", "peak_hour", "typical_window", "dwell_median_s", "dwell_max_s",
                "crops_per_visit"):
        assert key in p, f"missing key {key}"
    # values
    assert p["label"] == "raccoon"
    assert p["n_visits"] == 3
    assert p["active_days"] == 2                     # 06-07 and 06-08
    assert p["arrival_hours"] == {20: 1, 21: 1, 22: 1}
    assert p["dwell_median_s"] == 60                 # each visit lasts 60s
    assert p["dwell_max_s"] == 60
    assert p["crops_per_visit"] == round((4 + 2 + 6) / 3, 1)
    assert p["visits_per_day"] == round(3 / 2, 1)
    assert p["peak_hour"] in (20, 21, 22)            # all tied at 1; any is acceptable


def test_species_profile_none_when_no_visits(conn):
    assert behavior.species_profile(conn, "no-such-species") is None


def test_individual_profile_selects_by_individual(conn):
    _visit(conn, species="raccoon", individual_id="Notch",
           started_at="2026-06-07T20:00:00-07:00", ended_at="2026-06-07T20:00:30-07:00")
    _visit(conn, species="raccoon", individual_id="Gimpy",
           started_at="2026-06-07T21:00:00-07:00", ended_at="2026-06-07T21:00:30-07:00")
    conn.commit()
    p = behavior.individual_profile(conn, "Notch")
    assert p is not None
    assert p["n_visits"] == 1
    assert p["label"] == "Notch"


# --- co_occurrence ----------------------------------------------------------------------

def _det_in_visit(conn, *, visit_id, species, minutes):
    """Insert a detection already stamped with a visit_id and a classified species."""
    det_id = db.insert_detection(
        conn, timestamp=f"2026-06-07T20:{minutes:02d}:00-07:00",
        source=db.SOURCE_GLASS_DOOR_CAM, detection_class="animal", confidence=0.8,
        bbox=(0, 0, 10, 10), frame_w=100, frame_h=100, crop_path=f"c{minutes}.jpg",
        species=species,
    )
    db.assign_visit(conn, [det_id], visit_id)
    conn.commit()
    return det_id


def test_co_occurrence_pairs_species_sharing_a_visit(conn):
    """Two species among one visit's crops -> a sorted pair counted once. A solo-species visit
    contributes nothing."""
    v1 = _visit(conn, species="crow", started_at="2026-06-07T20:00:00-07:00",
                ended_at="2026-06-07T20:05:00-07:00")
    v2 = _visit(conn, species="raccoon", started_at="2026-06-07T21:00:00-07:00",
                ended_at="2026-06-07T21:05:00-07:00")
    # visit 1: crow + raccoon together.
    _det_in_visit(conn, visit_id=v1, species="crow", minutes=0)
    _det_in_visit(conn, visit_id=v1, species="raccoon", minutes=1)
    # visit 2: raccoon only.
    _det_in_visit(conn, visit_id=v2, species="raccoon", minutes=2)

    pairs = behavior.co_occurrence(conn)
    assert pairs[("crow", "raccoon")] == 1   # sorted tuple, counted once
    assert len(pairs) == 1                   # the solo visit added no pair


def test_co_occurrence_excludes_denylisted_labels(conn):
    """A non-critter label (e.g. 'door' / 'person') is dropped before pairing, so it never forms a
    spurious co-occurrence with a real animal."""
    v = _visit(conn, species="raccoon", started_at="2026-06-07T20:00:00-07:00",
               ended_at="2026-06-07T20:05:00-07:00")
    _det_in_visit(conn, visit_id=v, species="raccoon", minutes=0)
    _det_in_visit(conn, visit_id=v, species="door", minutes=1)      # denylisted
    pairs = behavior.co_occurrence(conn)
    assert pairs == {} or ("door", "raccoon") not in pairs
    assert len(pairs) == 0


def test_co_occurrence_empty_when_no_stamped_detections(conn):
    """Detections with no visit_id or no species don't enter co-occurrence."""
    db.insert_detection(
        conn, timestamp="2026-06-07T20:00:00-07:00", source=db.SOURCE_GLASS_DOOR_CAM,
        detection_class="animal", confidence=0.8, bbox=(0, 0, 10, 10), frame_w=100,
        frame_h=100, crop_path="c.jpg", species="raccoon",   # but no visit_id stamp
    )
    conn.commit()
    assert behavior.co_occurrence(conn) == {}
