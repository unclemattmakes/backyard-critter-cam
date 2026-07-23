"""
Tests for classify.py's visit-ledger refresh wiring.

THE regression this guards (2026-07-22): a trail-cam batch import builds visits while species
are still NULL; classify.py then labels the crops, and nothing rebuilt the ledger -- 90 of 113
trail-cam visits sat species-less until a manual `python visits.py`. classify now refreshes the
ledger itself: at the end of a one-shot run, and in --watch when a naming backlog drains (the
TRAILING edge, so live naming isn't rebuilding once per poll while crops stream in).

No ML here (suite rule): build_classifier / build_nonanimal_filter are monkeypatched to a stub
that labels every crop 'raccoon'. Crop files must exist on disk (classify_rows drops missing
paths), so tiny placeholder files stand in -- the stub never opens them. The naming status file
is pointed into tmp so tests never touch the project-root one a live dashboard may be reading.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta

import pytest

import classify
import config
import db
import visits

BASE = datetime(2026, 7, 22, 21, 0, 0, tzinfo=datetime.now().astimezone().tzinfo)


class _StubClassifier:
    """BioCLIP stand-in: labels every path 'raccoon' at 0.9 without touching a model or a file."""

    def predict(self, paths):
        return [{"file_name": p, "classification": "raccoon", "score": 0.9} for p in paths]


class _StopAfter:
    """threading.Event stand-in that lets watch_loop run exactly `polls` polls with zero sleep."""

    def __init__(self, polls: int):
        self.polls = polls

    def is_set(self) -> bool:
        self.polls -= 1
        return self.polls < 0

    def wait(self, timeout) -> None:
        pass


@pytest.fixture
def stub_naming(monkeypatch, tmp_path):
    monkeypatch.setattr(classify, "build_classifier", lambda device: (_StubClassifier(), "cpu"))
    monkeypatch.setattr(classify, "build_nonanimal_filter", lambda device: None)
    monkeypatch.setattr(config, "NAMING_STATUS_FILE", tmp_path / "naming_status.json")


def _unlabeled_det(conn, tmp_path, minutes: float) -> int:
    """One trail-cam detection with species NULL whose crop file exists (as a placeholder)."""
    crop = tmp_path / f"crop_{minutes}.jpg"
    crop.write_bytes(b"placeholder")
    return db.insert_detection(
        conn, timestamp=(BASE + timedelta(minutes=minutes)).isoformat(),
        source=db.SOURCE_TRAIL_CAM_SD, detection_class="animal", confidence=0.9,
        bbox=(0, 0, 10, 10), frame_w=100, frame_h=100, crop_path=str(crop),
    )


def _import_time_build(conn) -> None:
    """The state import_trailcam leaves behind: visits exist, but species-less."""
    visits.build_visits(conn, gap_minutes=5, verbose=False)
    assert conn.execute("SELECT species FROM visits").fetchone()[0] is None


def test_watch_drain_refreshes_visit_ledger(conn, tmp_path, stub_naming):
    """Poll 1 names the backlog; poll 2 finds nothing pending -- the trailing edge -- and must
    fold the fresh labels into the ledger, so an import processed by the rig's naming helper
    ends with labeled visits on its own."""
    _unlabeled_det(conn, tmp_path, minutes=0)
    _unlabeled_det(conn, tmp_path, minutes=1)
    _import_time_build(conn)

    classify.watch_loop(conn, device="cpu", interval=0, stop_event=_StopAfter(2))

    rows = conn.execute("SELECT species, detection_count FROM visits").fetchall()
    assert [tuple(r) for r in rows] == [("raccoon", 2)]


def test_watch_stop_mid_burst_still_refreshes(conn, tmp_path, stub_naming):
    """Stopped right after a naming pass, before any quiet poll (standalone --watch Ctrl-C during
    a backlog): the exit path must still fold the written labels in rather than strand them."""
    _unlabeled_det(conn, tmp_path, minutes=0)
    _import_time_build(conn)

    classify.watch_loop(conn, device="cpu", interval=0, stop_event=_StopAfter(1))

    assert conn.execute("SELECT species FROM visits").fetchone()[0] == "raccoon"


def test_one_shot_run_ends_with_labeled_visits(conn, db_path, tmp_path, stub_naming, monkeypatch):
    """The documented trail-cam pipeline: import (visits built, species NULL), then
    `python classify.py`. The one-shot run must end by refreshing the ledger."""
    _unlabeled_det(conn, tmp_path, minutes=0)
    _unlabeled_det(conn, tmp_path, minutes=1)
    _import_time_build(conn)
    conn.commit()

    monkeypatch.setattr(config.CONFIG, "db_path", db_path)  # main() opens its own connection
    monkeypatch.setattr(sys, "argv", ["classify.py"])
    assert classify.main() == 0

    assert conn.execute("SELECT species FROM visits").fetchone()[0] == "raccoon"
