"""
Shared pytest fixtures for the backyard-critter test suite.

Boring and robust, stdlib-first (matching the project): every test gets a throwaway
tempfile SQLite DB built by db.connect() -- the real backyard.db is NEVER touched. The
project modules (db, visits, behavior, twoaxis, clips, config) live in the project ROOT,
so we put that root on sys.path here rather than turning the package into something
installable.

Scope of the suite: pure logic only. Nothing here loads the camera, the network, a GPU, or
any ML model (MegaDetector / BioCLIP / MegaDescriptor) -- but "no model" is not the same as
"no dependency", and the difference bites in CI. The suite needs cv2 (clips, quality,
import_trailcam, backyard_cam), numpy (clipembed, clipmotion, eval, evalmetrics, individuals,
reid, reidutil) and torch (test_clipfilter, which hand-builds unit vectors rather than loading
open_clip). Everything else is stdlib. torch is the only one guarded by importorskip, so a
lean environment loses one test file instead of collapsing at collection.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

# The project root is the parent of this tests/ directory. Putting it first on sys.path lets
# `import db`, `import visits`, ... resolve to the live modules without an install step.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import db  # noqa: E402  (import after sys.path tweak, on purpose)


@pytest.fixture
def db_path(tmp_path) -> Path:
    """A path to a not-yet-created SQLite DB inside pytest's per-test tmp dir. The file (plus any
    -wal/-shm WAL siblings) is cleaned up with tmp_path, so the real backyard.db is never at risk."""
    return tmp_path / "test_backyard.db"


@pytest.fixture
def conn(db_path):
    """An open connection to a fresh DB with the full schema (db.connect runs SCHEMA + _migrate).
    row_factory is sqlite3.Row because visits.py / behavior.py / twoaxis.py read columns by name."""
    c = db.connect(db_path)
    c.row_factory = sqlite3.Row
    try:
        yield c
    finally:
        c.close()
