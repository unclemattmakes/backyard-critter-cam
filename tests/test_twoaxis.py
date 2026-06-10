"""
Tests for twoaxis.py -- the two-axis appearance/behaviour read-out.

We test the pure logic that doesn't need appearance vectors or a GPU:
  * _in_window(hour, win) -- circular membership for both normal (start <= end) and midnight-
    wrapping (start > end) typical windows.
  * species_fit(visit, prof) -- the behaviour verdict: FITS when arrival is inside the species'
    usual window, DISAGREES when it isn't, and 'thin' when the profile has < 3 visits to judge by.

Both functions read plain dicts (a `visit` exposes started_at / ended_at / species like a
sqlite3.Row; a `prof` is a behavior.species_profile-shaped dict), so no DB is required here.
Importing twoaxis is cheap -- it pulls behavior + stats only, never cv2/torch.
"""
from __future__ import annotations

import twoaxis


# --- _in_window: circular membership ----------------------------------------------------

def test_in_window_normal_block():
    """A non-wrapping window (start <= end) is a plain inclusive range."""
    win = {"start_hour": 9, "end_hour": 17}
    assert twoaxis._in_window(9, win) is True      # left edge inclusive
    assert twoaxis._in_window(13, win) is True
    assert twoaxis._in_window(17, win) is True     # right edge inclusive
    assert twoaxis._in_window(8, win) is False
    assert twoaxis._in_window(18, win) is False


def test_in_window_wrapping_midnight():
    """A window that wraps midnight (start > end) admits hours >= start OR <= end."""
    win = {"start_hour": 22, "end_hour": 1}
    assert twoaxis._in_window(22, win) is True
    assert twoaxis._in_window(23, win) is True
    assert twoaxis._in_window(0, win) is True
    assert twoaxis._in_window(1, win) is True      # right edge inclusive
    assert twoaxis._in_window(2, win) is False     # the daytime gap
    assert twoaxis._in_window(12, win) is False
    assert twoaxis._in_window(21, win) is False    # just before the window opens


def test_in_window_none_is_permissive():
    """No window (None / empty) -> everything is 'in' (can't judge arrival, so don't flag it)."""
    assert twoaxis._in_window(3, None) is True
    assert twoaxis._in_window(15, {}) is True


# --- species_fit verdicts ---------------------------------------------------------------

def _prof(**over):
    """A behaviour profile with enough visits to judge (n_visits >= 3) and a crepuscular window."""
    p = {
        "label": "raccoon",
        "n_visits": 10,
        "typical_window": {"start_hour": 20, "end_hour": 23, "width_hours": 4},
        "dwell_median_s": 60,
    }
    p.update(over)
    return p


def _visit(started_at, ended_at, species="raccoon"):
    return {"started_at": started_at, "ended_at": ended_at, "species": species}


def test_species_fit_fits_inside_window():
    """A raccoon arriving at 21h (inside the usual 20-23h window) FITS."""
    visit = _visit("2026-06-07T21:00:00-07:00", "2026-06-07T21:01:00-07:00")
    verdict, notes = twoaxis.species_fit(visit, _prof())
    assert verdict == "FITS"
    assert any("OK" in n for n in notes)            # arrival noted as OK


def test_species_fit_disagrees_outside_window():
    """A 'raccoon' at 11am DISAGREES with the species' crepuscular pattern (mis-class or unusual)."""
    visit = _visit("2026-06-07T11:00:00-07:00", "2026-06-07T11:01:00-07:00")
    verdict, notes = twoaxis.species_fit(visit, _prof())
    assert verdict == "DISAGREES"
    assert any("UNUSUAL" in n for n in notes)


def test_species_fit_thin_profile():
    """A profile with fewer than 3 visits can't be judged -> 'thin'."""
    visit = _visit("2026-06-07T21:00:00-07:00", "2026-06-07T21:01:00-07:00")
    verdict, notes = twoaxis.species_fit(visit, _prof(n_visits=2))
    assert verdict == "thin"
    assert notes == ["species profile too thin to judge"]


def test_species_fit_none_profile_is_thin():
    """No profile at all (unclassified / unseen species) is also 'thin'."""
    visit = _visit("2026-06-07T21:00:00-07:00", "2026-06-07T21:01:00-07:00")
    verdict, _ = twoaxis.species_fit(visit, None)
    assert verdict == "thin"


def test_species_fit_wrapping_window_accepts_after_midnight():
    """With a midnight-wrapping window (22-01h), a 00:30 arrival FITS -- the circular logic holds
    end to end through species_fit, not just in _in_window."""
    prof = _prof(typical_window={"start_hour": 22, "end_hour": 1, "width_hours": 4})
    visit = _visit("2026-06-07T00:30:00-07:00", "2026-06-07T00:31:00-07:00")
    verdict, _ = twoaxis.species_fit(visit, prof)
    assert verdict == "FITS"


def test_species_fit_verdict_driven_by_arrival_not_dwell():
    """Timing is the strong species signal: an in-window arrival FITS even with an odd dwell."""
    prof = _prof(dwell_median_s=60)
    # 30-minute dwell (1800s) is 30x the median, but arrival at 21h is in-window -> still FITS.
    visit = _visit("2026-06-07T21:00:00-07:00", "2026-06-07T21:30:00-07:00")
    verdict, notes = twoaxis.species_fit(visit, prof)
    assert verdict == "FITS"
    # the dwell note still reports the ratio (not OK), even though it doesn't flip the verdict.
    assert any("dwell" in n for n in notes)
