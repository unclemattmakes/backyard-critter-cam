"""
Power + USB-wedge guard for the live rig (the 2026-07-29..31 evening failures).

Two related defences, one module:

1. PowerMonitor -- warn LOUDLY while the box runs on battery. The rig's keep-awake
   (PowerRequestExecutionRequired, backyard_cam.keep_system_awake) only wins on AC power: on
   battery, Windows idle-naps into Modern Standby at the DC timeout (~15 min) no matter what
   requests we hold. Each nap suspend-cycles the USB camera, and the 2026-07 glass-door unit
   eventually comes back WEDGED: the UVC stream still delivers frames, but they're torn garbage.
   Every wedge so far followed an on-battery afternoon/evening; every on-AC overnight run was
   clean. So the cheapest real fix is the human one -- plug the charger in -- and this monitor
   makes that impossible to miss (console + preview HUD + dashboard banner).

2. WedgeDetector + SelfHealer -- when the wedge happens anyway, recognise it and reset the
   camera at the DEVICE level. App-level recovery is proven useless against a wedge: on
   2026-07-30 the rig re-opened the capture 19x and re-asserted AUTO_WB 5x against one (R/G
   pinned 0.42-0.44) and lost; a physical unplug/replug fixed it on the first try. pnputil can
   do the same disable/enable cycle in software, but needs admin -- so the heal path runs
   through a one-time-registered elevated scheduled task (usb_reset.ps1 -Setup, via
   setup_selfheal.bat) that a normal user may *start*. The rig just fires `schtasks /run`.

Wedge signatures (from the two observed variants -- see the glass-door notes):
  * 07-30 "striped static": R/G collapses to ~0.42-0.44 and STAYS there through AUTO_WB
    re-asserts (the ordinary manual-WB trap recovers in ~2 s) -> the failed-recovery streak
    the WhiteBalanceWatchdog now tracks.
  * 07-31 "torn slabs": the largest motion blob pins near the WHOLE frame (~98%) for minutes
    while MegaDetector finds nothing in the garbage -> the pegged-motion rule. A real animal
    filling the view is excluded twice over: it doesn't hold >=55% of the frame in one blob
    for a full minute, and it IS detected -- any MegaDetector hit inside the pegged window
    vetoes the trigger. (The 07-30 variant fails the no-detections test -- its noise scored
    "animal" 0.77-0.89 -- which is exactly why the WB rule exists alongside this one.)

This module is deliberately free of cv2/torch imports so its logic is unit-testable
(tests/test_powerguard.py) without the rig's heavyweight dependencies.
"""
from __future__ import annotations

import subprocess
import sys
import threading


# ---- AC / battery state ------------------------------------------------------------
def interpret_power(ac_line: int, pct: int) -> tuple[bool | None, int | None]:
    """Map GetSystemPowerStatus fields to (on_ac, battery_percent), None-ing the unknowns.
    ACLineStatus: 0 = battery, 1 = AC, 255 = unknown. BatteryLifePercent: 0..100, 255 = unknown."""
    on_ac = {0: False, 1: True}.get(int(ac_line))
    p = int(pct)
    return on_ac, (p if 0 <= p <= 100 else None)


def power_status() -> tuple[bool | None, int | None]:
    """(on_ac, battery_percent) via GetSystemPowerStatus. (None, None) off Windows or on any
    failure -- callers treat unknown as 'nothing to warn about' rather than guessing."""
    if sys.platform != "win32":
        return None, None
    try:
        import ctypes

        class _SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte),
                        ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte),
                        ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_uint32),
                        ("BatteryFullLifeTime", ctypes.c_uint32)]

        st = _SYSTEM_POWER_STATUS()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
            return None, None
        return interpret_power(st.ACLineStatus, st.BatteryLifePercent)
    except Exception:
        return None, None


class PowerMonitor:
    """Poll AC/battery, log transitions, and hold a one-line warning for the HUD/dashboard.

    The warning repeats in the console every cfg.power_warn_repeat_s while on battery (a single
    scroll-away line at 2 PM doesn't help anyone at 7 PM), but the HUD/banner line is simply
    present the whole time. All state is plain attribute reads cross-thread (GIL-atomic)."""

    def __init__(self, cfg, notify=print, reader=power_status):
        self.cfg = cfg
        self.notify = notify
        self._read = reader                  # injectable for tests
        self.enabled = bool(cfg.power_warn)
        self.on_ac: bool | None = None
        self.battery_pct: int | None = None
        self.warning: str | None = None      # one HUD/banner line while on battery, else None
        self._was_on_ac: bool | None = None
        self._last_warn = 0.0

    def poll(self, now: float) -> None:
        self.on_ac, self.battery_pct = self._read()
        if not self.enabled or self.on_ac is None:
            self.warning = None
            return
        if self.on_ac:
            if self._was_on_ac is False:
                self.notify("  power: back on AC -- the keep-awake holds again; standby naps "
                            "can't suspend the camera.")
            self.warning = None
        else:
            pct = f" ({self.battery_pct}%)" if self.battery_pct is not None else ""
            self.warning = f"ON BATTERY{pct} -- plug in: standby naps wedge the camera"
            first = self._was_on_ac is not False
            if first or (now - self._last_warn) >= self.cfg.power_warn_repeat_s:
                self._last_warn = now
                self.notify(
                    f"  power: WARNING -- running on BATTERY{pct}. The keep-awake only wins on "
                    "AC: on battery the box\n"
                    "  power: naps into Modern Standby (~15 min idle) and suspend-cycles the "
                    "camera until it wedges\n"
                    "  power: (the 07-29..31 evening failures). Plug the charger in.")
        self._was_on_ac = self.on_ac

    def monitor(self, stop_event: threading.Event) -> None:
        """Thread target: poll until shutdown. Uses stop_event.wait (not sleep) so 'q' ends it."""
        import time
        while not stop_event.is_set():
            self.poll(time.monotonic())
            stop_event.wait(self.cfg.power_poll_s)

    def snapshot(self) -> dict:
        """For the dashboard's /api/camera payload."""
        return {"on_ac": self.on_ac, "battery_pct": self.battery_pct, "warning": self.warning}


# ---- Device-level self-heal --------------------------------------------------------
class SelfHealer:
    """Fire the one-time-registered elevated task that pnputil-cycles the wedged camera.

    The task (cfg.wedge_heal_task, registered by usb_reset.ps1 -Setup) runs as SYSTEM and
    disable/enable-cycles the camera device -- the software equivalent of the unplug/replug
    that cured the 07-30 wedge. Starting it needs no elevation (setup grants normal users
    run rights), so the rig can trigger it unattended. Budgeted: at most
    cfg.wedge_heal_max_per_hour hardware resets an hour -- past that, something deeper is
    wrong and the rig asks for a human replug instead of churning the device."""

    def __init__(self, cfg, tag: str = "", notify=print, runner=subprocess.run):
        self.cfg = cfg
        self.tag = tag
        self.notify = notify
        self._runner = runner                # injectable for tests
        self._available: bool | None = None  # lazy schtasks probe, cached for the session
        self._attempts: list[float] = []     # monotonic times of triggered heals
        self.last_thread: threading.Thread | None = None

    def available(self) -> bool:
        """Is the heal path usable at all? (Windows + enabled + the task is registered.)"""
        if sys.platform != "win32" or not self.cfg.wedge_self_heal:
            return False
        if self._available is None:
            try:
                r = self._runner(["schtasks", "/query", "/tn", self.cfg.wedge_heal_task],
                                 capture_output=True, text=True, timeout=10)
                self._available = (r.returncode == 0)
            except Exception:
                self._available = False
        return self._available

    def budget_left(self, now: float) -> bool:
        self._attempts = [t for t in self._attempts if (now - t) < 3600.0]
        return len(self._attempts) < self.cfg.wedge_heal_max_per_hour

    def try_heal(self, now: float, reason: str | None) -> bool:
        """Trigger a device reset (async -- never blocks the capture loop). False when the
        task isn't registered or the hourly budget is spent; the caller banners instead."""
        if not self.available() or not self.budget_left(now):
            return False
        self._attempts.append(now)
        self.notify(f"{self.tag} triggering the elevated USB reset task "
                    f"'{self.cfg.wedge_heal_task}' (attempt {len(self._attempts)} this hour) "
                    f"-- reason: {reason}.")
        t = threading.Thread(target=self._run, name="usb-heal", daemon=True)
        self.last_thread = t
        t.start()
        return True

    def _run(self) -> None:
        try:
            r = self._runner(["schtasks", "/run", "/tn", self.cfg.wedge_heal_task],
                             capture_output=True, text=True, timeout=30)
        except Exception as e:
            self.notify(f"{self.tag} could not start the USB reset task: {e}")
            return
        if r.returncode != 0:
            err = ((r.stderr or "") + (r.stdout or "")).strip().splitlines()
            self.notify(f"{self.tag} USB reset task refused to start "
                        f"({err[0] if err else f'rc={r.returncode}'}). If that's 'Access is "
                        "denied', re-run setup_selfheal.bat once (it grants normal users the "
                        "right to start the task).")
        else:
            self.notify(f"{self.tag} USB reset task started -- expect the camera to drop and "
                        "reconnect within ~15 s (logs/usb_reset.log has the device-side story).")


# ---- Wedge detection ---------------------------------------------------------------
class WedgeDetector:
    """Recognise the wedged-camera state from the capture loop's own cheap signals and drive
    the recovery ladder: detect -> self-heal (device reset) -> ask for a human replug.

    Fed from the worker thread each loop:
      note_motion(now, frac)   largest-blob fraction of the frame (post-warmup reads only)
      note_detections(now)     MegaDetector returned >= 1 box (pre ignore-zone filtering)
      note_wb(streak)          WhiteBalanceWatchdog.failed_streak (recoveries that didn't take)
      note_reconnect(now)      the capture was reopened -- motion evidence is stale
      update(now)              advance the state machine (cheap; every loop)

    States: 'ok' -> 'healing' (reset task fired; wedged is a momentary step in between) ->
    'replug' (no task / resets didn't hold). Any state returns to 'ok' after
    cfg.wedge_clear_s of signature-free frames -- including 'replug', so the banner drops on
    its own once the human replug (or a lucky recovery) happens."""

    OK, WEDGED, HEALING, REPLUG = "ok", "wedged", "healing", "replug"

    def __init__(self, cfg, tag: str = "", healer: SelfHealer | None = None, notify=print):
        self.cfg = cfg
        self.tag = tag
        self.healer = healer
        self.notify = notify
        self.enabled = bool(cfg.wedge_guard)
        self.state = self.OK
        self.message: str | None = None      # one HUD/banner line while not ok, else None
        self.reason: str | None = None
        self._pegged_since: float | None = None
        self._last_det_t: float | None = None
        self._wb_failed = 0                  # mirror of the watchdog's failed-recovery streak
        self._wb_baseline = 0                # streak level already 'spent' by the last heal
        self._clean_since: float | None = None
        self._heal_started: float | None = None
        self._hint_shown = False

    # -- feeds -----------------------------------------------------------------------
    def note_motion(self, now: float, frac: float) -> None:
        if frac >= self.cfg.wedge_motion_frac:
            if self._pegged_since is None:
                self._pegged_since = now
        else:
            self._pegged_since = None

    def note_detections(self, now: float) -> None:
        self._last_det_t = now

    def note_wb(self, failed_streak: int) -> None:
        s = int(failed_streak)
        if s < self._wb_failed:
            # The watchdog itself reset (a good bright frame): the stale credit in the
            # baseline goes with it, so the NEXT failure counts from zero again.
            self._wb_baseline = 0
        self._wb_failed = s

    def note_reconnect(self, now: float) -> None:
        # A fresh capture invalidates the motion evidence (the caller also rebuilds the gate,
        # so post-reconnect frames re-learn the scene instead of reading as one giant blob).
        # A reconnect also means the device RE-ENUMERATED (a replug, a heal's reset, a USB
        # hiccup) -- WB failures measured against the old enumeration are stale too, so only
        # failures accrued from here on count. Without this, a human replug at NIGHT (when the
        # watchdog can't judge dark frames, so its streak stays frozen) leaves the wedge banner
        # stuck until morning.
        self._pegged_since = None
        self._wb_baseline = self._wb_failed

    # -- rules -----------------------------------------------------------------------
    def _wb_evidence(self) -> int:
        # Streak accrued SINCE the last heal attempt. The raw streak can survive a successful
        # night-time heal unrefuted (too dark for the watchdog to judge, so it never resets);
        # only failures beyond the baseline captured at heal time count as fresh evidence.
        return max(0, self._wb_failed - self._wb_baseline)

    def _signature(self, now: float) -> str | None:
        if self._wb_evidence() >= self.cfg.wedge_wb_failures:
            return (f"white balance pinned through {self._wb_failed} AUTO_WB re-assert(s) "
                    "(a real WB trap recovers in ~2 s)")
        if self._pegged_since is not None \
                and (now - self._pegged_since) >= self.cfg.wedge_motion_sustain_s \
                and not (self._last_det_t is not None and self._last_det_t >= self._pegged_since):
            return (f"largest motion blob >= {self.cfg.wedge_motion_frac:.0%} of the frame for "
                    f"{self.cfg.wedge_motion_sustain_s:.0f}s with zero detections in it")
        return None

    # -- state machine -----------------------------------------------------------------
    def update(self, now: float) -> None:
        if not self.enabled:
            return
        # The all-clear clock: no signature COMPONENT active at all (not merely un-triggered).
        if self._pegged_since is None and self._wb_evidence() == 0:
            if self._clean_since is None:
                self._clean_since = now
        else:
            self._clean_since = None

        if self.state == self.OK:
            sig = self._signature(now)
            if sig:
                self.reason = sig
                self.notify(
                    f"{self.tag} CAMERA WEDGE detected -- {sig}.\n"
                    f"{self.tag} the camera is delivering garbage frames (the 07-30/07-31 USB "
                    "wedge); app-level reopens cannot fix this.")
                self._escalate(now)
            return

        # Wedged/healing/replug: healthy again once the signature stays gone for a while.
        if self._clean_since is not None and (now - self._clean_since) >= self.cfg.wedge_clear_s:
            how = ("the USB device reset did it -- no replug needed"
                   if self.state == self.HEALING else "signature gone")
            self.notify(f"{self.tag} camera healthy again ({how}; clean for "
                        f"{self.cfg.wedge_clear_s:.0f}s).")
            self.state, self.message, self.reason = self.OK, None, None
            self._heal_started = None
            # Absorb (don't zero) the watchdog streak: after a night-time heal it stays frozen
            # at its pre-heal value until a bright frame resets it, and zeroing the baseline
            # here would read that frozen number as fresh evidence and re-wedge immediately.
            self._wb_baseline = self._wb_failed
            return

        if self.state == self.HEALING:
            # Give the reset time to happen (device cycle + reconnect + gate re-learn) before
            # believing a still-wedged signature; then escalate to the next rung.
            if (now - self._heal_started) >= self.cfg.wedge_heal_verify_s:
                sig = self._signature(now)
                if sig:
                    self.reason = sig
                    self.notify(f"{self.tag} the USB reset did not clear the wedge ({sig}).")
                    self._escalate(now)
            return
        # REPLUG is terminal until the signature clears (above) or the rig restarts: the wedge
        # already survived our resets, so more hardware churn without a human helps nothing.

    def _escalate(self, now: float) -> None:
        """One rung up the ladder: fire a device reset if we can, else ask for hands."""
        h = self.healer
        if h is not None and h.try_heal(now, self.reason):
            self.state = self.HEALING
            self._heal_started = now
            self._wb_baseline = self._wb_failed   # only NEW failures count against this heal
            self._pegged_since = None             # fresh evidence only, post-reset
            self.message = "CAMERA WEDGED -- resetting its USB device"
            return
        self.state = self.REPLUG
        if h is None or not h.available():
            self.message = "CAMERA WEDGED -- unplug/replug its USB cable"
            if not self._hint_shown:
                self._hint_shown = True
                self.notify(f"{self.tag} self-heal is not set up (scheduled task "
                            f"'{self.cfg.wedge_heal_task}' not found). Run setup_selfheal.bat "
                            "once, as admin, to let the rig reset the device itself next time.")
        else:
            self.message = "CAMERA WEDGED -- resets didn't hold; unplug/replug the cable"
            self.notify(f"{self.tag} out of reset attempts (max "
                        f"{self.cfg.wedge_heal_max_per_hour}/hour) -- the cable needs a human: "
                        "unplug/replug the camera's USB.")

    def snapshot(self) -> dict:
        """For the dashboard's /api/camera payload."""
        return {"state": self.state, "message": self.message}
