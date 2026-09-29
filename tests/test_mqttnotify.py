"""
Tests for mqttnotify.py -- live sighting alerts over MQTT.

Two halves. The TRANSPORT is a hand-rolled MQTT 3.1.1 publisher, so it is exercised against a
tiny in-process fake broker on a real localhost socket (no network beyond loopback, no broker
install). The DETECTOR -- what counts as an arrival, when a visit has left, what must stay quiet --
is tested with an injected sender, so each test reads as "given these crops at these times, these
events go out".
"""
from __future__ import annotations

import json
import socket
import struct
import threading
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

import db
import mqttnotify
from clipfilter import NONANIMAL_LABEL

NOW = datetime(2026, 9, 28, 21, 0, 0, tzinfo=datetime.now().astimezone().tzinfo)


def _cfg(**over):
    base = dict(mqtt_host="broker.test", mqtt_port=1883, mqtt_username="critter-cam",
                mqtt_password="pw", mqtt_topic_prefix="critter-cam", mqtt_alert_species="*",
                mqtt_min_confidence=0.8, mqtt_min_crops=2, mqtt_window_s=120.0,
                mqtt_sources=None, visit_gap_minutes=5.0)
    base.update(over)
    return SimpleNamespace(**base)


class _Sink:
    """Sender stand-in: records every batch of (topic, payload, retain)."""

    def __init__(self, fail=False):
        self.batches = []
        self.fail = fail

    def __call__(self, msgs):
        if self.fail:
            raise OSError("connection refused")
        self.batches.append([(t, json.loads(p), r) for t, p, r in msgs])

    def events(self):
        return [p for b in self.batches for t, p, r in b if "/sighting/" in t]


def _crop(conn, *, seconds_ago, species="raccoon", conf=0.95, source="glass_door_cam"):
    rid = db.insert_detection(
        conn, timestamp=(NOW - timedelta(seconds=seconds_ago)).isoformat(), source=source,
        detection_class="animal", confidence=0.9, bbox=(0, 0, 10, 10), frame_w=100, frame_h=100,
        crop_path="crops/x.jpg")
    db.set_species(conn, rid, species, conf)
    conn.commit()
    return rid


def _notifier(sink=None, **over):
    return mqttnotify.SightingNotifier(_cfg(**over), sender=sink if sink is not None else _Sink())


# ---- what counts as an arrival ---------------------------------------------------------------

def test_two_confident_crops_announce_an_arrival(conn):
    sink = _Sink()
    n = _notifier(sink)
    _crop(conn, seconds_ago=10)
    _crop(conn, seconds_ago=5)

    events = n.poll(conn, now=NOW)

    assert [(e["event"], e["species"], e["source"], e["crops"]) for e in events] == \
        [("arrived", "raccoon", "glass_door_cam", 2)]
    topics = {t: (p, r) for t, p, r in sink.batches[-1]}
    assert topics["critter-cam/sighting/raccoon"][1] is False          # events are not retained
    assert topics["critter-cam/present"] == (
        {"present": [{"species": "raccoon", "slug": "raccoon", "source": "glass_door_cam",
                      "since": (NOW - timedelta(seconds=10)).isoformat()}],
         "updated": NOW.isoformat()}, True)
    assert topics["critter-cam/last-sighting"][0]["event"] == "arrived"


def test_one_crop_is_not_enough(conn):
    n = _notifier()
    _crop(conn, seconds_ago=5)
    assert n.poll(conn, now=NOW) == []


def test_low_confidence_crops_do_not_count(conn):
    n = _notifier()
    _crop(conn, seconds_ago=10, conf=0.5)
    _crop(conn, seconds_ago=5, conf=0.79)
    _crop(conn, seconds_ago=4)
    assert n.poll(conn, now=NOW) == []


def test_old_crops_never_alert(conn):
    """A trail-cam import or a naming backlog labels crops from the past. Not 'right now'."""
    n = _notifier()
    _crop(conn, seconds_ago=3600)
    _crop(conn, seconds_ago=3590)
    _crop(conn, seconds_ago=121)
    _crop(conn, seconds_ago=5)
    assert n.poll(conn, now=NOW) == []


def test_species_filter_and_non_animal(conn):
    n = _notifier(mqtt_alert_species=["Raccoon"])
    for s in (10, 5):
        _crop(conn, seconds_ago=s, species="American crow")
        _crop(conn, seconds_ago=s, species=NONANIMAL_LABEL)
    assert n.poll(conn, now=NOW) == []
    _crop(conn, seconds_ago=4)
    _crop(conn, seconds_ago=3)
    assert [e["species"] for e in n.poll(conn, now=NOW)] == ["raccoon"]


def test_non_animal_label_is_excluded_even_with_wildcard(conn):
    n = _notifier()
    _crop(conn, seconds_ago=10, species=NONANIMAL_LABEL)
    _crop(conn, seconds_ago=5, species=NONANIMAL_LABEL)
    assert n.poll(conn, now=NOW) == []


def test_source_filter(conn):
    n = _notifier(mqtt_sources=["glass_door_cam"])
    _crop(conn, seconds_ago=10, source="yard_ir")
    _crop(conn, seconds_ago=5, source="yard_ir")
    assert n.poll(conn, now=NOW) == []


def test_two_species_at_once_are_two_arrivals(conn):
    n = _notifier()
    for s in (10, 5):
        _crop(conn, seconds_ago=s)
        _crop(conn, seconds_ago=s, species="Virginia opossum")
    assert sorted(e["slug"] for e in n.poll(conn, now=NOW)) == ["raccoon", "virginia-opossum"]


# ---- a visit's lifetime ----------------------------------------------------------------------

def test_a_lingering_visit_announces_once_then_leaves(conn):
    sink = _Sink()
    n = _notifier(sink)
    _crop(conn, seconds_ago=10)
    _crop(conn, seconds_ago=5)
    assert len(n.poll(conn, now=NOW)) == 1

    # Still here a minute later: more crops, no new event.
    later = NOW + timedelta(minutes=1)
    for s in (20, 10):
        _crop(conn, seconds_ago=-60 + s)
    assert n.poll(conn, now=later) == []

    # Five minutes of quiet after the last crop: it has left.
    gone = later + timedelta(minutes=5, seconds=1)
    events = n.poll(conn, now=gone)
    assert [(e["event"], e["crops"]) for e in events] == [("left", 4)]
    arrived = [p for b in sink.batches for t, p, r in b if t.endswith("/sighting/raccoon")][0]
    assert events[0]["visit_id"] == arrived["visit_id"], "one id across a visit's two events"
    assert events[0]["duration_s"] == 60                      # first crop -10s .. last crop +50s
    present = [p for t, p, r in sink.batches[-1] if t.endswith("/present")][0]
    assert present["present"] == []


def test_returning_after_leaving_is_a_new_arrival(conn):
    n = _notifier()
    _crop(conn, seconds_ago=10)
    _crop(conn, seconds_ago=5)
    n.poll(conn, now=NOW)
    t_left = NOW + timedelta(minutes=6)
    assert [e["event"] for e in n.poll(conn, now=t_left)] == ["left"]

    back = NOW + timedelta(minutes=10)
    _crop(conn, seconds_ago=-600 + 10)
    _crop(conn, seconds_ago=-600 + 5)
    again = n.poll(conn, now=back)
    assert [e["event"] for e in again] == ["arrived"]
    assert again[0]["visit_id"].startswith("glass_door_cam/raccoon/")
    assert again[0]["visit_id"] != "glass_door_cam/raccoon/" + \
        (NOW - timedelta(seconds=10)).strftime("%Y%m%dT%H%M%S"), "a return is a new visit"


def test_departed_visit_is_not_reannounced_from_its_own_crops(conn):
    """With a freshness window longer than the visit gap, the crops of a visit that just left
    are still 'fresh'. They must not turn straight back into an arrival."""
    n = _notifier(mqtt_window_s=900.0)
    _crop(conn, seconds_ago=10)
    _crop(conn, seconds_ago=5)
    n.poll(conn, now=NOW)
    assert [e["event"] for e in n.poll(conn, now=NOW + timedelta(minutes=6))] == ["left"]
    assert n.poll(conn, now=NOW + timedelta(minutes=7)) == []


def test_startup_clears_the_retained_present_list(conn):
    """After a crash the broker still holds the old retained list; the first poll replaces it,
    even with nothing in view."""
    sink = _Sink()
    n = _notifier(sink)
    n.poll(conn, now=NOW)
    assert [(t, p["present"], r) for t, p, r in sink.batches[0]] == \
        [("critter-cam/present", [], True)]
    n.poll(conn, now=NOW + timedelta(seconds=5))
    assert len(sink.batches) == 1, "a quiet poll after startup sends nothing"


# ---- failure modes ---------------------------------------------------------------------------

def test_disabled_without_a_host(conn):
    sink = _Sink()
    n = _notifier(sink, mqtt_host=None)
    _crop(conn, seconds_ago=10)
    _crop(conn, seconds_ago=5)
    assert n.poll(conn, now=NOW) == [] and sink.batches == []


def test_broker_down_never_raises_and_backs_off(conn):
    calls = []

    def down(msgs):
        calls.append(msgs)
        raise OSError("connection refused")

    n = mqttnotify.SightingNotifier(_cfg(), sender=down)
    _crop(conn, seconds_ago=10)
    _crop(conn, seconds_ago=5)
    assert [e["event"] for e in n.poll(conn, now=NOW)] == ["arrived"]   # detected, not sent
    assert len(calls) == 1
    n.poll(conn, now=NOW + timedelta(seconds=5))
    assert len(calls) == 1, "within the backoff, no reconnect attempt"
    n.poll(conn, now=NOW + timedelta(seconds=61))
    assert len(calls) == 2, "after the backoff, the retained present list is retried"


def test_slug():
    assert mqttnotify.slug("Steller's jay") == "stellers-jay"
    assert mqttnotify.slug("American crow") == "american-crow"
    assert mqttnotify.slug("a/b+#c") == "a-b-c"


# ---- the wire protocol, against a fake broker on loopback ------------------------------------

class _FakeBroker:
    """Accepts one connection, records CONNECT + PUBLISH packets, PUBACKs QoS 1."""

    def __init__(self, connack_rc=0):
        self.rc = connack_rc
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.connect = None
        self.publishes = []
        self.disconnected = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        s, _ = self.srv.accept()
        with s:
            s.settimeout(5)
            while True:
                try:
                    ptype, body = mqttnotify._read_packet(s)
                except (mqttnotify.MqttError, OSError):
                    return
                kind = ptype >> 4
                if kind == 1:
                    self.connect = body
                    s.sendall(bytes([0x20, 2, 0, self.rc]))
                    if self.rc:
                        return
                elif kind == 3:
                    tlen = struct.unpack("!H", body[:2])[0]
                    topic = body[2:2 + tlen].decode()
                    pid = body[2 + tlen:4 + tlen]
                    self.publishes.append((topic, body[4 + tlen:], bool(ptype & 1),
                                           (ptype >> 1) & 3))
                    s.sendall(b"\x40\x02" + pid)
                elif kind == 14:
                    self.disconnected = True
                    return

    def close(self):
        self.thread.join(5)
        self.srv.close()


def test_publish_speaks_mqtt_311():
    b = _FakeBroker()
    mqttnotify.publish("127.0.0.1", b.port,
                       [("critter-cam/sighting/raccoon", b'{"a":1}', False),
                        ("critter-cam/present", b"x" * 300, True)],   # >127: 2-byte length
                       username="critter-cam", password="s3cret")
    b.close()
    assert b.connect[:7] == b"\x00\x04MQTT\x04"                       # protocol + level 4
    assert b.connect[7] == 0xC2                                         # user, pass, clean
    assert b.connect.endswith(b"\x00\x0bcritter-cam\x00\x06s3cret")
    assert b.publishes == [("critter-cam/sighting/raccoon", b'{"a":1}', False, 1),
                           ("critter-cam/present", b"x" * 300, True, 1)]
    assert b.disconnected


def test_publish_reports_a_refused_login():
    b = _FakeBroker(connack_rc=5)
    with pytest.raises(mqttnotify.MqttError, match="not authorised"):
        mqttnotify.publish("127.0.0.1", b.port, [("t", b"x", False)], username="u", password="p")
    b.close()


def test_varint_boundaries():
    assert mqttnotify._varint(0) == b"\x00"
    assert mqttnotify._varint(127) == b"\x7f"
    assert mqttnotify._varint(128) == b"\x80\x01"
    assert mqttnotify._varint(16383) == b"\xff\x7f"


# ---- wired into the naming loop --------------------------------------------------------------

def test_watch_loop_publishes_live_arrivals(conn, tmp_path, monkeypatch):
    """End to end through classify.watch_loop: fresh crops get named, and the same poll
    announces the arrival."""
    import classify
    import config
    from test_classify_refresh import _StopAfter, _StubClassifier
    from PIL import Image

    monkeypatch.setattr(classify, "build_classifier", lambda device: (_StubClassifier(), "cpu"))
    monkeypatch.setattr(classify, "build_nonanimal_filter", lambda device: None)
    monkeypatch.setattr(config, "NAMING_STATUS_FILE", tmp_path / "naming_status.json")
    for k, v in vars(_cfg(mqtt_host="127.0.0.1")).items():
        monkeypatch.setattr(config.CONFIG, k, v, raising=False)
    sent = []
    monkeypatch.setattr(mqttnotify, "publish", lambda host, port, msgs, **kw: sent.extend(msgs))

    now = datetime.now().astimezone()
    for i in range(2):
        crop = tmp_path / f"live_{i}.jpg"
        Image.new("RGB", (1, 1)).save(crop, "JPEG")
        db.insert_detection(conn, timestamp=(now - timedelta(seconds=5 - i)).isoformat(),
                            source="glass_door_cam", detection_class="animal", confidence=0.9,
                            bbox=(0, 0, 10, 10), frame_w=100, frame_h=100, crop_path=str(crop))
    conn.commit()

    classify.watch_loop(conn, device="cpu", interval=0, stop_event=_StopAfter(1))

    arrived = [json.loads(p) for t, p, r in sent if t == "critter-cam/sighting/raccoon"]
    assert [(a["event"], a["crops"]) for a in arrived] == [("arrived", 2)]
