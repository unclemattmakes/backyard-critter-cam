"""
Live sighting alerts over MQTT: "a raccoon just arrived at the glass door", as a message any
device on the LAN can subscribe to (first consumer: the lantern LED controller).

Called once per poll by the naming loop (classify.watch_loop), right after fresh crops get their
species. It reads the DB, not the loop's in-memory results, so it doesn't matter which process or
thread did the naming. Off unless cfg.mqtt_host is set.

WHAT COUNTS AS A SIGHTING. Single crop labels are noisy, and one raccoon makes dozens of crops, so
this never fires per crop. An ARRIVAL is `mqtt_min_crops` crops of one species, each at
species_confidence >= `mqtt_min_confidence`, on one source, all inside the freshness window
(`mqtt_window_s`), and not continuing an arrival already announced. A crop within
cfg.visit_gap_minutes of the previous one for that (source, species) continues the visit; a longer
silence ends it and publishes a LEFT event. Measured over Sept 2026 glass-door visits,
2 crops >= 0.8 catches 70 of 80 raccoon visits and 94 of 113 Steller's jay visits, and never fires
on 'brown rat' (whose labels are mostly misreads).

The freshness window is what keeps history quiet. A trail-cam import, a naming backlog after a
restart and a --redo all label OLD crops; none of those are "right now", so none of them alert.

TOPICS (prefix = cfg.mqtt_topic_prefix, default "critter-cam"):
  <prefix>/sighting/<species-slug>   event JSON, QoS 1, not retained
                                     {"event": "arrived"|"left", "species", "slug", "source", ...}
  <prefix>/present                   retained JSON: what is in view right now, republished on
                                     every change (and once at startup, to clear a stale list
                                     left by a crash)
  <prefix>/last-sighting             retained JSON: the latest "arrived" event

TRANSPORT. MQTT 3.1.1 over a plain socket, stdlib only -- the same no-SDK stance as newsletter.py's
Resend call. One short connection per batch of messages (sightings are minutes apart, so a held
connection would buy nothing but reconnect logic). A broker that is down costs one log line per
outage and never raises into the naming loop.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import socket
import struct
from datetime import datetime, timedelta

import config
import db

# Crops labelled this are the non-animal gate's verdict (clipfilter), never a sighting.
from clipfilter import NONANIMAL_LABEL

_CONNECT_TIMEOUT_S = 3.0
_RETRY_S = 60.0


# ---------------------------------------------------------------------------
# Minimal MQTT 3.1.1 publisher
# ---------------------------------------------------------------------------

class MqttError(RuntimeError):
    pass


def _varint(n: int) -> bytes:
    """MQTT 'remaining length' encoding: 7 bits per byte, high bit = more follows."""
    out = bytearray()
    while True:
        b, n = n % 128, n // 128
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _str(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("!H", len(raw)) + raw


def _packet(first_byte: int, body: bytes) -> bytes:
    return bytes([first_byte]) + _varint(len(body)) + body


def connect_packet(client_id: str, username: str | None, password: str | None,
                   keepalive: int = 30) -> bytes:
    flags = 0x02                                   # clean session
    payload = _str(client_id)
    if username is not None:
        flags |= 0x80
        payload += _str(username)
        if password is not None:
            flags |= 0x40
            payload += _str(password)
    var = _str("MQTT") + bytes([4, flags]) + struct.pack("!H", keepalive)
    return _packet(0x10, var + payload)


def publish_packet(topic: str, payload: bytes, packet_id: int, *, retain: bool) -> bytes:
    """QoS 1 PUBLISH (the broker PUBACKs, so a send is confirmed rather than hoped for)."""
    return _packet(0x30 | 0x02 | (0x01 if retain else 0),
                   _str(topic) + struct.pack("!H", packet_id) + payload)


def _read_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise MqttError("broker closed the connection")
        buf += chunk
    return buf


def _read_packet(sock: socket.socket) -> tuple[int, bytes]:
    first = _read_exact(sock, 1)[0]
    length, shift = 0, 0
    while True:
        b = _read_exact(sock, 1)[0]
        length |= (b & 0x7F) << shift
        if not b & 0x80:
            break
        shift += 7
    return first, _read_exact(sock, length)


_CONNACK_CODES = {1: "unacceptable protocol version", 2: "client id rejected",
                  3: "server unavailable", 4: "bad username or password", 5: "not authorised"}


def publish(host: str, port: int, messages, *, username: str | None = None,
            password: str | None = None, timeout: float = _CONNECT_TIMEOUT_S) -> None:
    """Publish (topic, payload_bytes, retain) messages in one connection, QoS 1. Raises MqttError
    or OSError on failure. Note a PUBACK means the broker ACCEPTED the message; a topic the ACL
    denies is still PUBACKed (MQTT 3.1.1 has no way to say no) and silently dropped."""
    client_id = f"critter-cam-{os.getpid()}-{secrets.token_hex(3)}"
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(connect_packet(client_id, username, password))
        ptype, body = _read_packet(sock)
        if ptype >> 4 != 2 or len(body) < 2:
            raise MqttError(f"expected CONNACK, got packet type {ptype >> 4}")
        if body[1] != 0:
            raise MqttError(f"broker refused the connection: "
                            f"{_CONNACK_CODES.get(body[1], f'code {body[1]}')}")
        for pid, (topic, payload, retain) in enumerate(messages, start=1):
            sock.sendall(publish_packet(topic, payload, pid, retain=retain))
            ptype, body = _read_packet(sock)
            if ptype >> 4 != 4 or struct.unpack("!H", body[:2])[0] != pid:
                raise MqttError(f"expected PUBACK for message {pid}")
        sock.sendall(b"\xe0\x00")                  # DISCONNECT


# ---------------------------------------------------------------------------
# Sighting detection
# ---------------------------------------------------------------------------

def slug(species: str) -> str:
    """Topic-safe species name: "Steller's jay" -> "stellers-jay". MQTT wildcards (+ #) and the
    level separator can never appear in the result."""
    s = species.lower().replace("'", "")
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-") or "unknown"


def _wanted(species: str, alert_species) -> bool:
    if species == NONANIMAL_LABEL:
        return False
    if alert_species in (None, "*"):
        return True
    return species.lower() in {s.lower() for s in alert_species}


class SightingNotifier:
    """Turns freshly named crops into arrived/left events. One instance lives as long as the
    naming loop; its memory of what is in view is deliberately NOT persisted -- after a restart it
    re-announces anything still in view, once, and clears the retained present list."""

    def __init__(self, cfg=None, sender=None):
        self.cfg = cfg if cfg is not None else config.CONFIG
        # Injection point for tests; production publishes to the configured broker.
        self._send = sender if sender is not None else self._publish
        self.active: dict[tuple[str, str], dict] = {}    # (source, species) -> visit state
        # Last crop of each ended visit: those crops can still sit inside the freshness window
        # (if mqtt_window_s is set longer than the visit gap) and must not re-announce it.
        self._departed: dict[tuple[str, str], datetime] = {}
        self._announced_startup = False
        self._broker_down = False
        self._retry_at: datetime | None = None           # backoff while the broker is down

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.cfg, "mqtt_host", None))

    def _topic(self, *parts: str) -> str:
        return "/".join([getattr(self.cfg, "mqtt_topic_prefix", "critter-cam"), *parts])

    def _publish(self, messages) -> None:
        c = self.cfg
        publish(c.mqtt_host, int(getattr(c, "mqtt_port", 1883)), messages,
                username=getattr(c, "mqtt_username", None),
                password=getattr(c, "mqtt_password", None))

    def _fresh_crops(self, conn, now: datetime):
        """(source, species, dt, confidence, detection_id) for recent confident crops."""
        c = self.cfg
        window = timedelta(seconds=float(getattr(c, "mqtt_window_s", 120.0)))
        # String bound with an hour of slack (a DST change moves the stored offset), then the
        # exact test on parsed instants.
        bound = (now - window - timedelta(hours=1)).isoformat()
        rows = conn.execute(
            "SELECT id, source, timestamp, species, species_confidence FROM detections "
            "WHERE timestamp >= ? AND species IS NOT NULL AND species_confidence >= ?",
            (bound, float(getattr(c, "mqtt_min_confidence", 0.8)))).fetchall()
        sources = getattr(c, "mqtt_sources", None)
        alert_species = getattr(c, "mqtt_alert_species", "*")
        out = []
        for rid, src, ts, sp, conf in rows:
            dt = db.parse_local(ts)
            if dt is None or dt < now - window or dt > now + timedelta(minutes=5):
                continue
            if sources is not None and src not in sources:
                continue
            if not _wanted(sp, alert_species):
                continue
            out.append((src, sp, dt, float(conf), rid))
        return out

    def poll(self, conn, now: datetime | None = None) -> list[dict]:
        """Examine recent crops, publish any arrivals/departures. Returns the events (for tests
        and logging). Never raises."""
        if not self.enabled:
            return []
        now = now or datetime.now().astimezone()
        try:
            crops = self._fresh_crops(conn, now)
        except Exception as e:      # noqa: BLE001 -- a locked DB must not kill the naming loop
            print(f"[mqtt] couldn't read recent crops: {type(e).__name__}: {e}")
            return []

        gap = timedelta(minutes=float(getattr(self.cfg, "visit_gap_minutes", 5.0)))
        min_crops = int(getattr(self.cfg, "mqtt_min_crops", 2))
        events: list[dict] = []
        changed = not self._announced_startup

        grouped: dict[tuple[str, str], list] = {}
        for src, sp, dt, conf, rid in crops:
            grouped.setdefault((src, sp), []).append((dt, conf, rid))

        for key, items in grouped.items():
            items.sort()
            if key in self._departed:
                items = [it for it in items if it[0] > self._departed[key]]
                if not items:
                    continue
            state = self.active.get(key)
            if state is not None:
                newer = [it for it in items if it[0] > state["last_seen"]]
                if newer:
                    state["last_seen"] = newer[-1][0]
                    state["crops"] += len(newer)
                    state["confidence"] = max(state["confidence"], max(it[1] for it in newer))
                continue
            if len(items) < min_crops:
                continue
            src, sp = key
            self.active[key] = {"first_seen": items[0][0], "last_seen": items[-1][0],
                                "crops": len(items), "confidence": max(it[1] for it in items),
                                "detection_id": max(items, key=lambda it: it[1])[2]}
            events.append(self._event("arrived", key, now))
            changed = True

        for key in [k for k, s in self.active.items() if now - s["last_seen"] > gap]:
            events.append(self._event("left", key, now))
            self._departed[key] = self.active.pop(key)["last_seen"]
            changed = True

        if events or changed:
            self._emit(events, now)
        return events

    def _event(self, kind: str, key: tuple[str, str], now: datetime) -> dict:
        src, sp = key
        s = self.active[key]      # read before the caller pops a departing visit
        # visit_id: the same on a visit's "arrived" and "left", unique across visits (a subscriber
        # can de-duplicate on it). Not the DB's visits.id -- that table is rebuilt from scratch and
        # its ids are not stable.
        vid = f"{src}/{slug(sp)}/{s['first_seen'].strftime('%Y%m%dT%H%M%S')}"
        # KEY ORDER IS A CONTRACT: "event", "slug" and "visit_id" lead, in that order. The lantern
        # LED controller parses only the first 2048 bytes of a payload, so those three must land
        # inside it however long the other fields get. test_routing_keys_lead_the_payload holds this.
        ev = {"event": kind, "slug": slug(sp), "visit_id": vid, "species": sp, "source": src,
              "first_seen": s["first_seen"].isoformat(), "last_seen": s["last_seen"].isoformat(),
              "crops": s["crops"], "confidence": round(s["confidence"], 3),
              "detection_id": s["detection_id"], "sent_at": now.isoformat()}
        if kind == "left":
            ev["duration_s"] = round((s["last_seen"] - s["first_seen"]).total_seconds())
        return ev

    def _present(self, now: datetime) -> dict:
        items = sorted(self.active.items(), key=lambda kv: kv[1]["first_seen"])
        return {"present": [{"species": sp, "slug": slug(sp), "source": src,
                             "since": s["first_seen"].isoformat()} for (src, sp), s in items],
                "updated": now.isoformat()}

    def _emit(self, events: list[dict], now: datetime) -> None:
        enc = lambda obj: json.dumps(obj).encode("utf-8")   # noqa: E731
        msgs = [(self._topic("sighting", ev["slug"]), enc(ev), False) for ev in events]
        msgs.append((self._topic("present"), enc(self._present(now)), True))
        arrivals = [ev for ev in events if ev["event"] == "arrived"]
        if arrivals:
            msgs.append((self._topic("last-sighting"), enc(arrivals[-1]), True))
        # While the broker is down, try at most every _RETRY_S: an unreachable host costs a full
        # connect timeout, and the naming loop that calls this polls every few seconds. Events
        # raised in the meantime are logged and dropped (a late "a raccoon arrived" is noise);
        # the retained present list is republished by the first send that gets through.
        if self._retry_at is not None and now < self._retry_at:
            self._log(events, sent=False)
            return
        try:
            self._send(msgs)
        except Exception as e:      # noqa: BLE001 -- the broker is optional; naming is not
            if not self._broker_down:
                print(f"[mqtt] publish failed ({type(e).__name__}: {e}); "
                      f"retrying every {_RETRY_S:.0f}s.")
            self._broker_down = True
            self._retry_at = now + timedelta(seconds=_RETRY_S)
            self._log(events, sent=False)
            return
        if self._broker_down:
            print("[mqtt] broker reachable again.")
        self._broker_down = False
        self._retry_at = None
        self._announced_startup = True
        self._log(events, sent=True)

    @staticmethod
    def _log(events: list[dict], *, sent: bool) -> None:
        for ev in events:
            print(f"[mqtt] {ev['event']}: {ev['species']} on {ev['source']} "
                  f"({ev['crops']} crops, conf {ev['confidence']:.2f})"
                  f"{'' if sent else ' -- NOT sent, broker down'}")


def main() -> int:
    """`python mqttnotify.py --test`: publish one clearly-marked test sighting, to check the
    broker settings without waiting for an animal."""
    import argparse
    p = argparse.ArgumentParser(description="Live sighting alerts over MQTT.")
    p.add_argument("--test", action="store_true",
                   help="Publish a test 'arrived' event to <prefix>/sighting/test.")
    args = p.parse_args()
    cfg = config.CONFIG
    if not getattr(cfg, "mqtt_host", None):
        print("MQTT is off: set cfg.mqtt_host (and credentials) in config_local.py.")
        return 1
    if not args.test:
        p.print_help()
        return 0
    now = datetime.now().astimezone().isoformat()
    ev = {"event": "arrived", "slug": "test",
          "visit_id": f"mqttnotify-test/test/{datetime.now().strftime('%Y%m%dT%H%M%S')}",
          "species": "test", "source": "mqttnotify --test",
          "first_seen": now, "last_seen": now, "crops": 0, "confidence": 1.0,
          "detection_id": None, "sent_at": now, "test": True}
    topic = f"{getattr(cfg, 'mqtt_topic_prefix', 'critter-cam')}/sighting/test"
    try:
        SightingNotifier(cfg)._publish([(topic, json.dumps(ev).encode("utf-8"), False)])
    except Exception as e:      # noqa: BLE001
        print(f"Publish failed: {type(e).__name__}: {e}")
        return 1
    print(f"Published a test event to {topic} on {cfg.mqtt_host}:{getattr(cfg, 'mqtt_port', 1883)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
