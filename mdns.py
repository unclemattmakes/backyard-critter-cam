"""Give the rig a NAME on the LAN, so nobody has to be told a number.

WHY
---
Everything that points a person at this dashboard has, until now, pointed them at an IP
address. The LAN launcher computes one and prints it; newsletter.dashboard_base bakes one
into every link in the morning email. Both do that for a good reason -- a bare Windows
hostname does NOT resolve from a phone, because iOS and Android ask mDNS for `name.local`
and do not speak NetBIOS (newsletter.py says so at length) -- but an IP is a bad thing to
hand a human being:

  * It is not memorable. "192.168.1.50, port 8000" is a thing you write on a napkin.
  * It is not durable. It is the DHCP lease, and the lease is not a promise. The email heals
    itself by re-deriving the address every issue; a napkin does not.
  * It only exists where the launcher window is, and that window closes.

mDNS fixes all three at once. The rig answers multicast DNS for a name IT chooses, so anyone
on the Wi-Fi opens `http://critter-cam.local` -- and keeps opening it after the router hands
out a different lease.

The missing port is the other half, and it is why config.web_port defaults to 80: a browser
assumes 80, so serving there is what turns `critter-cam.local:8000` into `critter-cam.local`.
That is the difference between an address you dictate and one a child can type. Port 80 is
not always available (root on Linux/macOS, and anything web-shaped may hold it), so the bind
falls back -- see web._bind -- and every URL printed here reads the port off the socket.

WHY A RESPONDER OF OUR OWN
--------------------------
Windows 10 does answer mDNS for its own computer name, which would give you
`http://<this-pc>.local` for free. Two things are wrong with free: the name is whatever
the PC is called (rename the box, break every napkin), and you cannot have two of them or a
name that describes the thing. Running the responder in-process means the name is a config
value, it is published only while the dashboard is actually up -- an unanswerable name is
worse than no name -- and the rig also advertises `_http._tcp`, so it turns up BY NAME in
network browsers and discovery apps without anyone typing anything at all.

WHAT THIS IS NOT
----------------
Not a security boundary and not a way onto the internet. `.local` is resolvable only by
devices on the same link (RFC 6762); the dashboard's own guards -- lan_only, the peer-IP
check, the DNS-rebinding Host check -- are unchanged and still the thing standing between
the yard and the world. This module hands out a nicer spelling of an address those guards
already allow.

REQUIRES `zeroconf`, and treats it as optional: no zeroconf (or no LAN, or UDP 5353 blocked)
means no name, a one-line explanation, and a rig that runs exactly as before.
"""
from __future__ import annotations

import ipaddress
import re
import socket
import sys
import threading
import time as _time
from dataclasses import replace

# The label the rig answers to, minus the suffix. Overridable per-install (cfg.mdns_name)
# because a household may end up with two rigs, and "critter-cam" and "front-yard-cam" is a
# better conversation than two IP addresses.
DEFAULT_NAME = "critter-cam"

# RFC 6762's reserved top-level domain for link-local names. Not a choice -- it is the only
# suffix a phone will route to multicast instead of to the internet.
SUFFIX = "local"

# Advertising as a WEB server (rather than only publishing the A record) is what makes the rig
# appear by name in the discovery UIs people already have -- macOS Finder's Network, "Discovery"
# on iOS, Service Browser on Android, `avahi-browse` on Linux.
SERVICE_TYPE = "_http._tcp.local."

# How often to re-check that the address we are advertising is still this machine's address.
# The whole point of the name is that it outlives a DHCP lease; a responder pinned to a stale
# IP would make the name WORSE than the number it replaces.
REFRESH_S = 60.0

# How often to stop trusting our own bookkeeping and ask the LINK whether anything still answers
# for the name. Slower than REFRESH_S because it costs a browse rather than a local comparison,
# and because the failure it catches is measured in hours-until-somebody-notices, not seconds.
VERIFY_S = 300.0


def lan_ip() -> str | None:
    """This machine's address ON THE LAN, or None.

    Opens a UDP socket toward a routable address and reads back the local end. No packet is ever
    sent (UDP connect only sets the peer), and it resolves to the interface the OS would actually
    use -- which beats enumerating adapters and guessing, on a box that also carries WSL and
    virtual ones. Only a PRIVATE address is accepted: a public one here would mean the rig sits
    directly on the internet, which is not a thing to advertise.

    Lives here rather than in newsletter.py (its first home) so the address the email links to
    and the address the rig announces are decided by the same code. Creating the socket is INSIDE
    the try, unlike in that first home: "there is no network" has to be an answer this returns,
    not an exception it raises, now that the rig's startup asks the question -- there the only
    handler in reach catches OSError and blames the web server for failing to start."""
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 9))
        ip = s.getsockname()[0]
        addr = ipaddress.ip_address(ip)
        return ip if addr.is_private and not addr.is_loopback and not addr.is_link_local else None
    except Exception:
        return None
    finally:
        if s is not None:
            s.close()


def safe_label(raw) -> str:
    """`raw` reduced to something legal to the left of `.local`, or DEFAULT_NAME if nothing legal
    survives. A DNS label is letters, digits and hyphens, no leading or trailing hyphen, 63 max;
    a name that breaks those rules does not fail loudly at registration, it fails quietly at
    whichever resolver is strictest, on somebody else's phone, later."""
    label = re.sub(r"[^a-z0-9-]+", "-", str(raw or "").strip().lower())
    label = re.sub(r"-{2,}", "-", label).strip("-")[:63].strip("-")
    return label or DEFAULT_NAME


def host_name(cfg=None) -> str:
    """The full name the rig answers to -- 'critter-cam.local'."""
    return f"{safe_label(getattr(cfg, 'mdns_name', None) or DEFAULT_NAME)}.{SUFFIX}"


def lan_bound(cfg) -> bool:
    """Is the dashboard listening for other machines at all? True for a wildcard bind (the LAN
    launcher's --host 0.0.0.0), False for the loopback default. Separate from `enabled` because
    it answers a different question -- "can anyone else connect?" rather than "should we announce
    a name?" -- and the printed block needs the first one even when mDNS is switched off."""
    return str(getattr(cfg, "web_host", "")).strip() in ("0.0.0.0", "::")


def enabled(cfg) -> bool:
    """Publish a name only when the dashboard is actually reachable from other machines. Bound to
    loopback there is nobody to tell, and the name would resolve to an address that refuses every
    caller -- so the answer to "why doesn't critter-cam.local work?" stays "because the rig is in
    localhost mode", not a mystery."""
    return bool(getattr(cfg, "mdns", True)) and lan_bound(cfg)


def port_of(cfg) -> int:
    """The port to advertise. One place, because the printed block, the service record and the
    email all have to agree on it."""
    return int(getattr(cfg, "web_port", 80) or 80)


def local_candidates(cfg) -> list[int]:
    """Every port this rig might be serving on, in the order it would have tried them.

    One definition, used by the code that BINDS (web._bind) and by the launcher that waits for
    the bind to happen. Two lists would disagree the first time somebody changed one, and they
    would disagree exactly on the fallback path -- the path nobody exercises until port 80 is
    taken on a machine they cannot reach.

    Reads web_port RAW rather than through port_of, which is a display helper and maps 0 to 80.
    Here 0 means what it means to bind(): "any free port", the ephemeral bind the tests use. A
    list that quietly turned that into 80 would send the whole suite at the real HTTP port."""
    ports = [int(getattr(cfg, "web_port", 80))]
    fallback = int(getattr(cfg, "web_port_fallback", 0) or 0)
    if fallback and fallback not in ports:
        ports.append(fallback)
    return ports


def url(cfg, host: str | None = None, port: int | None = None) -> str:
    """The address to hand a person: 'http://critter-cam.local'.

    Port 80 is OMITTED, which is the entire point of serving on it -- a browser assumes 80, so
    printing it back would put the number we just worked to remove in front of the person we
    removed it for. Any other port is spelled out, because there it is load-bearing."""
    p = port_of(cfg) if port is None else int(port)
    return f"http://{host or host_name(cfg)}" + ("" if p == 80 else f":{p}")


def connect_lines(cfg, ip: str | None = None, name: str | None = None,
                  port: int | None = None) -> list[str]:
    """The "here is how to get in" block, as lines, ready to print.

    `port` overrides cfg for the case that made it necessary: the rig asked for 80, could not have
    it, and is serving on the fallback. The block has to print where the socket ACTUALLY is.

    Pure, so the text can be tested and so the launcher and the rig cannot drift apart. Prints
    the NAME first and the numeric address underneath rather than one or the other: the name is
    the memorable half, but Android's mDNS support is the weak link in an otherwise solid story
    (iOS, macOS and Windows resolve `.local` reliably; a stock Android browser may not), and a
    visitor whose phone shrugs at the name needs the number on the SAME screen, not a support
    conversation. Loopback comes last, labelled, so nobody reads 127.0.0.1 off this PC's screen
    and types it into a phone.

    A loopback-bound rig gets the short version. This machine HAS a LAN address whether or not
    the dashboard is listening on it, so printing one here would be a lie that sends someone to a
    connection their own rig refuses -- and it is the likeliest way to end up debugging a
    firewall for an hour over a missing --host flag."""
    p = port_of(cfg) if port is None else int(port)
    if not lan_bound(cfg):
        return [f"  On this PC only:       {url(cfg, '127.0.0.1', p)}",
                "    (localhost mode -- nobody else can connect;",
                "     start_critter_cam_lan.bat opens it to your Wi-Fi)"]
    out = []
    if name:
        out.append(f"  Others on your Wi-Fi:  {url(cfg, name, p)}")
        if ip:
            out.append(f"    ...or by number:     {url(cfg, ip, p)}"
                       "   (if a device can't find the name)")
    elif ip:
        out.append(f"  Others on your Wi-Fi:  {url(cfg, ip, p)}")
    out.append(f"  On this PC:            {url(cfg, '127.0.0.1', p)}")
    return out


class Publication:
    """A live mDNS registration, and the thread that keeps its address honest.

    Held by the caller only to close it. `name` is the published host name, so callers print
    what was actually announced rather than what they hoped for."""

    def __init__(self, zc, info, name: str, ip: str, port: int):
        self.zc, self.info, self.name, self.ip, self.port = zc, info, name, ip, port
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, name="mdns-refresh", daemon=True)
        self._thread.start()

    def _watch(self) -> None:
        """Re-announce whenever this machine's LAN address moves under us, or whenever the name
        stops being answered at an address that never moved.

        A DHCP reassignment is precisely the case the name exists to survive -- it is the whole
        reason the name beats the number -- so failing to notice one would defeat the feature at
        exactly the moment it was supposed to pay off, and quietly: the name would go on
        resolving, to nothing.

        The responder is torn down and rebuilt rather than merely updated. update_service alone
        would rewrite the A record but leave the socket bound to the OLD interface address, which
        by then is not an address this machine has -- so the correction would be announced from
        somewhere nobody is listening.

        A correct address is not the same as a working name, and watching only the address is how
        this went wrong in the field on 2026-08-25: a Wi-Fi drop at 23:14 took the responder's
        socket with it (the camera reader logged the same blip and recovered on its fifth reopen;
        the responder has no such loop), the address came back UNCHANGED, and so this watch --
        which compared the address, found it correct, and asked nothing else -- never fired.
        Nothing else in the rig reported it either, because nothing else was looking: the
        dashboard went on serving, the camera went on recording, and only the name was gone. It
        would have stayed gone until the next restart, which on that night meant restarting a rig
        mid-visit with raccoons in frame -- the last thing anybody wants to do at midnight.

        So every VERIFY_S this stops trusting its own bookkeeping and asks the link the only
        question with an unambiguous answer: does ANYTHING answer for this name? Silence is the
        one symptom the failure reliably has -- a responder cannot tell you its socket is deaf,
        but the network can tell you nobody is talking. Any answer at all counts as alive,
        including one carrying somebody else's address: a name another device is now defending is
        exactly the case `publish` refuses to stomp on, and this must not stomp on it either."""
        until_verify = VERIFY_S
        while not self._stop.wait(REFRESH_S):
            now = lan_ip()
            if not now:
                continue
            moved = now != self.ip
            if not moved:
                until_verify -= REFRESH_S
                if until_verify > 0:
                    continue
                if _held_by(self.name):
                    until_verify = VERIFY_S
                    continue
                # Nothing on the link answers for us, so whatever the responder we hold believes
                # about itself, it is not reaching anybody. Retire it BEFORE building the
                # replacement -- that frees its socket, and it keeps a half-alive corpse from
                # failing the name-conflict probe its own replacement is about to run.
                _tear_down(self.zc, self.info)
                self.zc, self.info = None, None
            try:
                new_zc, new_info = _register(self.name, now, self.port)
            except Exception as e:
                # until_verify is spent, so a name still dead here retries on the NEXT tick
                # rather than waiting out another full verify interval.
                print(f"  [mdns] could not {'re-announce' if moved else 'revive'} "
                      f"{self.name} at {now}: {e}")
                continue
            old_zc, old_info = self.zc, self.info
            self.zc, self.info, self.ip = new_zc, new_info, now
            if old_zc is not None:
                _tear_down(old_zc, old_info)
            until_verify = VERIFY_S
            if moved:
                print(f"  [mdns] address changed -- {self.name} now points at {now}")
            else:
                print(f"  [mdns] {self.name} had stopped answering -- re-announced at {now}")

    def close(self) -> None:
        self._stop.set()
        _tear_down(self.zc, self.info)


def _tear_down(zc, info) -> None:
    """Withdraw a registration and close its responder, surviving any state either is in. Called
    on shutdown and on every re-announce, so it must never raise -- a failure here would leave a
    thread holding UDP 5353 against the replacement."""
    try:
        zc.unregister_service(info)
    except Exception:
        pass
    try:
        zc.close()
    except Exception:
        pass


def _register(name: str, ip: str, port: int, cooperating: bool = False):
    """Build and start a responder answering for `name` at `ip`. Raises on failure; the two
    callers differ in what they do about that, so it is not swallowed here.

    Bound to the single LAN interface rather than all of them, because this box also carries WSL
    and virtual adapters and announcing the rig on those tells nobody anything.

    `cooperating` skips zeroconf's name-conflict probe. Only `publish` sets it, and only after
    establishing that the thing holding the name is this rig's own ghost -- see _held_by."""
    from zeroconf import IPVersion, ServiceInfo, Zeroconf

    label = name.rsplit("." + SUFFIX, 1)[0]
    info = ServiceInfo(
        SERVICE_TYPE,
        f"{label}.{SERVICE_TYPE}",
        addresses=[socket.inet_aton(ip)],
        port=port,
        # Advertised to discovery UIs, which show these next to the name.
        properties={"path": "/", "description": "Backyard Critter Cam dashboard"},
        server=f"{name}.",          # the A record -- this is the half that makes the URL work
    )
    zc = Zeroconf(interfaces=[ip], ip_version=IPVersion.V4Only)
    try:
        # allow_name_change stays FALSE on purpose: silently becoming 'critter-cam-2' would hand
        # everyone a name that is not the one written down. A clash is worth hearing about.
        zc.register_service(info, allow_name_change=False, cooperating_responders=cooperating)
    except Exception:
        try:
            zc.close()
        except Exception:
            pass
        raise
    return zc, info


def publish(cfg, port: int | None = None) -> Publication | None:
    """Start answering mDNS for cfg.mdns_name, returning a Publication (close it to stop) or None.

    `port` overrides cfg, so the SRV record advertises the port the rig actually bound rather than
    the one it asked for -- a discovery UI that sends someone to a dead port is worse than one
    that never listed the rig.

    Never raises and never fatal: every failure here costs a nicer spelling of an address that
    still works as a number, so each one prints its own fix and the rig carries on."""
    if not enabled(cfg):
        return None
    try:
        from zeroconf import NonUniqueNameException as NonUniqueName
    except ImportError:
        print("  [mdns] 'zeroconf' is not installed, so the dashboard has no name on the network "
              "-- the numeric address below still works.")
        print(r"  [mdns] fix: .venv\Scripts\pip install zeroconf")
        return None

    ip = lan_ip()
    if not ip:
        print("  [mdns] no LAN address found, so there is nothing to announce (is Wi-Fi up?).")
        return None

    name = host_name(cfg)
    port = port_of(cfg) if port is None else int(port)
    try:
        zc, info = _register(name, ip, port)
    except NonUniqueName:
        # The name is spoken for. Usually by THIS RIG'S OWN GHOST: zeroconf refuses when any
        # responder on the link still answers for the name, and a rig that died without sending
        # its goodbye packets -- a crash, a taskkill, a power cut -- leaves exactly that behind,
        # cached by every other responder until the record's TTL runs out. That is not a rare
        # case here. rigwatch.py exists to restart this rig after it dies, so the restart lands
        # inside the ghost's lifetime almost by construction, and refusing would mean the name
        # is missing precisely on the reboots nobody is watching.
        holders = _held_by(name)
        if holders and set(holders) <= {ip}:
            print(f"  [mdns] the network still remembers this rig's previous run; "
                  f"re-claiming {name}.")
            try:
                zc, info = _register(name, ip, port, cooperating=True)
            except Exception as e:
                print(f"  [mdns] could not re-claim {name} ({type(e).__name__}: {e}).")
                return None
        else:
            where = ", ".join(holders) if holders else "another device"
            print(f"  [mdns] {name} is already taken on this network (by {where}), so it was NOT "
                  "published -- the numeric address below still works.")
            print("  [mdns] fix: give this rig its own name with cfg.mdns_name in config_local.py")
            return None
    except Exception as e:
        # Most likely Windows Firewall eating UDP 5353. The operator's to fix, and not a reason
        # to stop the rig.
        print(f"  [mdns] could not publish {name} ({type(e).__name__}: {e}) -- "
              "is UDP 5353 blocked by the firewall?")
        return None
    return Publication(zc, info, name, ip, port)


def _held_by(name: str, timeout: float = 3.0) -> list[str] | None:
    """The addresses currently answering for `name`, or None if nothing does / we couldn't ask.

    Only called when registration has already been refused, so the cost of a browse lands on the
    failure path and never on a normal start. What it buys is the difference between the two
    reasons a name is taken -- this rig's own stale record, which we should reclaim, and a second
    device genuinely using the name, which we must not stomp on.

    A rig whose IP changed while its ghost was still cached reads as the second case and refuses.
    That is the safe direction to be wrong in, and the message names the holder, so the operator
    sees an address rather than a mystery."""
    try:
        from zeroconf import Zeroconf
    except ImportError:
        return None
    label = name.rsplit("." + SUFFIX, 1)[0]
    zc = None
    try:
        zc = Zeroconf()
        info = zc.get_service_info(SERVICE_TYPE, f"{label}.{SERVICE_TYPE}",
                                   timeout=int(max(0.1, timeout) * 1000))
        return list(info.parsed_addresses()) if info else None
    except Exception:
        return None
    finally:
        if zc is not None:
            try:
                zc.close()
            except Exception:
                pass


def unpublish(pub) -> None:
    """Stop answering. Safe on None, so callers need no shutdown branch."""
    if pub is not None:
        pub.close()


def resolves(name: str, timeout: float = 2.0) -> bool:
    """Does `name` actually resolve from THIS machine right now?

    Used by the launcher, which prints its block from outside the rig process and so cannot ask
    the Publication whether registration worked. A local resolve is not proof a phone will manage
    it -- Android is the weak link no check here can see -- but it does separate "published" from
    "we hoped", and printing a name that this very box cannot resolve is how a feature earns a
    reputation for not working."""
    old = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(timeout)
        socket.getaddrinfo(name, None, socket.AF_INET)
        return True
    except OSError:
        return False
    finally:
        socket.setdefaulttimeout(old)


def wait_local(cfg, timeout: float = 60.0, interval: float = 0.5) -> int | None:
    """Block until the dashboard accepts a connection on loopback, and return the port it answered
    on -- or None if `timeout` runs out first.

    Polling for an accepted connection rather than sleeping a fixed guess is the behaviour the
    launchers already had; what is new is that it no longer knows the port in advance. It cannot:
    the rig may be on 80 or, if 80 was taken, on the fallback, and only the socket knows which.
    Loopback only -- this asks "is my own rig up yet?", never anything about another machine."""
    deadline = _time.monotonic() + max(0.0, float(timeout))
    ports = local_candidates(cfg)
    while True:
        for port in ports:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    return port
            except OSError:
                pass
        if _time.monotonic() >= deadline:
            return None
        _time.sleep(interval)


def main() -> int:
    """`python mdns.py` -- print how to reach this rig, the same block the rig prints at startup.

    Exists so the LAN launcher (a .bat) does not have to reimplement any of this in batch, and so
    "what is the address again?" has an answer you can run at any time, not just a line that
    scrolled past at startup. --host/--port mirror backyard_cam.py's flags, because the launcher
    passes --host 0.0.0.0 on the command line and config.py alone would not know that."""
    import argparse

    import config

    ap = argparse.ArgumentParser(description="Print how to reach this rig's dashboard.")
    ap.add_argument("--host", default=None, help="the web_host the rig is served on (e.g. 0.0.0.0)")
    ap.add_argument("--port", type=int, default=None, help="the web port (default: config)")
    ap.add_argument("--publish", action="store_true",
                    help="publish the name and hold it until Ctrl+C (a standalone responder, for "
                         "testing the name without starting the rig)")
    ap.add_argument("--wait-local", action="store_true",
                    help="wait until the dashboard answers on this machine, then print the URL to "
                         "open here (used by the launchers, which must not guess the port)")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="how long --wait-local waits before giving up (default 60s)")
    args = ap.parse_args()

    cfg = config.CONFIG
    if args.host is not None:
        cfg = replace(cfg, web_host=args.host)
    if args.port is not None:
        cfg = replace(cfg, web_port=args.port)

    if args.wait_local:
        port = wait_local(cfg, timeout=args.timeout)
        # On a timeout, still print the port the rig MEANT to use: the launcher opens whatever
        # comes back either way (the first run downloads a detector model and can outlast any
        # sensible wait), and a browser that has to be refreshed once beats no browser at all.
        print(url(cfg, "127.0.0.1", port or local_candidates(cfg)[0]))
        return 0 if port else 1

    if args.publish:
        pub = publish(cfg)
        if pub is None:
            return 1
        print(f"answering for {pub.name} at {pub.ip} -- Ctrl+C to stop")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            print("\nstopping")
        finally:
            unpublish(pub)
        return 0

    name = host_name(cfg) if enabled(cfg) else None
    for line in connect_lines(cfg, ip=lan_ip(), name=name if name and resolves(name) else None):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
