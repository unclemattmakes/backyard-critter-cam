"""
Unit tests for mdns.py -- the rig's NAME on the local network.

Pure logic only: no multicast, no sockets, no zeroconf. Everything worth testing here is a
decision (what is a legal label, when is a name worth publishing at all, which addresses the
startup block is allowed to claim) rather than a packet, and those decisions are the ones that
go wrong silently -- on somebody else's phone, in another room, days later.

The parts NOT covered here are the parts a unit test cannot honestly cover: whether a phone
resolves the name is a property of the phone. That was verified by hand against the OS resolver
(`ping critter-cam.local`) and a service browse; see mdns.py's module docstring.
"""
from __future__ import annotations

import pytest

import mdns


class FakeCfg:
    """The four attributes mdns.py reads. A stand-in rather than a real config.Config so a test
    can express states config's defaults cannot -- an empty name, mDNS switched off."""

    def __init__(self, web_host="0.0.0.0", web_port=80, mdns=True, mdns_name="critter-cam",
                 web_port_fallback=8000):
        self.web_host, self.web_port = web_host, web_port
        self.mdns, self.mdns_name = mdns, mdns_name
        self.web_port_fallback = web_port_fallback


# ---- safe_label: what is legal to the left of .local ----------------------------------

@pytest.mark.parametrize("raw, want", [
    ("critter-cam", "critter-cam"),
    ("Critter Cam", "critter-cam"),          # spaces and case are the common human input
    ("Matt's Yard Cam!", "matt-s-yard-cam"),  # punctuation collapses rather than vanishing
    ("--front--yard--", "front-yard"),        # no leading/trailing hyphen, no doubles
    ("front___yard", "front-yard"),
    ("cam2", "cam2"),
])
def test_safe_label_normalises_human_input(raw, want):
    """A DNS label is letters, digits and hyphens only. Anything else has to be fixed HERE,
    because the alternative is a registration that succeeds and a name that fails at whichever
    resolver is strictest."""
    assert mdns.safe_label(raw) == want


@pytest.mark.parametrize("raw", ["", "   ", None, "!!!", "---", "___"])
def test_safe_label_falls_back_rather_than_returning_nothing(raw):
    """An empty label would publish ".local", which is not a name -- so a name that sanitises
    away becomes the default instead of becoming nothing."""
    assert mdns.safe_label(raw) == mdns.DEFAULT_NAME


def test_safe_label_respects_the_63_character_limit():
    """63 octets is the DNS label maximum, and the trailing strip must not leave a hyphen at the
    cut -- a name ending in '-' is as invalid as one starting with it."""
    label = mdns.safe_label("a" * 200)
    assert len(label) == 63 and label == "a" * 63
    cut = mdns.safe_label("b" * 62 + "-cam")     # the 63rd character lands ON a hyphen
    assert len(cut) <= 63 and not cut.endswith("-")


def test_host_name_is_the_label_plus_the_reserved_suffix():
    assert mdns.host_name(FakeCfg(mdns_name="Front Yard")) == "front-yard.local"
    assert mdns.host_name(FakeCfg(mdns_name="")) == "critter-cam.local"
    assert mdns.host_name(None) == "critter-cam.local"       # no config at all -> the default


def test_url_omits_port_80_and_spells_out_every_other():
    """Suppressing :80 IS the feature -- serving on the port a browser assumes is the only way the
    address gets to be a bare name a child can type, and printing the number back would undo it."""
    assert mdns.url(FakeCfg(web_port=80)) == "http://critter-cam.local"
    assert mdns.url(FakeCfg(web_port=8000)) == "http://critter-cam.local:8000"
    assert mdns.url(FakeCfg(web_port=9001)) == "http://critter-cam.local:9001"
    assert mdns.url(FakeCfg(web_port=80), "192.168.1.50") == "http://192.168.1.50"


def test_url_takes_an_explicit_port_over_the_config():
    """The rig asks for 80 and may not get it. Everything printed afterwards has to describe the
    socket that exists, not the one that was requested."""
    assert mdns.url(FakeCfg(web_port=80), port=8000) == "http://critter-cam.local:8000"


# ---- which ports the rig might be on -------------------------------------------------

def test_local_candidates_is_the_port_then_its_fallback():
    assert mdns.local_candidates(FakeCfg(web_port=80, web_port_fallback=8000)) == [80, 8000]
    assert mdns.local_candidates(FakeCfg(web_port=8000, web_port_fallback=8000)) == [8000]
    assert mdns.local_candidates(FakeCfg(web_port=80, web_port_fallback=0)) == [80]


def test_local_candidates_keeps_port_zero_meaning_ephemeral():
    """port_of maps 0 to 80 for display; this list must NOT. Here 0 is bind()'s "any free port" --
    the ephemeral bind the whole test suite relies on -- and turning it into 80 would point every
    test at the real HTTP port."""
    assert mdns.local_candidates(FakeCfg(web_port=0))[0] == 0
    assert mdns.port_of(FakeCfg(web_port=0)) == 80


# ---- when a name is worth publishing --------------------------------------------------

def test_enabled_only_when_the_dashboard_is_bound_to_the_network():
    """A loopback-bound rig has nobody to tell, and a name resolving to an address that refuses
    every caller is worse than no name -- it turns 'the rig is in localhost mode' into a mystery."""
    assert mdns.enabled(FakeCfg(web_host="0.0.0.0")) is True
    assert mdns.enabled(FakeCfg(web_host="::")) is True
    assert mdns.enabled(FakeCfg(web_host="127.0.0.1")) is False
    assert mdns.enabled(FakeCfg(web_host="192.168.1.50")) is False   # a specific bind, not wildcard


def test_enabled_respects_the_off_switch():
    assert mdns.enabled(FakeCfg(mdns=False)) is False


def test_lan_bound_is_independent_of_the_mdns_switch():
    """The two questions are different -- 'can anyone else connect?' vs 'should we announce a
    name?' -- and the printed block needs the first one even with mDNS switched off, or turning
    mDNS off would also hide the numeric address that still works."""
    off = FakeCfg(mdns=False, web_host="0.0.0.0")
    assert mdns.lan_bound(off) is True and mdns.enabled(off) is False


def test_publish_declines_without_starting_anything_when_disabled(monkeypatch):
    """The disabled path must not import zeroconf or touch a socket -- it is the path a
    localhost rig takes on every single start."""
    monkeypatch.setattr(mdns, "lan_ip", lambda: pytest.fail("must not look for a LAN address"))
    assert mdns.publish(FakeCfg(web_host="127.0.0.1")) is None
    assert mdns.publish(FakeCfg(mdns=False)) is None


def test_unpublish_tolerates_nothing_to_do():
    """Callers keep `None` when publishing was skipped or failed, and must not need a branch in
    their shutdown path to say so."""
    mdns.unpublish(None)


# ---- a name that is already taken: our own ghost, or somebody else's rig ---------------
# zeroconf refuses to register while ANY responder on the link still answers for the name, and a
# rig that died without sending goodbye packets -- crash, taskkill, power cut -- leaves exactly
# that, cached elsewhere until the TTL expires. rigwatch.py restarts this rig after it dies, so
# the restart lands inside the ghost's lifetime almost by construction. Refusing there would drop
# the name precisely on the reboots nobody is watching. Reclaiming somebody ELSE's name, though,
# is a different act entirely, so the two are told apart rather than blurred.

def _force_conflict(monkeypatch, holders, ip="192.168.1.50"):
    """Make the first _register raise NonUniqueName and the second succeed, recording whether the
    retry asked to cooperate. Returns the list of `cooperating` flags seen."""
    zeroconf = pytest.importorskip("zeroconf")
    seen = []

    def fake_register(name, addr, port, cooperating=False):
        seen.append(cooperating)
        if not cooperating:
            raise zeroconf.NonUniqueNameException
        return object(), object()

    monkeypatch.setattr(mdns, "lan_ip", lambda: ip)
    monkeypatch.setattr(mdns, "_register", fake_register)
    monkeypatch.setattr(mdns, "_held_by", lambda name, **kw: holders)
    monkeypatch.setattr(mdns, "Publication", lambda zc, info, name, a, p: ("pub", name, a, p))
    return seen


def test_a_name_held_by_this_rigs_own_ghost_is_reclaimed(monkeypatch, capsys):
    """Same address = the network is remembering us, not describing a rival."""
    seen = _force_conflict(monkeypatch, holders=["192.168.1.50"])
    pub = mdns.publish(FakeCfg())
    assert pub is not None, "a rig must not lose its name to its own previous run"
    assert seen == [False, True], "the retry should be the cooperating one"
    assert "previous run" in capsys.readouterr().out


def test_a_name_held_by_another_device_is_left_alone(monkeypatch, capsys):
    """A different address is a different rig. Taking the name would give two machines one name
    and make which-one-you-reach a coin flip -- worse than having no name."""
    seen = _force_conflict(monkeypatch, holders=["192.168.1.99"])
    assert mdns.publish(FakeCfg()) is None
    assert seen == [False], "must not retry onto somebody else's name"
    out = capsys.readouterr().out
    assert "already taken" in out and "192.168.1.99" in out    # name the holder, not a mystery
    assert "mdns_name" in out                                   # and say how to fix it


def test_a_name_held_by_nobody_findable_is_left_alone(monkeypatch):
    """Registration was refused but the browse turned up nothing to blame. Unexplained is not
    the same as ours, so this declines rather than forcing its way in."""
    seen = _force_conflict(monkeypatch, holders=None)
    assert mdns.publish(FakeCfg()) is None
    assert seen == [False]


def test_tear_down_never_raises_whatever_state_it_finds():
    """_tear_down runs on shutdown AND on every re-announce, where a raise would leave a dead
    thread still holding UDP 5353 against its own replacement -- so it has to survive a responder
    that is already closed, half-built, or broken."""
    class Exploding:
        def unregister_service(self, info):
            raise RuntimeError("already closed")

        def close(self):
            raise OSError("socket is gone")

    mdns._tear_down(Exploding(), object())


# ---- the printed block: the thing a person actually reads ------------------------------

def test_connect_lines_lead_with_the_name_and_keep_the_number():
    """Both, always, on the same screen. The name is the memorable half; the number is what a
    visitor needs the moment their Android phone shrugs at the name, and a support conversation
    is a bad substitute for a line of text."""
    lines = mdns.connect_lines(FakeCfg(web_port=8000), ip="192.168.1.50", name="critter-cam.local")
    assert "http://critter-cam.local:8000" in lines[0]
    assert "http://192.168.1.50:8000" in lines[1]
    assert any("127.0.0.1" in ln for ln in lines)


def test_connect_lines_on_port_80_print_a_bare_name():
    """The whole errand: what the block hands a person on the default port is a name with nothing
    after it. A stray ':80' anywhere here is the feature failing quietly."""
    lines = mdns.connect_lines(FakeCfg(web_port=80), ip="192.168.1.50", name="critter-cam.local")
    assert "http://critter-cam.local" in lines[0]
    assert not any(":80" in ln for ln in lines)
    assert any("http://192.168.1.50" in ln for ln in lines)


def test_connect_lines_report_the_port_actually_bound():
    """Asked for 80, got the fallback: the block must describe the socket, not the request. This
    is the difference between a wrong address printed confidently and a right one."""
    lines = mdns.connect_lines(FakeCfg(web_port=80), ip="192.168.1.50",
                               name="critter-cam.local", port=8000)
    assert "http://critter-cam.local:8000" in lines[0]
    assert not any(ln.rstrip().endswith("critter-cam.local") for ln in lines)


def test_connect_lines_without_a_published_name_fall_back_to_the_number():
    """Publishing can fail (no zeroconf, firewall, name clash) and the rig serves on anyway, so
    the block must never advertise a name nothing is answering for."""
    lines = mdns.connect_lines(FakeCfg(), ip="192.168.1.50", name=None)
    assert not any(".local" in ln for ln in lines)
    assert any("http://192.168.1.50" in ln for ln in lines)


def test_connect_lines_never_claim_the_lan_in_localhost_mode():
    """This machine HAS a LAN address whether or not the dashboard is listening on it. Printing
    it while bound to loopback sends someone to a connection their own rig refuses -- the
    likeliest way to spend an hour on a firewall over a missing --host flag."""
    lines = mdns.connect_lines(FakeCfg(web_host="127.0.0.1"), ip="192.168.1.50",
                               name="critter-cam.local")
    assert not any("192.168.1.50" in ln for ln in lines)
    assert not any(".local" in ln for ln in lines)
    assert any("http://127.0.0.1" in ln for ln in lines)
    assert any("localhost mode" in ln for ln in lines)


def test_connect_lines_carry_a_non_default_port():
    lines = mdns.connect_lines(FakeCfg(web_port=9001), ip="10.0.0.7", name="cam.local")
    assert all(":8000" not in ln for ln in lines)
    assert any("http://cam.local:9001" in ln for ln in lines)


# ---- lan_ip: which addresses may be announced -----------------------------------------

class FakeSock:
    def __init__(self, ip):
        self.ip = ip

    def settimeout(self, t):
        pass

    def connect(self, addr):
        pass

    def getsockname(self):
        return (self.ip, 9)

    def close(self):
        pass


@pytest.mark.parametrize("ip, want", [
    ("192.168.1.101", "192.168.1.101"),
    ("10.0.0.5", "10.0.0.5"),
    ("172.20.1.1", "172.20.1.1"),
    ("8.8.8.8", None),            # public: the rig would be sitting on the internet
    ("127.0.0.1", None),          # loopback tells a visitor nothing
    ("169.254.3.4", None),        # link-local = DHCP failed; announcing it advertises the failure
])
def test_lan_ip_accepts_only_a_private_address(monkeypatch, ip, want):
    monkeypatch.setattr(mdns.socket, "socket", lambda *a: FakeSock(ip))
    assert mdns.lan_ip() == want


def test_lan_ip_is_none_when_there_is_no_network(monkeypatch):
    """No LAN is a normal state (Wi-Fi down, rig on a bench), not an error to raise into a
    startup path."""
    def boom(*a):
        raise OSError("network is down")
    monkeypatch.setattr(mdns.socket, "socket", boom)
    assert mdns.lan_ip() is None
