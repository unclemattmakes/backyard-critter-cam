"""
Tests for secondopinion.py, the shadow-mode Claude second opinion on uncertain species.

No network and no Anthropic SDK: a fake client stands in for the API and records what would have
been sent. Pinned here: which crops are chosen (people never, reviewed crops only for the eval),
that BioCLIP's guess never reaches the prompt, how answers and refusals land in the sidecar, that
backyard.db is not written, and how the grade treats each kind of human verdict.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

import db
import secondopinion as so
from clipfilter import NONANIMAL_LABEL

TZ = datetime.now().astimezone().tzinfo
NOW = datetime(2026, 10, 16, 22, 0, 0, tzinfo=TZ)
LABELS = so.allowed_labels(["raccoon", "Virginia opossum", "domestic cat"])


def _crop(tmp_path, name, size=(64, 48)):
    from PIL import Image
    path = tmp_path / f"{name}.jpg"
    Image.new("RGB", size, (90, 90, 90)).save(path)
    return str(path)


def _det(conn, tmp_path, *, cls="animal", species="raccoon", conf=0.6, hours_ago=1.0,
         verified=None, corrected_to=None, crop_size=(64, 48)):
    det_id = db.insert_detection(
        conn, timestamp=(NOW - timedelta(hours=hours_ago)).isoformat(),
        source=db.SOURCE_GLASS_DOOR_CAM, detection_class=cls, confidence=0.9,
        bbox=(0, 0, 10, 10), frame_w=100, frame_h=100,
        crop_path=_crop(tmp_path, f"d{conn.execute('SELECT COUNT(*) FROM detections').fetchone()[0]}",
                        crop_size))
    if species is not None:
        db.set_species(conn, det_id, species, conf)
    if verified is not None:
        db.set_species_verified(conn, det_id, verified)
    if corrected_to is not None:
        db.correct_species(conn, det_id, corrected_to)
    conn.commit()
    return det_id


class FakeClient:
    """Stands in for anthropic.Anthropic: client.beta.messages.create(**req)."""

    def __init__(self, answer=None, stop_reason="end_turn", error=None):
        self.requests = []
        self._answer = answer or {"species": "raccoon", "certainty": "high", "reason": "mask"}
        self._stop = stop_reason
        self._error = error
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **req):
        self.requests.append(req)
        if self._error is not None:
            raise self._error
        content = [SimpleNamespace(type="thinking", thinking=""),
                   SimpleNamespace(type="text", text=json.dumps(self._answer))]
        return SimpleNamespace(stop_reason=self._stop, content=content, model=req["model"],
                               usage=SimpleNamespace(input_tokens=700, output_tokens=120))


@pytest.fixture
def sidecar(tmp_path):
    c = so.open_sidecar(tmp_path / "second_opinion.db")
    yield c
    c.close()


# ---- choosing crops ------------------------------------------------------------------------

def test_graded_candidates_are_reviewed_uncertain_animal_crops(conn, tmp_path):
    want = _det(conn, tmp_path, conf=0.6, verified=1)
    corrected = _det(conn, tmp_path, conf=0.3, corrected_to="Virginia opossum")
    _det(conn, tmp_path, conf=0.9, verified=1)                  # confident: not uncertain
    _det(conn, tmp_path, conf=0.6)                              # nobody has reviewed it
    _det(conn, tmp_path, cls="person", conf=0.6, verified=1)    # people are never sent
    ids = {r["id"] for r in so.candidates(conn, graded=True, max_conf=0.8)}
    assert ids == {want, corrected}


def test_recent_candidates_are_unreviewed_and_inside_the_window(conn, tmp_path):
    fresh = _det(conn, tmp_path, conf=0.6, hours_ago=2)
    _det(conn, tmp_path, conf=0.6, hours_ago=30)                # too old
    _det(conn, tmp_path, conf=0.6, hours_ago=2, verified=1)     # already reviewed
    ids = {r["id"] for r in so.candidates(conn, graded=False, max_conf=0.8, recent_hours=24,
                                          now=NOW)}
    assert ids == {fresh}


def test_pick_gives_a_second_model_the_same_sample(conn, tmp_path):
    rows = [{"id": i} for i in range(50)]
    first = so.pick(rows, limit=10, already=set(), graded=True, seed=3)
    again = so.pick(rows, limit=10, already=set(), graded=True, seed=3)
    longer = so.pick(rows, limit=20, already=set(), graded=True, seed=3)
    assert [r["id"] for r in first] == [r["id"] for r in again]
    assert [r["id"] for r in longer[:10]] == [r["id"] for r in first]
    done = {r["id"] for r in first[:4]}
    assert [r["id"] for r in so.pick(rows, limit=10, already=done, graded=True, seed=3)] == \
        [r["id"] for r in first[4:]]


# ---- asking ----------------------------------------------------------------------------------

def test_prompt_never_carries_bioclips_guess(conn, tmp_path, sidecar):
    _det(conn, tmp_path, species="domestic cat", conf=0.6, verified=1)
    rows = so.candidates(conn, graded=True, max_conf=0.8)
    client = FakeClient()
    so.ask_all(rows, client=client, model="claude-opus-5-5", labels=LABELS, sidecar=sidecar,
               log=lambda *_: None)
    req = client.requests[0]
    text = next(b["text"] for b in req["messages"][0]["content"] if b["type"] == "text")
    assert "0.6" not in text and "BioCLIP" not in text and "bioclip" not in json.dumps(req)
    # the label list is offered, so "domestic cat" appears once there and nowhere else
    assert text.count("domestic cat") == 1


def test_request_shape_fallbacks_and_downscaling(tmp_path):
    image = so.encode_crop(_crop(tmp_path, "big", size=(2000, 1000)))
    from PIL import Image
    import base64
    import io
    with Image.open(io.BytesIO(base64.b64decode(image))) as im:
        assert max(im.size) == so.IMAGE_MAX_SIDE
    opus = so.build_request("claude-opus-5-5", LABELS, image, NOW.isoformat())
    assert opus["fallbacks"] == "default" and opus["betas"] == [so.FALLBACK_BETA]
    assert opus["output_config"]["format"]["schema"]["properties"]["species"]["enum"] == LABELS
    assert "thinking" not in opus and "tool_choice" not in opus
    haiku = so.build_request("claude-haiku-5-5", LABELS, image, NOW.isoformat())
    assert "fallbacks" not in haiku and "betas" not in haiku


def test_answers_refusals_and_unreadable_crops_land_in_the_sidecar(conn, tmp_path, sidecar):
    ok = _det(conn, tmp_path, verified=1)
    refused = _det(conn, tmp_path, verified=1)
    broken = _det(conn, tmp_path, verified=1)
    rows = {r["id"]: r for r in so.candidates(conn, graded=True, max_conf=0.8)}
    with open(rows[broken]["crop_path"], "wb") as f:
        f.write(b"\0" * 4096)            # a host hang leaves NUL-filled crops

    so.ask_all([rows[ok], rows[broken]], client=FakeClient(), model="m", labels=LABELS,
               sidecar=sidecar, log=lambda *_: None)
    so.ask_all([rows[refused]], client=FakeClient(stop_reason="refusal"), model="m",
               labels=LABELS, sidecar=sidecar, log=lambda *_: None)
    got = {r["detection_id"]: dict(r) for r in sidecar.execute("SELECT * FROM opinions")}
    assert got[ok]["outcome"] == "answered" and got[ok]["species"] == "raccoon"
    assert got[ok]["input_tokens"] == 700
    assert got[refused]["outcome"] == "refusal" and got[refused]["species"] is None
    assert got[broken]["outcome"] == "unreadable"
    assert so.already_asked(sidecar, "m") == {ok, refused, broken}


def test_a_transient_error_is_retried_next_run_and_a_bad_key_stops_the_run(conn, tmp_path,
                                                                           sidecar):
    _det(conn, tmp_path, verified=1)
    rows = so.candidates(conn, graded=True, max_conf=0.8)
    so.ask_all(rows, client=FakeClient(error=ConnectionError("reset")), model="m",
               labels=LABELS, sidecar=sidecar, log=lambda *_: None)
    assert sidecar.execute("SELECT outcome FROM opinions").fetchone()[0] == "error"
    assert so.already_asked(sidecar, "m") == set()

    bad_key = RuntimeError("invalid x-api-key")
    bad_key.status_code = 401
    with pytest.raises(RuntimeError):
        so.ask_all(rows, client=FakeClient(error=bad_key), model="m", labels=LABELS,
                   sidecar=sidecar, log=lambda *_: None)


def test_an_answer_off_the_list_is_an_error_not_a_label():
    resp = FakeClient(answer={"species": "wolverine", "certainty": "high", "reason": ""})
    out = so.parse_response(resp._create(model="m"), LABELS)
    assert out["outcome"] == "error" and "species" not in out


def test_backyard_db_is_not_written(conn, db_path, tmp_path, sidecar):
    _det(conn, tmp_path, verified=1)
    before = conn.execute("SELECT * FROM detections").fetchall()
    ro = db.connect_readonly(db_path)
    so.ask_all(so.candidates(ro, graded=True, max_conf=0.8), client=FakeClient(), model="m",
               labels=LABELS, sidecar=sidecar, log=lambda *_: None)
    ro.close()
    assert conn.execute("SELECT * FROM detections").fetchall() == before


# ---- grading ---------------------------------------------------------------------------------

def _answer(sidecar, det_id, species, certainty="high", model="m"):
    so.record(sidecar, det_id, model, {"outcome": "answered", "species": species,
                                       "certainty": certainty})


def test_grade_uses_each_kind_of_human_verdict(conn, tmp_path, sidecar):
    confirmed = _det(conn, tmp_path, species="raccoon", conf=0.6, verified=1)
    corrected = _det(conn, tmp_path, species="raccoon", conf=0.3, corrected_to="Virginia opossum")
    rejected_echo = _det(conn, tmp_path, species="domestic cat", conf=0.6, verified=0)
    rejected_other = _det(conn, tmp_path, species="domestic cat", conf=0.6, verified=0)
    off_list = _det(conn, tmp_path, species="raccoon", conf=0.4, corrected_to="coyote")
    _answer(sidecar, confirmed, "raccoon")
    _answer(sidecar, corrected, "Virginia opossum", certainty="low")
    _answer(sidecar, rejected_echo, "domestic cat")
    _answer(sidecar, rejected_other, NONANIMAL_LABEL)
    _answer(sidecar, off_list, so.OTHER_LABEL)
    sidecar.commit()

    rep = so.grade(conn, sidecar, "m", LABELS, period_fn=lambda dt: "night")
    every = rep["slices"]["all:all"]
    assert every["n"] == 3                       # confirmed, corrected, off-list coyote
    assert every["bioclip_acc"] == 0.333
    assert every["claude_acc"] == 1.0            # "other animal" is right for a coyote
    assert every["claude_high_n"] == 2
    assert every["agree_n"] == 1 and every["agree_acc"] == 1.0
    assert every["rejected_n"] == 2 and every["rejected_claude_echoed"] == 1
    assert rep["slices"]["band:<0.5"]["n"] == 2
    assert rep["slices"]["band:0.5-0.8"]["n"] == 1
    assert rep["outcomes"] == {"answered": 5}


def test_anthropic_api_key_is_a_secret_field():
    import config
    assert "anthropic_api_key" in config.SECRET_FIELDS
    assert config.Config().anthropic_api_key is None


def test_crop_paths_resolve_from_the_database_folder(tmp_path, monkeypatch):
    # crop_path is stored relative to the project root, the folder backyard.db lives in; a run
    # from another checkout must not look for the crops beside itself.
    rig = tmp_path / "rig"
    (rig / "crops").mkdir(parents=True)
    from PIL import Image
    Image.new("RGB", (64, 48)).save(rig / "crops" / "a.jpg")
    conn = db.connect(rig / "backyard.db")
    det = db.insert_detection(
        conn, timestamp=NOW.isoformat(), source=db.SOURCE_GLASS_DOOR_CAM,
        detection_class="animal", confidence=0.9, bbox=(0, 0, 1, 1), frame_w=10, frame_h=10,
        crop_path="crops/a.jpg")
    db.set_species(conn, det, "raccoon", 0.6)
    db.set_species_verified(conn, det, 1)
    conn.commit()
    conn.close()
    monkeypatch.chdir(tmp_path)
    assert so.main(["--graded", "--dry-run", "--db", str(rig / "backyard.db")]) == 0
    assert not (rig / "second_opinion.db").exists()      # a dry run writes nothing
    (rig / "crops" / "a.jpg").unlink()
    assert so.main(["--graded", "--dry-run", "--db", str(rig / "backyard.db")]) == 1


def test_a_graded_sample_takes_at_most_three_crops_from_one_visit():
    rows = ([{"id": i, "visit_id": 7} for i in range(100)]
            + [{"id": 100 + i, "visit_id": 100 + i} for i in range(20)]
            + [{"id": 200, "visit_id": None}, {"id": 201, "visit_id": None}])
    sample = so.pick(rows, limit=None, already=set(), graded=True, seed=0)
    assert sum(1 for r in sample if r["visit_id"] == 7) == 3
    assert len(sample) == 3 + 20 + 2            # unstamped crops each count as their own visit
    uncapped = so.pick(rows, limit=None, already=set(), graded=True, seed=0, per_visit=0)
    assert len(uncapped) == len(rows)


def test_grade_counts_only_the_sample(conn, tmp_path, sidecar):
    inside = _det(conn, tmp_path, species="raccoon", conf=0.6, verified=1)
    outside = _det(conn, tmp_path, species="raccoon", conf=0.6, verified=1)
    _answer(sidecar, inside, "raccoon")
    _answer(sidecar, outside, NONANIMAL_LABEL)  # left over from an earlier, wider sample
    sidecar.commit()
    rep = so.grade(conn, sidecar, "m", LABELS, period_fn=lambda dt: "day", sample_ids={inside})
    assert rep["slices"]["all:all"]["n"] == 1
    assert rep["slices"]["all:all"]["claude_acc"] == 1.0
    assert rep["graded_answers"] == 1 and rep["asked"] == 2
