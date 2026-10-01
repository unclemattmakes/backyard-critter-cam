"""
The secrets file (config.secrets_path / load_secrets / apply_secrets): credentials kept in a JSON
file outside the project, so backup.py's meta zip -- which carries config_local.py -- never carries
them. Every value here is a made-up placeholder.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent.parent


def test_a_missing_file_is_no_secrets_not_an_error(tmp_path, capsys):
    assert config.load_secrets(tmp_path / "absent.json") == {}
    assert capsys.readouterr().err == ""


def test_only_non_empty_strings_survive(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"email_resend_api_key": "re_placeholder", "operator_token": "",
                             "mqtt_password": 1234, "yard_ir_url": "rtsp://u:p@10.0.0.9/s"}),
                 encoding="utf-8")
    assert config.load_secrets(p) == {"email_resend_api_key": "re_placeholder",
                                      "yard_ir_url": "rtsp://u:p@10.0.0.9/s"}


def test_a_malformed_file_is_reported_without_its_contents(tmp_path, capsys):
    p = tmp_path / "s.json"
    p.write_text('{"operator_token": "tok_placeholder",,}', encoding="utf-8")
    assert config.load_secrets(p) == {}
    err = capsys.readouterr().err
    assert str(p) in err and "tok_placeholder" not in err
    p.write_text('["not", "an", "object"]', encoding="utf-8")
    assert config.load_secrets(p) == {}


def test_path_comes_from_the_env_var_else_the_home_folder(monkeypatch, tmp_path):
    monkeypatch.setenv(config.SECRETS_ENV, str(tmp_path / "elsewhere.json"))
    assert config.secrets_path() == tmp_path / "elsewhere.json"
    monkeypatch.delenv(config.SECRETS_ENV)
    assert config.secrets_path() == Path.home() / ".critter-cam" / "secrets.json"


def test_apply_fills_only_the_documented_fields():
    cfg = config.Config()
    filled = config.apply_secrets(cfg, {"email_resend_api_key": "re_placeholder",
                                        "mqtt_password": "mqtt_placeholder",
                                        "web_host": "0.0.0.0", "yard_ir_url": "rtsp://x"})
    assert filled == ["email_resend_api_key", "mqtt_password"]
    assert cfg.email_resend_api_key == "re_placeholder"
    assert cfg.mqtt_password == "mqtt_placeholder"
    assert cfg.operator_token is None
    assert cfg.web_host == "127.0.0.1"          # not a secret field: never touched


def test_the_secrets_file_is_never_in_the_meta_zip(monkeypatch):
    import backup
    monkeypatch.delenv(config.SECRETS_ENV)       # the default location, not conftest's stand-in
    target = config.secrets_path().resolve()
    for item in backup.meta_items():
        item = item.resolve()
        assert target != item and item not in target.parents


def _load_config_in(cwd: Path, secrets: Path) -> dict:
    """Import config fresh in a subprocess (cwd first on sys.path, so the config_local.py there is
    the one it finds, never the operator's) and report the three secret fields."""
    env = dict(os.environ, PYTHONPATH=str(ROOT), CRITTER_CAM_SECRETS=str(secrets))
    code = ("import json, config; c = config.CONFIG; print(json.dumps("
            "[c.email_resend_api_key, c.operator_token, c.mqtt_password]))")
    out = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True,
                         text=True, timeout=60, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_config_loads_the_file_and_config_local_still_wins(tmp_path):
    secrets = tmp_path / "secrets.json"
    secrets.write_text(json.dumps({"email_resend_api_key": "re_file",
                                   "operator_token": "tok_file"}), encoding="utf-8")
    bare = tmp_path / "bare"
    bare.mkdir()
    # A no-op config_local.py here shadows any real one in the project root (PYTHONPATH).
    (bare / "config_local.py").write_text("def apply(cfg):\n    pass\n", encoding="utf-8")
    assert _load_config_in(bare, secrets) == ["re_file", "tok_file", None]

    local = tmp_path / "local"
    local.mkdir()
    (local / "config_local.py").write_text(
        "from config import secret\n"
        "def apply(cfg):\n"
        "    cfg.operator_token = 'tok_local'\n"
        "    cfg.mqtt_password = secret('mqtt_password') or 'mqtt_local'\n", encoding="utf-8")
    assert _load_config_in(local, secrets) == ["re_file", "tok_local", "mqtt_local"]
