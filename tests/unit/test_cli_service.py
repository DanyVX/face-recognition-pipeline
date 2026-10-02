"""CLI commands for keys and the recognition operations (service replaced by a fake-backed one)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from typer.testing import CliRunner

from frp import cli
from frp.logging_setup import ALLOWED_EXTRAS, JsonFormatter, configure_logging
from tests.unit.service_fixtures import make_service, person_png

runner = CliRunner()


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(
        cli,
        "build_service",
        lambda: make_service(tmp_path, verify_threshold=0.9, identify_threshold=0.9),
    )
    return tmp_path


def write_png(path: Path, seed: int, noise: float = 0.0, noise_seed: int = 0) -> Path:
    path.write_bytes(person_png(seed, noise, noise_seed))
    return path


def test_enroll_identify_verify_list_delete_through_the_cli(env: Path) -> None:
    alice = write_png(env / "alice.png", 1)
    alice2 = write_png(env / "alice2.png", 1, 4, 1)
    bob = write_png(env / "bob.png", 2)

    out = runner.invoke(
        cli.app, ["enroll", str(alice), str(alice2), "--name", "Alice", "--identity-id", "alice"]
    )
    assert out.exit_code == 0, out.output
    assert json.loads(out.stdout)["embeddings_added"] == 2
    assert (
        runner.invoke(
            cli.app, ["enroll", str(bob), "--name", "Bob", "--identity-id", "bob"]
        ).exit_code
        == 0
    )

    listed = json.loads(runner.invoke(cli.app, ["list"]).stdout)
    assert [r["identity_id"] for r in listed] == ["alice", "bob"] and listed[0]["embeddings"] == 2
    assert 'embedding"' not in runner.invoke(cli.app, ["list"]).stdout  # counts only

    found = json.loads(
        runner.invoke(cli.app, ["identify", str(write_png(env / "q.png", 1, 4, 5))]).stdout
    )
    assert found["match"]["identity_id"] == "alice"

    same = json.loads(runner.invoke(cli.app, ["verify", str(alice), str(alice2)]).stdout)
    different = json.loads(runner.invoke(cli.app, ["verify", str(alice), str(bob)]).stdout)
    assert same["accepted"] is True and different["accepted"] is False
    against = json.loads(
        runner.invoke(cli.app, ["verify", str(alice2), "--identity", "alice"]).stdout
    )
    assert against["accepted"] is True

    deleted = json.loads(runner.invoke(cli.app, ["delete", "alice"]).stdout)
    assert deleted == {"identity_id": "alice", "embeddings_removed": 2}


def test_errors_print_a_machine_readable_code_and_exit_nonzero(env: Path) -> None:
    bad = env / "bad.jpg"
    bad.write_bytes(b"not an image")
    result = runner.invoke(cli.app, ["enroll", str(bad), "--name", "X"])
    assert result.exit_code == 1
    assert "UNSUPPORTED_IMAGE_FORMAT" in result.output and '"image_index": 0' in result.output
    missing = runner.invoke(cli.app, ["identify", str(env / "nope.png")])
    assert missing.exit_code == 1 and "Cannot read" in missing.output
    ghost = runner.invoke(cli.app, ["delete", "ghost"])
    assert ghost.exit_code == 1 and "IDENTITY_NOT_FOUND" in ghost.output


def test_verify_needs_exactly_one_comparison_target(env: Path) -> None:
    a = write_png(env / "a.png", 1)
    assert runner.invoke(cli.app, ["verify", str(a)]).exit_code == 2
    b = write_png(env / "b.png", 2)
    assert runner.invoke(cli.app, ["verify", str(a), str(b), "--identity", "x"]).exit_code == 2


def test_api_key_commands(env: Path) -> None:
    created = runner.invoke(cli.app, ["keys", "create", "--name", "ci"])
    assert created.exit_code == 0
    key = created.stdout.strip()
    assert key.startswith("frp_") and "key id:" in created.stderr
    listed = json.loads(runner.invoke(cli.app, ["keys", "list"]).stdout)
    assert listed[0]["name"] == "ci" and listed[0]["revoked"] is False
    assert key not in runner.invoke(cli.app, ["keys", "list"]).output  # never printed again
    key_id = listed[0]["key_id"]
    assert runner.invoke(cli.app, ["keys", "revoke", key_id]).exit_code == 0
    assert runner.invoke(cli.app, ["keys", "revoke", key_id]).exit_code == 1  # already revoked
    assert json.loads(runner.invoke(cli.app, ["keys", "list"]).stdout)[0]["revoked"] is True


# --- logging -------------------------------------------------------------------------------------


def test_json_logs_have_structure_and_drop_unknown_extras() -> None:
    record = logging.LogRecord("frp.test", logging.INFO, __file__, 1, "enrolled %s", ("x",), None)
    record.request_id = "abc12345"
    record.name_value = "Alice Example"  # a name logged by mistake
    record.embedding = [0.1, 0.2]
    line = JsonFormatter().format(record)
    data = json.loads(line)
    assert (
        data["message"] == "enrolled x"
        and data["request_id"] == "abc12345"
        and data["level"] == "INFO"
    )
    assert "Alice" not in line and "0.1" not in line  # unknown extras are dropped, not serialized
    assert set(ALLOWED_EXTRAS) >= {"request_id", "key_id"}


def test_exception_messages_are_not_included_in_json_logs() -> None:
    try:
        raise ValueError("contains private data: Alice")
    except ValueError:
        import sys

        record = logging.LogRecord("t", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
    line = JsonFormatter().format(record)
    assert json.loads(line)["exception"] == "ValueError" and "Alice" not in line


def test_configure_logging_installs_one_handler_in_the_requested_format(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO", "json")
    configure_logging("INFO", "json")  # idempotent: handlers are replaced, not stacked
    assert len(logging.getLogger().handlers) == 1
    logging.getLogger("frp.demo").info("hello", extra={"request_id": "rid-1"})
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and json.loads(err[0])["request_id"] == "rid-1"
    configure_logging("WARNING", "text")
