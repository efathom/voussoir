import json

import pytest

from voussoir.observability.logging_setup import configure_logging, get_logger


def test_configure_logging_dev_human_readable(capsys):
    configure_logging(level="DEBUG", format="dev")
    log = get_logger("voussoir.test")
    log.info("hello", agent="researcher", tokens_in=42)

    captured = capsys.readouterr()
    assert "hello" in captured.err
    assert "researcher" in captured.err
    # Dev format is human-readable, NOT JSON
    with pytest.raises(json.JSONDecodeError):
        json.loads(captured.err.strip().splitlines()[-1])


def test_configure_logging_prod_emits_json(capsys):
    configure_logging(level="INFO", format="json")
    log = get_logger("voussoir.test")
    log.info("agent_started", agent="researcher", run_id="r1")

    captured = capsys.readouterr()
    payload = json.loads(captured.err.strip().splitlines()[-1])
    assert payload["event"] == "agent_started"
    assert payload["agent"] == "researcher"
    assert payload["run_id"] == "r1"


def test_get_logger_returns_logger_with_name():
    log = get_logger("voussoir.foo.bar")
    assert log is not None
