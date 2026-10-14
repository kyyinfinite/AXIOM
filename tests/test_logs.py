import json

from axiom.config import LoggingConfig
from axiom.logs import get_logger, redact, setup_logging, shutdown_logging


def test_redact_known_patterns():
    text = "sk-abcdefgh12345678 Bearer abcdefghijkl token=abc123 gsk_abcdefgh1234"
    cleaned = redact(text)
    for leaked in ("sk-abcdefgh", "abcdefghijkl", "abc123", "gsk_abcdefgh"):
        assert leaked not in cleaned


def test_redact_explicit_secrets():
    assert "supersecretvalue" not in redact("v supersecretvalue", ["supersecretvalue"])


def test_file_log_is_json_and_redacted(tmp_path):
    target = tmp_path / "logs" / "axiom.log"
    try:
        logger = setup_logging(LoggingConfig("INFO", str(target), True), secrets=["topsecret123"])
        get_logger("test").info("hello topsecret123")
        for handler in logger.handlers:
            handler.flush()
        line = json.loads(target.read_text(encoding="utf-8").strip().splitlines()[-1])
    finally:
        shutdown_logging()
    assert line["level"] == "INFO"
    assert "hello" in line["message"]
    assert "topsecret123" not in line["message"]
