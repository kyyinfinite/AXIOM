from pathlib import Path

from eurus.tui.capabilities import detect, detect_color, detect_unicode
from eurus.tui.commands import COMMANDS, resolve, suggest
from eurus.tui.layout import Segment, clip, fit_segments, format_tokens, shorten_path
from eurus.tui.redact import redact_arguments, redact_text, summarize_arguments


def test_color_levels():
    assert detect_color({"TERM": "xterm-256color", "COLORTERM": "truecolor"}, True) == "truecolor"
    assert detect_color({"TERM": "xterm-256color"}, True) == "256"
    assert detect_color({"TERM": "xterm"}, True) == "basic"
    assert detect_color({"TERM": "xterm-256color", "NO_COLOR": "1"}, True) == "none"
    assert detect_color({"TERM": "dumb"}, True) == "none"
    assert detect_color({"TERM": "xterm"}, False) == "none"


def test_unicode_detection():
    assert detect_unicode({}, "utf-8")
    assert detect_unicode({"LANG": "en_US.UTF-8"}, "ascii")
    assert not detect_unicode({"LANG": "C"}, "ascii")


def test_width_classes():
    assert detect({}, width=40, encoding="utf-8", is_tty=True).narrow
    caps = detect({}, width=50, encoding="utf-8", is_tty=True)
    assert caps.compact and not caps.narrow
    assert not detect({}, width=80, encoding="utf-8", is_tty=True).compact


def test_clip_and_tokens():
    assert clip("a  b\nc", 10) == "a b c"
    assert clip("x" * 50, 10) == "x" * 7 + "..."
    assert clip("abc", 0) == ""
    assert format_tokens(999) == "999" and format_tokens(1500) == "1.5k"


def test_shorten_path_keeps_tail():
    long = "/a/very/long/project/path/that/keeps/going/src"
    result = shorten_path(long, 20)
    assert len(result) == 20 and result.endswith("going/src")
    assert shorten_path(Path.home() / "work", 40) == "~/work"


def test_fit_segments_drops_low_priority_fields():
    segments = [Segment("a idle"), Segment("m" * 40, flexible=True), Segment("/help")]
    narrow = fit_segments(segments, 24)
    assert sum(len(part) for part in narrow) + 3 * (len(narrow) - 1) <= 24
    assert narrow[0] == "a idle"
    wide = fit_segments([Segment("a"), Segment("b"), Segment("/help")], 80)
    assert wide == ["a", "b", "/help"]


def test_redaction():
    args = {"api_key": "x", "nested": {"Authorization": "Bearer abcdefghij"}, "path": "a.py"}
    safe = redact_arguments(args)
    assert safe["api_key"] == "REDACTED" and safe["nested"]["Authorization"] == "REDACTED"
    assert safe["path"] == "a.py"
    assert "sk-abcdefghijkl" not in redact_text("use sk-abcdefghijkl now")
    assert "hunter2" not in redact_text("export API_KEY=hunter2")
    assert "REDACTED" in summarize_arguments({"password": "p"})


def test_command_resolution():
    assert [c.name for c in resolve("/model")] == ["/model"]
    assert {c.name for c in resolve("/mo")} == {"/model", "/models"}
    assert [c.name for c in resolve("/quit")] == ["/exit"]
    assert len(resolve("/")) == len(COMMANDS)
    assert resolve("/zzz") == []
    assert "/status" in [c.name for c in suggest("/staus")]


def test_banner_rows_have_gradient_and_fit():
    from eurus.tui.brand import BANNER, BANNER_MIN_WIDTH, banner

    text = banner()
    colors = {span.style for span in text.spans if span.style}
    assert len(colors) > 20
    assert text.plain.splitlines() == list(BANNER)
    assert BANNER_MIN_WIDTH <= 46
