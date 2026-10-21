from __future__ import annotations

from dataclasses import dataclass

from eurus.tui.state import AgentState


@dataclass(frozen=True, slots=True)
class Glyphs:
    prompt: str
    cont: str
    dot: str
    rule: str
    arrow: str
    ok: str
    fail: str
    ellipsis: str
    frames: str
    bar: str
    spark: str
    ascii: bool = False


UNICODE_GLYPHS = Glyphs("›", "…", "●", "─", "▸", "✓", "✗", "…", "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", "▎", "✦")
ASCII_GLYPHS = Glyphs(">", ".", "*", "-", ">", "+", "x", "...", "|/-\\", "|", "*", True)
NERD_GLYPHS = Glyphs(
    "❯",
    "…",
    "\uf111",
    "─",
    "\uf054",
    "\uf00c",
    "\uf00d",
    "…",
    "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
    "▎",
    "\uf0e7",
)

GLYPH_SETS: dict[str, Glyphs] = {
    "unicode": UNICODE_GLYPHS,
    "ascii": ASCII_GLYPHS,
    "nerd": NERD_GLYPHS,
}
GLYPH_KEYS = (
    "prompt",
    "cont",
    "dot",
    "rule",
    "arrow",
    "ok",
    "fail",
    "ellipsis",
    "frames",
    "bar",
    "spark",
)
DEFAULT_CODE_THEME = "gruvbox-dark"


@dataclass(frozen=True, slots=True)
class Theme:
    teal: str = "#4fb3bf"
    steel: str = "#c2b59b"
    copper: str = "#d08a6a"
    amber: str = "#e0a45c"
    green: str = "#8bc98f"
    yellow: str = "#e3c26b"
    red: str = "#e07a7a"
    text: str = "#e6dfd2"
    dim: str = "#9a948a"
    border: str = "#4a4743"
    surface: str = "#2b2824"
    background: str = "#121110"
    glyphs: Glyphs = UNICODE_GLYPHS
    code_theme: str = DEFAULT_CODE_THEME

    @classmethod
    def for_unicode(cls, unicode: bool) -> Theme:
        return cls(glyphs=UNICODE_GLYPHS if unicode else ASCII_GLYPHS)

    @property
    def unicode(self) -> bool:
        return not self.glyphs.ascii

    def state_style(self, state: AgentState) -> str:
        return {
            AgentState.IDLE: self.green,
            AgentState.THINKING: self.copper,
            AgentState.RUNNING: self.steel,
            AgentState.APPROVAL: self.amber,
            AgentState.ERROR: self.red,
        }[state]
