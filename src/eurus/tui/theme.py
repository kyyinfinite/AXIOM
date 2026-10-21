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


UNICODE_GLYPHS = Glyphs("›", "…", "●", "─", "▸", "✓", "✗", "…", "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", "▎", "✦")
ASCII_GLYPHS = Glyphs(">", ".", "*", "-", ">", "+", "x", "...", "|/-\\", "|", "*")


@dataclass(frozen=True, slots=True)
class Theme:
    teal: str = "#4fb3bf"
    steel: str = "#8fb4cf"
    copper: str = "#d08a6a"
    amber: str = "#e0a45c"
    green: str = "#8bc98f"
    yellow: str = "#e3c26b"
    red: str = "#e07a7a"
    text: str = "#e6dfd2"
    dim: str = "#9a948a"
    border: str = "#4a4743"
    surface: str = "#2b2824"
    glyphs: Glyphs = UNICODE_GLYPHS

    @classmethod
    def for_unicode(cls, unicode: bool) -> Theme:
        return cls(glyphs=UNICODE_GLYPHS if unicode else ASCII_GLYPHS)

    def state_style(self, state: AgentState) -> str:
        return {
            AgentState.IDLE: self.green,
            AgentState.THINKING: self.copper,
            AgentState.RUNNING: self.steel,
            AgentState.APPROVAL: self.amber,
            AgentState.ERROR: self.red,
        }[state]
