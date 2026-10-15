from __future__ import annotations

from dataclasses import dataclass

from axiom.tui.state import AgentState


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


UNICODE_GLYPHS = Glyphs("›", "…", "●", "─", "▸", "✓", "✗", "…", "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
ASCII_GLYPHS = Glyphs(">", ".", "*", "-", ">", "+", "x", "...", "|/-\\")


@dataclass(frozen=True, slots=True)
class Theme:
    cyan: str = "#22d3ee"
    blue: str = "#60a5fa"
    violet: str = "#a78bfa"
    orange: str = "#fb923c"
    green: str = "#4ade80"
    yellow: str = "#facc15"
    red: str = "#f87171"
    text: str = "#e5e7eb"
    dim: str = "#9ca3af"
    border: str = "#4b5563"
    glyphs: Glyphs = UNICODE_GLYPHS

    @classmethod
    def for_unicode(cls, unicode: bool) -> Theme:
        return cls(glyphs=UNICODE_GLYPHS if unicode else ASCII_GLYPHS)

    def state_style(self, state: AgentState) -> str:
        return {
            AgentState.IDLE: self.green,
            AgentState.THINKING: self.violet,
            AgentState.RUNNING: self.blue,
            AgentState.APPROVAL: self.orange,
            AgentState.ERROR: self.red,
        }[state]
