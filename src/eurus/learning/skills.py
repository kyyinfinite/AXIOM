from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from eurus.models import Message
from eurus.paths import ensure_private_dir, home_dir
from eurus.providers.router import Router

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SECRET = re.compile(
    r"(?:sk-|gsk_|nvapi-|AIza|ghp_|github_pat_|xox[baprs]-)[\w-]{10,}|\b[A-Za-z0-9_-]{40,}\b",
    re.I,
)
_DANGEROUS = re.compile(
    r"(?:rm\s+-rf\s+(?:/|~|\$HOME)"
    r"|curl[^\n|]*\|\s*(?:sudo\s+)?(?:ba)?sh\b"
    r"|wget[^\n|]*\|\s*(?:sudo\s+)?(?:ba)?sh\b"
    r"|chmod\s+-R\s+777"
    r"|sudo\s+\w+"
    r"|powershell[^\n]*-enc(?:odedcommand)?\b"
    r"|base64\s+-d[^\n|]*\|\s*sh\b)",
    re.I,
)
_POLICY_OVERRIDE = re.compile(
    r"(?:ignore|disregard|override)\s+(?:all\s+)?(?:previous|prior|system|safety|security)"
    r"(?:\s+\w{1,20}){0,2}\s+instructions"
    r"|(?:bypass|disable|skip)\s+(?:approval|permission|safety|security|policy)"
    r"|reveal\s+(?:the\s+)?(?:system prompt|secrets|credentials)",
    re.I,
)


@dataclass(frozen=True, slots=True)
class SkillScan:
    safe: bool
    findings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SkillDraft:
    name: str
    description: str
    content: str
    path: Path
    sha256: str
    findings: tuple[str, ...]


def scan_skill(name: str, description: str, content: str) -> SkillScan:
    findings: list[str] = []
    if not _NAME.fullmatch(name):
        findings.append("invalid skill name")
    if not description.strip() or len(description) > 300:
        findings.append("invalid skill description")
    if not content.strip() or len(content.encode("utf-8")) > 128 * 1024:
        findings.append("empty or oversized skill content")
    if _SECRET.search(description + "\n" + content):
        findings.append("possible secret or credential")
    if _DANGEROUS.search(content):
        findings.append("dangerous command pattern")
    if _POLICY_OVERRIDE.search(content):
        findings.append("attempt to override safety or approval policy")
    return SkillScan(not findings, tuple(dict.fromkeys(findings)))


def render_skill(name: str, description: str, content: str) -> str:
    return f"---\nname: {name}\ndescription: {description.strip()}\n---\n\n{content.strip()}\n"


class SkillDraftStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or home_dir() / "skills" / "quarantine"

    def draft(self, name: str, description: str, content: str) -> SkillDraft:
        scan = scan_skill(name, description, content)
        quarantine = ensure_private_dir(self.root)
        path = quarantine / f"{name if _NAME.fullmatch(name) else 'invalid'}.SKILL.md"
        rendered = render_skill(name, description, content)
        path.write_text(rendered, encoding="utf-8")
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        return SkillDraft(name, description, content, path, digest, scan.findings)

    def read(self, name: str) -> SkillDraft:
        if not _NAME.fullmatch(name):
            raise ValueError("invalid skill name")
        path = self.root / f"{name}.SKILL.md"
        rendered = path.read_text(encoding="utf-8")
        match = re.match(
            r"\A---\nname: ([a-z0-9_-]+)\ndescription: (.*?)\n---\n\n(.*)\Z", rendered, re.S
        )
        if not match:
            raise ValueError("quarantined skill metadata is malformed")
        actual_name, description, content = match.groups()
        if actual_name != name:
            raise ValueError("quarantined skill name mismatch")
        scan = scan_skill(actual_name, description, content)
        return SkillDraft(
            actual_name,
            description,
            content,
            path,
            hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            scan.findings,
        )

    def promote(self, name: str, workspace: Path) -> Path:
        draft = self.read(name)
        if draft.findings:
            raise ValueError("skill failed quarantine scan: " + ", ".join(draft.findings))
        target_root = (workspace / ".eurus" / "skills").resolve()
        target = target_root / name / "SKILL.md"
        if not target.resolve().is_relative_to(target_root):
            raise ValueError("skill destination escapes workspace")
        if target.exists():
            raise ValueError("skill already exists; use skill_patch to update it")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            render_skill(draft.name, draft.description, draft.content), encoding="utf-8"
        )
        return target


INSTRUCTIONS = (
    "Create a reusable Eurus skill from a completed task. "
    "Return one JSON object with name, description, and content. "
    "The skill must contain only general repeatable workflow guidance, not a transcript, "
    "secrets, personal data, or project-specific credentials. "
    "Never include commands that destroy broad paths, execute downloaded scripts, "
    "bypass approval, override safety rules, or expose secrets. "
    "Keep content concise and actionable. "
    "If no reusable skill exists, return {}."
)


def _transcript(task: str, messages: Sequence[Message]) -> str:
    lines = [f"Task: {task[:600]}"]
    for message in messages:
        if message.role == "assistant" and message.content.strip():
            lines.append(f"Assistant: {message.content.strip()[:350]}")
        elif message.role == "tool":
            lines.append(f"Tool {message.name or 'unknown'}: {message.content.strip()[:250]}")
        if len(lines) >= 18:
            break
    return "\n".join(lines)[:8000]


class SkillLearner:
    def __init__(
        self, router: Router, store: SkillDraftStore | None = None, *, min_tool_calls: int = 4
    ) -> None:
        self.router = router
        self.store = store or SkillDraftStore()
        self.min_tool_calls = min_tool_calls

    async def draft(self, task: str, transcript: Sequence[Message]) -> SkillDraft | None:
        calls = sum(len(message.tool_calls) for message in transcript)
        if calls < self.min_tool_calls:
            return None
        response = await self.router.chat(
            [Message.system(INSTRUCTIONS), Message.user(_transcript(task, transcript))],
            tools=(),
            temperature=0,
            max_tokens=1200,
        )
        text = response.message.content.strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict) or not {"name", "description", "content"}.issubset(data):
            return None
        values = (data["name"], data["description"], data["content"])
        if not all(isinstance(value, str) for value in values):
            return None
        name = data["name"].strip().lower()
        description = " ".join(data["description"].split())
        content = data["content"].strip()
        if not _NAME.fullmatch(name) or len(content) < 80:
            return None
        return self.store.draft(name, description[:300], content)
