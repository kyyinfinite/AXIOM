from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from eurus import paths

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MAX_SKILL_BYTES = 128 * 1024


class SkillError(ValueError):
    """Raised when a skill has invalid metadata or cannot be read safely."""


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    content: str
    source: str
    path: Path


def _metadata(text: str, path: Path) -> tuple[str, str, str]:
    if not text.startswith("---\n"):
        raise SkillError(f"{path}: SKILL.md must start with YAML frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise SkillError(f"{path}: unterminated YAML frontmatter")
    header, body = text[4:end], text[end + 5 :]
    fields: dict[str, str] = {}
    for line in header.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise SkillError(f"{path}: invalid metadata line")
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        if key not in {"name", "description"}:
            continue
        if key in fields:
            raise SkillError(f"{path}: duplicate metadata field {key!r}")
        fields[key] = value
    name, description = fields.get("name", ""), fields.get("description", "")
    if not _NAME.fullmatch(name):
        raise SkillError(f"{path}: invalid or missing skill name")
    if not description:
        raise SkillError(f"{path}: missing skill description")
    if not body.strip():
        raise SkillError(f"{path}: skill body is empty")
    return name, description, body.strip()


def _candidate_roots(workspace: Path | None) -> list[tuple[Path, str]]:
    roots: list[tuple[Path, str]] = []
    builtin = Path(__file__).parent / "builtin"
    roots.append((builtin, "builtin"))
    roots.append((paths.home_dir() / "skills", "user"))
    if workspace is not None:
        roots.append((workspace / ".eurus" / "skills", "workspace"))
        roots.append((workspace / "skills", "workspace"))
    return roots


def _read_skill(path: Path, source: str) -> Skill:
    try:
        if path.stat().st_size > _MAX_SKILL_BYTES:
            raise SkillError(f"{path}: skill exceeds {_MAX_SKILL_BYTES} bytes")
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillError(f"cannot read skill {path}: {exc}") from exc
    name, description, content = _metadata(text, path)
    return Skill(name=name, description=description, content=content, source=source, path=path)


def list_skills(workspace: Path | None = None) -> tuple[Skill, ...]:
    """Return discovered skills; workspace definitions override user and built-in names."""
    found: dict[str, Skill] = {}
    for root, source in _candidate_roots(workspace):
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/SKILL.md")):
            try:
                skill = _read_skill(path, source)
            except SkillError:
                # Invalid skills are ignored by discovery and reported when explicitly loaded.
                continue
            found[skill.name] = skill
    return tuple(found[name] for name in sorted(found))


def load_skill(name: str, workspace: Path | None = None) -> Skill:
    """Load one skill by name, preferring workspace, then user, then built-in."""
    if not _NAME.fullmatch(name):
        raise SkillError(f"invalid skill name: {name!r}")
    roots = list(reversed(_candidate_roots(workspace)))
    for root, source in roots:
        path = root / name / "SKILL.md"
        if path.is_file():
            skill = _read_skill(path, source)
            if skill.name != name:
                raise SkillError(f"skill directory {name!r} contains skill named {skill.name!r}")
            return skill
    raise SkillError(f"skill not found: {name}")


def resolve_skills(names: Iterable[str], workspace: Path | None = None) -> tuple[Skill, ...]:
    """Resolve configured skill names in order; duplicate names are included only once."""
    result: list[Skill] = []
    seen: set[str] = set()
    for name in names:
        clean = name.strip()
        if not clean or clean in seen:
            continue
        result.append(load_skill(clean, workspace))
        seen.add(clean)
    return tuple(result)
