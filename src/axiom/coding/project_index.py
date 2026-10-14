from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
}
MAX_INDEX_FILE_BYTES = 1_000_000


@dataclass(frozen=True, slots=True)
class ModuleInfo:
    path: str
    imports: tuple[str, ...]
    definitions: tuple[str, ...]
    parse_error: str = ""


def _inside(root: Path, path: Path) -> bool:
    return path == root or root in path.parents


def index_project(workspace: Path, *, max_files: int = 2_000) -> list[ModuleInfo]:
    """Create a lightweight, stdlib-only index of Python files in a workspace."""
    root = workspace.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("workspace must be an existing directory")
    if not 1 <= max_files <= 20_000:
        raise ValueError("max_files must be between 1 and 20000")

    candidates: list[Path] = []
    for path in root.rglob("*.py"):
        try:
            relative_path = path.relative_to(root)
            if any(part in IGNORED_DIRS for part in relative_path.parts):
                continue
            if path.is_symlink() or not path.is_file():
                continue
            resolved = path.resolve()
            if not _inside(root, resolved) or path.stat().st_size > MAX_INDEX_FILE_BYTES:
                continue
            candidates.append(path)
        except (OSError, ValueError):
            continue
        if len(candidates) >= max_files:
            break

    results: list[ModuleInfo] = []
    for path in sorted(candidates):
        relative_name = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=relative_name)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            results.append(ModuleInfo(relative_name, (), (), f"{type(exc).__name__}: {exc}"))
            continue

        imports: set[str] = set()
        definitions: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions.append(node.name)
        results.append(
            ModuleInfo(
                path=relative_name,
                imports=tuple(sorted(imports)),
                definitions=tuple(sorted(set(definitions))),
            )
        )
    return results
