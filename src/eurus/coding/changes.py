from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from eurus.coding.patchset import AppliedChange, restore_changes


@dataclass(frozen=True, slots=True)
class ChangeReview:
    changed_files: tuple[str, ...]
    can_restore: bool
    notes: tuple[str, ...] = ()


class WorkspaceChangeManager:
    """Track a task's own changes for review and guarded restore."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.expanduser().resolve()
        if not self.workspace.is_dir():
            raise ValueError("workspace must be an existing directory")
        self._changes: list[AppliedChange] = []

    def record(self, changes: tuple[AppliedChange, ...] | list[AppliedChange]) -> None:
        positions = {item.path: index for index, item in enumerate(self._changes)}
        for change in changes:
            if change.path in positions:
                index = positions[change.path]
                original = self._changes[index]
                self._changes[index] = AppliedChange(
                    change.path, original.before_content, change.after_sha256
                )
            else:
                positions[change.path] = len(self._changes)
                self._changes.append(change)

    def review(self) -> ChangeReview:
        notes: list[str] = []
        can_restore = True
        for item in self._changes:
            target = (self.workspace / item.path).resolve()
            if target != self.workspace and self.workspace not in target.parents:
                notes.append(f"Outside workspace: {item.path}")
                can_restore = False
                continue
            try:
                current = target.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                notes.append(f"Cannot verify: {item.path}")
                can_restore = False
                continue
            digest = hashlib.sha256(current.encode("utf-8")).hexdigest()
            if digest != item.after_sha256:
                notes.append(f"Changed since agent edit: {item.path}")
                can_restore = False
            else:
                notes.append(f"Changed: {item.path}")
        return ChangeReview(tuple(item.path for item in self._changes), can_restore, tuple(notes))

    def restore(self) -> list[str]:
        restored = restore_changes(self.workspace, self._changes)
        self._changes.clear()
        return restored
