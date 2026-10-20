"""Workspace-confined coding tools and multi-file workflow primitives."""

from eurus.coding.changes import ChangeReview, WorkspaceChangeManager
from eurus.coding.patchset import (
    AppliedChange,
    FilePatch,
    PatchError,
    PatchResult,
    apply_patchset,
    preview_patchset,
    restore_changes,
)
from eurus.coding.project_index import ModuleInfo, index_project
from eurus.coding.tools import coding_tools
from eurus.coding.validation import ValidationResult, run_validation, validate_command
from eurus.coding.workflow_tools import workflow_tools

__all__ = [
    "AppliedChange",
    "ChangeReview",
    "FilePatch",
    "ModuleInfo",
    "PatchError",
    "PatchResult",
    "ValidationResult",
    "WorkspaceChangeManager",
    "apply_patchset",
    "coding_tools",
    "index_project",
    "workflow_tools",
    "preview_patchset",
    "restore_changes",
    "run_validation",
    "validate_command",
]
