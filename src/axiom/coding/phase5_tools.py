"""Compatibility wrapper for the multi-file workflow tools.

Prefer importing :func:`workflow_tools` from ``axiom.coding.workflow_tools``.
"""

from axiom.coding.workflow_tools import workflow_tools
from axiom.tools.base import Tool


def phase5_tools() -> list[Tool]:
    """Return the current workflow tools for legacy callers."""
    return workflow_tools()
