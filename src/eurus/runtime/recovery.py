from __future__ import annotations

from enum import StrEnum
from typing import Any

from eurus.runtime.models import TERMINAL_STATUSES, Run, RunStatus


class RecoveryAction(StrEnum):
    START_FRESH = "start_fresh"
    RESUME_CHECKPOINT = "resume_checkpoint"
    RECONCILE_UNKNOWN_OUTCOME = "reconcile_unknown_outcome"
    TERMINAL = "terminal"
    MANUAL_REVIEW = "manual_review"


def classify_recovery(run: Run, checkpoint: dict[str, Any] | None) -> RecoveryAction:
    if run.status in TERMINAL_STATUSES:
        return RecoveryAction.TERMINAL
    if run.status == RunStatus.QUEUED:
        return RecoveryAction.START_FRESH
    if checkpoint is None:
        return RecoveryAction.MANUAL_REVIEW
    state = checkpoint.get("state", {})
    if not isinstance(state, dict):
        return RecoveryAction.MANUAL_REVIEW
    if state.get("in_flight_side_effect") is True:
        return RecoveryAction.RECONCILE_UNKNOWN_OUTCOME
    side_effect_status = state.get("side_effect_status")
    if isinstance(side_effect_status, str) and side_effect_status in {
        "started",
        "unknown",
        "pending",
    }:
        return RecoveryAction.RECONCILE_UNKNOWN_OUTCOME
    if state.get("tool_execution") == "in_flight":
        return RecoveryAction.RECONCILE_UNKNOWN_OUTCOME
    if run.status in {RunStatus.RUNNING, RunStatus.INTERRUPTED, RunStatus.RETRY_WAIT}:
        return RecoveryAction.RESUME_CHECKPOINT
    return RecoveryAction.MANUAL_REVIEW
