from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def utc_now() -> datetime:
    return datetime.now(UTC)


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_INPUT = "waiting_input"
    RETRY_WAIT = "retry_wait"
    VERIFYING = "verifying"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED})


@dataclass(slots=True)
class Run:
    owner_id: str
    objective: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_run_id: str | None = None
    session_id: str | None = None
    status: RunStatus = RunStatus.QUEUED
    stop_reason: str | None = None
    budget: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    schema_version: int = 1
    checkpoint_version: int = 0
    result_ref: str | None = None
    status_version: int = 0

    def __post_init__(self) -> None:
        if not self.owner_id.strip():
            raise ValueError("owner_id must be non-empty")
        if not self.objective.strip():
            raise ValueError("objective must be non-empty")
        if self.schema_version < 1 or self.checkpoint_version < 0 or self.status_version < 0:
            raise ValueError("schema and state versions must be non-negative")


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    run_id: str
    event_type: str
    sequence_no: int
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload_version: int = 1
    created_at: datetime = field(default_factory=utc_now)
    step_id: str | None = None
    attempt_id: str | None = None
    parent_run_id: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id or not self.event_type:
            raise ValueError("run_id and event_type must be non-empty")
        if self.sequence_no < 1 or self.payload_version < 1:
            raise ValueError("sequence_no and payload_version must be positive")
