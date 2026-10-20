from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    max_iterations: int = 500
    max_runtime_seconds: float = 1200.0
    max_tool_calls: int = 1000
    max_parallel_tools: int = 4
    max_child_runs: int = 0
    max_delegation_depth: int = 0
    max_tokens: int | None = None
    max_cost: float | None = None
    deadline_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_iterations",
            "max_tool_calls",
            "max_parallel_tools",
            "max_child_runs",
            "max_delegation_depth",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.max_iterations < 1 or self.max_parallel_tools < 1:
            raise ValueError("max_iterations and max_parallel_tools must be at least 1")
        if self.max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be positive")
        if self.max_tokens is not None and self.max_tokens < 0:
            raise ValueError("max_tokens cannot be negative")
        if self.max_cost is not None and self.max_cost < 0:
            raise ValueError("max_cost cannot be negative")
        if self.deadline_at is not None and self.deadline_at.tzinfo is None:
            raise ValueError("deadline_at must be timezone-aware")


@dataclass(slots=True)
class BudgetTracker:
    spec: BudgetSpec
    iterations_used: int = 0
    tool_calls_used: int = 0
    tokens_used: int = 0
    cost_used: float = 0.0
    started_at: datetime | None = None

    def start(self) -> None:
        self.started_at = datetime.now(UTC)

    def consume_iteration(self) -> bool:
        if self.iterations_used >= self.spec.max_iterations or self.exhausted:
            return False
        self.iterations_used += 1
        return True

    def consume_tool_call(self) -> bool:
        return self.consume_tool_calls(1)

    def consume_tool_calls(self, count: int) -> bool:
        if count < 0:
            raise ValueError("tool call count cannot be negative")
        if self.exhausted or self.tool_calls_used + count > self.spec.max_tool_calls:
            return False
        self.tool_calls_used += count
        return True

    def can_continue(self) -> bool:
        return not self.exhausted

    def add_usage(self, *, tokens: int = 0, cost: float = 0.0) -> None:
        if tokens < 0 or cost < 0:
            raise ValueError("usage cannot be negative")
        self.tokens_used += tokens
        self.cost_used += cost

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, (datetime.now(UTC) - self.started_at).total_seconds())

    @property
    def exhausted(self) -> bool:
        if self.started_at is not None and self.elapsed_seconds >= self.spec.max_runtime_seconds:
            return True
        if self.spec.deadline_at is not None and datetime.now(UTC) >= self.spec.deadline_at:
            return True
        if self.spec.max_tokens is not None and self.tokens_used >= self.spec.max_tokens:
            return True
        return self.spec.max_cost is not None and self.cost_used >= self.spec.max_cost
