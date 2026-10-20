from datetime import datetime

import pytest

from eurus.runtime.budgets import BudgetSpec, BudgetTracker
from eurus.runtime.models import Run, RunStatus, RuntimeEvent
from eurus.runtime.state_machine import InvalidRunTransition, transition_run


def test_run_lifecycle_and_terminal_guard():
    run = Run(owner_id="test", objective="check")
    transition_run(run, RunStatus.RUNNING)
    assert run.started_at is not None
    assert run.status_version == 1
    transition_run(run, RunStatus.VERIFYING)
    transition_run(run, RunStatus.COMPLETED, stop_reason="stop")
    assert run.finished_at is not None
    with pytest.raises(InvalidRunTransition):
        transition_run(run, RunStatus.RUNNING)


def test_budget_tracker_enforces_limits():
    tracker = BudgetTracker(BudgetSpec(max_iterations=2, max_tool_calls=1))
    assert tracker.consume_iteration()
    assert tracker.consume_iteration()
    assert not tracker.consume_iteration()
    assert tracker.consume_tool_call()
    assert not tracker.consume_tool_call()


def test_budget_rejects_naive_deadline():
    with pytest.raises(ValueError):
        BudgetSpec(deadline_at=datetime(2026, 1, 1))


def test_runtime_event_requires_positive_sequence():
    with pytest.raises(ValueError):
        RuntimeEvent(run_id="r", event_type="run.started", sequence_no=0)


def test_budget_tracker_reserves_tool_batch_atomically():
    tracker = BudgetTracker(BudgetSpec(max_tool_calls=2))
    assert tracker.consume_tool_calls(2)
    assert not tracker.consume_tool_calls(1)
    assert tracker.tool_calls_used == 2
