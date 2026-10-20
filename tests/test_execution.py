from __future__ import annotations

import asyncio

import pytest

from eurus.agent.execution import TaskExecutionController
from eurus.agent.planner import TaskPlan, TaskPlanner
from eurus.agent.task_state import TaskState
from eurus.agent.verification import VerificationResult


def test_controller_runs_steps_sequentially_and_verifies_each_step():
    async def scenario():
        plan = TaskPlan(
            "complete work",
            TaskPlanner(max_steps=2).create_plan("explain a concept").steps[:2],
        )
        calls: list[tuple[int, int, str]] = []
        checks: list[str] = []

        async def execute(step, attempt, previous_error):
            calls.append((step.number, attempt, previous_error))
            return f"output-{step.number}"

        async def verify(title, criteria, output):
            checks.append(output)
            return VerificationResult(True, "verified")

        state = await TaskExecutionController().run(plan, execute, verify)
        assert state.status == "completed"
        assert [step.status for step in state.steps] == ["succeeded", "succeeded"]
        assert calls == [(1, 1, ""), (2, 1, "")]
        assert checks == ["output-1", "output-2"]
        assert state.current_step is None

    asyncio.run(scenario())


def test_controller_retries_failed_verification_with_bounded_attempts():
    async def scenario():
        plan = TaskPlanner(max_steps=1).create_plan("explain a concept")
        calls: list[tuple[int, str]] = []

        async def execute(step, attempt, previous_error):
            calls.append((attempt, previous_error))
            return "candidate"

        async def verify(title, criteria, output):
            if len(calls) < 3:
                return VerificationResult(False, "verification did not pass")
            return VerificationResult(True, "all checks passed")

        state = await TaskExecutionController(max_attempts=3).run(plan, execute, verify)
        assert state.status == "completed"
        assert state.steps[0].attempts == 3
        assert calls[0] == (1, "")
        assert calls[1] == (2, "verification did not pass")
        assert state.steps[0].verification == "all checks passed"

    asyncio.run(scenario())


def test_controller_stops_after_retry_budget_is_exhausted():
    async def scenario():
        plan = TaskPlanner(max_steps=2).create_plan("explain a concept")
        executed: list[int] = []

        async def execute(step, attempt, previous_error):
            executed.append(step.number)
            return "bad output"

        async def verify(title, criteria, output):
            return VerificationResult(False, "not acceptable")

        state = await TaskExecutionController(max_attempts=2).run(plan, execute, verify)
        assert state.status == "failed"
        assert state.steps[0].status == "failed"
        assert state.steps[0].attempts == 2
        assert state.steps[1].status == "pending"
        assert set(executed) == {1}
        assert "after 2 attempts" in state.summary

    asyncio.run(scenario())


def test_controller_handles_timeout_as_retryable_failure():
    async def scenario():
        plan = TaskPlanner(max_steps=1).create_plan("explain a concept")
        calls = 0

        async def execute(step, attempt, previous_error):
            nonlocal calls
            calls += 1
            if calls == 1:
                await asyncio.sleep(0.05)
            return "ready"

        async def verify(title, criteria, output):
            return VerificationResult(True, "ok")

        state = await TaskExecutionController(max_attempts=2, step_timeout=0.01).run(
            plan, execute, verify
        )
        assert state.status == "completed"
        assert state.steps[0].attempts == 2
        assert state.steps[0].verification == "ok"

    asyncio.run(scenario())


def test_task_state_round_trip_and_rejects_invalid_data():
    plan = TaskPlanner(max_steps=2).create_plan("explain a concept")
    state = TaskState.from_plan(plan)
    restored = TaskState.from_dict(state.to_dict())
    assert restored.to_dict() == state.to_dict()

    invalid = state.to_dict()
    invalid["status"] = "mystery"
    with pytest.raises(ValueError):
        TaskState.from_dict(invalid)


def test_controller_rejects_state_for_a_different_plan():
    async def scenario():
        first = TaskPlanner(max_steps=1).create_plan("explain a concept")
        other = TaskPlanner(max_steps=1).create_plan("fix a bug")

        async def execute(step, attempt, previous_error):
            return "output"

        async def verify(title, criteria, output):
            return VerificationResult(True, "ok")

        with pytest.raises(ValueError, match="does not match"):
            await TaskExecutionController().run(
                first, execute, verify, state=TaskState.from_plan(other)
            )

    asyncio.run(scenario())


def test_resumed_state_cannot_exceed_total_attempt_budget():
    async def scenario():
        plan = TaskPlanner(max_steps=1).create_plan("explain a concept")
        state = TaskState.from_plan(plan)
        state.status = "failed"
        state.steps[0].status = "failed"
        state.steps[0].attempts = 2
        state.steps[0].error = "previous failure"

        async def execute(step, attempt, previous_error):
            assert attempt == 3
            return "good output"

        async def verify(title, criteria, output):
            return VerificationResult(True, "verified")

        result = await TaskExecutionController(max_attempts=3).run(
            plan, execute, verify, state=state
        )
        assert result.status == "completed"
        assert result.steps[0].attempts == 3

        exhausted = TaskState.from_plan(plan)
        exhausted.status = "failed"
        exhausted.steps[0].status = "failed"
        exhausted.steps[0].attempts = 3

        async def unexpected_execute(step, attempt, previous_error):
            raise AssertionError("must not execute after retry budget is exhausted")

        stopped = await TaskExecutionController(max_attempts=3).run(
            plan, unexpected_execute, verify, state=exhausted
        )
        assert stopped.status == "failed"
        assert stopped.steps[0].attempts == 3

    asyncio.run(scenario())
