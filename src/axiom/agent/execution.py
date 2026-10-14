from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from axiom.agent.planner import PlanStep, TaskPlan
from axiom.agent.task_state import TaskState
from axiom.agent.verification import StepVerifier, VerificationResult

StepExecutor = Callable[[PlanStep, int, str], Awaitable[str]]


class TaskExecutionController:
    """Run a task plan sequentially with bounded retries and explicit verification.

    The executor is responsible for using AXIOM's existing tools and approval hooks.
    This controller never runs shell commands or modifies files directly.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        step_timeout: float = 120.0,
    ) -> None:
        if not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
        if step_timeout <= 0:
            raise ValueError("step_timeout must be greater than zero")
        self.max_attempts = max_attempts
        self.step_timeout = step_timeout

    async def run(
        self,
        plan: TaskPlan,
        executor: StepExecutor,
        verifier: StepVerifier,
        *,
        state: TaskState | None = None,
    ) -> TaskState:
        """Execute each step in order and stop safely at the first exhausted step."""
        task = state or TaskState.from_plan(plan)
        self._validate_state_matches_plan(task, plan)
        if task.status not in ("pending", "failed"):
            raise ValueError(f"cannot run a task in {task.status!r} state")
        if any(step.status == "succeeded" for step in task.steps) and task.status == "pending":
            raise ValueError("a pending task cannot contain completed steps")

        task.status = "running"
        task.summary = ""
        for plan_step, step_state in zip(plan.steps, task.steps, strict=True):
            if step_state.status == "succeeded":
                continue
            task.current_step = plan_step.number
            step_state.status = "running"
            previous_error = step_state.error
            if step_state.attempts >= self.max_attempts:
                step_state.status = "failed"
                task.status = "failed"
                task.summary = (
                    f"Stopped at step {plan_step.number} ({plan_step.title}) after "
                    f"{step_state.attempts} attempts: retry budget already exhausted"
                )
                return task

            while step_state.attempts < self.max_attempts:
                step_state.attempts += 1
                attempt = step_state.attempts
                try:
                    output = await asyncio.wait_for(
                        executor(plan_step, attempt, previous_error), timeout=self.step_timeout
                    )
                    if not isinstance(output, str):
                        raise TypeError("step executor must return a string")
                    step_state.output = output
                    result = await asyncio.wait_for(
                        verifier(plan_step.title, plan_step.success_criteria, output),
                        timeout=self.step_timeout,
                    )
                    if not isinstance(result, VerificationResult):
                        raise TypeError("step verifier must return VerificationResult")
                    step_state.verification = result.summary
                    if result.passed:
                        step_state.status = "succeeded"
                        step_state.error = ""
                        break
                    previous_error = result.summary or "verification failed"
                    step_state.error = previous_error
                except asyncio.CancelledError:
                    step_state.status = "cancelled"
                    task.status = "cancelled"
                    task.summary = f"Execution cancelled during step {plan_step.number}."
                    raise
                except TimeoutError:
                    previous_error = f"step timed out after {self.step_timeout:g} seconds"
                    step_state.error = previous_error
                except Exception as exc:
                    previous_error = f"{type(exc).__name__}: {exc}"
                    step_state.error = previous_error

                if step_state.attempts < self.max_attempts:
                    continue
                step_state.status = "failed"
                task.status = "failed"
                task.summary = (
                    f"Stopped at step {plan_step.number} ({plan_step.title}) after "
                    f"{step_state.attempts} attempts: {step_state.error}"
                )
                return task

        task.current_step = None
        task.status = "completed"
        task.summary = f"Completed all {len(task.steps)} planned steps."
        return task

    @staticmethod
    def _validate_state_matches_plan(task: TaskState, plan: TaskPlan) -> None:
        expected = [(step.number, step.title, step.success_criteria) for step in plan.steps]
        actual = [(step.number, step.title, step.success_criteria) for step in task.steps]
        if task.goal != plan.goal or actual != expected:
            raise ValueError("task state does not match the supplied plan")
