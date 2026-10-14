from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, cast

from axiom.agent.planner import TaskPlan

StepStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]
TaskStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


@dataclass(slots=True)
class StepState:
    number: int
    title: str
    success_criteria: str
    status: StepStatus = "pending"
    attempts: int = 0
    output: str = ""
    error: str = ""
    verification: str = ""


@dataclass(slots=True)
class TaskState:
    goal: str
    steps: list[StepState]
    status: TaskStatus = "pending"
    current_step: int | None = None
    summary: str = ""

    @classmethod
    def from_plan(cls, plan: TaskPlan) -> TaskState:
        return cls(
            goal=plan.goal,
            steps=[
                StepState(step.number, step.title, step.success_criteria) for step in plan.steps
            ],
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot suitable for persistence."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> TaskState:
        """Restore a task snapshot while validating its basic structure."""
        goal = value.get("goal")
        raw_steps = value.get("steps")
        status = value.get("status", "pending")
        current_step = value.get("current_step")
        summary = value.get("summary", "")
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("task state goal must be a non-empty string")
        if not isinstance(raw_steps, list):
            raise ValueError("task state steps must be a list")
        valid_task_statuses = {"pending", "running", "completed", "failed", "cancelled"}
        if not isinstance(status, str) or status not in valid_task_statuses:
            raise ValueError("task state has an invalid status")
        if current_step is not None and (not isinstance(current_step, int) or current_step < 1):
            raise ValueError("task state current_step must be a positive integer or null")
        if not isinstance(summary, str):
            raise ValueError("task state summary must be a string")

        steps: list[StepState] = []
        valid_step_statuses = {"pending", "running", "succeeded", "failed", "cancelled"}
        for raw in raw_steps:
            if not isinstance(raw, dict):
                raise ValueError("each task step must be an object")
            number = raw.get("number")
            title = raw.get("title")
            criteria = raw.get("success_criteria")
            step_status = raw.get("status", "pending")
            attempts = raw.get("attempts", 0)
            output = raw.get("output", "")
            error = raw.get("error", "")
            verification = raw.get("verification", "")
            if not isinstance(number, int) or number < 1:
                raise ValueError("step number must be a positive integer")
            if not isinstance(title, str) or not isinstance(criteria, str):
                raise ValueError("step title and success_criteria must be strings")
            if not isinstance(step_status, str) or step_status not in valid_step_statuses:
                raise ValueError("task step has an invalid status")
            if not isinstance(attempts, int) or attempts < 0:
                raise ValueError("step attempts must be a non-negative integer")
            if not all(isinstance(item, str) for item in (output, error, verification)):
                raise ValueError("step output, error and verification must be strings")
            steps.append(
                StepState(
                    number=number,
                    title=title,
                    success_criteria=criteria,
                    status=cast(StepStatus, step_status),
                    attempts=attempts,
                    output=output,
                    error=error,
                    verification=verification,
                )
            )
        if len({step.number for step in steps}) != len(steps):
            raise ValueError("task step numbers must be unique")
        return cls(
            goal=goal,
            steps=steps,
            status=cast(TaskStatus, status),
            current_step=current_step,
            summary=summary,
        )
