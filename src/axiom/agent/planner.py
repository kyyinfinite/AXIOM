from __future__ import annotations

from dataclasses import dataclass

MAX_PLAN_STEPS = 8


@dataclass(frozen=True, slots=True)
class PlanStep:
    number: int
    title: str
    success_criteria: str


@dataclass(frozen=True, slots=True)
class TaskPlan:
    goal: str
    steps: tuple[PlanStep, ...]


class TaskPlanner:
    """Lightweight deterministic planner; it does not call a second model."""

    def __init__(self, max_steps: int = MAX_PLAN_STEPS) -> None:
        if not 1 <= max_steps <= MAX_PLAN_STEPS:
            raise ValueError(f"max_steps must be between 1 and {MAX_PLAN_STEPS}")
        self.max_steps = max_steps

    def create_plan(self, goal: str) -> TaskPlan:
        cleaned = goal.strip()
        if not cleaned:
            raise ValueError("task goal must not be empty")
        lowered = cleaned.lower()
        if any(word in lowered for word in ("bug", "fix", "refactor", "implement", "code", "test")):
            actions = [
                ("Inspect workspace", "Identify relevant files and project conventions"),
                ("Trace the implementation", "Locate the code and tests related to the goal"),
                ("Propose a minimal change", "Explain the intended patch before modifying files"),
                ("Apply approved change", "Only edit files inside the configured workspace"),
                ("Run focused checks", "Run relevant tests and inspect failures"),
                ("Review the diff", "Confirm only intended files and lines changed"),
                ("Report outcome", "Summarize changes, checks, and remaining risks"),
            ]
        else:
            actions = [
                ("Clarify the goal", "Identify the requested result and constraints"),
                ("Gather context", "Use available tools rather than guessing"),
                ("Perform the task", "Complete the smallest safe set of actions"),
                ("Verify the result", "Check the result against the original goal"),
                ("Report outcome", "Summarize the result and any limitations"),
            ]
        actions = actions[: self.max_steps]
        steps = tuple(
            PlanStep(i, title, criteria) for i, (title, criteria) in enumerate(actions, 1)
        )
        return TaskPlan(cleaned, steps)
