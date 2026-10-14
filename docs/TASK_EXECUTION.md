# Task execution and verification

AXIOM exposes a small orchestration API for running a bounded plan one step at a time. It is designed to be composed with existing workspace tools and approval callbacks rather than bypassing them.

## Example

```python
import asyncio

from axiom.agent import (
    TaskExecutionController,
    TaskPlanner,
    VerificationResult,
)


async def execute_step(step, attempt, previous_error):
    # Call your existing agent/tools here. Keep mutations behind AXIOM's
    # configured approval flow. Use previous_error to inform a repair attempt.
    return f"Completed: {step.title}"


async def verify_step(title, criteria, output):
    # Replace this example with real tests or checks appropriate to the task.
    passed = output.startswith("Completed:")
    return VerificationResult(passed, "output shape checked")


async def main():
    plan = TaskPlanner().create_plan("implement a feature and test it")
    state = await TaskExecutionController(max_attempts=3).run(plan, execute_step, verify_step)
    print(state.status, state.summary)
    # Persist state.to_dict() as JSON if resumable state is required.


asyncio.run(main())
```

The example verifier is illustrative only. Production integrations should run relevant tests, static analysis, or other concrete checks before returning `passed=True`.

## Execution guarantees

- Steps execute sequentially; the controller stops at the first step that exhausts its retry budget.
- Attempts per step are bounded (1–5, default 3), and each executor and verifier call has a timeout.
- A failed verification summary or exception is passed to the executor on the next attempt as repair context.
- Cancellation marks the active step and task as cancelled, then propagates cancellation to the caller.
- State snapshots can be serialized with `TaskState.to_dict()` and restored with `TaskState.from_dict()`.
- Restored state is checked against the supplied plan before execution.
- The controller itself does not invoke shell commands, modify files, or grant permissions. The executor must use existing AXIOM tools and approval controls.

## Limitations

This API provides orchestration primitives, not automatic model-driven planning or a durable background job service. The application is responsible for storing snapshots, supplying a meaningful verifier, and wiring execution to tools with the appropriate workspace and approval policy.
