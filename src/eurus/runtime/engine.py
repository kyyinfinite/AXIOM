from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from eurus.agent.events import AgentEvent
from eurus.agent.loop import Agent
from eurus.runtime.budgets import BudgetSpec, BudgetTracker
from eurus.runtime.cancellation import CancellationToken
from eurus.runtime.models import Run, RunStatus, RuntimeEvent
from eurus.runtime.state_machine import transition_run
from eurus.storage.events import EventStore
from eurus.storage.runs import RunStore

Verifier = Callable[[str, str], Awaitable[bool]]


def _current_status(run: Run) -> RunStatus:
    return run.status


@dataclass(frozen=True, slots=True)
class RunResult:
    run: Run
    events: tuple[AgentEvent, ...]
    output: str
    verified: bool | None = None
    runtime_events: tuple[RuntimeEvent, ...] = ()


class AgentEngine:
    def __init__(
        self,
        agent: Agent,
        *,
        verifier: Verifier | None = None,
        run_store: RunStore | None = None,
        event_store: EventStore | None = None,
    ) -> None:
        self.agent = agent
        self.verifier = verifier
        if run_store is None and event_store is not None:
            run_store = RunStore(event_store.database)
        self.run_store = run_store
        self.event_store = event_store
        self._sequences: dict[str, int] = {}

    def _event(
        self, run: Run, event_type: str, payload: dict[str, object] | None = None
    ) -> RuntimeEvent:
        previous_sequence = self._sequences.get(run.run_id, 0)
        if self.event_store is not None:
            previous_sequence = max(previous_sequence, self.event_store.last_sequence(run.run_id))
        sequence = previous_sequence + 1
        self._sequences[run.run_id] = sequence
        return RuntimeEvent(
            run_id=run.run_id,
            event_type=event_type,
            sequence_no=sequence,
            payload=payload or {},
            parent_run_id=run.parent_run_id,
        )

    async def run(
        self,
        objective: str,
        *,
        owner_id: str = "local",
        run: Run | None = None,
        budget: BudgetSpec | None = None,
        cancellation: CancellationToken | None = None,
        success_criteria: str | None = None,
    ) -> RunResult:
        spec = budget or BudgetSpec()
        if spec.max_cost is not None:
            raise ValueError(
                "max_cost cannot be enforced because the provider usage contract "
                "does not report cost"
            )
        runtime_run = run or Run(owner_id=owner_id, objective=objective, budget=asdict(spec))
        if self.run_store is not None:
            stored_run = self.run_store.get(runtime_run.run_id)
            if stored_run is None:
                self.run_store.create(runtime_run)
            else:
                if stored_run.status != RunStatus.QUEUED:
                    raise ValueError("AgentEngine only starts queued runs")
                if stored_run.status_version != runtime_run.status_version:
                    raise ValueError("run state is stale compared with durable storage")
                if (
                    stored_run.owner_id != runtime_run.owner_id
                    or stored_run.objective != runtime_run.objective
                ):
                    raise ValueError("run identity differs from durable storage")
        if runtime_run.status != RunStatus.QUEUED:
            raise ValueError("AgentEngine only starts queued runs")
        tracker = BudgetTracker(spec)
        tracker.start()
        agent_events: list[AgentEvent] = []
        runtime_events: list[RuntimeEvent] = []
        output_parts: list[str] = []

        def persist_run() -> None:
            if self.run_store is not None:
                self.run_store.save(
                    runtime_run,
                    expected_status_version=runtime_run.status_version - 1,
                )

        def record_event(event_type: str, payload: dict[str, object] | None = None) -> None:
            event = self._event(runtime_run, event_type, payload)
            if self.event_store is not None:
                self.event_store.append(event)
            runtime_events.append(event)

        verified: bool | None = None
        cancellation_watcher: asyncio.Task[None] | None = None
        control_watcher: asyncio.Task[None] | None = None
        cancellation_token = cancellation
        if cancellation_token is None and self.run_store is not None:
            cancellation_token = CancellationToken()
        transition_run(runtime_run, RunStatus.RUNNING)
        persist_run()
        record_event("run.started", {"objective": runtime_run.objective})
        if cancellation_token is not None and cancellation_token.cancelled:
            transition_run(runtime_run, RunStatus.CANCELLED, stop_reason="cancelled_before_start")
            persist_run()
            record_event("run.cancelled")
            return RunResult(runtime_run, (), "", None, tuple(runtime_events))
        if cancellation_token is not None:
            cancellation_watcher = asyncio.create_task(self._watch_cancellation(cancellation_token))
        if self.run_store is not None and cancellation_token is not None:
            control_watcher = asyncio.create_task(
                self._watch_persisted_cancellation(runtime_run.run_id, cancellation_token)
            )
        previous_parallel = self.agent.max_parallel_tools
        previous_limit = self.agent.iteration_budget.limit
        previous_tracker: BudgetTracker | None = getattr(self.agent, "runtime_budget_tracker", None)
        self.agent.max_parallel_tools = spec.max_parallel_tools
        self.agent.iteration_budget.limit = min(previous_limit, spec.max_iterations)
        self.agent.runtime_budget_tracker = tracker
        try:
            async with asyncio.timeout(self._effective_timeout(spec)):
                async for event in self.agent.run(runtime_run.objective):
                    agent_events.append(event)
                    if event.kind == "text":
                        output_parts.append(event.text)
                    if event.kind == "done":
                        if cancellation_token is not None and cancellation_token.cancelled:
                            transition_run(
                                runtime_run,
                                RunStatus.CANCELLED,
                                stop_reason="operator_cancelled",
                            )
                            persist_run()
                        elif event.stop_reason in {"interrupted", "timeout"}:
                            target = (
                                RunStatus.INTERRUPTED
                                if event.stop_reason == "interrupted"
                                else RunStatus.FAILED
                            )
                            transition_run(runtime_run, target, stop_reason=event.stop_reason)
                            persist_run()
                        elif event.stop_reason in {
                            "iteration_budget",
                            "max_steps",
                            "tool_budget",
                            "budget_exhausted",
                        }:
                            reason = (
                                "runtime_budget_exhausted"
                                if event.stop_reason == "budget_exhausted"
                                else event.stop_reason
                            )
                            transition_run(runtime_run, RunStatus.FAILED, stop_reason=reason)
                            persist_run()
                            record_event("budget.exhausted", {"stop_reason": reason})
                        else:
                            transition_run(runtime_run, RunStatus.VERIFYING)
                            persist_run()
                            verified = True
                            if self.verifier is not None:
                                verified = await self.verifier(
                                    success_criteria or runtime_run.objective,
                                    "".join(output_parts),
                                )
                            if verified:
                                transition_run(
                                    runtime_run, RunStatus.COMPLETED, stop_reason=event.stop_reason
                                )
                            else:
                                transition_run(
                                    runtime_run, RunStatus.FAILED, stop_reason="verification_failed"
                                )
                            persist_run()
        except TimeoutError:
            if runtime_run.status in {RunStatus.RUNNING, RunStatus.VERIFYING}:
                transition_run(
                    runtime_run, RunStatus.FAILED, stop_reason="runtime_budget_exhausted"
                )
                persist_run()
                record_event("budget.exhausted", {"stop_reason": "runtime_budget_exhausted"})
        except asyncio.CancelledError:
            if runtime_run.status not in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }:
                transition_run(runtime_run, RunStatus.CANCELLED, stop_reason="cancelled")
                persist_run()
            record_event("run.cancelled")
            raise
        except Exception as exc:
            if runtime_run.status not in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }:
                transition_run(
                    runtime_run, RunStatus.FAILED, stop_reason=f"engine_error:{type(exc).__name__}"
                )
                persist_run()
                record_event("run.failed", {"stop_reason": runtime_run.stop_reason})
            raise
        finally:
            watchers = [
                task for task in (cancellation_watcher, control_watcher) if task is not None
            ]
            for task in watchers:
                task.cancel()
            if watchers:
                await asyncio.gather(*watchers, return_exceptions=True)
            self.agent.max_parallel_tools = previous_parallel
            self.agent.iteration_budget.limit = previous_limit
            self.agent.runtime_budget_tracker = previous_tracker
        if runtime_run.status in {RunStatus.RUNNING, RunStatus.VERIFYING}:
            transition_run(
                runtime_run, RunStatus.FAILED, stop_reason="agent_ended_without_terminal_event"
            )
            persist_run()
        if _current_status(runtime_run) == RunStatus.INTERRUPTED:
            record_event("run.interrupted", {"stop_reason": runtime_run.stop_reason})
        elif _current_status(runtime_run) == RunStatus.COMPLETED:
            record_event("run.completed", {"stop_reason": runtime_run.stop_reason})
        elif _current_status(runtime_run) == RunStatus.FAILED:
            record_event("run.failed", {"stop_reason": runtime_run.stop_reason})
        elif _current_status(runtime_run) == RunStatus.CANCELLED:
            record_event("run.cancelled", {"stop_reason": runtime_run.stop_reason})
        return RunResult(
            runtime_run,
            tuple(agent_events),
            "".join(output_parts),
            verified,
            tuple(runtime_events),
        )

    async def _watch_cancellation(self, cancellation: CancellationToken) -> None:
        await cancellation.wait()
        self.agent.interrupt()

    async def _watch_persisted_cancellation(
        self, run_id: str, cancellation: CancellationToken
    ) -> None:
        if self.run_store is None:
            return
        while not cancellation.cancelled:
            if self.run_store.cancellation_requested(run_id):
                cancellation.cancel()
                return
            await asyncio.sleep(0.2)

    @staticmethod
    def _effective_timeout(spec: BudgetSpec) -> float:
        timeout = spec.max_runtime_seconds
        if spec.deadline_at is not None:
            remaining = (spec.deadline_at - datetime.now(UTC)).total_seconds()
            timeout = min(timeout, max(0.0, remaining))
        return timeout
