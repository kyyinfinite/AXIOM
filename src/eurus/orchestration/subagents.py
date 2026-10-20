from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from eurus.orchestration.aggregation import AggregationResult, aggregate_results
from eurus.orchestration.concurrency import ConcurrencyLimiter
from eurus.orchestration.graph import TaskGraph, TaskNode
from eurus.runtime.budgets import BudgetSpec
from eurus.runtime.cancellation import CancellationToken
from eurus.runtime.models import Run, RunStatus
from eurus.runtime.state_machine import transition_run
from eurus.storage.runs import RunStore


class OrchestrationError(RuntimeError):
    pass


ChildExecutor = Callable[
    [TaskNode, dict[str, dict[str, Any]], CancellationToken], Awaitable[dict[str, Any]]
]


class SubagentOrchestrator:
    def __init__(
        self,
        run_store: RunStore,
        *,
        max_concurrency: int = 4,
        max_child_runs: int = 16,
        allowed_tools: frozenset[str] = frozenset(),
        allow_nested: bool = False,
    ) -> None:
        if max_child_runs < 1:
            raise ValueError("max_child_runs must be at least 1")
        self.run_store = run_store
        self.max_child_runs = max_child_runs
        self.allowed_tools = allowed_tools
        self.allow_nested = allow_nested
        self.limiter = ConcurrencyLimiter(max_concurrency)

    def list_children(self, parent_run_id: str) -> list[Run]:
        return [
            run
            for run in self.run_store.list_runs(limit=1000)
            if run.parent_run_id == parent_run_id
        ]

    def recover_orphaned_children(self, parent_run_id: str) -> list[str]:
        recovered: list[str] = []
        for child in self.list_children(parent_run_id):
            if child.status == RunStatus.RUNNING:
                transition_run(
                    child, RunStatus.INTERRUPTED, stop_reason="orchestrator_restart_recovery"
                )
                self.run_store.save(child, expected_status_version=child.status_version - 1)
                recovered.append(child.run_id)
        return recovered

    async def execute(
        self,
        parent: Run,
        graph: TaskGraph,
        executor: ChildExecutor,
        *,
        budget: BudgetSpec,
        cancellation: CancellationToken | None = None,
        error_policy: str = "fail_fast",
        verifier: Callable[[AggregationResult], Awaitable[bool]] | None = None,
    ) -> AggregationResult:
        if error_policy not in {"fail_fast", "continue_independent", "retry_bounded"}:
            raise ValueError("unsupported error policy")
        if parent.parent_run_id is not None and not self.allow_nested:
            raise PermissionError("nested delegation is disabled")
        if len(graph.nodes) > min(self.max_child_runs, budget.max_child_runs):
            raise OrchestrationError("child-run budget exceeded")
        for node in graph.nodes:
            if node.allowed_tools and not node.allowed_tools.issubset(self.allowed_tools):
                raise PermissionError(
                    f"node {node.node_id!r} requests tools outside parent allowlist"
                )
        token = cancellation or CancellationToken()
        nodes = graph.by_id()
        remaining = set(nodes)
        results: dict[str, dict[str, Any]] = {}
        child_runs: dict[str, Run] = {}
        failed_required: set[str] = set()
        budget_remaining = budget.max_iterations
        lock = asyncio.Lock()

        async def run_node(node: TaskNode) -> None:
            nonlocal budget_remaining
            token.raise_if_cancelled()
            async with lock:
                requested = int(
                    node.budget.get(
                        "max_iterations", max(1, budget.max_iterations // len(graph.nodes))
                    )
                )
                if requested < 1 or budget_remaining < 1:
                    raise OrchestrationError("shared iteration budget exhausted")
                allocation = min(requested, budget_remaining)
                budget_remaining -= allocation
            child = Run(
                owner_id=parent.owner_id,
                objective=node.objective,
                parent_run_id=parent.run_id,
                session_id=parent.session_id,
                budget={
                    **node.budget,
                    "max_iterations": allocation,
                    "allowed_tools": sorted(node.allowed_tools),
                },
            )
            self.run_store.create(child)
            child_runs[node.node_id] = child
            transition_run(child, RunStatus.RUNNING)
            self.run_store.save(child, expected_status_version=0)
            try:

                async def invoke() -> dict[str, Any]:
                    return await asyncio.wait_for(
                        executor(node, dict(results), token), timeout=node.timeout_seconds
                    )

                raw = await self.limiter.run(invoke)
                status = str(raw.get("status", "completed"))
                verified = bool(raw.get("verified", False))
                if status == "completed" and verified:
                    transition_run(child, RunStatus.VERIFYING)
                    self.run_store.save(child, expected_status_version=child.status_version - 1)
                    transition_run(
                        child,
                        RunStatus.COMPLETED,
                        result_ref=str(raw.get("result_ref", "")) or None,
                    )
                    raw = {**raw, "status": "completed", "verified": True}
                else:
                    transition_run(child, RunStatus.FAILED, stop_reason="child_result_unverified")
                    raw = {**raw, "status": "failed", "verified": False}
                self.run_store.save(child, expected_status_version=child.status_version - 1)
                results[node.node_id] = {
                    "role": str(raw.get("role", node.node_id)),
                    "status": str(raw.get("status", "failed")),
                    "summary": str(raw.get("summary", "")),
                    "evidence_refs": list(raw.get("evidence_refs", [])),
                    "artifact_refs": list(raw.get("artifact_refs", [])),
                    "verified": bool(raw.get("verified", False)),
                    "usage": dict(raw.get("usage", {})),
                    "error_class": raw.get("error_class"),
                    "run_id": child.run_id,
                }
                if results[node.node_id]["status"] != "completed" and node.required:
                    failed_required.add(node.node_id)
            except asyncio.CancelledError:
                if child.status == RunStatus.RUNNING:
                    transition_run(
                        child, RunStatus.CANCELLED, stop_reason="orchestration_cancelled"
                    )
                    self.run_store.save(child, expected_status_version=child.status_version - 1)
                raise
            except Exception as exc:
                if child.status == RunStatus.RUNNING:
                    transition_run(child, RunStatus.FAILED, stop_reason=type(exc).__name__)
                    self.run_store.save(child, expected_status_version=child.status_version - 1)
                results[node.node_id] = {
                    "role": node.node_id,
                    "status": "failed",
                    "summary": "",
                    "evidence_refs": [],
                    "artifact_refs": [],
                    "verified": False,
                    "usage": {},
                    "error_class": type(exc).__name__,
                    "run_id": child.run_id,
                }
                if node.required:
                    failed_required.add(node.node_id)
                if error_policy == "fail_fast":
                    token.cancel()
                    raise

        tasks: dict[str, asyncio.Task[None]] = {}
        try:
            while remaining:
                token.raise_if_cancelled()
                ready = [
                    node_id
                    for node_id in graph.topological_order()
                    if node_id in remaining
                    and all(dep in results for dep in nodes[node_id].dependencies)
                ]
                if not ready:
                    raise OrchestrationError(
                        "no runnable nodes remain; dependencies are unresolved"
                    )
                if failed_required and error_policy == "fail_fast":
                    raise OrchestrationError("required child failed under fail_fast policy")
                batch: list[asyncio.Task[None]] = []
                for node_id in ready[: self.limiter.limit]:
                    node = nodes[node_id]
                    if any(dep in failed_required for dep in node.dependencies):
                        results[node_id] = {
                            "role": node_id,
                            "status": "skipped",
                            "summary": "dependency failed",
                            "evidence_refs": [],
                            "artifact_refs": [],
                            "verified": False,
                            "usage": {},
                            "error_class": "DependencyFailed",
                        }
                        if node.required:
                            failed_required.add(node_id)
                        remaining.remove(node_id)
                        continue
                    task = asyncio.create_task(run_node(node), name=f"eurus-child-{node_id}")
                    tasks[node_id] = task
                    batch.append(task)
                    remaining.remove(node_id)
                if batch:
                    outcomes = await asyncio.gather(*batch, return_exceptions=True)
                    errors = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
                    if errors and error_policy == "fail_fast":
                        raise errors[0]
            aggregate = aggregate_results(
                results, required_node_ids={n.node_id for n in graph.nodes if n.required}
            )
            if (
                aggregate.status == "completed"
                and verifier is not None
                and not await verifier(aggregate)
            ):
                return AggregationResult(
                    "incomplete", "Parent aggregation verification failed", results, ()
                )
            return aggregate
        except BaseException:
            token.cancel()
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            raise
