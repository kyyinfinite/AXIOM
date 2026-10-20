from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from eurus.orchestration import (
    OrchestrationError,
    SubagentOrchestrator,
    TaskGraph,
    TaskNode,
    aggregate_results,
)
from eurus.runtime.budgets import BudgetSpec
from eurus.runtime.models import Run, RunStatus
from eurus.runtime.state_machine import transition_run
from eurus.storage.database import Database
from eurus.storage.runs import RunStore


def _store(tmp_path: Path) -> RunStore:
    return RunStore(Database(tmp_path / "orchestration.sqlite3"))


def _parent() -> Run:
    return Run(owner_id="owner", objective="coordinate work")


def _budget(*, children: int = 8, iterations: int = 100) -> BudgetSpec:
    return BudgetSpec(max_child_runs=children, max_iterations=iterations)


def test_graph_rejects_cycles_and_unknown_dependencies() -> None:
    with pytest.raises(ValueError, match="cycle"):
        TaskGraph((TaskNode("a", "A", ("b",)), TaskNode("b", "B", ("a",))))
    with pytest.raises(ValueError, match="unknown dependencies"):
        TaskGraph((TaskNode("a", "A", ("missing",)),))


@pytest.mark.asyncio
async def test_dag_respects_dependencies_and_persists_child_runs(tmp_path: Path) -> None:
    store = _store(tmp_path)
    parent = _parent()
    store.create(parent)
    graph = TaskGraph(
        (
            TaskNode("first", "first task"),
            TaskNode("second", "second task", ("first",)),
        )
    )
    calls: list[tuple[str, dict[str, dict[str, Any]]]] = []

    async def executor(
        node: TaskNode, previous: dict[str, dict[str, Any]], _token: Any
    ) -> dict[str, Any]:
        calls.append((node.node_id, previous))
        return {"status": "completed", "verified": True, "summary": node.objective}

    result = await SubagentOrchestrator(store).execute(parent, graph, executor, budget=_budget())
    assert result.status == "completed"
    assert [name for name, _ in calls] == ["first", "second"]
    assert "first" in calls[1][1]
    children = [run for run in store.list_runs() if run.parent_run_id == parent.run_id]
    assert len(children) == 2
    assert all(run.status == RunStatus.COMPLETED for run in children)


@pytest.mark.asyncio
async def test_parallel_execution_obeys_concurrency_limit(tmp_path: Path) -> None:
    parent = _parent()
    store = _store(tmp_path)
    store.create(parent)
    orchestrator = SubagentOrchestrator(store, max_concurrency=2)

    async def executor(
        node: TaskNode, _previous: dict[str, dict[str, Any]], _token: Any
    ) -> dict[str, Any]:
        await asyncio.sleep(0.01)
        return {"status": "completed", "verified": True, "summary": node.node_id}

    graph = TaskGraph(tuple(TaskNode(f"n{i}", f"task {i}") for i in range(5)))
    result = await orchestrator.execute(parent, graph, executor, budget=_budget())
    assert result.status == "completed"
    assert orchestrator.limiter.peak_active <= 2
    assert orchestrator.limiter.peak_active == 2


@pytest.mark.asyncio
async def test_child_tool_permission_and_shared_budget_are_enforced(tmp_path: Path) -> None:
    store = _store(tmp_path)
    parent = _parent()
    store.create(parent)
    graph = TaskGraph((TaskNode("a", "A", allowed_tools=frozenset({"write_file"})),))
    orchestrator = SubagentOrchestrator(store, allowed_tools=frozenset({"read_file"}))

    async def executor(
        _node: TaskNode, _previous: dict[str, dict[str, Any]], _token: Any
    ) -> dict[str, Any]:
        return {"status": "completed", "verified": True}

    with pytest.raises(PermissionError, match="allowlist"):
        await orchestrator.execute(parent, graph, executor, budget=_budget())
    with pytest.raises(OrchestrationError, match="child-run budget"):
        await orchestrator.execute(
            parent, TaskGraph((TaskNode("a", "A"),)), executor, budget=_budget(children=0)
        )


@pytest.mark.asyncio
async def test_unverified_child_cannot_complete_required_aggregation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    parent = _parent()
    store.create(parent)

    async def executor(
        _node: TaskNode, _previous: dict[str, dict[str, Any]], _token: Any
    ) -> dict[str, Any]:
        return {"status": "completed", "verified": False, "summary": "unverified"}

    result = await SubagentOrchestrator(store).execute(
        parent, TaskGraph((TaskNode("a", "A"),)), executor, budget=_budget()
    )
    assert result.status == "incomplete"
    assert result.missing_required == ("a",)


def test_aggregation_rejects_missing_required_results() -> None:
    result = aggregate_results({}, required_node_ids={"required"})
    assert result.status == "incomplete"
    assert result.missing_required == ("required",)


def test_recovery_marks_running_children_interrupted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    parent = _parent()
    store.create(parent)
    child = Run(owner_id="owner", objective="child", parent_run_id=parent.run_id)
    store.create(child)
    transition_run(child, RunStatus.RUNNING)
    store.save(child, expected_status_version=0)
    orchestrator = SubagentOrchestrator(store)
    assert orchestrator.recover_orphaned_children(parent.run_id) == [child.run_id]
    recovered = store.get(child.run_id)
    assert recovered is not None
    assert recovered.status == RunStatus.INTERRUPTED
    assert recovered.stop_reason == "orchestrator_restart_recovery"
