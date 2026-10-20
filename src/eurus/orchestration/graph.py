from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskNode:
    node_id: str
    objective: str
    dependencies: tuple[str, ...] = ()
    required: bool = True
    allowed_tools: frozenset[str] = frozenset()
    read_scope: tuple[str, ...] = ()
    write_scope: tuple[str, ...] = ()
    budget: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 300.0
    output_contract: str = "summary"

    def __post_init__(self) -> None:
        if not self.node_id.strip() or not self.objective.strip():
            raise ValueError("node_id and objective must be non-empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.node_id in self.dependencies:
            raise ValueError("a node cannot depend on itself")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("dependencies must be unique")


@dataclass(frozen=True, slots=True)
class TaskGraph:
    nodes: tuple[TaskNode, ...]

    def __post_init__(self) -> None:
        ids = [node.node_id for node in self.nodes]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("graph must contain uniquely identified nodes")
        known = set(ids)
        for node in self.nodes:
            missing = set(node.dependencies) - known
            if missing:
                raise ValueError(
                    f"node {node.node_id!r} has unknown dependencies: {sorted(missing)}"
                )
        self.topological_order()

    def topological_order(self) -> tuple[str, ...]:
        by_id = {node.node_id: node for node in self.nodes}
        pending = {node.node_id: set(node.dependencies) for node in self.nodes}
        ordered: list[str] = []
        while pending:
            ready = [node_id for node_id, deps in pending.items() if not deps]
            if not ready:
                raise ValueError("task graph contains a dependency cycle")
            for node_id in ready:
                ordered.append(node_id)
                pending.pop(node_id)
                for deps in pending.values():
                    deps.discard(node_id)
        if set(ordered) != set(by_id):
            raise ValueError("task graph is invalid")
        return tuple(ordered)

    def by_id(self) -> dict[str, TaskNode]:
        return {node.node_id: node for node in self.nodes}
