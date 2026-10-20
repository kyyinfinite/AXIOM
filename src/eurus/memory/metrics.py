from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    query_count: int
    returned_count: int
    relevant_count: int
    precision_at_k: float | None
    recall_at_k: float | None
    scope_violations: int

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def evaluate_retrieval(
    results: list[dict[str, Any]],
    relevant_ids: set[int | str],
    *,
    requested_scope: str | None,
    query_count: int = 1,
) -> RetrievalMetrics:
    if query_count < 1:
        raise ValueError("query_count must be at least one")
    returned = len(results)
    relevant = sum(str(row.get("id")) in {str(value) for value in relevant_ids} for row in results)
    violations = sum(
        requested_scope is not None
        and row.get("scope", "global") not in {"global", requested_scope}
        for row in results
    )
    return RetrievalMetrics(
        query_count=query_count,
        returned_count=returned,
        relevant_count=relevant,
        precision_at_k=relevant / returned if returned else None,
        recall_at_k=relevant / len(relevant_ids) if relevant_ids else None,
        scope_violations=violations,
    )
