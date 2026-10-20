from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AggregationResult:
    status: str
    summary: str
    results: dict[str, dict[str, Any]]
    missing_required: tuple[str, ...] = ()


def aggregate_results(
    results: dict[str, dict[str, Any]], *, required_node_ids: set[str]
) -> AggregationResult:
    missing = tuple(
        sorted(
            node_id
            for node_id in required_node_ids
            if node_id not in results
            or results[node_id].get("status") != "completed"
            or not results[node_id].get("verified", False)
        )
    )
    if missing:
        return AggregationResult(
            status="incomplete",
            summary="Required child results are missing or not verified: " + ", ".join(missing),
            results=results,
            missing_required=missing,
        )
    summaries = [
        f"{node_id}: {str(results[node_id].get('summary', '')).strip()}"
        for node_id in sorted(results)
        if str(results[node_id].get("summary", "")).strip()
    ]
    return AggregationResult(status="completed", summary="\n".join(summaries), results=results)
