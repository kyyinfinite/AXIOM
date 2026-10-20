from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SkillRegressionReport:
    passed: bool
    baseline_revision: str
    candidate_revision: str
    baseline_pass_rate: float | None
    candidate_pass_rate: float | None
    regressions: tuple[str, ...]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "baseline_revision": self.baseline_revision,
            "candidate_revision": self.candidate_revision,
            "baseline_pass_rate": self.baseline_pass_rate,
            "candidate_pass_rate": self.candidate_pass_rate,
            "regressions": list(self.regressions),
            "fingerprint": self.fingerprint,
        }


def evaluate_skill_regression(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    baseline_revision: str,
    candidate_revision: str,
) -> SkillRegressionReport:
    if not baseline_revision.strip() or not candidate_revision.strip():
        raise ValueError("baseline and candidate revisions are required")
    before = {str(row["task"]): bool(row["passed"]) for row in baseline}
    after = {str(row["task"]): bool(row["passed"]) for row in candidate}
    if not before or set(before) != set(after):
        raise ValueError("baseline and candidate must contain the same non-empty task set")
    regressions = tuple(
        sorted(name for name, passed in before.items() if passed and not after[name])
    )
    baseline_rate = sum(before.values()) / len(before)
    candidate_rate = sum(after.values()) / len(after)
    payload = {
        "baseline_revision": baseline_revision,
        "candidate_revision": candidate_revision,
        "baseline": sorted(before.items()),
        "candidate": sorted(after.items()),
        "regressions": regressions,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SkillRegressionReport(
        passed=not regressions and candidate_rate >= baseline_rate,
        baseline_revision=baseline_revision,
        candidate_revision=candidate_revision,
        baseline_pass_rate=baseline_rate,
        candidate_pass_rate=candidate_rate,
        regressions=regressions,
        fingerprint=fingerprint,
    )
