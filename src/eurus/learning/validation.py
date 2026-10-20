from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from eurus.learning.regression import SkillRegressionReport
from eurus.learning.skills import SkillDraftStore, scan_skill


@dataclass(frozen=True, slots=True)
class LearningCandidate:
    name: str
    description: str
    procedure: str
    preconditions: tuple[str, ...]
    expected_outcomes: tuple[str, ...]
    unsafe_steps: tuple[str, ...]
    source_task_id: str
    evidence: tuple[str, ...]
    created_at: str
    sha256: str
    validated: bool = False
    validation_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _candidate_digest(
    name: str,
    description: str,
    procedure: str,
    source_task_id: str,
    evidence: tuple[str, ...],
    preconditions: tuple[str, ...],
    expected_outcomes: tuple[str, ...],
    unsafe_steps: tuple[str, ...],
) -> str:
    rendered = json.dumps(
        {
            "name": name,
            "description": description,
            "procedure": procedure,
            "source": source_task_id,
            "evidence": evidence,
            "preconditions": preconditions,
            "expected_outcomes": expected_outcomes,
            "unsafe_steps": unsafe_steps,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode()).hexdigest()


def create_candidate(
    *,
    name: str,
    description: str,
    procedure: str,
    source_task_id: str,
    evidence: tuple[str, ...],
    preconditions: tuple[str, ...] = (),
    expected_outcomes: tuple[str, ...] = (),
    unsafe_steps: tuple[str, ...] = (),
) -> LearningCandidate:
    if not source_task_id.strip() or len(source_task_id) > 200:
        raise ValueError("source_task_id must be non-empty and at most 200 characters")
    if not evidence or any(not item.strip() for item in evidence):
        raise ValueError("candidate requires non-empty source evidence")
    if not procedure.strip() or len(procedure.encode("utf-8")) > 128 * 1024:
        raise ValueError("procedure is empty or too large")
    scan = scan_skill(name, description, procedure)
    if scan.findings:
        raise ValueError("candidate failed safety scan: " + ", ".join(scan.findings))
    if not expected_outcomes:
        raise ValueError("candidate requires expected outcomes")
    clean_preconditions = tuple(item.strip() for item in preconditions if item.strip())
    clean_outcomes = tuple(item.strip() for item in expected_outcomes if item.strip())
    clean_unsafe = tuple(item.strip() for item in unsafe_steps if item.strip())
    clean_evidence = tuple(item.strip()[:500] for item in evidence if item.strip())
    digest = _candidate_digest(
        name,
        description,
        procedure,
        source_task_id,
        clean_evidence,
        clean_preconditions,
        clean_outcomes,
        clean_unsafe,
    )
    return LearningCandidate(
        name,
        description,
        procedure,
        clean_preconditions,
        clean_outcomes,
        clean_unsafe,
        source_task_id,
        clean_evidence,
        datetime.now(UTC).isoformat(),
        digest,
    )


def validate_candidate(
    candidate: LearningCandidate, *, checks: tuple[str, ...]
) -> LearningCandidate:
    notes: list[str] = []
    if not checks:
        notes.append("at least one explicit validation check is required")
    if not candidate.expected_outcomes:
        notes.append("expected outcomes are missing")
    if not candidate.evidence:
        notes.append("source evidence is missing")
    scan = scan_skill(candidate.name, candidate.description, candidate.procedure)
    notes.extend(scan.findings)
    if candidate.unsafe_steps:
        notes.append("candidate includes steps requiring human safety review")
    if any(not item.strip() for item in checks):
        notes.append("validation checks must be non-empty")
    return replace(
        candidate,
        validated=not notes,
        validation_notes=tuple(notes),
    )


def promote_candidate(
    candidate: LearningCandidate,
    store: SkillDraftStore,
    workspace: Path,
    *,
    approved: bool = False,
    regression_report: SkillRegressionReport | None = None,
) -> Path:
    if not approved:
        raise PermissionError("owner approval is required to promote a learning candidate")
    if regression_report is not None and not regression_report.passed:
        details = ", ".join(regression_report.regressions) or "pass rate decreased"
        raise ValueError("skill regression gate failed: " + details)
    if not candidate.validated or candidate.validation_notes:
        raise ValueError("learning candidate must pass validation before promotion")
    current = store.draft(candidate.name, candidate.description, candidate.procedure)
    if current.findings:
        raise ValueError("candidate no longer passes skill safety scan")
    expected = _candidate_digest(
        candidate.name,
        candidate.description,
        candidate.procedure,
        candidate.source_task_id,
        candidate.evidence,
        candidate.preconditions,
        candidate.expected_outcomes,
        candidate.unsafe_steps,
    )
    if candidate.sha256 != expected:
        raise ValueError("candidate provenance hash mismatch")
    return store.promote(candidate.name, workspace)
