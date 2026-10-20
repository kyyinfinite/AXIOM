from dataclasses import replace
from pathlib import Path

import pytest

from eurus.learning.skills import SkillDraftStore
from eurus.learning.validation import create_candidate, promote_candidate, validate_candidate


def candidate():
    return create_candidate(
        name="safe-workflow",
        description="Reusable safe workflow",
        procedure=(
            "Inspect the input, validate its structure, write to a new output file, "
            "then reopen and verify the output."
        ),
        source_task_id="task-123",
        evidence=("verification: output file reopened",),
        preconditions=("Input path is within workspace",),
        expected_outcomes=("Output exists and validates",),
    )


def test_learning_candidate_requires_evidence_and_expected_outcome() -> None:
    with pytest.raises(ValueError, match="evidence"):
        create_candidate(
            name="safe-workflow",
            description="workflow",
            procedure="x" * 100,
            source_task_id="task",
            evidence=(),
        )
    validated = validate_candidate(candidate(), checks=("output validates",))
    assert validated.validated


def test_learning_candidate_with_unsafe_steps_does_not_validate() -> None:
    value = create_candidate(
        name="safe-workflow",
        description="Reusable safe workflow",
        procedure=(
            "Inspect the input, validate its structure, write to a new output file, "
            "then reopen and verify the output."
        ),
        source_task_id="task-123",
        evidence=("checked output",),
        expected_outcomes=("Output validates",),
        unsafe_steps=("delete unrelated user files",),
    )
    assert not validate_candidate(value, checks=("output validates",)).validated


def test_promotion_requires_validation(tmp_path: Path) -> None:
    value = candidate()
    store = SkillDraftStore(tmp_path / "quarantine")
    with pytest.raises(ValueError, match="pass validation"):
        promote_candidate(value, store, tmp_path / "workspace", approved=True)
    valid = validate_candidate(value, checks=("output validates",))
    target = promote_candidate(valid, store, tmp_path / "workspace", approved=True)
    assert target.is_file()


def test_candidate_integrity_is_checked_before_promotion(tmp_path: Path) -> None:
    valid = validate_candidate(candidate(), checks=("output validates",))
    altered = replace(valid, procedure=valid.procedure + " Add another unreviewed step.")
    store = SkillDraftStore(tmp_path / "quarantine")
    with pytest.raises(ValueError, match="hash mismatch"):
        promote_candidate(altered, store, tmp_path / "workspace", approved=True)


def test_candidate_promotion_fails_closed_without_owner_approval(tmp_path: Path) -> None:
    valid = validate_candidate(candidate(), checks=("output validates",))
    store = SkillDraftStore(tmp_path / "quarantine")
    with pytest.raises(PermissionError, match="owner approval"):
        promote_candidate(valid, store, tmp_path / "workspace")


def test_skill_regression_gate_rejects_regressed_candidate(tmp_path: Path) -> None:
    from eurus.learning.regression import evaluate_skill_regression

    report = evaluate_skill_regression(
        [{"task": "a", "passed": True}, {"task": "b", "passed": True}],
        [{"task": "a", "passed": True}, {"task": "b", "passed": False}],
        baseline_revision="base-1",
        candidate_revision="candidate-2",
    )
    assert not report.passed
    assert report.regressions == ("b",)
    valid = validate_candidate(candidate(), checks=("output validates",))
    with pytest.raises(ValueError, match="regression gate failed"):
        promote_candidate(
            valid,
            SkillDraftStore(tmp_path / "quarantine"),
            tmp_path / "workspace",
            approved=True,
            regression_report=report,
        )


def test_skill_regression_fingerprint_is_reproducible() -> None:
    from eurus.learning.regression import evaluate_skill_regression

    baseline = [{"task": "a", "passed": True}, {"task": "b", "passed": False}]
    candidate_rows = [{"task": "a", "passed": True}, {"task": "b", "passed": True}]
    first = evaluate_skill_regression(
        baseline, candidate_rows, baseline_revision="base", candidate_revision="next"
    )
    second = evaluate_skill_regression(
        baseline, candidate_rows, baseline_revision="base", candidate_revision="next"
    )
    assert first.passed
    assert first.fingerprint == second.fingerprint


def test_memory_retrieval_metrics_count_scope_violations() -> None:
    from eurus.memory.metrics import evaluate_retrieval

    metrics = evaluate_retrieval(
        [
            {"id": 1, "scope": "project-a"},
            {"id": 2, "scope": "project-b"},
        ],
        {1},
        requested_scope="project-a",
    )
    assert metrics.precision_at_k == 0.5
    assert metrics.recall_at_k == 1.0
    assert metrics.scope_violations == 1
