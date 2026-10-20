from pathlib import Path

from eurus.evals.results import EvaluationRecord, EvaluationStore


def test_evaluation_records_and_revision_comparison(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path / "evals.db")
    store.record(EvaluationRecord("core", "abc123", "task-a", True, 3, 120, 1.5, "completed"))
    store.record(EvaluationRecord("core", "def456", "task-a", False, 5, 190, 2.5, "limit"))
    comparison = store.compare("core", "abc123", "def456")
    assert comparison["baseline"]["pass_rate"] == 1.0
    assert comparison["candidate"]["pass_rate"] == 0.0


def test_evaluation_rejects_negative_metrics(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path / "evals.db")
    try:
        store.record(EvaluationRecord("core", "rev", "task", True, -1, 0, 0, "completed"))
    except ValueError as exc:
        assert "negative" in str(exc)
    else:
        raise AssertionError("negative measurements were accepted")
