import json

from click.testing import CliRunner

from eurus.cli import main
from eurus.runtime.models import Run, RunStatus, RuntimeEvent
from eurus.runtime.state_machine import transition_run
from eurus.storage.events import EventStore
from eurus.storage.runs import RunStore


def stores(tmp_path, monkeypatch):
    monkeypatch.setenv("EURUS_HOME", str(tmp_path))
    database = tmp_path / "sessions.sqlite3"
    return RunStore(database), EventStore(database)


def test_inspect_includes_run_checkpoint_and_child_ids(tmp_path, monkeypatch):
    runs, _ = stores(tmp_path, monkeypatch)
    parent = Run(owner_id="operator-test", objective="parent")
    child = Run(owner_id="operator-test", objective="child", parent_run_id=parent.run_id)
    runs.create(parent)
    runs.create(child)
    runs.save_checkpoint(parent.run_id, {"progress": 2})

    result = CliRunner().invoke(main, ["inspect", parent.run_id])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["run_id"] == parent.run_id
    assert payload["child_run_ids"] == [child.run_id]
    assert payload["checkpoint"]["version"] == 1


def test_history_filters_status_and_events_redact_sensitive_payload(tmp_path, monkeypatch):
    runs, events = stores(tmp_path, monkeypatch)
    run = Run(owner_id="operator-test", objective="history")
    runs.create(run)
    events.append(
        RuntimeEvent(
            run_id=run.run_id,
            event_type="run.started",
            sequence_no=1,
            payload={"api_key": "secret-value", "progress": 1},
        )
    )

    history = CliRunner().invoke(main, ["history", "--status", "queued"])
    timeline = CliRunner().invoke(main, ["events", run.run_id])

    assert history.exit_code == 0, history.output
    assert run.run_id in history.output
    assert timeline.exit_code == 0, timeline.output
    assert "[REDACTED]" in timeline.output
    assert "secret-value" not in timeline.output


def test_cancel_only_marks_non_active_run_and_is_idempotent(tmp_path, monkeypatch):
    runs, events = stores(tmp_path, monkeypatch)
    queued = Run(owner_id="operator-test", objective="queued")
    active = Run(owner_id="operator-test", objective="active")
    runs.create(queued)
    runs.create(active)
    transition_run(active, RunStatus.RUNNING)
    runs.save(active, expected_status_version=0)

    runner = CliRunner()
    cancelled = runner.invoke(main, ["cancel", queued.run_id])
    repeated = runner.invoke(main, ["cancel", queued.run_id])
    active_result = runner.invoke(main, ["cancel", active.run_id])

    assert cancelled.exit_code == 0, cancelled.output
    assert runs.get(queued.run_id).status == RunStatus.CANCELLED
    assert repeated.exit_code == 0
    assert len(events.list_for_run(queued.run_id)) == 1
    assert active_result.exit_code == 0, active_result.output
    assert "cancellation requested" in active_result.output
    assert runs.get(active.run_id).status == RunStatus.RUNNING
    assert runs.cancellation_requested(active.run_id)


def test_resume_fails_closed_when_checkpoint_restore_is_unavailable(tmp_path, monkeypatch):
    runs, _ = stores(tmp_path, monkeypatch)
    run = Run(owner_id="operator-test", objective="interrupted", status=RunStatus.INTERRUPTED)
    runs.create(run)
    runs.save_checkpoint(run.run_id, {"progress": 1, "in_flight_side_effect": False})

    result = CliRunner().invoke(main, ["resume", run.run_id])

    assert result.exit_code != 0
    assert "restoration adapter" in result.output
    assert runs.get(run.run_id).status == RunStatus.INTERRUPTED


def test_benchmark_writes_explicit_unavailable_metrics(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import eurus.cli as cli
    from eurus.evals.runner import EvalResult

    monkeypatch.setenv("EURUS_HOME", str(tmp_path))

    class RouterStub:
        async def aclose(self):
            return None

    config = SimpleNamespace(agent=SimpleNamespace(model="test/model"))
    monkeypatch.setattr(cli, "_state", lambda ctx: SimpleNamespace(config=config))
    monkeypatch.setattr(cli, "build_router", lambda config: RouterStub())

    async def run_task(task, config, router, lessons=False):
        return EvalResult(task.name, True, 2, 10, 0.25, "completed")

    monkeypatch.setattr(cli, "run_task", run_task)
    result = CliRunner().invoke(main, ["benchmark", "--task", "fix_off_by_one"])

    assert result.exit_code == 0, result.output
    report_path = next((tmp_path / "evals").glob("benchmark-*.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["metrics"]["task_success_rate"] == 1.0
    assert report["metrics"]["recovery_success_rate"] is None
    assert (
        "does not inject runtime faults" in report["unavailable_metrics"]["recovery_success_rate"]
    )
