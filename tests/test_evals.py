import asyncio
import sys

import pytest

from eurus.coding.validation import run_validation
from eurus.config import load_config
from eurus.evals import TASKS, EvalTask, run_task
from eurus.evals.runner import write_files
from eurus.providers.testing import openai_message, scripted_router

COMPILE = (sys.executable, "-m", "compileall", "-q", ".")
BROKEN = "def f(:\n"
FIXED = "def f():\n    pass\n"


def config_for(tmp_path):
    return load_config(cwd=tmp_path, environ={"EURUS_HOME": str(tmp_path)})


def broken_task():
    return EvalTask("broken", "fix bad.py", {"bad.py": BROKEN}, {"bad.py": FIXED}, COMPILE)


def run(tmp_path, responses):
    router, transport = scripted_router(json_responses=responses)
    result = asyncio.run(run_task(broken_task(), config_for(tmp_path), router))
    return result, transport


def test_run_task_passes_after_agent_fixes_the_file(tmp_path):
    patch = {"patches": [{"path": "bad.py", "expected_content": BROKEN, "new_content": FIXED}]}
    result, transport = run(
        tmp_path,
        [openai_message("", [("c1", "apply_patchset", patch)]), openai_message("fixed")],
    )
    assert result.passed
    assert result.steps == 2
    assert result.stop_reason == "completed"
    assert result.tokens == 30


def test_run_task_fails_when_nothing_is_fixed(tmp_path):
    result, _ = run(tmp_path, [openai_message("I could not do it")])
    assert not result.passed
    assert "bad.py" in result.detail


def test_eval_agent_has_no_shell_or_memory_tools(tmp_path):
    result, transport = run(tmp_path, [openai_message("nothing")])
    names = {tool["function"]["name"] for tool in transport.calls[0].payload["tools"]}
    assert "shell" not in names
    assert "remember_memory" not in names
    assert "apply_patchset" in names and "run_checks" in names


def test_task_names_are_unique():
    names = [task.name for task in TASKS]
    assert len(names) == len(set(names)) >= 4


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_builtin_task_fails_before_and_passes_with_solution(tmp_path, task):
    write_files(tmp_path, task.files)
    assert not run_validation(tmp_path, task.check, timeout_seconds=60).passed
    write_files(tmp_path, task.solution)
    assert run_validation(tmp_path, task.check, timeout_seconds=60).passed
