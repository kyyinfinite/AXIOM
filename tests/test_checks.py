import asyncio
import os

import pytest

from eurus.coding.checks import check_tools
from eurus.models import ToolCall
from eurus.tools import ToolContext, ToolRegistry, default_registry


def run_tool(workspace, arguments):
    registry = ToolRegistry(check_tools())
    context = ToolContext(workspace=workspace, max_output=4000, timeout=30.0)
    call = ToolCall("1", "run_checks", arguments)
    return asyncio.run(registry.execute(call, context))


def test_run_checks_requires_approval():
    tool = check_tools()[0]
    assert tool.requires_approval


def test_compile_passes(tmp_path):
    (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
    result = run_tool(tmp_path, {"checks": ["compile"]})
    assert not result.is_error
    assert "[compile] passed" in result.output
    assert "all checks passed" in result.output


def test_compile_failure_is_reported_to_model(tmp_path):
    (tmp_path / "bad.py").write_text("def broken(:\n", encoding="utf-8")
    result = run_tool(tmp_path, {"checks": ["compile"]})
    assert not result.is_error
    assert "[compile] failed" in result.output
    assert "SyntaxError" in result.output
    assert "1 failed" in result.output


def test_environment_is_scrubbed(tmp_path):
    (tmp_path / "env_probe.py").write_text(
        "import os\nassert 'EURUS_PROBE_API_KEY' not in os.environ\n", encoding="utf-8"
    )
    os.environ["EURUS_PROBE_API_KEY"] = "secret"
    try:
        result = run_tool(tmp_path, {"checks": ["compile"]})
    finally:
        del os.environ["EURUS_PROBE_API_KEY"]
    assert "passed" in result.output


@pytest.mark.parametrize("target", ["-k", "../outside", "/etc", "missing.py"])
def test_bad_targets_are_rejected(tmp_path, target):
    result = run_tool(tmp_path, {"checks": ["compile"], "targets": [target]})
    assert result.is_error


def test_unknown_check_is_rejected(tmp_path):
    result = run_tool(tmp_path, {"checks": ["rm"]})
    assert result.is_error


def test_registered_in_default_workflow_registry():
    registry = default_registry(include_workflow_tools=True)
    assert "run_checks" in registry.names()
