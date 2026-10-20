import asyncio
import re
from pathlib import Path

from eurus.agent import Agent
from eurus.coding.tools import coding_tools
from eurus.errors import ToolError
from eurus.models import ToolCall
from eurus.providers.testing import openai_message, scripted_router
from eurus.tools import Tool, ToolContext, ToolRegistry

OLD = "def f():\n    return 1\n"
NEW = "def f():\n    return 2\n"


def run_tool(registry, name, arguments, workspace):
    context = ToolContext(workspace=workspace, max_output=5000, timeout=5.0)
    return asyncio.run(registry.execute(ToolCall("1", name, arguments), context))


def workspace_with(tmp_path):
    (tmp_path / "m.py").write_text(OLD, encoding="utf-8")
    return ToolRegistry(coding_tools())


def test_expected_equal_to_new_content_gets_a_specific_error(tmp_path):
    registry = workspace_with(tmp_path)
    result = run_tool(
        registry,
        "apply_patch",
        {"path": "m.py", "expected_content": NEW, "new_content": NEW},
        tmp_path,
    )
    assert result.is_error
    assert "equals new_content" in result.output
    assert (tmp_path / "m.py").read_text(encoding="utf-8") == OLD


def test_base_sha256_flow_applies_without_resending_original(tmp_path):
    registry = workspace_with(tmp_path)
    proposal = run_tool(registry, "propose_patch", {"path": "m.py", "new_content": NEW}, tmp_path)
    digest = re.search(r"base_sha256: ([0-9a-f]{64})", proposal.output)
    assert digest is not None
    result = run_tool(
        registry,
        "apply_patch",
        {"path": "m.py", "new_content": NEW, "base_sha256": digest.group(1)},
        tmp_path,
    )
    assert not result.is_error
    assert (tmp_path / "m.py").read_text(encoding="utf-8") == NEW


def test_base_sha256_prefix_is_accepted_and_wrong_hash_rejected(tmp_path):
    registry = workspace_with(tmp_path)
    proposal = run_tool(registry, "propose_patch", {"path": "m.py", "new_content": NEW}, tmp_path)
    digest = re.search(r"base_sha256: ([0-9a-f]{64})", proposal.output).group(1)
    bad = run_tool(
        registry,
        "apply_patch",
        {"path": "m.py", "new_content": NEW, "base_sha256": "0" * 64},
        tmp_path,
    )
    short = run_tool(
        registry,
        "apply_patch",
        {"path": "m.py", "new_content": NEW, "base_sha256": digest[:6]},
        tmp_path,
    )
    assert bad.is_error and short.is_error
    assert (tmp_path / "m.py").read_text(encoding="utf-8") == OLD
    ok = run_tool(
        registry,
        "apply_patch",
        {"path": "m.py", "new_content": NEW, "base_sha256": digest[:16]},
        tmp_path,
    )
    assert not ok.is_error


def test_apply_patch_needs_some_base(tmp_path):
    registry = workspace_with(tmp_path)
    result = run_tool(registry, "apply_patch", {"path": "m.py", "new_content": NEW}, tmp_path)
    assert result.is_error and "base_sha256" in result.output


async def always_fails(arguments, context):
    raise ToolError("boom")


async def noop(arguments, context):
    return "ok"


async def allow(call, tool):
    return True


EMPTY = {"type": "object", "properties": {}}


def agent_for(responses):
    router, transport = scripted_router(json_responses=responses)
    registry = ToolRegistry(
        [
            Tool("fail", "always fails", EMPTY, always_fails),
            Tool("noop", "does nothing", EMPTY, noop),
            Tool("write", "mutates", EMPTY, noop, mutates=True, requires_approval=True),
        ]
    )
    context = ToolContext(workspace=Path("."), max_output=1000, timeout=5.0)
    agent = Agent(router, registry, context, system_prompt="sys", approver=allow, stream=False)
    return agent, transport


def run_agent(agent):
    async def go():
        return [event async for event in agent.run("go")]

    return asyncio.run(go())


def call(name, index):
    return openai_message("", [(f"c{index}", name, {})])


def tool_outputs(events):
    return [e.tool_result.output for e in events if e.kind == "tool_result"]


def test_same_failing_call_is_blocked_even_when_not_consecutive():
    names = ["fail", "noop", "fail", "noop", "fail", "noop"]
    agent, _ = agent_for([call(n, i) for i, n in enumerate(names)] + [openai_message("done")])
    outputs = tool_outputs(run_agent(agent))
    assert outputs[0] == "error: boom" and outputs[2] == "error: boom"
    assert "already failed 2 times" in outputs[4]
    assert "boom" in outputs[4]


def test_successful_mutation_resets_the_failure_memory():
    names = ["fail", "fail", "write", "fail"]
    agent, _ = agent_for([call(n, i) for i, n in enumerate(names)] + [openai_message("done")])
    outputs = tool_outputs(run_agent(agent))
    assert outputs[3] == "error: boom"


def test_patchset_error_explains_expected_equal_to_new(tmp_path):
    from eurus.coding.patchset import FilePatch, apply_patchset

    (tmp_path / "m.py").write_text(OLD, encoding="utf-8")
    try:
        apply_patchset(tmp_path, [FilePatch("m.py", NEW, NEW)])
    except ValueError as exc:
        assert "file changed since planning" in str(exc)
        assert "equals new_content" in str(exc)
    else:
        raise AssertionError("expected a failure")
