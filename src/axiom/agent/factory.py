from __future__ import annotations

from pathlib import Path

from axiom.agent.loop import Agent, Approver
from axiom.agent.prompt import build_system_prompt
from axiom.config import Config
from axiom.execution.sandbox import ContainerSandbox
from axiom.providers.router import Router
from axiom.tools.base import ToolContext, ToolRegistry
from axiom.tools.builtin import default_registry
from axiom.tools.shell import shell_tool


def build_agent(
    config: Config,
    router: Router,
    *,
    registry: ToolRegistry | None = None,
    approver: Approver | None = None,
    workspace: Path | None = None,
    stream: bool = True,
) -> Agent:
    root = (workspace or Path(config.permissions.workspace)).expanduser().resolve()
    context = ToolContext(
        workspace=root,
        max_output=config.tools.max_output_chars,
        timeout=config.tools.timeout_seconds,
    )
    tools = (
        registry
        if registry is not None
        else default_registry(include_capability_tools=True, include_workflow_tools=True)
    )
    if "shell" not in tools:
        sandbox = ContainerSandbox(
            root,
            runtime=config.sandbox.runtime,
            image=config.sandbox.image,
            memory_mb=config.sandbox.memory_mb,
            cpu_limit=config.sandbox.cpu_limit,
            pids_limit=config.sandbox.pids_limit,
            network_enabled=config.sandbox.network,
        )
        tools.register(shell_tool(sandbox))
    return Agent(
        router,
        tools,
        context,
        system_prompt=build_system_prompt(root, config.agent.system_prompt),
        max_steps=config.agent.max_steps,
        approver=approver,
        stream=stream,
        max_parallel_tools=config.tools.max_parallel_tools,
    )
