from __future__ import annotations

from pathlib import Path

from axiom import paths
from axiom.agent.loop import Agent, Approver
from axiom.agent.prompt import build_system_prompt
from axiom.coding.project_index import render_project_index
from axiom.config import Config
from axiom.execution.host import HostSandbox
from axiom.execution.sandbox import ContainerSandbox, Sandbox
from axiom.learning import LessonStore, Reflector
from axiom.memory.manager import PromptMemory
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
    shell: bool = True,
    lessons: bool = False,
    learn: bool = False,
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
    if shell and "shell" not in tools:
        sandbox: Sandbox
        if config.sandbox.backend == "host":
            sandbox = HostSandbox(root)
        else:
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
    settings = config.learning
    store = LessonStore(
        paths.home_dir() / "sessions.sqlite3",
        limit=settings.lessons_in_prompt,
        max_lessons=settings.max_lessons,
    )
    reflect = learn and settings.mode != "off"
    system_prompt = build_system_prompt(
        root, config.agent.system_prompt, skills=config.agent.skills
    )
    if root.is_dir():
        try:
            project_context = render_project_index(root)
        except (OSError, ValueError):
            project_context = ""
        if project_context and len(project_context) <= 6_000:
            system_prompt = f"{system_prompt}\n\n{project_context}"
    return Agent(
        router,
        tools,
        context,
        system_prompt=system_prompt,
        max_steps=config.agent.max_steps,
        approver=approver,
        stream=stream,
        max_parallel_tools=config.tools.max_parallel_tools,
        context_tokens=config.agent.context_tokens,
        lessons=PromptMemory(paths.home_dir() / "sessions.sqlite3", store) if lessons else None,
        learner=Reflector(router, store, settings.mode) if reflect else None,
    )
