from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from eurus import paths
from eurus.agent.context import ExtractiveContextEngine, LLMSummarizingContextEngine
from eurus.agent.delegation import delegation_tools
from eurus.agent.events import AgentEvent
from eurus.agent.loop import Agent, Approver, IterationBudget
from eurus.agent.prompt import build_system_prompt_tiers
from eurus.agent.team_tools import document_team_tools
from eurus.coding.project_index import render_project_index
from eurus.config import Config
from eurus.execution.host import HostSandbox
from eurus.execution.sandbox import ContainerSandbox, Sandbox
from eurus.learning import LessonStore, Reflector
from eurus.learning.skills import SkillLearner
from eurus.memory.manager import PromptMemory
from eurus.providers.factory import build_auxiliary_router
from eurus.providers.router import Router
from eurus.skills.tools import skill_tools
from eurus.tools.base import ToolContext, ToolRegistry
from eurus.tools.builtin import default_registry
from eurus.tools.shell import shell_tool


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
    event_recorder: Callable[[AgentEvent], object] | None = None,
) -> Agent:
    root = (workspace or Path(config.permissions.workspace)).expanduser().resolve()
    context = ToolContext(
        workspace=root,
        max_output=config.tools.max_output_chars,
        timeout=config.tools.timeout_seconds,
    )
    auxiliary_router = build_auxiliary_router(config)
    tools = (
        registry
        if registry is not None
        else default_registry(
            include_capability_tools=True,
            include_workflow_tools=True,
            memory_router=auxiliary_router,
        )
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
    for tool in [
        *skill_tools(),
        *document_team_tools(router, tools),
        *delegation_tools(router, tools),
    ]:
        if tool.name not in tools:
            tools.register(tool)
    settings = config.learning
    store = LessonStore(
        paths.home_dir() / "sessions.sqlite3",
        limit=settings.lessons_in_prompt,
        max_lessons=settings.max_lessons,
    )
    reflect = learn and settings.mode != "off"
    stable_prompt, context_prompt, volatile_prompt = build_system_prompt_tiers(
        root, config.agent.system_prompt
    )
    if root.is_dir():
        try:
            project_context = render_project_index(root)
        except (OSError, ValueError):
            project_context = ""
        if project_context and len(project_context) <= 6_000:
            context_prompt = f"{context_prompt}\n\n{project_context}"
    context_engine = (
        LLMSummarizingContextEngine(auxiliary_router, keep_recent=20)
        if auxiliary_router is not None
        else ExtractiveContextEngine(keep_recent=20)
    )
    return Agent(
        router,
        tools,
        context,
        system_prompt=stable_prompt,
        context_prompt=context_prompt,
        volatile_prompt=volatile_prompt,
        iteration_budget=IterationBudget(config.agent.iteration_budget),
        request_timeout=config.network.timeout_seconds,
        approver=approver,
        stream=stream,
        max_parallel_tools=config.tools.max_parallel_tools,
        context_tokens=config.agent.context_tokens,
        context_engine=context_engine,
        lessons=PromptMemory(paths.home_dir() / "sessions.sqlite3", store, scope=str(root))
        if lessons
        else None,
        learner=Reflector(router, store, settings.mode) if reflect else None,
        skill_learner=SkillLearner(auxiliary_router or router) if learn else None,
        event_recorder=event_recorder,
    )
