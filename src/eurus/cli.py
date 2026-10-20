from __future__ import annotations

import asyncio
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from eurus import __version__, paths
from eurus.agent.events import AgentEvent
from eurus.agent.factory import build_agent
from eurus.agent.loop import Agent
from eurus.catalog import is_ready, resolve_api_key
from eurus.config import LOG_LEVELS, Config, load_config, sample_config
from eurus.diagnose import probe_all
from eurus.dotenv import load_dotenv
from eurus.errors import ConfigError, EurusError
from eurus.evals import TASKS, EvalResult, run_task
from eurus.logs import collect_secrets, setup_logging
from eurus.mcp.manager import MCPServerConfig, MCPToolManager
from eurus.models import Message, ModelRef, ToolCall, Usage
from eurus.providers.factory import build_router, resolve_chain
from eurus.providers.router import Router
from eurus.runtime.models import RunStatus, RuntimeEvent
from eurus.runtime.recovery import RecoveryAction, classify_recovery
from eurus.runtime.state_machine import InvalidRunTransition, transition_run
from eurus.security.approval import approval_request
from eurus.sessions import SessionStore
from eurus.storage.events import EventStore
from eurus.storage.runs import ConcurrentRunUpdate, RunStore
from eurus.tools.base import Tool
from eurus.tools.builtin import default_registry


class State:
    def __init__(self, config_path: Path | None, overrides: dict[str, Any]) -> None:
        self.config_path = config_path
        self.overrides = overrides
        self._config: Config | None = None

    @property
    def config(self) -> Config:
        if self._config is None:
            try:
                config = load_config(config_path=self.config_path, overrides=self.overrides)
            except ConfigError as exc:
                raise click.ClickException(str(exc)) from exc
            setup_logging(config.logging, secrets=collect_secrets(config))
            self._config = config
        return self._config


def _state(ctx: click.Context) -> State:
    obj = ctx.find_root().obj
    if not isinstance(obj, State):
        raise click.ClickException("internal error: state not initialised")
    return obj


@click.group(invoke_without_command=True)
@click.version_option(__version__, prog_name="eurus")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Use this config file instead of the user and project files.",
)
@click.option("--model", default=None, help="Override agent.model, as provider/model.")
@click.option(
    "--log-level",
    type=click.Choice(LOG_LEVELS, case_sensitive=False),
    default=None,
    help="Override logging.level.",
)
@click.pass_context
def main(
    ctx: click.Context,
    config_path: Path | None,
    model: str | None,
    log_level: str | None,
) -> None:
    overrides: dict[str, Any] = {}
    if model:
        overrides["agent"] = {"model": model}
    if log_level:
        overrides["logging"] = {"level": log_level.upper()}
    if os.environ.get("EURUS_NO_DOTENV", "").strip() not in ("1", "true"):
        load_dotenv([Path.cwd() / ".env", paths.home_dir() / ".env"], os.environ)
    ctx.obj = State(config_path, overrides)
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    config = _state(ctx).config
    keyed = [name for name, spec in sorted(config.providers.items()) if resolve_api_key(spec)]
    table = Table(title="eurus doctor")
    table.add_column("Check")
    table.add_column("Value")
    table.add_row("eurus", __version__)
    table.add_row("python", sys.version.split()[0])
    table.add_row("platform", platform.platform())
    table.add_row("home", str(paths.home_dir()))
    table.add_row("model", config.agent.model or "(not set)")
    table.add_row("permission mode", config.permissions.mode)
    table.add_row("local only", str(config.network.local_only).lower())
    table.add_row("providers with keys", ", ".join(keyed) or "none")
    if config.agent.model:
        spec = config.providers[ModelRef.parse(config.agent.model).provider]
        if not is_ready(spec):
            table.add_row("warning", f"{spec.api_key_env} is not set for the default model")
    Console().print(table)


@main.command()
@click.pass_context
def providers(ctx: click.Context) -> None:
    config = _state(ctx).config
    table = Table(title="providers")
    for column in ("Name", "Protocol", "Base URL", "Key env", "Status"):
        table.add_column(column)
    for name in sorted(config.providers):
        spec = config.providers[name]
        if not spec.api_key_env:
            status = "no key needed"
        elif resolve_api_key(spec):
            status = "ready"
        else:
            status = "missing key"
        table.add_row(spec.name, spec.protocol, spec.base_url, spec.api_key_env or "-", status)
    Console().print(table)


@main.command()
@click.argument("models", nargs=-1)
@click.pass_context
def diagnose(ctx: click.Context, models: tuple[str, ...]) -> None:

    config = _state(ctx).config
    try:
        if models:
            refs = [ModelRef.parse(text) for text in models]
            unknown = [ref.provider for ref in refs if ref.provider not in config.providers]
            if unknown:
                raise ConfigError(f"unknown provider '{unknown[0]}'")
        else:
            refs = resolve_chain(config)
        results = asyncio.run(probe_all(config, refs))
    except EurusError as exc:
        raise click.ClickException(str(exc)) from exc
    table = Table(title="eurus diagnose")
    for column in ("Model", "Result", "Seconds", "Detail"):
        table.add_column(column)
    for result in results:
        status = "[green]ok[/green]" if result.ok else "[red]failed[/red]"
        table.add_row(result.model, status, f"{result.seconds:.1f}", result.detail)
    Console().print(table)
    if not all(result.ok for result in results):
        ctx.exit(1)


async def _ask(config: Config, messages: list[Message], stream: bool) -> None:
    router = build_router(config)
    usage = Usage()
    try:
        if stream:
            async for event in router.stream(messages):
                if event.kind == "text":
                    click.echo(event.text, nl=False)
                elif event.kind == "usage" and event.usage is not None:
                    usage = event.usage
            click.echo()
        else:
            response = await router.chat(messages)
            click.echo(response.message.content)
            usage = response.usage
        if router.last_used is not None:
            click.echo(
                f"[{router.last_used}] in={usage.input_tokens} out={usage.output_tokens}",
                err=True,
            )
    finally:
        await router.aclose()


@main.command()
@click.argument("prompt", nargs=-1, required=True)
@click.option("--system", "system_prompt", default=None, help="Optional system prompt.")
@click.option("--stream/--no-stream", default=True, help="Stream tokens as they arrive.")
@click.pass_context
def ask(
    ctx: click.Context,
    prompt: tuple[str, ...],
    system_prompt: str | None,
    stream: bool,
) -> None:
    config = _state(ctx).config
    messages: list[Message] = []
    if system_prompt:
        messages.append(Message.system(system_prompt))
    messages.append(Message.user(" ".join(prompt)))
    try:
        asyncio.run(_ask(config, messages, stream))
    except EurusError as exc:
        raise click.ClickException(str(exc)) from exc


def _render(event: AgentEvent, router: Router) -> None:
    if event.kind == "text":
        click.echo(event.text, nl=False)
    elif event.kind == "tool_call" and event.tool_call is not None:
        shown = json.dumps(event.tool_call.arguments, ensure_ascii=False)[:200]
        click.echo(f"\n-> {event.tool_call.name} {shown}", err=True)
    elif event.kind == "tool_result" and event.tool_result is not None:
        result = event.tool_result
        preview = result.output.replace("\n", " ")[:200]
        click.echo(f"<- {'error' if result.is_error else 'ok'}: {preview}", err=True)
    elif event.kind == "lesson":
        click.echo(f"saved lesson: {event.text}", err=True)
    elif event.kind == "skill":
        click.echo(f"skill draft quarantined: {event.text}", err=True)
    elif event.kind == "done":
        click.echo()
        usage = event.usage or Usage()
        used = f"[{router.last_used}] " if router.last_used is not None else ""
        click.echo(
            f"{used}{event.stop_reason} in={usage.input_tokens} out={usage.output_tokens}",
            err=True,
        )


async def _approve(call: ToolCall, tool: Tool) -> bool:
    request = approval_request(call, tool)
    click.echo(f"\nApproval required: {request.tool_name}", err=True)
    click.echo(request.summary, err=True)
    click.echo(f"Action fingerprint: {request.action_fingerprint[:16]}", err=True)
    allowed = click.confirm("Allow this action?", default=False, err=True)
    SessionStore(paths.home_dir() / "sessions.sqlite3").record_event(
        "approval_accepted" if allowed else "approval_rejected",
        metadata={"tool": call.name, "mutating": tool.mutates},
    )
    return allowed


async def _turn(agent: Agent, router: Router, text: str) -> None:
    async for event in agent.run(text):
        _render(event, router)


async def _run_once(config: Config, text: str, stream: bool) -> None:
    router = build_router(config)
    mcp = MCPToolManager.load(paths.home_dir() / "mcp.json")
    try:
        await mcp.connect()
        registry = default_registry(include_capability_tools=True, include_workflow_tools=True)
        mcp.register_tools(registry)
        agent = build_agent(
            config,
            router,
            registry=registry,
            approver=_approve,
            stream=stream,
            lessons=True,
            learn=True,
        )
        await _turn(agent, router, text)
    finally:
        await mcp.close()
        await router.aclose()


@main.command()
@click.argument("prompt", nargs=-1, required=True)
@click.option("--stream/--no-stream", default=True, help="Stream tokens as they arrive.")
@click.pass_context
def run(ctx: click.Context, prompt: tuple[str, ...], stream: bool) -> None:
    config = _state(ctx).config
    try:
        asyncio.run(_run_once(config, " ".join(prompt), stream))
    except EurusError as exc:
        raise click.ClickException(str(exc)) from exc


CHAT_HELP = (
    "/tools list tools, /usage token usage, /clear reset conversation, "
    "/privacy export|purge, /exit quit"
)


@main.command()
@click.option("--stream/--no-stream", default=True, help="Stream tokens as they arrive.")
@click.option("--session", "session_id", default=None, help="Persist this chat session in SQLite.")
@click.pass_context
def chat(ctx: click.Context, stream: bool, session_id: str | None) -> None:
    config = _state(ctx).config
    with asyncio.Runner() as runner:
        try:
            router = build_router(config)
        except EurusError as exc:
            raise click.ClickException(str(exc)) from exc
        mcp = MCPToolManager.load(paths.home_dir() / "mcp.json")
        try:
            runner.run(mcp.connect())
            registry = default_registry(include_capability_tools=True, include_workflow_tools=True)
            mcp.register_tools(registry)
        except (RuntimeError, ValueError) as exc:
            runner.run(mcp.close())
            runner.run(router.aclose())
            raise click.ClickException(str(exc)) from exc
        agent = build_agent(
            config,
            router,
            registry=registry,
            approver=_approve,
            stream=stream,
            lessons=True,
            learn=True,
        )
        privacy_store = SessionStore(paths.home_dir() / "sessions.sqlite3")
        store = privacy_store if session_id else None
        if store is not None and session_id is not None:
            resume_id = store.latest_session_descendant(session_id)
            agent.history.extend(store.load_messages(resume_id))
            agent.context_store = store
            agent.session_id = resume_id
        if privacy_store.memory_curation_due():
            click.echo(
                "Memory curation is due. Review saved notes with `eurus memory list`, "
                "then update or remove stale entries.",
                err=True,
            )
            privacy_store.record_event("memory_curation_nudge", session_id=agent.session_id)
        click.echo(f"eurus chat ({config.agent.model}). {CHAT_HELP}", err=True)
        try:
            while True:
                try:
                    text = input("you> ").strip()
                except (EOFError, KeyboardInterrupt):
                    click.echo()
                    break
                if not text:
                    continue
                if text in ("/exit", "/quit"):
                    break
                if text == "/clear":
                    agent.reset()
                    if store is not None and session_id is not None:
                        store.save_messages(
                            agent.session_id or session_id,
                            agent.history,
                            project_scope=str(agent.context.workspace.resolve()),
                        )
                    click.echo("conversation cleared", err=True)
                elif text == "/tools":
                    click.echo(", ".join(agent.registry.names()), err=True)
                elif text == "/usage":
                    usage = agent.usage
                    click.echo(f"in={usage.input_tokens} out={usage.output_tokens}", err=True)
                elif text == "/privacy export":
                    click.echo(
                        json.dumps(
                            privacy_store.export_privacy_data(),
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                elif text == "/privacy purge":
                    prompt = (
                        "Type PURGE to clear saved sessions, memories, and local activity "
                        "metadata: "
                    )
                    if input(prompt).strip() == "PURGE":
                        counts = privacy_store.purge_privacy_data()
                        agent.reset()
                        click.echo(json.dumps({"purged": counts}, ensure_ascii=False), err=True)
                    else:
                        click.echo("privacy purge cancelled", err=True)
                else:
                    try:
                        runner.run(_turn(agent, router, text))
                        if store is not None and session_id is not None:
                            store.save_messages(
                                agent.session_id or session_id,
                                agent.history,
                                project_scope=str(agent.context.workspace.resolve()),
                            )
                    except EurusError as exc:
                        click.echo(f"error: {exc}", err=True)
                    except KeyboardInterrupt:
                        click.echo("\ninterrupted", err=True)
        finally:
            runner.run(mcp.close())
            runner.run(router.aclose())


@main.command("tui")
@click.pass_context
def tui_command(ctx: click.Context) -> None:
    from eurus.tui.app import run_tui

    try:
        run_tui(ctx)
    except EurusError as exc:
        raise click.ClickException(str(exc)) from exc


def agent_main() -> None:
    main(args=[*sys.argv[1:], "tui"], prog_name="eurus")


@main.group("gateway")
def gateway_group() -> None:
    pass


@gateway_group.command("telegram")
@click.option("--max-concurrent", type=click.IntRange(1, 8), default=2, show_default=True)
@click.pass_context
def gateway_telegram(ctx: click.Context, max_concurrent: int) -> None:
    from eurus.gateway.security import GatewayAccessPolicy, parse_id_allowlist
    from eurus.gateway.tasks import TaskStore
    from eurus.gateway.telegram import run_telegram_polling, token_from_environment

    try:
        token = token_from_environment()
        allowed_users = parse_id_allowlist(os.environ.get("EURUS_TELEGRAM_ALLOWED_USER_IDS", ""))
        allowed_groups = parse_id_allowlist(
            os.environ.get("EURUS_TELEGRAM_ALLOWED_GROUP_IDS", ""), allow_negative=True
        )
        if not allowed_users:
            raise RuntimeError(
                "Eurus Telegram access configuration must contain numeric Telegram user IDs"
            )
        policy = GatewayAccessPolicy(allowed_users, allowed_groups)
        config = _state(ctx).config
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    store = TaskStore(paths.home_dir() / "sessions.sqlite3")
    session_store = SessionStore(paths.home_dir() / "sessions.sqlite3")
    session_locks: dict[str, asyncio.Lock] = {}

    async def runner(task: Any) -> str:
        lock = session_locks.setdefault(task.session_id, asyncio.Lock())
        async with lock:
            router = build_router(config)
            mcp = MCPToolManager.load(paths.home_dir() / "mcp.json")
            try:
                await mcp.connect()
                registry = default_registry(
                    include_capability_tools=True, include_workflow_tools=True
                )
                mcp.register_tools(registry)
                agent = build_agent(
                    config,
                    router,
                    registry=registry,
                    approver=lambda call, tool: _deny_remote_approval(call, tool),
                    stream=False,
                    lessons=True,
                    learn=False,
                )
                agent.history.extend(session_store.load_messages(task.session_id))
                agent.session_id = task.session_id
                agent.context_store = session_store
                output: list[str] = []
                async for event in agent.run(task.objective):
                    if event.kind == "text" and event.text:
                        output.append(event.text)
                session_store.save_messages(
                    task.session_id,
                    agent.history,
                    project_scope=str(agent.context.workspace.resolve()),
                )
                answer = "".join(output).strip()
                if not answer:
                    answer = next(
                        (
                            message.content
                            for message in reversed(agent.history)
                            if message.role == "assistant" and message.content
                        ),
                        "Task ended without a verifiable text response.",
                    )
                return answer
            finally:
                await mcp.close()
                await router.aclose()

    try:
        asyncio.run(
            run_telegram_polling(
                token=token,
                policy=policy,
                store=store,
                runner=runner,
                max_concurrent=max_concurrent,
            )
        )
    except (RuntimeError, EurusError) as exc:
        raise click.ClickException(str(exc)) from exc


async def _deny_remote_approval(call: ToolCall, tool: Tool) -> bool:
    return False


@main.group("schedule")
def schedule_group() -> None:
    pass


@schedule_group.command("add")
@click.argument("objective")
@click.option("--owner", default="local", show_default=True)
@click.option("--timezone", default="UTC", show_default=True)
@click.option("--at", "run_at", default=None, help="One-time ISO-8601 datetime.")
@click.option(
    "--every",
    "interval_seconds",
    type=click.IntRange(60, 31536000),
    default=None,
    help="Recurring interval in seconds, minimum 60.",
)
@click.option("--profile", "profile_name", default=None, help="Saved task profile name.")
@click.pass_context
def schedule_add(
    ctx: click.Context,
    objective: str,
    owner: str,
    timezone: str,
    run_at: str | None,
    interval_seconds: int | None,
    profile_name: str | None,
) -> None:
    from eurus.scheduler import ScheduleStore

    if (run_at is None) == (interval_seconds is None):
        raise click.ClickException("provide exactly one of --at or --every")
    store = ScheduleStore(paths.home_dir() / "sessions.sqlite3")
    try:
        profile = store.get_profile(owner, profile_name) if profile_name else None
        if profile_name and profile is None:
            raise ValueError("saved task profile not found for this owner")
        schedule = store.create(
            owner,
            objective,
            kind="once" if run_at else "interval",
            run_at=run_at,
            interval_seconds=interval_seconds,
            timezone=timezone,
            profile=profile,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Schedule created: {schedule.schedule_id} (next run {schedule.next_run_at})")


@schedule_group.command("profile-save")
@click.argument("name")
@click.option("--owner", default="local", show_default=True)
@click.option("--timezone", default="UTC", show_default=True)
@click.option("--max-runtime", type=click.IntRange(1, 86400), default=1800, show_default=True)
@click.option("--max-steps", type=click.IntRange(1, 1000), default=20, show_default=True)
@click.option("--no-notifications", is_flag=True, default=False)
def schedule_profile_save(
    name: str,
    owner: str,
    timezone: str,
    max_runtime: int,
    max_steps: int,
    no_notifications: bool,
) -> None:
    from eurus.scheduler import ScheduleStore, TaskProfile

    store = ScheduleStore(paths.home_dir() / "sessions.sqlite3")
    profile = TaskProfile(
        name=name,
        timezone=timezone,
        max_runtime_seconds=max_runtime,
        max_steps=max_steps,
        notifications=not no_notifications,
    )
    try:
        store.save_profile(owner, profile)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Saved task profile: {name}")


@schedule_group.command("list")
@click.option("--owner", default="local", show_default=True)
def schedule_list(owner: str) -> None:
    from eurus.scheduler import ScheduleStore

    schedules = ScheduleStore(paths.home_dir() / "sessions.sqlite3").list_owner(owner)
    table = Table(title=f"Schedules for {owner}")
    for column in ("ID", "State", "Kind", "Next run (UTC)", "Objective"):
        table.add_column(column)
    for item in schedules:
        table.add_row(
            item.schedule_id[:8],
            "enabled" if item.enabled else "disabled",
            item.kind,
            item.next_run_at,
            item.objective[:72],
        )
    Console().print(table)


@schedule_group.command("enable")
@click.argument("schedule_id")
@click.option("--owner", default="local", show_default=True)
def schedule_enable(schedule_id: str, owner: str) -> None:
    from eurus.scheduler import ScheduleStore

    store = ScheduleStore(paths.home_dir() / "sessions.sqlite3")
    item = next(
        (entry for entry in store.list_owner(owner) if entry.schedule_id.startswith(schedule_id)),
        None,
    )
    if item is None or not store.set_enabled(item.schedule_id, owner, True):
        raise click.ClickException("schedule not found")
    click.echo(f"Enabled {item.schedule_id}")


@schedule_group.command("disable")
@click.argument("schedule_id")
@click.option("--owner", default="local", show_default=True)
def schedule_disable(schedule_id: str, owner: str) -> None:
    from eurus.scheduler import ScheduleStore

    store = ScheduleStore(paths.home_dir() / "sessions.sqlite3")
    item = next(
        (entry for entry in store.list_owner(owner) if entry.schedule_id.startswith(schedule_id)),
        None,
    )
    if item is None or not store.set_enabled(item.schedule_id, owner, False):
        raise click.ClickException("schedule not found")
    click.echo(f"Disabled {item.schedule_id}")


@schedule_group.command("history")
@click.argument("schedule_id")
@click.option("--owner", default="local", show_default=True)
def schedule_history(schedule_id: str, owner: str) -> None:
    from eurus.scheduler import ScheduleStore

    store = ScheduleStore(paths.home_dir() / "sessions.sqlite3")
    item = next(
        (entry for entry in store.list_owner(owner) if entry.schedule_id.startswith(schedule_id)),
        None,
    )
    if item is None:
        raise click.ClickException("schedule not found")
    for run in store.history(item.schedule_id):
        click.echo(f"{run.started_at} {run.status} {run.result[:120]}")


@schedule_group.command("cancel-run")
@click.argument("job_id")
@click.option("--owner", default="local", show_default=True)
def schedule_cancel_run(job_id: str, owner: str) -> None:
    from eurus.scheduler import ScheduleStore
    from eurus.storage.jobs import JobStore

    database_path = paths.home_dir() / "sessions.sqlite3"
    jobs = JobStore(database_path)
    job = next(
        (
            item
            for item in jobs.list_jobs(owner_id=owner, limit=1000)
            if item.job_id.startswith(job_id)
        ),
        None,
    )
    if job is None or not jobs.request_cancel(job.job_id, owner_id=owner):
        raise click.ClickException("queued/running job not found or cannot be cancelled")
    if job.status == "queued":
        ScheduleStore(database_path).finish_run(
            job.job_id, status="cancelled", result="cancelled by owner"
        )
    click.echo(f"Cancellation requested for job {job.job_id}")


@schedule_group.command("daemon")
@click.option("--poll-seconds", type=click.IntRange(1, 300), default=15, show_default=True)
@click.option("--workers", type=click.IntRange(1, 16), default=2, show_default=True)
@click.option("--owner", default="local", show_default=True)
@click.pass_context
def schedule_daemon(ctx: click.Context, poll_seconds: int, workers: int, owner: str) -> None:
    from eurus.automation.worker import AutomationWorkerPool
    from eurus.runtime.budgets import BudgetSpec
    from eurus.runtime.engine import AgentEngine
    from eurus.runtime.models import Run, RunStatus
    from eurus.scheduler import ScheduleStore
    from eurus.storage.events import EventStore
    from eurus.storage.jobs import Job, JobStore

    config = _state(ctx).config
    database_path = paths.home_dir() / "sessions.sqlite3"
    store = ScheduleStore(database_path)
    jobs = JobStore(database_path)
    events = EventStore(jobs.database)

    async def execute_job(job: Job) -> str:
        store.mark_run_started(job.job_id)
        schedule = store.get(job.schedule_id or "", owner_id=job.owner_id)
        if schedule is None:
            raise RuntimeError("schedule_unavailable")
        profile = job.payload.get("profile", {})
        timeout = int(profile.get("max_runtime_seconds", 1800))
        router = build_router(config)
        mcp = MCPToolManager.load(paths.home_dir() / "mcp.json")
        try:
            await mcp.connect()
            registry = default_registry(include_capability_tools=True, include_workflow_tools=True)
            mcp.register_tools(registry)
            agent = build_agent(
                config,
                router,
                registry=registry,
                approver=lambda call, tool: _deny_remote_approval(call, tool),
                stream=False,
                lessons=True,
                learn=False,
            )
            runtime_run = Run(
                run_id=job.job_id,
                owner_id=job.owner_id,
                objective=job.objective,
                budget={
                    "max_iterations": int(profile.get("max_steps", 20)),
                    "max_runtime_seconds": timeout,
                },
            )
            engine = AgentEngine(agent, event_store=events)
            result = await engine.run(
                job.objective,
                owner_id=job.owner_id,
                run=runtime_run,
                budget=BudgetSpec(
                    max_iterations=int(profile.get("max_steps", 20)),
                    max_runtime_seconds=timeout,
                ),
            )
            if result.run.status != RunStatus.COMPLETED or not result.output.strip():
                raise RuntimeError(result.run.stop_reason or "no_verifiable_text_result")
            return result.output.strip()
        finally:
            await mcp.close()
            await router.aclose()

    def finish_job(job: Job, status: str, result: str, error: str | None) -> None:
        schedule_status = status
        if error and status == "failed":
            result = error
        store.finish_run(job.job_id, status=schedule_status, result=result)

    pool = AutomationWorkerPool(
        jobs,
        execute_job,
        workers=workers,
        poll_seconds=float(poll_seconds),
        lease_seconds=60,
        on_finish=finish_job,
    )

    async def dispatch_loop() -> None:
        while True:
            store.dispatch_due(limit=100, owner_id=owner)
            for job in jobs.list_jobs(
                owner_id=owner, statuses=("unknown_outcome", "failed", "cancelled"), limit=100
            ):
                store.finish_run(job.job_id, status=job.status, result=job.error or job.result)
            await asyncio.sleep(poll_seconds)

    async def run_daemon() -> None:
        dispatcher = asyncio.create_task(dispatch_loop())
        workers_task = asyncio.create_task(pool.run())
        try:
            await asyncio.gather(dispatcher, workers_task)
        finally:
            await pool.stop()
            dispatcher.cancel()
            workers_task.cancel()
            await asyncio.gather(dispatcher, workers_task, return_exceptions=True)

    click.echo(f"Scheduler running for owner {owner} with {workers} workers. Press Ctrl+C to stop.")
    try:
        asyncio.run(run_daemon())
    except KeyboardInterrupt:
        click.echo("Scheduler stopped.")


@main.command("tools")
def tools_command() -> None:
    table = Table(title="tools")
    for column in ("Name", "Changes things", "Description"):
        table.add_column(column)
    for item in default_registry(include_capability_tools=True).all():
        table.add_row(item.name, "yes" if item.mutates else "no", item.description)
    Console().print(table)


@main.group("mcp")
def mcp_group() -> None:
    pass


@mcp_group.command("list")
def mcp_list() -> None:

    manager = MCPToolManager.load(paths.home_dir() / "mcp.json")
    table = Table(title="MCP servers")
    for column in ("Name", "Transport", "Enabled", "Configured"):
        table.add_column(column)
    for item in manager.status():
        table.add_row(
            str(item["name"]),
            str(item["transport"]),
            "yes" if item["enabled"] else "no",
            "connected" if item["connected"] else "not connected",
        )
    Console().print(table)


@mcp_group.command("add-stdio")
@click.argument("name")
@click.argument("command")
@click.option(
    "--arg",
    "arguments",
    multiple=True,
    help="Argument passed to the server; repeat as needed.",
)
def mcp_add_stdio(name: str, command: str, arguments: tuple[str, ...]) -> None:

    path = paths.home_dir() / "mcp.json"
    manager = MCPToolManager.load(path)
    if name in manager.configs:
        raise click.ClickException(f"MCP server already exists: {name}")
    configs = [
        *manager.configs.values(),
        MCPServerConfig(name=name, command=command, args=arguments),
    ]
    MCPToolManager.save(path, configs)
    click.echo(f"Configured MCP server '{name}'. Review its command before connecting.")


@mcp_group.command("add-http")
@click.argument("name")
@click.argument("url")
def mcp_add_http(name: str, url: str) -> None:

    path = paths.home_dir() / "mcp.json"
    manager = MCPToolManager.load(path)
    if name in manager.configs:
        raise click.ClickException(f"MCP server already exists: {name}")
    configs = [
        *manager.configs.values(),
        MCPServerConfig(name=name, transport="http", url=url),
    ]
    MCPToolManager.save(path, configs)
    click.echo(
        f"Configured MCP server '{name}'. Authentication headers are not set by this command."
    )


@mcp_group.command("remove")
@click.argument("name")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def mcp_remove(name: str, yes: bool) -> None:

    path = paths.home_dir() / "mcp.json"
    manager = MCPToolManager.load(path)
    if name not in manager.configs:
        raise click.ClickException(f"Unknown MCP server: {name}")
    if not yes:
        click.confirm(f"Remove MCP server '{name}'?", abort=True)
    MCPToolManager.save(path, [item for key, item in manager.configs.items() if key != name])
    click.echo(f"Removed MCP server '{name}'.")


@mcp_group.command("test")
@click.argument("name", required=False)
def mcp_test(name: str | None) -> None:

    async def run_test() -> None:
        manager = MCPToolManager.load(paths.home_dir() / "mcp.json")
        if name is not None:
            if name not in manager.configs:
                raise click.ClickException(f"Unknown MCP server: {name}")
            manager.configs = {name: manager.configs[name]}
        await manager.connect()
        try:
            table = Table(title="MCP connection test")
            for column in ("Name", "Transport", "Connected", "Tools"):
                table.add_column(column)
            for item in manager.status():
                table.add_row(
                    str(item["name"]),
                    str(item["transport"]),
                    "yes" if item["connected"] else "no",
                    str(item["tools"]),
                )
            Console().print(table)
        finally:
            await manager.close()

    try:
        asyncio.run(run_test())
    except (RuntimeError, ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.group("sessions")
def sessions_group() -> None:
    pass


@sessions_group.command("list")
def sessions_list() -> None:
    table = Table(title="sessions")
    for column in ("ID", "Title", "Messages", "Updated"):
        table.add_column(column)
    for item in SessionStore(paths.home_dir() / "sessions.sqlite3").list_sessions():
        table.add_row(
            str(item["id"]),
            str(item["title"]),
            str(item["message_count"]),
            str(item["updated_at"]),
        )
    Console().print(table)


@sessions_group.command("show")
@click.argument("session_id")
def sessions_show(session_id: str) -> None:
    store = SessionStore(paths.home_dir() / "sessions.sqlite3")
    messages = store.load_messages(session_id)
    if not messages:
        raise click.ClickException(f"session not found or empty: {session_id}")
    for message in messages:
        click.echo(f"[{message.role}] {message.content}")
        for call in message.tool_calls:
            click.echo(f"  tool call: {call.name} {json.dumps(call.arguments, ensure_ascii=False)}")


@sessions_group.command("summarize")
@click.argument("session_id")
def sessions_summarize(session_id: str) -> None:
    store = SessionStore(paths.home_dir() / "sessions.sqlite3")
    messages = store.load_messages(session_id)
    if not messages:
        raise click.ClickException(f"session not found or empty: {session_id}")
    try:
        summary = store.build_session_summary(session_id)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    store.save_session_summary(session_id, summary)
    click.echo(summary)


@main.group("learn")
def learn_group() -> None:
    pass


@learn_group.command("consolidate")
@click.option(
    "--apply",
    is_flag=True,
    help="Delete exact duplicate memories after showing the plan.",
)
def learn_consolidate(apply: bool) -> None:
    store = SessionStore(paths.home_dir() / "sessions.sqlite3")
    result = store.consolidate_memories(dry_run=not apply)
    action = "deleted" if apply else "would delete"
    click.echo(
        f"scanned={result['scanned']} duplicates={result['duplicates']} "
        f"{action}={result['deleted'] if apply else result['duplicates']}"
    )
    if result["duplicates"] and not apply:
        click.echo("Review this plan, then rerun with --apply to remove exact duplicates.")


@sessions_group.command("delete")
@click.argument("session_id")
@click.confirmation_option(prompt="Delete this session and its messages?")
def sessions_delete(session_id: str) -> None:
    if not SessionStore(paths.home_dir() / "sessions.sqlite3").delete_session(session_id):
        raise click.ClickException(f"session not found: {session_id}")
    click.echo(f"deleted session {session_id}")


@main.command("eval")
@click.option("--task", "names", multiple=True, help="Task name to run; repeat for several.")
@click.option("--lessons/--no-lessons", default=False, help="Inject saved lessons into prompts.")
@click.pass_context
def eval_command(ctx: click.Context, names: tuple[str, ...], lessons: bool) -> None:

    config = _state(ctx).config
    known = {task.name for task in TASKS}
    unknown = sorted(set(names) - known)
    if unknown:
        raise click.ClickException(f"unknown task: {', '.join(unknown)}")
    selected = [task for task in TASKS if not names or task.name in names]

    async def go() -> list[EvalResult]:
        router = build_router(config)
        try:
            return [await run_task(task, config, router, lessons=lessons) for task in selected]
        except EurusError as exc:
            raise click.ClickException(str(exc)) from exc
        finally:
            await router.aclose()

    results = asyncio.run(go())
    for item in results:
        mark = "pass" if item.passed else "FAIL"
        click.echo(
            f"{mark}  {item.task:<22} steps={item.steps:<3} tokens={item.tokens:<7} "
            f"{item.seconds:.1f}s {item.stop_reason}"
        )
    passed = sum(item.passed for item in results)
    click.echo(f"{passed}/{len(results)} passed (lessons={'on' if lessons else 'off'})")
    target = paths.home_dir() / "evals"
    target.mkdir(parents=True, exist_ok=True)
    report = {
        "model": config.agent.model,
        "lessons": lessons,
        "metrics": {
            "task_count": len(results),
            "passed": passed,
            "pass_rate": passed / len(results) if results else 0.0,
            "average_steps": (
                sum(item.steps for item in results) / len(results) if results else 0.0
            ),
            "average_tokens": (
                sum(item.tokens for item in results) / len(results) if results else 0.0
            ),
            "average_seconds": (
                sum(item.seconds for item in results) / len(results) if results else 0.0
            ),
        },
        "results": [item.to_dict() for item in results],
    }
    file = target / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    click.echo(f"saved {file}", err=True)


def _runtime_stores() -> tuple[RunStore, EventStore]:
    database = paths.home_dir() / "sessions.sqlite3"
    return RunStore(database), EventStore(database)


def _safe_payload(value: Any, key: str = "") -> Any:
    sensitive_parts = ("secret", "password", "credential", "api_key", "token", "authorization")
    if any(part in key.lower() for part in sensitive_parts):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(name): _safe_payload(item, str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_safe_payload(item) for item in value]
    if isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in ("bearer ", "sk-", "api_key=", "password=")):
            return "[REDACTED]"
        return value[:500]
    return value


@main.command("inspect")
@click.argument("run_id")
def inspect_run(run_id: str) -> None:
    runs, events = _runtime_stores()
    run = runs.get(run_id)
    if run is None:
        raise click.ClickException(f"run not found: {run_id}")
    checkpoint = runs.latest_checkpoint(run_id)
    children = runs.list_runs(parent_run_id=run_id, limit=1000)
    payload = {
        "run_id": run.run_id,
        "owner_id": run.owner_id,
        "objective": _safe_payload(run.objective),
        "parent_run_id": run.parent_run_id,
        "child_run_ids": [item.run_id for item in children],
        "session_id": run.session_id,
        "status": run.status.value,
        "stop_reason": run.stop_reason,
        "budget": run.budget,
        "created_at": run.created_at.isoformat(),
        "updated_at": run.updated_at.isoformat(),
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "checkpoint": (
            {
                "version": checkpoint["checkpoint_version"],
                "type": checkpoint["checkpoint_type"],
                "created_at": checkpoint["created_at"].isoformat(),
            }
            if checkpoint
            else None
        ),
        "event_count": len(events.list_for_run(run_id, limit=10000)),
        "result_ref": run.result_ref,
    }
    click.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@main.command("history")
@click.option("--owner", "owner_id", default=None, help="Filter by run owner.")
@click.option(
    "--status",
    "statuses",
    multiple=True,
    type=click.Choice([item.value for item in RunStatus]),
)
@click.option("--limit", type=click.IntRange(1, 1000), default=50, show_default=True)
def run_history(owner_id: str | None, statuses: tuple[str, ...], limit: int) -> None:
    runs, _ = _runtime_stores()
    items = runs.list_runs(
        owner_id=owner_id,
        statuses=tuple(RunStatus(item) for item in statuses),
        limit=limit,
    )
    if not items:
        click.echo("No runtime runs found.")
        return
    click.echo(
        json.dumps(
            [
                {
                    "run_id": item.run_id,
                    "status": item.status.value,
                    "owner_id": item.owner_id,
                    "created_at": item.created_at.isoformat(),
                    "parent_run_id": item.parent_run_id,
                    "stop_reason": item.stop_reason,
                }
                for item in items
            ],
            indent=2,
            ensure_ascii=False,
        )
    )


@main.command("events")
@click.argument("run_id")
@click.option("--after", "after_sequence", type=click.IntRange(0), default=0, show_default=True)
@click.option("--limit", type=click.IntRange(1, 10000), default=1000, show_default=True)
def run_events(run_id: str, after_sequence: int, limit: int) -> None:
    runs, events = _runtime_stores()
    if runs.get(run_id) is None:
        raise click.ClickException(f"run not found: {run_id}")
    items = events.list_for_run(run_id, after_sequence=after_sequence, limit=limit)
    for item in items:
        payload = {
            "sequence": item.sequence_no,
            "event_id": item.event_id,
            "type": item.event_type,
            "timestamp": item.created_at.isoformat(),
            "run_id": item.run_id,
            "parent_run_id": item.parent_run_id,
            "step_id": item.step_id,
            "attempt_id": item.attempt_id,
            "payload": _safe_payload(item.payload),
        }
        click.echo(json.dumps(payload, ensure_ascii=False))
    if not items:
        click.echo("No runtime events found for this run.")


@main.command("cancel")
@click.argument("run_id")
def cancel_run(run_id: str) -> None:
    runs, events = _runtime_stores()
    run = runs.get(run_id)
    if run is None:
        raise click.ClickException(f"run not found: {run_id}")
    if run.status == RunStatus.CANCELLED:
        click.echo(f"run already cancelled: {run_id}")
        return
    if run.status in {RunStatus.COMPLETED, RunStatus.FAILED}:
        raise click.ClickException(f"terminal run cannot be cancelled: {run.status.value}")
    if run.status == RunStatus.RUNNING:
        if not runs.request_cancel(run_id):
            raise click.ClickException(
                "cancellation request was not accepted because the run state changed"
            )
        click.echo(f"cancellation requested: {run_id}")
        return
    previous_version = run.status_version
    try:
        transition_run(run, RunStatus.CANCELLED, stop_reason="operator_cancelled")
        runs.save(run, expected_status_version=previous_version)
    except (InvalidRunTransition, ConcurrentRunUpdate) as exc:
        raise click.ClickException(f"cancellation not applied: {exc}") from exc
    sequence = events.last_sequence(run_id) + 1
    events.append(
        RuntimeEvent(
            run_id=run_id,
            event_type="run.cancelled",
            sequence_no=sequence,
            parent_run_id=run.parent_run_id,
            payload={"source": "operator"},
        )
    )
    click.echo(f"cancelled run: {run_id}")


@main.command("resume")
@click.argument("run_id")
def resume_run(run_id: str) -> None:
    runs, _ = _runtime_stores()
    run = runs.get(run_id)
    if run is None:
        raise click.ClickException(f"run not found: {run_id}")
    checkpoint = runs.latest_checkpoint(run_id)
    action = classify_recovery(run, checkpoint)
    if action == RecoveryAction.TERMINAL:
        raise click.ClickException(f"terminal run cannot be resumed: {run.status.value}")
    if action == RecoveryAction.RECONCILE_UNKNOWN_OUTCOME:
        raise click.ClickException(
            "resume blocked: checkpoint indicates an uncertain side effect; "
            "reconcile it manually first"
        )
    if action == RecoveryAction.MANUAL_REVIEW:
        raise click.ClickException(
            "resume blocked: recovery state or checkpoint is insufficient; manual review required"
        )
    if action == RecoveryAction.RESUME_CHECKPOINT:
        raise click.ClickException(
            "resume blocked: checkpoint is classifiable, but AgentEngine has no "
            "checkpoint restoration adapter; replay is unsafe"
        )
    raise click.ClickException(
        "run is queued but no persisted dispatch/resume worker is available from this command"
    )


@main.command("benchmark")
@click.option("--task", "names", multiple=True, help="Task name to run; repeat for several.")
@click.option("--lessons/--no-lessons", default=False, help="Inject saved lessons into prompts.")
@click.pass_context
def benchmark_command(ctx: click.Context, names: tuple[str, ...], lessons: bool) -> None:
    config = _state(ctx).config
    known = {task.name for task in TASKS}
    unknown = sorted(set(names) - known)
    if unknown:
        raise click.ClickException(f"unknown task: {', '.join(unknown)}")
    selected = [task for task in TASKS if not names or task.name in names]

    async def go() -> list[EvalResult]:
        router = build_router(config)
        try:
            return [await run_task(task, config, router, lessons=lessons) for task in selected]
        except EurusError as exc:
            raise click.ClickException(str(exc)) from exc
        finally:
            await router.aclose()

    results = asyncio.run(go())
    passed = sum(item.passed for item in results)
    report = {
        "schema_version": 1,
        "benchmark_kind": "runtime_task_fixtures",
        "model": config.agent.model,
        "lessons": lessons,
        "metrics": {
            "task_count": len(results),
            "passed": passed,
            "task_success_rate": passed / len(results) if results else None,
            "mean_latency_seconds": (
                sum(item.seconds for item in results) / len(results) if results else None
            ),
            "mean_steps": (sum(item.steps for item in results) / len(results) if results else None),
            "mean_tokens": (
                sum(item.tokens for item in results) / len(results) if results else None
            ),
            "recovery_success_rate": None,
            "duplicate_side_effect_rate": None,
            "provider_cost": None,
        },
        "unavailable_metrics": {
            "recovery_success_rate": "This benchmark does not inject runtime faults.",
            "duplicate_side_effect_rate": "The fixtures do not instrument external side effects.",
            "provider_cost": "The provider usage contract does not expose cost.",
        },
        "results": [item.to_dict() for item in results],
    }
    target = paths.home_dir() / "evals"
    target.mkdir(parents=True, exist_ok=True)
    file = target / f"benchmark-{time.time_ns()}.json"
    file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    click.echo(json.dumps(report["metrics"], indent=2, ensure_ascii=False))
    click.echo(f"saved {file}", err=True)
    if passed != len(results):
        raise click.ClickException(f"benchmark failures: {len(results) - passed}/{len(results)}")


@main.command("metrics")
@click.option("--days", type=click.IntRange(1, 3650), default=30, show_default=True)
def metrics_command(days: int) -> None:

    store = SessionStore(paths.home_dir() / "sessions.sqlite3")
    metrics = store.event_metrics(days=days)
    rates = store.event_rates(days=days)
    click.echo(f"Lifecycle events in the last {days} days:")
    if not metrics:
        click.echo("No lifecycle events recorded in this window.")
    else:
        for event_type, count in metrics.items():
            click.echo(f"{event_type}: {count}")
    click.echo(f"Approval decisions: {rates['approval_decisions']}")
    click.echo(f"Approval rejection rate: {rates['approval_rejection_rate']:.1%}")
    click.echo(f"Correction proposals: {rates['correction_proposals']}")
    click.echo(f"Correction decisions: {rates['correction_decisions']}")
    click.echo(f"Correction acceptance rate: {rates['correction_acceptance_rate']:.1%}")
    click.echo(f"Correction rejection rate: {rates['correction_rejection_rate']:.1%}")


@main.group("memory")
def memory_group() -> None:
    pass


@memory_group.command("add")
@click.argument("content")
@click.option("--tag", "tags", default="", help="Comma-separated tags.")
@click.option("--category", default="general", help="Memory category, e.g. preference or project.")
@click.option("--scope", default="global", help="Memory scope: global or an explicit project path.")
@click.option(
    "--type",
    "memory_type",
    type=click.Choice(
        ["profile", "preference", "project_fact", "agent_note", "lesson", "episode", "fact"]
    ),
    default="fact",
)
def memory_add(content: str, tags: str, category: str, scope: str, memory_type: str) -> None:
    memory_id = SessionStore(paths.home_dir() / "sessions.sqlite3").add_memory(
        content, tags, category=category, scope=scope, memory_type=memory_type
    )
    click.echo(f"saved memory {memory_id}")


@memory_group.command("search")
@click.argument("query")
@click.option("--limit", type=click.IntRange(1, 100), default=10)
@click.option("--scope", default=None, help="Limit results to global notes and this project scope.")
def memory_search(query: str, limit: int, scope: str | None) -> None:
    results = SessionStore(paths.home_dir() / "sessions.sqlite3").search_memories(
        query, limit, scope=scope
    )
    for item in results:
        click.echo(f"#{item['id']} [{item['category']}] [{item['tags']}] {item['content']}")
    if not results:
        click.echo("no matching memories")


@memory_group.command("list")
@click.option("--category", default=None, help="Only show one category, e.g. lesson.")
@click.option("--limit", type=click.IntRange(1, 500), default=50)
@click.option("--scope", default=None, help="Limit results to global notes and this project scope.")
def memory_list(category: str | None, limit: int, scope: str | None) -> None:
    store = SessionStore(paths.home_dir() / "sessions.sqlite3")
    for item in store.list_memories(category=category, limit=limit, scope=scope):
        click.echo(f"#{item['id']} [{item['category']}] [{item['tags']}] {item['content']}")


@memory_group.command("update")
@click.argument("memory_id", type=int)
@click.argument("content")
@click.option("--tag", "tags", default=None, help="Replace tags.")
@click.option("--category", default=None, help="Replace category.")
def memory_update(memory_id: int, content: str, tags: str | None, category: str | None) -> None:
    store = SessionStore(paths.home_dir() / "sessions.sqlite3")
    if not store.update_memory(memory_id, content, tags=tags, category=category):
        raise click.ClickException(f"memory not found: {memory_id}")
    click.echo(f"updated memory {memory_id}")


@memory_group.command("delete")
@click.argument("memory_id", type=int)
@click.confirmation_option(prompt="Delete this memory?")
def memory_delete(memory_id: int) -> None:
    if not SessionStore(paths.home_dir() / "sessions.sqlite3").delete_memory(memory_id):
        raise click.ClickException(f"memory not found: {memory_id}")
    click.echo(f"deleted memory {memory_id}")


memory_group.add_command(memory_update, "replace")
memory_group.add_command(memory_delete, "remove")


@main.group("privacy")
def privacy_group() -> None:
    pass


@privacy_group.command("export")
def privacy_export() -> None:
    data = SessionStore(paths.home_dir() / "sessions.sqlite3").export_privacy_data()
    click.echo(json.dumps(data, ensure_ascii=False, indent=2))


@privacy_group.command("purge")
@click.confirmation_option(prompt="Clear saved sessions, memories, and local activity metadata?")
def privacy_purge() -> None:
    counts = SessionStore(paths.home_dir() / "sessions.sqlite3").purge_privacy_data()
    click.echo(json.dumps({"purged": counts}, ensure_ascii=False))


@main.group("config")
def config_group() -> None:
    pass


@config_group.command("show")
@click.pass_context
def config_show(ctx: click.Context) -> None:
    click.echo(json.dumps(_state(ctx).config.to_dict(), indent=2))


@config_group.command("path")
def config_path_command() -> None:
    project = paths.project_config_path(Path.cwd())
    click.echo(f"home: {paths.home_dir()}")
    click.echo(f"user: {paths.user_config_path()}")
    click.echo(f"project: {project if project else '-'}")


@config_group.command("init")
@click.option("--force", is_flag=True, help="Overwrite an existing user config.")
def config_init(force: bool) -> None:
    target = paths.user_config_path()
    if target.exists() and not force:
        raise click.ClickException(f"{target} already exists, use --force to overwrite")
    paths.ensure_private_dir(target.parent)
    target.write_text(sample_config(), encoding="utf-8")
    click.echo(f"written {target}")


if __name__ == "__main__":
    main()
