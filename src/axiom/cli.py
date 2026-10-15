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

from axiom import __version__, paths
from axiom.agent.events import AgentEvent
from axiom.agent.factory import build_agent
from axiom.agent.loop import Agent
from axiom.catalog import is_ready, resolve_api_key
from axiom.config import LOG_LEVELS, Config, load_config, sample_config
from axiom.diagnose import probe_all
from axiom.dotenv import load_dotenv
from axiom.errors import AxiomError, ConfigError
from axiom.evals import TASKS, EvalResult, run_task
from axiom.logs import collect_secrets, setup_logging
from axiom.models import Message, ModelRef, ToolCall, Usage
from axiom.providers.factory import build_router, resolve_chain
from axiom.providers.router import Router
from axiom.security.approval import approval_request
from axiom.sessions import SessionStore
from axiom.tools.base import Tool
from axiom.tools.builtin import default_registry


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
@click.version_option(__version__, prog_name="axiom")
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
    if os.environ.get("AXIOM_NO_DOTENV", "").strip() not in ("1", "true"):
        load_dotenv([Path.cwd() / ".env", paths.home_dir() / ".env"], os.environ)
    ctx.obj = State(config_path, overrides)
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    config = _state(ctx).config
    keyed = [name for name, spec in sorted(config.providers.items()) if resolve_api_key(spec)]
    table = Table(title="axiom doctor")
    table.add_column("Check")
    table.add_column("Value")
    table.add_row("axiom", __version__)
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
    """Send a tiny live request to each model and explain failures (403, key, quota)."""
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
    except AxiomError as exc:
        raise click.ClickException(str(exc)) from exc
    table = Table(title="axiom diagnose")
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
    except AxiomError as exc:
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
    return click.confirm("Allow this action?", default=False, err=True)


async def _turn(agent: Agent, router: Router, text: str) -> None:
    async for event in agent.run(text):
        _render(event, router)


async def _run_once(config: Config, text: str, stream: bool) -> None:
    router = build_router(config)
    try:
        agent = build_agent(
            config, router, approver=_approve, stream=stream, lessons=True, learn=True
        )
        await _turn(agent, router, text)
    finally:
        await router.aclose()


@main.command()
@click.argument("prompt", nargs=-1, required=True)
@click.option("--stream/--no-stream", default=True, help="Stream tokens as they arrive.")
@click.pass_context
def run(ctx: click.Context, prompt: tuple[str, ...], stream: bool) -> None:
    config = _state(ctx).config
    try:
        asyncio.run(_run_once(config, " ".join(prompt), stream))
    except AxiomError as exc:
        raise click.ClickException(str(exc)) from exc


CHAT_HELP = "/tools list tools, /usage token usage, /clear reset conversation, /exit quit"


@main.command()
@click.option("--stream/--no-stream", default=True, help="Stream tokens as they arrive.")
@click.option("--session", "session_id", default=None, help="Persist this chat session in SQLite.")
@click.pass_context
def chat(ctx: click.Context, stream: bool, session_id: str | None) -> None:
    config = _state(ctx).config
    with asyncio.Runner() as runner:
        try:
            router = build_router(config)
        except AxiomError as exc:
            raise click.ClickException(str(exc)) from exc
        agent = build_agent(
            config, router, approver=_approve, stream=stream, lessons=True, learn=True
        )
        store = SessionStore(paths.home_dir() / "sessions.sqlite3") if session_id else None
        if store is not None and session_id is not None:
            agent.history.extend(store.load_messages(session_id))
        click.echo(f"axiom chat ({config.agent.model}). {CHAT_HELP}", err=True)
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
                        store.save_messages(session_id, agent.history)
                    click.echo("conversation cleared", err=True)
                elif text == "/tools":
                    click.echo(", ".join(agent.registry.names()), err=True)
                elif text == "/usage":
                    usage = agent.usage
                    click.echo(f"in={usage.input_tokens} out={usage.output_tokens}", err=True)
                else:
                    try:
                        runner.run(_turn(agent, router, text))
                        if store is not None and session_id is not None:
                            store.save_messages(session_id, agent.history)
                    except AxiomError as exc:
                        click.echo(f"error: {exc}", err=True)
                    except KeyboardInterrupt:
                        click.echo("\ninterrupted", err=True)
        finally:
            runner.run(router.aclose())


@main.command("tui")
@click.pass_context
def tui_command(ctx: click.Context) -> None:
    from axiom.tui.app import run_tui

    try:
        run_tui(ctx)
    except AxiomError as exc:
        raise click.ClickException(str(exc)) from exc


def agent_main() -> None:
    main(args=[*sys.argv[1:], "tui"], prog_name="axagent")


@main.command("tools")
def tools_command() -> None:
    table = Table(title="tools")
    for column in ("Name", "Changes things", "Description"):
        table.add_column(column)
    for item in default_registry(include_capability_tools=True).all():
        table.add_row(item.name, "yes" if item.mutates else "no", item.description)
    Console().print(table)


@main.group("sessions")
def sessions_group() -> None:
    """Inspect and manage persisted chat sessions."""


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
    entries: list[str] = []
    for message in messages:
        content = " ".join(message.content.split())
        if message.role in {"user", "assistant"} and content:
            label = "User" if message.role == "user" else "Agent"
            entries.append(f"- {label}: {content[:220]}")
    summary = "\n".join(entries[-20:])[:6000]
    if not summary:
        raise click.ClickException("session has no summarizable text")
    store.save_session_summary(session_id, summary)
    click.echo(summary)


@main.group("learn")
def learn_group() -> None:
    """Review and consolidate locally stored learning data."""


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
    """Run small coding tasks against the configured model and score them with tests."""
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
        except AxiomError as exc:
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


@main.group("memory")
def memory_group() -> None:
    """Store and search local notes using SQLite full-text search."""


@memory_group.command("add")
@click.argument("content")
@click.option("--tag", "tags", default="", help="Comma-separated tags.")
@click.option("--category", default="general", help="Memory category, e.g. preference or project.")
def memory_add(content: str, tags: str, category: str) -> None:
    memory_id = SessionStore(paths.home_dir() / "sessions.sqlite3").add_memory(
        content, tags, category=category
    )
    click.echo(f"saved memory {memory_id}")


@memory_group.command("search")
@click.argument("query")
@click.option("--limit", type=click.IntRange(1, 100), default=10)
def memory_search(query: str, limit: int) -> None:
    results = SessionStore(paths.home_dir() / "sessions.sqlite3").search_memories(query, limit)
    for item in results:
        click.echo(f"#{item['id']} [{item['category']}] [{item['tags']}] {item['content']}")
    if not results:
        click.echo("no matching memories")


@memory_group.command("list")
@click.option("--category", default=None, help="Only show one category, e.g. lesson.")
@click.option("--limit", type=click.IntRange(1, 500), default=50)
def memory_list(category: str | None, limit: int) -> None:
    store = SessionStore(paths.home_dir() / "sessions.sqlite3")
    for item in store.list_memories(category=category, limit=limit):
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
