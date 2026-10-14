from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from axiom import __version__
from axiom.agent.events import AgentEvent
from axiom.agent.factory import build_agent
from axiom.config import Config
from axiom.errors import AxiomError
from axiom.models import Message, ModelRef, ToolCall
from axiom.providers.factory import build_router
from axiom.tools import Tool, ToolRegistry, default_registry


class TerminalRenderer:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def startup(self, config: Config) -> None:
        model = config.agent.model or "not configured"
        try:
            provider = ModelRef.parse(model).provider if config.agent.model else "not configured"
        except ValueError:
            provider = "invalid model"
        title = Text("AXIOM", style="bold white")
        title.append(f"  v{__version__}", style="dim")
        description = (
            "[bold]Interactive coding agent[/bold]\n"
            f"[dim]Provider[/dim]  {provider}   [dim]Model[/dim]  {model}\n"
            f"[dim]Workspace[/dim]  {Path(config.permissions.workspace).expanduser()}\n"
            "[dim]AGENT TERMINAL · Enter: submit · /help: commands · "
            "Ctrl+C: cancel/exit · /exit: quit[/dim]"
        )
        self.console.print(Panel(description, title=title, border_style="blue", padding=(0, 1)))

    def user(self, prompt: str) -> None:
        self.console.print("\n[bold cyan]YOU[/bold cyan]")
        self.console.print(prompt, soft_wrap=True)

    def assistant(self, content: str) -> None:
        self.console.print("\n[bold white]AXIOM[/bold white]")
        self.console.print(Markdown(content, code_theme="monokai"))

    def activity(self, event: AgentEvent) -> None:
        if event.kind == "tool_call" and event.tool_call is not None:
            args = event.tool_call.arguments
            sensitive_tokens = ("key", "token", "password", "secret", "cookie", "authorization")
            redacted = {
                key: (
                    "REDACTED" if any(token in key.lower() for token in sensitive_tokens) else value
                )
                for key, value in args.items()
            }
            summary = json.dumps(redacted, ensure_ascii=False)
            self.console.print(
                f"  [blue]→[/blue] [bold]{event.tool_call.name}[/bold] [dim]{summary[:180]}[/dim]"
            )
        elif event.kind == "tool_result" and event.tool_result is not None:
            result = event.tool_result
            marker = "[red]✗[/red]" if result.is_error else "[green]✓[/green]"
            detail = result.output.replace("\n", " ")[:180]
            self.console.print(f"  {marker} {result.name} [dim]{detail}[/dim]")
        elif event.kind == "done" and event.stop_reason == "max_steps":
            self.console.print("[yellow]Agent step limit reached.[/yellow]")

    def status(self, message: str) -> None:
        self.console.print(f"[dim]{message}[/dim]")

    def error(self, message: str, detail: str = "") -> None:
        body = message + (f"\n[dim]{detail}[/dim]" if detail else "")
        self.console.print(Panel(body, title="Request failed", border_style="red"))


Handler = Callable[[str], bool]


class InteractiveSession:
    def __init__(self, config: Config, renderer: TerminalRenderer | None = None) -> None:
        self.config = config
        self.renderer = renderer or TerminalRenderer()
        self.history: list[Message] = []
        self.running = True
        self.registry: ToolRegistry = default_registry()
        self.commands: dict[str, tuple[str, Handler]] = {
            "/help": ("Show available commands", self._help),
            "/clear": ("Clear visible conversation and context", self._clear),
            "/exit": ("Exit AXIOM", self._exit),
            "/quit": ("Exit AXIOM", self._exit),
            "/model": ("Show active provider and model", self._model),
            "/models": ("List configured model choices", self._models),
            "/status": ("Show session and workspace status", self._status),
            "/tools": ("List registered tools", self._tools),
            "/workspace": ("Show active workspace", self._workspace),
            "/compact": ("Trim old conversation context", self._compact),
        }

    def _help(self, _: str) -> bool:
        rows = "\n".join(
            f"[bold cyan]{key}[/bold cyan]  {value[0]}" for key, value in self.commands.items()
        )
        self.renderer.console.print(Panel(rows, title="Commands", border_style="dim"))
        return True

    def _clear(self, _: str) -> bool:
        self.history.clear()
        self.renderer.console.clear()
        self.renderer.startup(self.config)
        self.renderer.status("Conversation cleared; project files unchanged.")
        return True

    def _exit(self, _: str) -> bool:
        self.running = False
        self.renderer.status("Goodbye.")
        return True

    def _model(self, _: str) -> bool:
        self.renderer.status(f"Provider/model: {self.config.agent.model or 'not configured'}")
        return True

    def _models(self, _: str) -> bool:
        models = list(
            dict.fromkeys(
                model for model in (self.config.agent.model, *self.config.agent.fallback) if model
            )
        )
        content = "\n".join(models) or "No models configured."
        self.renderer.console.print(Panel(content, title="Configured models", border_style="dim"))
        return True

    def _status(self, _: str) -> bool:
        turns = sum(message.role == "user" for message in self.history)
        workspace = Path(self.config.permissions.workspace).expanduser()
        content = (
            f"Session: {'active' if self.running else 'stopping'}\n"
            f"Turns in context: {turns}\n"
            f"Model: {self.config.agent.model or 'not configured'}\n"
            f"Workspace: {workspace}\n"
            f"Tools: {len(self.registry.names())}"
        )
        self.renderer.console.print(Panel(content, title="Status", border_style="dim"))
        return True

    def _tools(self, _: str) -> bool:
        for name in self.registry.names():
            tool = self.registry.get(name)
            if tool is not None:
                self.renderer.console.print(
                    f"[bold cyan]{name}[/bold cyan]  [dim]{tool.description}[/dim]"
                )
        return True

    def _workspace(self, _: str) -> bool:
        workspace = Path(self.config.permissions.workspace).expanduser().resolve()
        self.renderer.status(str(workspace))
        return True

    def _compact(self, _: str) -> bool:
        user_turns = [index for index, message in enumerate(self.history) if message.role == "user"]
        if len(user_turns) > 1:
            self.history = self.history[user_turns[-1] :]
        self.renderer.status(
            f"Context compacted to the latest complete user turn ({len(self.history)} messages)."
        )
        return True

    async def submit(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt:
            self.renderer.status("Input is empty; enter a task or /help.")
            return
        if prompt.startswith("/"):
            command, _, argument = prompt.partition(" ")
            entry = self.commands.get(command.lower())
            if entry is None:
                self.renderer.error(
                    f"Unknown command: {command}", "Use /help to see available commands."
                )
            else:
                entry[1](argument.strip())
            return
        if not self.config.agent.model:
            self.renderer.error(
                "No model configured.", "Run `axiom config init` and set [agent].model."
            )
            return

        self.renderer.user(prompt)
        started = time.monotonic()
        router = None
        try:
            router = build_router(self.config)
            agent = build_agent(
                self.config,
                router,
                registry=self.registry,
                approver=self._approve,
                lessons=True,
            )
            agent.history.extend(self.history)
            with self.renderer.console.status("Waiting for model…", spinner="dots"):
                async for event in agent.run(prompt):
                    self.renderer.activity(event)
            self.history = agent.history[1:]
            final = next(
                (
                    message.content
                    for message in reversed(agent.history)
                    if message.role == "assistant" and message.content
                ),
                "",
            )
            self.renderer.assistant(final)
            tool_calls = sum(len(message.tool_calls) for message in agent.history)
            self.renderer.status(
                f"Completed in {time.monotonic() - started:.1f}s · {tool_calls} tool call(s)"
            )
        except (AxiomError, ValueError, KeyError) as exc:
            self.renderer.error(str(exc))
        finally:
            if router is not None:
                await router.aclose()

    async def _approve(self, call: ToolCall, tool: Tool) -> bool:
        from axiom.security.approval import approval_request

        request = approval_request(call, tool)
        self.renderer.console.print(f"\nApproval required: {request.tool_name}")
        self.renderer.console.print(request.summary)
        answer = self.renderer.console.input("Allow this action? [y/N] ").strip().lower()
        return answer in {"y", "yes"}


async def chat_async(ctx: click.Context) -> None:
    config: Config = ctx.find_root().obj.config
    session = InteractiveSession(config)
    session.renderer.startup(config)
    while session.running:
        try:
            prompt = session.renderer.console.input("[bold blue]›[/bold blue] ")
        except (KeyboardInterrupt, EOFError):
            session.renderer.console.print()
            session.renderer.status("Goodbye.")
            break
        await session.submit(prompt)
