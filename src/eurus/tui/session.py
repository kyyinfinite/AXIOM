from __future__ import annotations

import asyncio
import inspect
import logging
import re
import signal
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from eurus import paths
from eurus.agent.events import AgentEvent
from eurus.agent.factory import build_agent
from eurus.catalog import resolve_api_key
from eurus.config import Config
from eurus.errors import EurusError
from eurus.mcp.manager import MCPToolManager
from eurus.memory.manager import MemoryManager
from eurus.models import Message, ModelRef, ToolCall, Usage
from eurus.providers.discovery import Discovery, discover
from eurus.providers.factory import build_router
from eurus.providers.policy import is_local_url
from eurus.security.approval import approval_request
from eurus.sessions import SessionStore
from eurus.skills import list_skills
from eurus.tools import Tool, ToolRegistry, default_registry
from eurus.tui.commands import COMMANDS, resolve, suggest
from eurus.tui.info import config_rows, credential_note, describe, shell_label
from eurus.tui.picker import (
    active_ref,
    filter_models,
    page_bounds,
    page_size,
    parse_choice,
    provider_rows,
    with_active_first,
)
from eurus.tui.renderer import ALLOW_ANSWERS, StreamView, TerminalRenderer
from eurus.tui.state import AgentState

log = logging.getLogger(__name__)

Handler = Callable[[str], Awaitable[None] | None]


@dataclass(slots=True)
class TurnStats:
    tool_calls: int = 0
    pending: int = 0
    spoke: bool = False
    usage: Usage = field(default_factory=Usage)


class SigintCancel:
    def __init__(self, task: asyncio.Task[None]) -> None:
        self._task = task
        self._armed = False

    def arm(self) -> None:
        if self._armed:
            return
        try:
            asyncio.get_running_loop().add_signal_handler(signal.SIGINT, self._task.cancel)
        except (NotImplementedError, RuntimeError, ValueError):
            return
        self._armed = True

    def disarm(self) -> None:
        if not self._armed:
            return
        asyncio.get_running_loop().remove_signal_handler(signal.SIGINT)
        self._armed = False


class InteractiveSession:
    def __init__(self, config: Config, renderer: TerminalRenderer | None = None) -> None:
        self.config = config
        self.renderer = renderer or TerminalRenderer()
        self.history: list[Message] = []
        self.session_id = str(uuid.uuid4())
        self.session_store = SessionStore(paths.home_dir() / "sessions.sqlite3")
        self.memory = MemoryManager(paths.home_dir() / "sessions.sqlite3")
        self.running = True
        self.state = AgentState.IDLE
        self.usage = Usage()
        self.retry_prompt = ""
        self.registry: ToolRegistry = default_registry(
            include_capability_tools=True, include_workflow_tools=True
        )
        self.mcp = MCPToolManager.load(paths.home_dir() / "mcp.json")
        self._mcp_registered = False
        self._cancel: SigintCancel | None = None
        self._view: StreamView | None = None
        self._catalog: dict[str, Discovery] = {}
        self._handlers: dict[str, Handler] = {
            "/help": self._help,
            "/model": self._model,
            "/models": self._models,
            "/provider": self._provider,
            "/skills": self._skills,
            "/memory": self._memory,
            "/sessions": self._sessions,
            "/resume": self._resume,
            "/tools": self._tools,
            "/status": self._status,
            "/config": self._config,
            "/workspace": self._workspace,
            "/compact": self._compact,
            "/retry": self._retry,
            "/clear": self._clear,
            "/exit": self._exit,
        }

    @property
    def workspace(self) -> Path:
        return Path(self.config.permissions.workspace).expanduser().resolve()

    def _help(self, _: str) -> None:
        self.renderer.palette(COMMANDS)
        self.renderer.console.print(self.renderer.shortcuts())

    def _ask(self, label: str) -> str | None:
        if self._cancel is not None:
            self._cancel.disarm()
        try:
            return self.renderer.ask(self.renderer.menu_prompt(label))
        except (KeyboardInterrupt, EOFError):
            return None
        finally:
            if self._cancel is not None:
                self._cancel.arm()

    def _switch(self, text: str) -> bool:
        try:
            ref = ModelRef.parse(text)
        except ValueError as exc:
            self.renderer.error(str(exc))
            return False
        if ref.provider not in self.config.providers:
            known = ", ".join(sorted(self.config.providers)) or "none"
            self.renderer.error(f"Unknown provider: {ref.provider}", f"Configured: {known}")
            return False
        self.config = replace(self.config, agent=replace(self.config.agent, model=str(ref)))
        self.renderer.system(f"Active model: {ref} (this session)")
        if describe(self.config).credential == "no key":
            self.renderer.warning("No API key found for this provider.")
        return True

    async def _discover(self, provider: str, force: bool = False) -> Discovery:
        if not force and provider in self._catalog:
            return self._catalog[provider]
        spec = self.config.providers[provider]
        if self.config.network.local_only and not is_local_url(spec.base_url):
            return Discovery(provider, error="blocked by network.local_only")
        self.renderer.system(f"Fetching models from {provider}... (Ctrl+C cancels)")
        result = await discover(spec, resolve_api_key(spec))
        if not result.error:
            self._catalog[provider] = result
        return result

    async def _model(self, argument: str) -> None:
        if "/" in argument:
            self._switch(argument)
            return
        if argument:
            name = argument.lower()
            if name not in self.config.providers:
                self.renderer.error(f"Unknown provider: {argument}")
                return
            await self._model_menu(name)
            return
        await self._provider_menu()

    async def _provider_menu(self) -> None:
        while True:
            rows = provider_rows(self.config)
            self.renderer.provider_page(rows, self.config.agent.model)
            answer = self._ask("provider>")
            if answer is None:
                return
            choice = parse_choice(answer, len(rows))
            if choice.kind == "quit":
                return
            if choice.kind == "select":
                row = rows[choice.index]
            else:
                found = [item for item in rows if item.name == choice.text.lower()]
                if not found:
                    self.renderer.error(f"Unknown provider or number: {choice.text}")
                    continue
                row = found[0]
            if not row.ready:
                reason = f"set {row.env} and restart Eurus" if row.env else row.note
                self.renderer.warning(f"{row.name} is unavailable: {reason}.")
                continue
            if await self._model_menu(row.name) != "back":
                return

    async def _model_menu(self, provider: str) -> str:
        discovery = await self._discover(provider)
        query = ""
        page = 0
        while True:
            ref = active_ref(self.config)
            active_id = ref.model if ref is not None and ref.provider == provider else ""
            models = filter_models(with_active_first(discovery.models, active_id), query)
            size = page_size(self.renderer.console.height)
            page, pages, start, end = page_bounds(len(models), page, size)
            visible = models[start:end]
            self.renderer.model_page(
                provider,
                visible,
                start,
                page,
                pages,
                len(models),
                active_id,
                query,
                discovery.error,
            )
            answer = self._ask("model>")
            if answer is None:
                return "quit"
            choice = parse_choice(answer, len(visible))
            if choice.kind == "quit":
                return "quit"
            if choice.kind == "back":
                return "back"
            if choice.kind == "next":
                page += 1
            elif choice.kind == "prev":
                page -= 1
            elif choice.kind == "refresh":
                discovery = await self._discover(provider, force=True)
                page = 0
            elif choice.kind == "select":
                if self._switch(f"{provider}/{visible[choice.index].id}"):
                    return "chosen"
            elif choice.text.startswith("="):
                if self._switch(f"{provider}/{choice.text[1:].strip()}"):
                    return "chosen"
            elif choice.text.isdigit():
                self.renderer.warning("Number is not on this page.")
            else:
                query = choice.text.lstrip("/")
                page = 0

    def _models(self, _: str) -> None:
        models = list(
            dict.fromkeys(
                model for model in (self.config.agent.model, *self.config.agent.fallback) if model
            )
        )
        rows = [("-", model) for model in models]
        self.renderer.table("Configured models", rows or [("-", "none")])

    def _provider(self, _: str) -> None:
        rows = [
            (name, credential_note(describe(self.config, f"{name}/x").credential))
            for name in sorted(self.config.providers)
        ]
        self.renderer.table("Providers", rows or [("-", "none configured")])

    def _skills(self, _: str) -> None:
        active = set(self.config.agent.skills)
        rows = [
            (skill.name, f"{self._skill_state(skill.name, active)} | {skill.description}")
            for skill in list_skills(self.workspace)
        ]
        self.renderer.table("Skills", rows or [("-", "none found")])

    @staticmethod
    def _skill_state(name: str, active: set[str]) -> str:
        return "active" if name in active else "available"

    def _memory(self, argument: str) -> None:
        action, _, rest = argument.partition(" ")
        action = action.lower() or "list"
        if action == "list":
            rows = self.memory.store.list_memories(limit=50)
            entries = [
                (str(row["id"]), f"[{row.get('memory_type', 'fact')}] {row['content'][:100]}")
                for row in rows
            ]
            self.renderer.table("Saved memories", entries or [("-", "No saved memories")])
        elif action == "search":
            rows = self.memory.recall(rest)
            entries = [
                (str(row["id"]), f"[{row.get('memory_type', 'fact')}] {row['content'][:100]}")
                for row in rows
            ]
            self.renderer.table("Memory search", entries or [("-", "No matches")])
        elif action == "add":
            memory_type, _, content = rest.partition(" ")
            valid_types = {"profile", "preference", "project_fact", "lesson", "episode", "fact"}
            if memory_type not in valid_types or not content.strip():
                self.renderer.system("Usage: /memory add <type> <content>")
                return
            answer = self._ask("Save this memory? [y/N]")
            if answer and answer.strip().lower() in {"y", "yes"}:
                memory_id = self.memory.store.add_memory(
                    content,
                    category="user",
                    memory_type=memory_type,
                    source="user",
                    confidence=1.0,
                )
                self.renderer.system(f"Memory saved: #{memory_id}")
            else:
                self.renderer.system("Memory not saved.")
        elif action in {"delete", "edit"}:
            identifier, _, content = rest.partition(" ")
            if not identifier.isdigit():
                suffix = " <new content>" if action == "edit" else ""
                self.renderer.system(f"Usage: /memory {action} <id>{suffix}")
                return
            memory_id = int(identifier)
            if action == "delete":
                answer = self._ask(f"Delete memory #{memory_id}? [y/N]")
                if answer and answer.strip().lower() in {"y", "yes"}:
                    message = (
                        "Memory deleted." if self.memory.forget(memory_id) else "Memory not found."
                    )
                    self.renderer.system(message)
                else:
                    self.renderer.system("Memory kept.")
            elif not content.strip():
                self.renderer.system("Usage: /memory edit <id> <new content>")
            else:
                answer = self._ask(f"Update memory #{memory_id}? [y/N]")
                if answer and answer.strip().lower() in {"y", "yes"}:
                    message = (
                        "Memory updated."
                        if self.memory.update(memory_id, content)
                        else "Memory not found."
                    )
                    self.renderer.system(message)
                else:
                    self.renderer.system("Memory unchanged.")
        else:
            self.renderer.system(
                "Usage: /memory [list|search <query>|add <type> <content>|"
                "edit <id> <content>|delete <id>]"
            )

    def _sessions(self, _: str) -> None:
        rows = self.session_store.list_sessions()
        entries = [
            (str(row["id"]), f"{row['title']} ({row['message_count']} messages)")
            for row in rows[:20]
        ]
        self.renderer.table("Saved sessions", entries or [("-", "No saved sessions")])

    def _resume(self, argument: str) -> None:
        sessions = self.session_store.list_sessions()
        if not sessions:
            self.renderer.system("No saved sessions.")
            return
        target = argument.strip()
        selected: dict[str, Any] | None
        if not target or target == "latest":
            selected = sessions[0]
        else:
            selected = next((item for item in sessions if item["id"] == target), None)
        if selected is None:
            self.renderer.error("Session not found. Use /sessions to list saved sessions.")
            return
        loaded = self.session_store.load_messages(str(selected["id"]))
        self.history = loaded
        self.session_id = str(selected["id"])
        self.usage = Usage()
        self.retry_prompt = ""
        self.renderer.system(f"Resumed: {selected['title']} ({len(loaded)} messages)")

    def _tools(self, _: str) -> None:
        rows = []
        for name in self.registry.names():
            tool = self.registry.get(name)
            if tool is not None:
                rows.append((name, tool.description))
        self.renderer.table("Tools", rows)

    def _status(self, _: str) -> None:
        info = describe(self.config)
        turns = sum(message.role == "user" for message in self.history)
        rows = [
            ("state", self.state.value),
            ("model", info.model),
            ("credential", credential_note(info.credential)),
            ("workspace", str(self.workspace)),
            ("turns in context", str(turns)),
            ("tools", str(len(self.registry.names()))),
            ("shell", shell_label(self.config)),
        ]
        if self.usage.total_tokens:
            rows.append(
                ("tokens", f"in {self.usage.input_tokens} / out {self.usage.output_tokens}")
            )
        self.renderer.table("Status", rows)

    def _config(self, _: str) -> None:
        self.renderer.table("Config", config_rows(self.config))

    def _workspace(self, _: str) -> None:
        self.renderer.system(str(self.workspace))

    def _compact(self, _: str) -> None:
        turns = [index for index, message in enumerate(self.history) if message.role == "user"]
        if len(turns) > 1:
            self.history = self.history[turns[-1] :]
        count = len(self.history)
        self.renderer.system(f"Context compacted to the latest turn ({count} messages).")

    def _retry(self, _: str) -> None:
        if not self.retry_prompt:
            self.renderer.system("Nothing to retry.")

    def _clear(self, _: str) -> None:
        self.history.clear()
        self.session_id = str(uuid.uuid4())
        self.usage = Usage()
        self.retry_prompt = ""
        self.renderer.clear()
        self.renderer.startup(self.config)
        self.renderer.system("Conversation cleared; project files unchanged.")

    def _exit(self, _: str) -> None:
        self.running = False
        self.renderer.system("Goodbye.")

    async def _command(self, text: str) -> str:
        name, _, argument = text.partition(" ")
        matches = resolve(name)
        if len(matches) == 1:
            command = matches[0].name
            if command == "/retry" and self.retry_prompt:
                return self.retry_prompt
            outcome = self._handlers[command](argument.strip())
            if inspect.isawaitable(outcome):
                await outcome
        elif matches:
            self.renderer.palette(matches)
        else:
            close = ", ".join(command.name for command in suggest(name))
            hint = f"Did you mean: {close}" if close else "Use /help to see available commands."
            self.renderer.error(f"Unknown command: {name}", hint)
        return ""

    def _set_state(self, state: AgentState) -> None:
        self.state = state
        if self._view is not None:
            self._view.state = state

    async def run_cancellable(self, prompt: str) -> None:
        task = asyncio.create_task(self.submit(prompt))
        self._cancel = SigintCancel(task)
        self._cancel.arm()
        try:
            await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
        finally:
            self._cancel.disarm()
            self._cancel = None

    async def submit(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt:
            self.renderer.status("Input is empty; enter a task or /help.")
            return
        if prompt.startswith("/"):
            prompt = await self._command(prompt)
            if not prompt:
                return
        if not self.config.agent.model:
            self.renderer.error(
                "No model configured.",
                "Run `eurus config init` and set [agent].model, or use /model provider/name.",
            )
            return
        if self._is_explicit_correction(prompt):
            self.session_store.record_event("correction_proposed", session_id=self.session_id)
            answer = self._ask("This sounds like a lasting preference. Save it to memory? [y/N]")
            accepted = bool(answer and answer.strip().lower() in {"y", "yes"})
            self.session_store.record_event(
                "correction_accepted" if accepted else "correction_rejected",
                session_id=self.session_id,
            )
            if accepted:
                try:
                    memory_id = self.memory.store.add_memory(
                        prompt,
                        category="user",
                        memory_type="preference",
                        source="explicit_correction",
                        confidence=0.8,
                    )
                    self.renderer.system(f"Preference saved: #{memory_id}")
                except ValueError as exc:
                    self.renderer.warning(str(exc))
        await self._turn(prompt)

    @staticmethod
    def _is_explicit_correction(prompt: str) -> bool:
        patterns = (
            r"\b(jangan|hindari|tidak boleh|selalu|mulai sekarang|ke depannya|"
            r"saya lebih suka|saya maunya)\b",
            r"\b(don't|do not|never|always|from now on|going forward|i prefer|please avoid)\b",
        )
        return any(re.search(pattern, prompt, re.IGNORECASE) for pattern in patterns)

    def _flush(self, view: StreamView, stats: TurnStats) -> None:
        text = view.take()
        if text.strip():
            self.renderer.assistant(text, label=not stats.spoke)
            stats.spoke = True

    def _consume(self, event: AgentEvent, view: StreamView, stats: TurnStats) -> None:
        if event.kind == "step":
            self._set_state(AgentState.THINKING)
        elif event.kind == "text":
            view.feed(event.text)
        elif event.kind == "tool_call":
            self._flush(view, stats)
            stats.tool_calls += 1
            stats.pending += 1
            self._set_state(AgentState.RUNNING)
            self.renderer.activity(event)
        elif event.kind == "tool_result":
            stats.pending = max(stats.pending - 1, 0)
            self.renderer.activity(event)
            self._set_state(AgentState.RUNNING if stats.pending else AgentState.THINKING)
        elif event.kind == "done":
            self._flush(view, stats)
            if event.usage is not None:
                stats.usage = event.usage
            self.renderer.activity(event)
        else:
            self.renderer.activity(event)

    async def close(self) -> None:
        await self.mcp.close()

    async def _turn(self, prompt: str) -> None:
        started = time.monotonic()
        router = None
        stats = TurnStats()
        self.renderer.user(prompt)
        self._set_state(AgentState.THINKING)
        try:
            router = build_router(self.config)
            if not self._mcp_registered:
                await self.mcp.connect()
                self.mcp.register_tools(self.registry)
                self._mcp_registered = True
            agent = build_agent(
                self.config,
                router,
                registry=self.registry,
                approver=self._approve,
                lessons=True,
                learn=True,
            )
            agent.history.extend(self.history)
            saved_summary = self.session_store.load_session_summary(self.session_id)
            if saved_summary and agent.history:
                agent.history[0] = replace(
                    agent.history[0],
                    content=(
                        f"{agent.history[0].content}\n\n"
                        "Saved summary of this session (context only; verify details in history):\n"
                        f"{saved_summary}"
                    ),
                )
            with self.renderer.generating() as view:
                self._view = view
                try:
                    async for event in agent.run(prompt):
                        self._consume(event, view, stats)
                finally:
                    self._flush(view, stats)
                    self._view = None
            self.history = agent.history[1:]
            self.session_store.save_messages(self.session_id, agent.history)
            self.usage += stats.usage
            self.retry_prompt = ""
            self._set_state(AgentState.IDLE)
            self.renderer.turn_summary(time.monotonic() - started, stats.tool_calls, stats.usage)
        except asyncio.CancelledError:
            self.retry_prompt = prompt
            self._set_state(AgentState.IDLE)
            self.renderer.cancelled()
            raise
        except (EurusError, ValueError, KeyError) as exc:
            self._set_state(AgentState.ERROR)
            self.renderer.error(str(exc), "You can edit the request and try again.")
        except Exception as exc:
            log.debug("unexpected failure", exc_info=True)
            self._set_state(AgentState.ERROR)
            self.renderer.error(f"Unexpected error: {type(exc).__name__}: {exc}")
        finally:
            self._view = None
            if router is not None:
                await router.aclose()

    async def _approve(self, call: ToolCall, tool: Tool) -> bool:
        request = approval_request(call, tool)
        previous = self.state
        self._set_state(AgentState.APPROVAL)
        self.renderer.pause_live()
        unisolated = call.name == "shell" and self.config.sandbox.backend == "host"
        note = "Runs directly on this device; not isolated." if unisolated else ""
        self.renderer.approval(request, call.arguments, note)
        if self._cancel is not None:
            self._cancel.disarm()
        answer = ""
        interrupted = False
        try:
            answer = self.renderer.ask(self.renderer.approval_prompt())
        except KeyboardInterrupt:
            interrupted = True
        except EOFError:
            answer = ""
        finally:
            if self._cancel is not None:
                self._cancel.arm()
            self.renderer.resume_live()
            self._set_state(previous)
        allowed = not interrupted and answer.strip().lower() in ALLOW_ANSWERS
        self.renderer.approval_result(allowed)
        self.session_store.record_event(
            "approval_accepted" if allowed else "approval_rejected",
            session_id=self.session_id,
            metadata={"tool": call.name, "mutating": tool.mutates},
        )
        if interrupted:
            raise asyncio.CancelledError
        return allowed
