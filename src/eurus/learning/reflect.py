from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from eurus.learning.lessons import LessonStore
from eurus.models import Message, ToolCall
from eurus.providers.router import Router
from eurus.tools.base import Tool, ToolContext

MAX_LESSONS_PER_RUN = 3
MIN_LENGTH = 15
MAX_LENGTH = 200
SECRET = re.compile(r"(sk-|gsk_|nvapi-|AIza|ghp_|xox[bp]-)[\w-]{10,}|[A-Za-z0-9_-]{32,}")

INSTRUCTIONS = (
    "You review a finished agent run and extract lessons for future tasks. Return only a JSON "
    f"array of at most {MAX_LESSONS_PER_RUN} strings. Each lesson is a general, reusable rule "
    f"about tools, workflow, or this kind of project, under {MAX_LENGTH} characters. Do not "
    "include secrets, file contents, personal data, or one-off facts. Return [] when nothing "
    "is worth keeping."
)

Approver = Callable[[ToolCall, Tool], Awaitable[bool]]


async def _unused(arguments: dict[str, Any], context: ToolContext) -> str:
    raise RuntimeError("save_lesson is only used for approval")


SAVE_LESSON = Tool(
    "save_lesson",
    "Save a lesson learned from this task so it can be used in future runs.",
    {"type": "object", "properties": {"lesson": {"type": "string"}}},
    _unused,
    mutates=True,
    requires_approval=True,
)


def transcript_text(task: str, messages: Sequence[Message]) -> str:
    lines = [f"Task: {task[:500]}", "Steps:"]
    results = {item.tool_call_id: item.content for item in messages if item.role == "tool"}
    final = ""
    for message in messages:
        if message.role != "assistant":
            continue
        final = message.content or final
        for call in message.tool_calls:
            shown = json.dumps(call.arguments, ensure_ascii=False, default=str)[:120]
            outcome = results.get(call.id, "")
            status = outcome[:200].replace("\n", " ") if outcome.startswith("error") else "ok"
            lines.append(f"- {call.name} {shown} -> {status}")
    lines.append(f"Final answer: {final[:500]}")
    return "\n".join(lines)


def parse_lessons(text: str) -> list[str]:
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    lessons: list[str] = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, str):
            continue
        lesson = " ".join(item.split())
        if MIN_LENGTH <= len(lesson) <= MAX_LENGTH and not SECRET.search(lesson):
            lessons.append(lesson)
    return lessons[:MAX_LESSONS_PER_RUN]


class Reflector:
    def __init__(
        self, router: Router, store: LessonStore, mode: str, *, min_tool_calls: int = 2
    ) -> None:
        self.router = router
        self.store = store
        self.mode = mode
        self.min_tool_calls = min_tool_calls

    async def learn(
        self, task: str, transcript: Sequence[Message], approver: Approver | None
    ) -> list[str]:
        used = sum(len(message.tool_calls) for message in transcript)
        if self.mode == "off" or used < self.min_tool_calls:
            return []
        response = await self.router.chat(
            [Message.system(INSTRUCTIONS), Message.user(transcript_text(task, transcript))]
        )
        saved: list[str] = []
        for number, lesson in enumerate(parse_lessons(response.message.content), start=1):
            if self.store.is_duplicate(lesson):
                continue
            if self.mode == "ask":
                if approver is None:
                    continue
                call = ToolCall(f"lesson-{number}", "save_lesson", {"lesson": lesson})
                if not await approver(call, SAVE_LESSON):
                    continue
            if self.store.add(lesson) is not None:
                saved.append(lesson)
        return saved
