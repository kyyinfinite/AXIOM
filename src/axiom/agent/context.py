from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace

from axiom.models import Message
from axiom.tools.base import truncate_text

CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD = 8
MIN_TOOL_CHARS = 200


def estimate_tokens(message: Message) -> int:
    size = len(message.content)
    for call in message.tool_calls:
        size += len(call.name) + len(json.dumps(call.arguments, default=str))
    return size // CHARS_PER_TOKEN + MESSAGE_OVERHEAD


def _blocks(messages: Sequence[Message]) -> list[list[Message]]:
    blocks: list[list[Message]] = []
    for message in messages:
        if message.role == "tool" and blocks and blocks[-1][0].tool_calls:
            blocks[-1].append(message)
        else:
            blocks.append([message])
    return blocks


def _shrink(block: list[Message], limit: int) -> list[Message]:
    return [
        replace(item, content=truncate_text(item.content, limit))
        if item.role == "tool" and len(item.content) > limit
        else item
        for item in block
    ]


class ContextWindow:
    def __init__(
        self, max_tokens: int, *, keep_recent: int = 6, tool_output_chars: int = 2000
    ) -> None:
        if max_tokens < 1 or keep_recent < 1 or tool_output_chars < MIN_TOOL_CHARS:
            raise ValueError("invalid context window settings")
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.tool_output_chars = tool_output_chars

    def cost(self, messages: Sequence[Message]) -> int:
        return sum(estimate_tokens(item) for item in messages)

    def fit(self, history: Sequence[Message]) -> list[Message]:
        messages = list(history)
        if self.cost(messages) <= self.max_tokens:
            return messages
        lead_size = 0
        while lead_size < len(messages) and messages[lead_size].role == "system":
            lead_size += 1
        lead = messages[:lead_size]
        blocks = _blocks(messages[lead_size:])
        dropped = 0
        rolling_summary: list[str] = []

        def summarize(blocks_to_summarize: Sequence[Sequence[Message]]) -> None:
            for block in blocks_to_summarize:
                for item in block:
                    text = " ".join(item.content.split())
                    if item.role == "user" and text:
                        rolling_summary.append(f"User: {text[:180]}")
                    elif item.role == "assistant" and text and not item.tool_calls:
                        rolling_summary.append(f"Agent: {text[:180]}")
            del rolling_summary[:-8]

        def render() -> list[Message]:
            head = list(lead)
            if dropped and head:
                note = f"[{dropped} earlier messages were omitted to fit the context window]"
                if rolling_summary:
                    summary_text = "\n".join(rolling_summary)
                    note += "\nEarlier context summary (approximate):\n" + summary_text
                head[0] = replace(head[0], content=f"{head[0].content}\n\n{note}")
            return head + [item for block in blocks for item in block]

        def fits() -> bool:
            return self.cost(render()) <= self.max_tokens

        def user_starts() -> list[int]:
            return [index for index, block in enumerate(blocks) if block[0].role == "user"]

        protected = max(0, len(blocks) - self.keep_recent)
        for index in range(protected):
            blocks[index] = _shrink(blocks[index], self.tool_output_chars)

        while not fits():
            starts = user_starts()
            cut = starts[0] if starts and starts[0] > 0 else (starts[1] if len(starts) > 1 else 0)
            if cut == 0:
                break
            summarize(blocks[:cut])
            dropped += sum(len(block) for block in blocks[:cut])
            del blocks[:cut]

        while not fits():
            starts = user_starts()
            anchor = starts[-1] if starts else -1
            if len(blocks) - anchor <= 2:
                break
            dropped += len(blocks[anchor + 1])
            del blocks[anchor + 1]

        if not fits():
            limit = max(MIN_TOOL_CHARS, self.tool_output_chars // 4)
            blocks[:] = [_shrink(block, limit) for block in blocks]
        return render()
