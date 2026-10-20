from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from eurus.errors import ModelRefError

Role = Literal["system", "user", "assistant", "tool"]
ROLES: tuple[str, ...] = ("system", "user", "assistant", "tool")


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "arguments": dict(self.arguments),
        }
        if self.extra:
            data["extra"] = dict(self.extra)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            arguments=dict(data.get("arguments") or {}),
            extra=dict(data.get("extra") or {}),
        )


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"invalid role: {self.role!r}")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool message requires tool_call_id")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only assistant messages may carry tool_calls")

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: tuple[ToolCall, ...] = ()) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, call_id: str, content: str, name: str | None = None) -> Message:
        return cls(role="tool", content=content, tool_call_id=call_id, name=name)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            data["name"] = self.name
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            role=data["role"],
            content=str(data.get("content", "")),
            tool_calls=tuple(ToolCall.from_dict(item) for item in data.get("tool_calls", ())),
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    name: str
    output: str
    is_error: bool = False

    def to_message(self) -> Message:
        return Message.tool(self.call_id, self.output, self.name)


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
        )


@dataclass(frozen=True, slots=True)
class ModelRef:
    provider: str
    model: str

    @classmethod
    def parse(cls, text: str) -> ModelRef:
        provider, separator, model = text.strip().partition("/")
        provider = provider.strip()
        model = model.strip()
        if not separator or not provider or not model:
            raise ModelRefError(f"model must look like 'provider/model', got {text!r}")
        return cls(provider=provider, model=model)

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"
