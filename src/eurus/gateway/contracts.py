from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class GatewayMessage:
    platform: str
    user_id: str
    conversation_id: str
    message_id: str
    text: str
    is_group: bool = False


class GatewayAdapter(Protocol):
    @property
    def platform(self) -> str: ...

    async def send_text(self, conversation_id: str, text: str) -> None: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...
