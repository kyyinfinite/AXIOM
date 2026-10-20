from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

from eurus.models import ToolCall
from eurus.tools.base import Tool


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    request_id: str
    tool_name: str
    action_fingerprint: str
    summary: str
    risk_level: str = "high"


class ApprovalProvider(Protocol):
    async def request_approval(self, request: ApprovalRequest) -> bool: ...


def action_fingerprint(call: ToolCall) -> str:
    payload = json.dumps(
        {"name": call.name, "arguments": call.arguments},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def approval_request(call: ToolCall, tool: Tool) -> ApprovalRequest:
    summary = json.dumps(call.arguments, sort_keys=True, ensure_ascii=False, default=str)
    return ApprovalRequest(
        request_id=call.id,
        tool_name=tool.name,
        action_fingerprint=action_fingerprint(call),
        summary=summary,
        risk_level="high" if tool.requires_approval else "medium",
    )
