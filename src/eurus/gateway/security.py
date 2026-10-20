from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GatewayIdentity:
    user_id: int
    chat_id: int
    is_group: bool = False


class GatewayAccessPolicy:
    def __init__(
        self, allowed_user_ids: set[int], allowed_group_ids: set[int] | None = None
    ) -> None:
        self.allowed_user_ids = frozenset(allowed_user_ids)
        self.allowed_group_ids = frozenset(allowed_group_ids or set())
        if any(value <= 0 for value in self.allowed_user_ids):
            raise ValueError("Telegram user IDs must be positive integers")

    def permits(self, identity: GatewayIdentity) -> bool:
        if identity.user_id not in self.allowed_user_ids:
            return False
        return not (identity.is_group and identity.chat_id not in self.allowed_group_ids)


def parse_id_allowlist(value: str, *, allow_negative: bool = False) -> set[int]:
    if not value.strip():
        return set()
    result: set[int] = set()
    for part in value.split(","):
        item = part.strip()
        try:
            parsed = int(item)
        except ValueError as exc:
            raise ValueError("allowlist must contain numeric IDs") from exc
        if parsed == 0 or (parsed < 0 and not allow_negative):
            raise ValueError("allowlist contains an ID with an invalid sign")
        result.add(parsed)
    return result
