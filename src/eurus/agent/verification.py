from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class VerificationResult:
    passed: bool
    summary: str


class StepVerifier(Protocol):
    async def __call__(self, title: str, success_criteria: str, output: str) -> VerificationResult:
        """Check a step's output against its stated success criteria."""
