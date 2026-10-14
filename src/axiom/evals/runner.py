from __future__ import annotations

import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from axiom.agent.factory import build_agent
from axiom.coding.validation import run_validation
from axiom.config import Config
from axiom.errors import AxiomError
from axiom.evals.tasks import EvalTask
from axiom.execution.host import scrubbed_env
from axiom.models import ToolCall
from axiom.providers.router import Router
from axiom.tools.base import Tool
from axiom.tools.builtin import default_registry

CHECK_TIMEOUT = 120.0


@dataclass(frozen=True, slots=True)
class EvalResult:
    task: str
    passed: bool
    steps: int
    tokens: int
    seconds: float
    stop_reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


async def _allow(call: ToolCall, tool: Tool) -> bool:
    return True


def write_files(root: Path, files: dict[str, str]) -> None:
    for name, content in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


async def run_task(
    task: EvalTask, config: Config, router: Router, *, lessons: bool = False
) -> EvalResult:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"axiom-eval-{task.name}-") as tmp:
        root = Path(tmp)
        write_files(root, task.files)
        agent = build_agent(
            config,
            router,
            registry=default_registry(include_workflow_tools=True),
            approver=_allow,
            workspace=root,
            stream=False,
            shell=False,
            lessons=lessons,
        )
        steps = 0
        stop_reason = "error"
        detail = ""
        try:
            async for event in agent.run(task.prompt):
                if event.kind == "step":
                    steps = event.step
                elif event.kind == "done":
                    stop_reason = event.stop_reason
        except AxiomError as exc:
            detail = str(exc)
        result = run_validation(root, task.check, timeout_seconds=CHECK_TIMEOUT, env=scrubbed_env())
        if not result.passed and not detail:
            detail = (result.stdout + result.stderr).strip()[-300:]
        return EvalResult(
            task.name,
            result.passed,
            steps,
            agent.usage.total_tokens,
            time.monotonic() - started,
            stop_reason,
            detail,
        )
