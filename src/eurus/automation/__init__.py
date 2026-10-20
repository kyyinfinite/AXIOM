from typing import Any

__all__ = ["AutomationWorkerPool"]


def __getattr__(name: str) -> Any:
    if name == "AutomationWorkerPool":
        from eurus.automation.worker import AutomationWorkerPool

        return AutomationWorkerPool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
