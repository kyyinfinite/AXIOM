from importlib import import_module
from typing import Any

__all__ = [
    "AgentEngine",
    "BudgetSpec",
    "BudgetTracker",
    "ConcurrentRunUpdate",
    "Database",
    "EventStore",
    "InvalidRunTransition",
    "RecoveryAction",
    "Run",
    "RunResult",
    "RunStatus",
    "RunStore",
    "RuntimeEvent",
    "classify_recovery",
    "transition_run",
]

_MODULES = {
    "AgentEngine": "eurus.runtime.engine",
    "RunResult": "eurus.runtime.engine",
    "BudgetSpec": "eurus.runtime.budgets",
    "BudgetTracker": "eurus.runtime.budgets",
    "Run": "eurus.runtime.models",
    "RunStatus": "eurus.runtime.models",
    "RuntimeEvent": "eurus.runtime.models",
    "RecoveryAction": "eurus.runtime.recovery",
    "classify_recovery": "eurus.runtime.recovery",
    "InvalidRunTransition": "eurus.runtime.state_machine",
    "transition_run": "eurus.runtime.state_machine",
    "ConcurrentRunUpdate": "eurus.storage.runs",
    "RunStore": "eurus.storage.runs",
    "Database": "eurus.storage.database",
    "EventStore": "eurus.storage.events",
}


def __getattr__(name: str) -> Any:
    module_name = _MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)
