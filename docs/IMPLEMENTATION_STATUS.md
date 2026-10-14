# Implementation status

## Batch 1: approval and sandbox foundation

Implemented:
- Mutating tools fail closed when no approval callback is configured.
- Tool metadata supports explicit approval requirements.
- Approval requests include a SHA-256 fingerprint for tool name and arguments.
- Shell commands execute through a container runtime with no host fallback, network disabled by default, read-only root, dropped capabilities, resource limits, and timeout handling.
- CLI `run` and `chat` request approval before mutating actions.
- Sandbox settings are validated through TOML configuration.
- A JSONL audit journal API is available.

## Batch 2: concurrent tool execution and compatibility fixes

Implemented:
- Independent calls from one model response execute concurrently, with a configurable cap.
- Results and tool messages are restored to original call order and correlated by `tool_call_id`.
- Mutating tool approvals are collected before concurrent execution begins.
- Cancellation and `BaseException` paths append a result message for every call before propagating interruption.
- `tools.max_parallel_tools` is validated to the range 1–32.
- Removed the obsolete duplicate `src/axiom/agent.py` module.
- Updated the TUI to use the current Agent/Router APIs.
- Fixed provider registry imports and construction to use the current provider class and transport APIs.
- Updated stale provider/TUI tests to cover the active architecture.

## Validation

Latest recorded local run before task orchestration: `PYTHONPATH=src python -m pytest -q` — 253 passed after task execution tests were added.

`python -m compileall -q src tests` passed.

Ruff and mypy are not installed in this execution environment, and package installation could not download build dependencies because network access was unavailable. Docker/container-backed end-to-end execution has not been validated here.

## Not implemented yet

- MCP client and server management.
- Worker-agent orchestration and persistent background jobs.
- Persistent sessions and journal wiring into every execution path.
- Remote MCP egress controls.


## Agent capabilities capability milestone

Implemented in this branch: bounded `TaskPlanner`, `MemoryManager` with additive SQLite schema migration, coding tools (`list_files`, `read_file`, `search_code`, `propose_patch`, `apply_patch`), and memory tools (`recall_memory`, `remember_memory`, `forget_memory`). Agent factory enables Agent capabilities tools by default; `default_registry()` itself retains the original two-tool default for compatibility. See [AGENT_CAPABILITIES.md](AGENT_CAPABILITIES.md).


## Task execution and verification

Implemented: sequential plan execution, per-step verification, bounded retries, timeout handling, cancellation state, plan/state matching, and JSON-serializable state snapshots. This is an orchestration primitive; model-driven repair planning and durable job persistence remain application responsibilities. See [TASK_EXECUTION.md](TASK_EXECUTION.md).


## Provider hardening (0.0.8)

Implemented: 403 classification (`edge_blocked`, `model_blocked`, `region_blocked`, `forbidden`), explicit User-Agent with one Cloudflare fallback, per-provider `user_agent` and `headers`, embedded-error and empty-stream detection, adaptive `max_tokens` and `temperature`, NVIDIA NIM provider, optional `groq-sdk` and `genai-sdk` protocols, `axiom diagnose`, restricted `.env` loading. Not verified against live services or real SDK packages in the build environment. See [PROVIDERS.md](PROVIDERS.md).
