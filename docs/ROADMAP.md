# Roadmap

| Area | Scope | Status |
|---|---|---|
| Repository | Packaging, license, CI, documentation, clean-room policy | Done |
| Core | Configuration, logging, types, CLI foundation | Done |
| Providers | OpenAI-compatible, Anthropic, Gemini, streaming, retry, fallback | Done |
| Agent runtime | Agent loop, tool registry, approval hooks, concurrent tool execution | Implemented |
| Learning | Lessons from finished tasks, approval, prompt injection, eval harness | Implemented; skills pending |
| Sandbox | Container isolation and resource limits | Container isolation implemented; opt-in host mode for Termux |
| Sessions and memory | SQLite sessions and FTS5-backed memory | Implemented |
| Coding workflow | Workspace tools, planning, patch previews, multi-file patchsets | Implemented |
| Skills and routing | Skills and hybrid local/cloud model routing | Planned |
| MCP | STDIO and remote HTTP clients | Planned |
| Orchestration | Bounded sequential task execution and verification primitives | Implemented; model-driven orchestration and durable jobs remain planned |
| Release readiness | Evaluation, hardening, cross-platform validation, v0.1.0 | In progress |


## Agent capabilities

The agent includes bounded task planning, categorized SQLite/FTS5 memory management, workspace-confined coding tools, patch previews, stale-content checks, and approval metadata for mutations. See [AGENT_CAPABILITIES.md](AGENT_CAPABILITIES.md).
