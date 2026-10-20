<p align="center">
  <img src=".github/asset/eurus-banner.jpg" alt="EURUS — Autonomous AI Agent" width="100%">
</p>

<h1 align="center">EURUS</h1>

<p align="center">
  <strong>Provider-agnostic autonomous AI coding agent</strong><br>
  Shell, files, patches, tests, persistent memory, skills, and MCP workflows — from your terminal.
</p>

<p align="center">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="License MIT" src="https://img.shields.io/badge/License-MIT-blue.svg">
  <img alt="Status Pre--Alpha" src="https://img.shields.io/badge/Status-Pre--Alpha-orange">
</p>

<p align="center">
  <a href="#goals">Goals</a> ·
  <a href="#install">Install</a> ·
  <a href="#termux">Termux</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#agent">Agent</a> ·
  <a href="#development">Development</a>
</p>

> **Project status:** Pre-alpha. Features and interfaces may change. See the [roadmap](docs/ROADMAP.md).

## Overview

EURUS is a provider-agnostic AI coding agent designed to work in your terminal. It connects to OpenAI, Anthropic, Gemini, Groq, NVIDIA NIM, and OpenAI-compatible endpoints. It can work with shell commands, files, patches, tests, web research, persistent memory, reusable skills, and optional Model Context Protocol (MCP) integrations.

<p align="center">
  <img src=".github/asset/eurus-wordmark.jpg" alt="EURUS wordmark" width="80%">
</p>

## At a glance

- **Provider-agnostic:** switch providers and configure fallback chains.
- **Workspace-aware coding:** inspect and edit files, propose patches, and run bounded validation workflows.
- **Approval-gated actions:** mutating tools use approval hooks and guarded file changes.
- **Persistent context:** local SQLite sessions, memory, and session search.
- **Extensible:** reusable skills, optional MCP integrations, and custom tools.
- **Small-environment friendly:** designed to run in a terminal, including Termux with its host-execution limitations.

### Learning skills safely

When learning is enabled, Eurus can draft a reusable `SKILL.md` after a task involving at least four tool calls. Drafts are written to `~/.eurus/skills/quarantine/`, scanned for likely secrets, dangerous command patterns, and attempts to override safety or approval policy. A draft is not activated automatically. Use the `skill_promote` tool, which requires owner approval, to place a clean draft in the current workspace under `.eurus/skills/`. Existing workspace skills can be changed with `skill_patch`; the caller must provide the current SHA-256 and a unified diff, and the tool rechecks the hash and safety scan before writing. Skill instructions remain advisory and never grant permissions.

## Goals

- One agent, many providers, with fallback and local/cloud hybrid routing
- Real work on your machine: shell, file edits, search, tests
- Safe by default: workspace confinement, permission modes, action journal
- Light enough to run on Termux and small servers

## Install

```bash
git clone https://github.com/<your-username>/eurus.git
cd eurus
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
set -a; source .env; set +a
eurus doctor
```

## Termux

```bash
pkg install python git
pip install -e .
```

Core dependencies are pure Python, so no compiler toolchain is required.

Termux has no container runtime. To use `shell` there, opt in to host execution:

```toml
[sandbox]
backend = "host"
required = false
```

Host mode is not isolated. Every command still needs approval, and the environment passed to
commands only carries `PATH`, `HOME`, `LANG`, `TERM`, `TMPDIR`, `PREFIX`, `VIRTUAL_ENV`, and
`PYTHONPATH`, so API keys are not exposed.

## Configuration

Sources, from lowest to highest priority:

1. Built-in defaults
2. `~/.eurus/config.toml` (or `$EURUS_HOME/config.toml`)
3. `eurus.toml` or `.eurus/config.toml` in the project directory
4. `EURUS_*` environment variables
5. Command line flags

```bash
eurus config init
eurus config show
eurus config path
eurus providers
eurus doctor
```

Models are written as `provider/model`, for example `groq/llama-3.3-70b-versatile`. API keys are read from environment variables and are never stored in config files. Custom providers:

```toml
[providers.myhost]
protocol = "openai"
base_url = "https://example.com/v1"
api_key_env = "MYHOST_API_KEY"
```

## Talking to a model

```bash
export GROQ_API_KEY=...
eurus ask --model groq/llama-3.3-70b-versatile "explain what a mutex is"
eurus ask --no-stream --system "answer in one line" "capital of Indonesia?"
```

Providers:

| Protocol | Providers |
|---|---|
| `openai` | OpenAI, Groq, NVIDIA NIM (`nvidia`), OpenRouter, DeepSeek, Mistral, Together, xAI, Ollama and any OpenAI-compatible endpoint |
| `anthropic` | Anthropic |
| `gemini` | Google Gemini over REST |
| `groq-sdk` | Groq through the official `groq` SDK (`pip install 'eurus-agent[groq]'`) |
| `genai-sdk` | Gemini through the official `google-genai` SDK (`pip install 'eurus-agent[genai]'`) |

The SDK protocols are optional; the default REST providers need no extra dependency. Use them as `groq-sdk/<model>` and `gemini-sdk/<model>`, or switch an existing provider with `[providers.groq] protocol = "groq-sdk"`.

NVIDIA NIM (key from [build.nvidia.com](https://build.nvidia.com), starts with `nvapi-`):

```bash
export NVIDIA_API_KEY=nvapi-...
eurus ask --model nvidia/meta/llama-3.3-70b-instruct "hello"
```

For a self-hosted NIM set `NVIDIA_BASE_URL=http://localhost:8000/v1` and any placeholder `NVIDIA_API_KEY`.

Troubleshooting a provider that fails with 403 even though the key is new:

```bash
eurus diagnose                       # probes the configured model chain
eurus diagnose groq/llama-3.3-70b-versatile nvidia/meta/llama-3.3-70b-instruct
```

`diagnose` reports why a request failed: `edge_blocked` (Cloudflare rejected the client before the key was checked), `model_blocked` (model disabled for your org or project), `region_blocked`, or `forbidden` (key rejected). See `docs/PROVIDERS.md`.

Keys placed in `.env` (current directory or `EURUS_HOME`) are loaded automatically, but only variables ending in `_API_KEY` plus `OLLAMA_BASE_URL` and `NVIDIA_BASE_URL`; set `EURUS_NO_DOTENV=1` to disable. Existing environment values always win.

Behavior:

- Rate limits, timeouts and 5xx responses are retried with backoff, honoring `Retry-After`
- If a model still fails, the next entry in `agent.fallback` is tried
- A stream is only retried or failed over before its first token arrives
- With `network.local_only = true`, only local endpoints (loopback, private ranges, Tailscale) are allowed

```toml
[agent]
model = "groq/llama-3.3-70b-versatile"
fallback = ["gemini/gemini-2.0-flash", "ollama/qwen2.5-coder"]
iteration_budget = 500

[auxiliary]
model = "openai/gpt-4o-mini"
fallback = ["groq/qwen-qwq-32b"]

[network]
timeout_seconds = 120.0
max_retries = 3
local_only = false
```

## Agent

```bash
eurus run "what is 17 * 23, and what time is it in Asia/Jakarta?"
eurus chat
eurus chat --session project-notes
eurus sessions list
eurus sessions show project-notes
eurus memory add "Prefer small, tested changes" --tag workflow
eurus memory search "tested changes"
eurus tools
```

`eurus run` executes one task. `eurus chat` keeps a conversation going; inside it, `/tools`, `/usage`, `/clear` and `/exit` are available. Pass `--session ID` to persist and restore chat history in `$EURUS_HOME/sessions.sqlite3`. The `eurus sessions` commands list, inspect, and delete sessions. `eurus memory add/search` stores local notes in SQLite and searches them with FTS5; memory retrieval is explicit and is not silently injected into model prompts.

The agent loop sends the conversation and tool list to the model, executes requested tools, feeds results back, and repeats until the model answers without calling a tool or the configured iteration budget is reached. `agent.iteration_budget` defaults to 500; an explicitly configured legacy `agent.max_steps` remains supported when `iteration_budget` is omitted.

- Tool arguments are validated against each tool's JSON schema; mistakes are returned to the model as error results so it can correct itself
- Tool output is truncated to `tools.max_output_chars` and every call has a timeout
- A call repeated identically three times in a row is blocked
- Model requests can be interrupted with `Agent.interrupt()` or stopped by the configured request timeout; incomplete model responses are not committed to history
- Independent tools run concurrently only when their metadata allows it; interactive/non-parallel-safe tools are serialized and mutating tools that declare the same path key share a per-path lock
- Iteration-budget exhaustion returns a compact summary of completed assistant and tool work
- Auxiliary tasks can use a separate `[auxiliary]` model and fallback chain
- Tools flagged as mutating pass through an approval hook before they run; the permission modes that use it arrive with the shell and file tools

Built-in tools include `calculate` and `get_time`. Agent runs also enable web research tools (`web_search`, `web_fetch`, `deep_search`) using DDGS and Trafilatura without a search API key. They require an internet connection and may be affected by search-engine rate limits. See [docs/WEB_RESEARCH.md](docs/WEB_RESEARCH.md). Agent runs also enable additional tools: `plan_task`, memory tools, workspace coding tools, and document tools (`document_read`, `document_create`, `document_replace`). `document_team_review` runs three parallel reviewers—proofreader, engagement editor, and critical reviewer—then a lead editor synthesizes feedback and proposes a revision. The team never writes the revision automatically; document creation/replacement requires owner approval and replacement uses a SHA-256 freshness check. `apply_patch`, `remember_memory`, `replace_memory`, `forget_memory`, `document_create`, and `document_replace` require owner approval. `propose_patch` previews a unified diff without writing files. `apply_patch` only writes if the file content still exactly matches the content that was reviewed. The public `default_registry()` keeps a minimal default toolset; use `default_registry(include_capability_tools=True)` to opt into planning, memory, and workspace tools.

Memory remains local in `$EURUS_HOME/sessions.sqlite3`; bounded profile and preference notes can be injected when relevant, and the agent can call `recall_memory` or `session_search` to retrieve prior context. Existing SQLite memory tables are upgraded additively.

Add your own tools:

```python
from eurus.tools import ToolRegistry

registry = ToolRegistry()


@registry.tool(
    "word_count",
    "Count the words in a text.",
    {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)
async def word_count(arguments, context):
    return str(len(arguments["text"].split()))
```

```toml
[agent]
system_prompt = ""

[tools]
max_output_chars = 20000
timeout_seconds = 120.0
```

## Task execution and verification

Use the bounded orchestration API to run planned steps sequentially, verify each result, retry failures within a fixed budget, and serialize progress for application-managed persistence. The controller does not bypass existing tool approval or workspace boundaries. See [docs/TASK_EXECUTION.md](docs/TASK_EXECUTION.md).

## Multi-file workflow

Multi-file workflow adds a lightweight Python project index, approved multi-file patchsets, guarded restore, and an allowlisted validation runner. See [docs/MULTI_FILE_WORKFLOW.md](docs/MULTI_FILE_WORKFLOW.md) for the architecture and verification commands.

## Development

```bash
ruff check .
ruff format --check .
mypy
pytest
```

## Origins

Eurus is implemented from scratch. Its design is informed by studying open-source agent projects. See [docs/CLEAN_ROOM.md](docs/CLEAN_ROOM.md).

## License

MIT. See [LICENSE](LICENSE).


## MCP integrations

Eurus supports optional Model Context Protocol clients. Install the integration with
`pip install -e '.[mcp]'`, then use `eurus mcp list`, `eurus mcp add-stdio`,
`eurus mcp add-http`, `eurus mcp test`, and `eurus mcp remove`. Review local server
commands before connecting; external MCP tools require approval by default. See
[`docs/MCP.md`](docs/MCP.md) for configuration and security details.
## Office artifacts

Eurus can create and read Word (`.docx`), Excel (`.xlsx`), and PowerPoint (`.pptx`) files. Install support with `python -m pip install -e ".[documents]"`. Use `office_team_create` for parallel proofreader, engagement-editor, and critical-review roles plus a lead producer. PowerPoint generation also creates a responsive HTML Canvas preview. See [`docs/OFFICE_ARTIFACTS.md`](docs/OFFICE_ARTIFACTS.md).

## Prompt and skill loading

Eurus keeps the stable system instructions separate from workspace context and per-turn volatile context. The stable tier contains a compact skill index; the full contents of a skill are loaded on demand with `skill_view(name)`. Skills are advisory and cannot grant permissions or bypass approval. The default configured skill list is empty, but the skill index remains available for discovery.

Anthropic requests attach ephemeral prompt caching to the stable system tier. Cache creation and cache-read token counts are retained in usage data and emitted at debug log level when reported by the provider. Context budgeting uses a replaceable `TokenCounter` interface and falls back to a character-based heuristic when no tokenizer is configured.

## Context compaction

Eurus preflights the context window before model requests. Normal mode compacts above 50% of the configured window; gateway mode uses an 85% threshold. The default context engine keeps the latest 20 conversation blocks and creates an extractive summary of older content. When an auxiliary model is configured, Eurus attempts an LLM summary and falls back to the extractive version on errors. Tool-call/result pairs are kept together. Compaction can invoke a memory flush callback and records parent/child session lineage through additive SQLite storage.

## Memory and session search

Eurus keeps persistent memory and conversation history in the local SQLite database at `$EURUS_HOME/sessions.sqlite3`. Session messages are indexed with SQLite FTS5 and can be searched by the `session_search` tool. When an auxiliary model is configured, retrieved snippets are summarized with source message references; otherwise the tool returns bounded matching excerpts. Project-scoped memories and session search are filtered to the active workspace plus global entries.

Memory providers are selected through `build_memory_provider`; SQLite is the supported provider, and unknown provider names fail closed. Memory types include profile, preference, project fact, agent note, lesson, episode, and fact. Persistent memory is not automatically created from web pages, tool output, or retrieved conversation text. `remember_memory`, `replace_memory`, and `forget_memory` require owner approval in the agent runtime; direct CLI memory commands are explicit user actions.

Use `eurus memory add`, `eurus memory replace`, `eurus memory remove`, `eurus memory search`, and `eurus memory list` to manage saved notes. Use `eurus privacy export` to export local session and memory records, or `eurus privacy purge` to clear saved sessions, summaries, lineage, memories, and local lifecycle events after confirmation. This removes records from the active database; it is not a forensic secure-erase guarantee. In interactive chat, `/privacy export` prints the export and `/privacy purge` requires typing `PURGE`.

## Delegated subagents

Eurus provides an approval-gated `delegate_task` tool for bounded read-only work. Each subagent receives an isolated conversation, a restricted allowlist of read-only tools, and a maximum of 50 iterations. Subagents cannot invoke `delegate_task`, use shell access, modify files, or persist memories. `document_team_review` uses isolated subagents for its proofreader, engagement-editor, and critical-reviewer roles when built through the standard agent factory. Delegated outputs are advisory and should be verified by the parent agent.

## GitHub branding and discovery

The repository banner is stored at `.github/asset/eurus-banner.jpg`. For GitHub's repository search cards and link previews, also upload that same image in **Settings → General → Social preview**. The README image and the GitHub social preview are separate settings.

Suggested GitHub repository topics: `ai-agent`, `autonomous-agent`, `coding-agent`, `llm`, `python`, `cli`, `mcp`, `termux`.

