# AXIOM

Provider-agnostic AI coding agent that works in your terminal: shell, files, patches and test runs, with OpenAI, Anthropic, Gemini, Groq, NVIDIA NIM and any OpenAI-compatible endpoint.

> Status: pre-alpha. Includes workspace coding tools, test and lint runs, context trimming, bounded task planning, and categorized persistent memory. It can save lessons from finished tasks (with your approval) and reuse them, and `axiom eval` scores it on small coding tasks. Skills, MCP, and release hardening are planned, not implemented. See [docs/ROADMAP.md](docs/ROADMAP.md).

## Goals

- One agent, many providers, with fallback and local/cloud hybrid routing
- Real work on your machine: shell, file edits, search, tests
- Safe by default: workspace confinement, permission modes, action journal
- Light enough to run on Termux and small servers

## Install

```bash
git clone https://github.com/<your-username>/axiom.git
cd axiom
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
set -a; source .env; set +a
axiom doctor
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
2. `~/.axiom/config.toml` (or `$AXIOM_HOME/config.toml`)
3. `axiom.toml` or `.axiom/config.toml` in the project directory
4. `AXIOM_*` environment variables
5. Command line flags

```bash
axiom config init
axiom config show
axiom config path
axiom providers
axiom doctor
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
axiom ask --model groq/llama-3.3-70b-versatile "explain what a mutex is"
axiom ask --no-stream --system "answer in one line" "capital of Indonesia?"
```

Providers:

| Protocol | Providers |
|---|---|
| `openai` | OpenAI, Groq, NVIDIA NIM (`nvidia`), OpenRouter, DeepSeek, Mistral, Together, xAI, Ollama and any OpenAI-compatible endpoint |
| `anthropic` | Anthropic |
| `gemini` | Google Gemini over REST |
| `groq-sdk` | Groq through the official `groq` SDK (`pip install 'axiom-agent[groq]'`) |
| `genai-sdk` | Gemini through the official `google-genai` SDK (`pip install 'axiom-agent[genai]'`) |

The SDK protocols are optional; the default REST providers need no extra dependency. Use them as `groq-sdk/<model>` and `gemini-sdk/<model>`, or switch an existing provider with `[providers.groq] protocol = "groq-sdk"`.

NVIDIA NIM (key from [build.nvidia.com](https://build.nvidia.com), starts with `nvapi-`):

```bash
export NVIDIA_API_KEY=nvapi-...
axiom ask --model nvidia/meta/llama-3.3-70b-instruct "hello"
```

For a self-hosted NIM set `NVIDIA_BASE_URL=http://localhost:8000/v1` and any placeholder `NVIDIA_API_KEY`.

Troubleshooting a provider that fails with 403 even though the key is new:

```bash
axiom diagnose                       # probes the configured model chain
axiom diagnose groq/llama-3.3-70b-versatile nvidia/meta/llama-3.3-70b-instruct
```

`diagnose` reports why a request failed: `edge_blocked` (Cloudflare rejected the client before the key was checked), `model_blocked` (model disabled for your org or project), `region_blocked`, or `forbidden` (key rejected). See `docs/PROVIDERS.md`.

Keys placed in `.env` (current directory or `AXIOM_HOME`) are loaded automatically, but only variables ending in `_API_KEY` plus `OLLAMA_BASE_URL` and `NVIDIA_BASE_URL`; set `AXIOM_NO_DOTENV=1` to disable. Existing environment values always win.

Behavior:

- Rate limits, timeouts and 5xx responses are retried with backoff, honoring `Retry-After`
- If a model still fails, the next entry in `agent.fallback` is tried
- A stream is only retried or failed over before its first token arrives
- With `network.local_only = true`, only local endpoints (loopback, private ranges, Tailscale) are allowed

```toml
[agent]
model = "groq/llama-3.3-70b-versatile"
fallback = ["gemini/gemini-2.0-flash", "ollama/qwen2.5-coder"]

[network]
timeout_seconds = 120.0
max_retries = 3
local_only = false
```

## Agent

```bash
axiom run "what is 17 * 23, and what time is it in Asia/Jakarta?"
axiom chat
axiom chat --session project-notes
axiom sessions list
axiom sessions show project-notes
axiom memory add "Prefer small, tested changes" --tag workflow
axiom memory search "tested changes"
axiom tools
```

`axiom run` executes one task. `axiom chat` keeps a conversation going; inside it, `/tools`, `/usage`, `/clear` and `/exit` are available. Pass `--session ID` to persist and restore chat history in `$AXIOM_HOME/sessions.sqlite3`. The `axiom sessions` commands list, inspect, and delete sessions. `axiom memory add/search` stores local notes in SQLite and searches them with FTS5; memory retrieval is explicit and is not silently injected into model prompts.

The agent loop sends the conversation and the tool list to the model, runs the tool calls the model asks for, feeds the results back and repeats until the model answers without calling a tool or `agent.max_steps` is reached.

- Tool arguments are validated against each tool's JSON schema; mistakes are returned to the model as error results so it can correct itself
- Tool output is truncated to `tools.max_output_chars` and every call has a timeout
- A call repeated identically three times in a row is blocked
- An interrupted run leaves the conversation in a valid state
- Tools flagged as mutating pass through an approval hook before they run; the permission modes that use it arrive with the shell and file tools

Built-in tools include `calculate` and `get_time`. Agent runs also enable additional tools: `plan_task`, `recall_memory`, `remember_memory`, `forget_memory`, `list_files`, `read_file`, `search_code`, `propose_patch`, and `apply_patch`. `apply_patch`, `remember_memory`, and `forget_memory` require owner approval. `propose_patch` previews a unified diff without writing files. `apply_patch` only writes if the file content still exactly matches the content that was reviewed. The public `default_registry()` keeps a minimal default toolset; use `default_registry(include_capability_tools=True)` to opt into planning, memory, and workspace tools.

Memory remains local in `$AXIOM_HOME/sessions.sqlite3` and is not automatically injected into prompts; the agent can call `recall_memory` when relevant. Existing SQLite memory tables are upgraded additively.

Add your own tools:

```python
from axiom.tools import ToolRegistry

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

AXIOM is implemented from scratch. Its design is informed by studying open-source agent projects. See [docs/CLEAN_ROOM.md](docs/CLEAN_ROOM.md).

## License

MIT. See [LICENSE](LICENSE).
