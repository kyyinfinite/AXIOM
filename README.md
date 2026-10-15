<div align="center">

# ⚡ AXagent

### Your terminal. Your models. Your workflow.

A provider-flexible AI agent for the command line, built for practical coding workflows—from exploring a codebase to proposing changes, running checks, and keeping useful project context close at hand.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Termux-111827?style=for-the-badge&logo=linux&logoColor=white)](https://termux.dev/)
[![Status](https://img.shields.io/badge/Status-Pre--alpha-f59e0b?style=for-the-badge)](docs/ROADMAP.md)

**Designed and built by [kyyinfinite](https://github.com/kyyinfinite).**

</div>

---

## ✦ What is AXagent?

AXagent is a terminal-first interface for the AXIOM agent engine. It connects to different language-model providers while keeping tools, configuration, sessions, and local memory in one workflow. Use the interactive terminal UI for hands-on work, or run a task directly from your shell.

The project is designed around three principles:

- **Flexible:** choose a supported model provider, configure fallbacks, or connect an OpenAI-compatible endpoint.
- **Practical:** work with files, inspect code, propose patches, and run verification commands as part of an agent workflow.
- **Controlled:** keep workspace boundaries and approval steps in the loop for actions that can modify files or affect the environment.

> **Project status:** Pre-alpha. Interfaces and behavior may change. Review commands and proposed changes before using AXagent on important projects.

## ✨ Highlights

| Capability | What it does |
|---|---|
| **Interactive terminal UI** | A branded TUI for working with the agent from your terminal. |
| **Multiple model providers** | Supports OpenAI-compatible APIs, Anthropic, Gemini, Groq, NVIDIA NIM, Ollama, and compatible endpoints. |
| **Provider fallback** | Try configured models in sequence when a provider request fails. |
| **Coding tools** | Explore files, read source, search code, propose patches, and apply approved changes. |
| **Task planning** | Break a task into bounded steps and verify results within a limited retry budget. |
| **Approval-aware execution** | Route mutating actions through approval controls and keep workspace restrictions in place. |
| **Persistent sessions** | Save and resume chat history using a local SQLite database. |
| **Local memory and lessons** | Save searchable notes and retain task lessons for later workflows. |
| **Project indexing and checks** | Build a lightweight Python project index and run allowlisted validation checks. |
| **Termux-friendly** | Lightweight Python dependencies and a setup path for Android terminals. |

## 📸 Interface

The interactive interface is launched with `axagent` or `axiom tui`. Add a screenshot or terminal recording here when you're ready to showcase the current UI:

<!-- Replace this comment with a committed image, for example:
<p align="center"><img src="docs/images/axagent-tui.png" alt="AXagent terminal interface" width="900"></p>
-->

## 🚀 Quick start

### Requirements

- Python **3.11 or newer**
- Git
- An API key for at least one configured model provider, or a reachable local model endpoint

### Install from source

```bash
git clone https://github.com/kyyinfinite/axiom.git
cd axiom
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If your repository uses a different GitHub URL or name, replace the clone URL and directory above with your repository's actual URL.

### Configure a provider

Initialize the default configuration and inspect your environment:

```bash
axiom config init
axiom config show
axiom providers
axiom doctor
```

Set the API key for the provider you plan to use. For example, with NVIDIA NIM:

```bash
export NVIDIA_API_KEY="nvapi-your-key"
```

You can also place supported provider keys in a local `.env` file. Keep secrets out of Git—never commit `.env` or API keys.

### Launch AXagent

```bash
axagent
```

Or launch the TUI through the main CLI:

```bash
axiom tui
```

Run a single task without entering the interactive interface:

```bash
axiom run "Inspect this project and explain its main components"
```

Start a regular chat session:

```bash
axiom chat
```

Persist a chat session so it can be resumed later:

```bash
axiom chat --session project-notes
```

## 🧠 Model providers

AXagent uses the `provider/model` format. Examples:

| Provider | Example model reference |
|---|---|
| NVIDIA NIM | `nvidia/meta/llama-3.3-70b-instruct` |
| Groq | `groq/llama-3.3-70b-versatile` |
| Gemini | `gemini/gemini-2.0-flash` |
| Ollama | `ollama/qwen2.5-coder` |
| OpenAI-compatible endpoint | Configure a custom provider in TOML |

Provider model availability depends on the provider account, region, endpoint, and current model catalogue. Use `axiom providers` to inspect the providers configured in your environment.

Example: set NVIDIA NIM as the active model for one command:

```bash
axiom ask --model nvidia/meta/llama-3.3-70b-instruct "Explain this Python function"
```

For provider setup details and troubleshooting, see [`docs/PROVIDERS.md`](docs/PROVIDERS.md).

## 🛠️ Useful commands

| Command | Purpose |
|---|---|
| `axagent` | Launch the AXagent terminal UI. |
| `axiom tui` | Launch the terminal UI from the main CLI. |
| `axiom run "..."` | Execute one task. |
| `axiom chat` | Start an interactive chat. |
| `axiom providers` | Inspect configured providers. |
| `axiom doctor` | Check the local setup. |
| `axiom tools` | List available agent tools. |
| `axiom sessions list` | List saved sessions. |
| `axiom sessions show <id>` | Inspect a saved session. |
| `axiom memory add "..." --tag workflow` | Save a local memory note. |
| `axiom memory search "..."` | Search local memory. |
| `axiom eval` | Run the built-in evaluation workflow. |

Run `axiom --help` or `axiom <command> --help` for the available options in your installed version.

## 🔐 Safety and privacy

AXagent is built to keep important actions visible and controlled, but it is still pre-alpha software. Treat it like a powerful developer tool, not an unattended production operator.

- Review proposed diffs before approving changes.
- Keep API keys in environment variables or a local, untracked `.env` file.
- Persistent sessions and memory are stored locally in the AXIOM data directory, including `sessions.sqlite3`.
- Memory retrieval is explicit; saved notes are not silently injected into every model prompt.
- Shell execution in **host mode** is not a container or security boundary. Use it only when you understand the implications.
- Do not point the agent at sensitive workspaces until you have reviewed its permissions and configuration.

For security reports, see [`SECURITY.md`](SECURITY.md). For execution details and limitations, see [`docs/TASK_EXECUTION.md`](docs/TASK_EXECUTION.md) and [`docs/MULTI_FILE_WORKFLOW.md`](docs/MULTI_FILE_WORKFLOW.md).

## 📱 Running on Termux

AXagent is designed to be usable on Android through Termux.

```bash
pkg update
pkg install python git
git clone https://github.com/kyyinfinite/axiom.git
cd axiom
python -m pip install --upgrade pip
pip install -e .
axagent
```

Termux does not provide a standard container runtime by default. If you enable host execution, understand that it is **not isolated**. Keep approval enabled and review the sandbox settings before running commands.

## 🧑‍💻 Development

Create a development environment and install the test and lint tools:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the project checks:

```bash
pytest -q
ruff check src tests
ruff format --check src tests
mypy src
python -m compileall -q src tests
```

Please report the exact command and output when opening an issue for a reproducible bug. Before submitting a pull request, keep changes focused and add tests for changed behavior.

## 🗺️ Roadmap

The project is actively evolving. Current work focuses on reliability, a stronger terminal experience, persistent learning workflows, coding-tool safety, and verification across desktop Linux and Termux.

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for implementation status and planned work. Planned items should not be assumed to be available until documented as implemented.

## 🤝 Contributing

Bug reports, focused improvements, and thoughtful feedback are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.

## 📄 License

AXagent is released under the [MIT License](LICENSE).

---

<div align="center">

**AXagent** · Built for the terminal by **kyyinfinite**

</div>
