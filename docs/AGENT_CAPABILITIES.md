# Agent capabilities

This document describes lightweight planning, workspace-confined coding tools, and categorized persistent memory while while keeping the default tool registry minimal.

## Features

- `TaskPlanner` creates a bounded deterministic plan (up to 8 steps) without requiring a second model or extra dependencies.
- `MemoryManager` wraps the existing SQLite/FTS5 store with categories, recall, update, and delete operations.
- Coding tools: `list_files`, `read_file`, `search_code`, `propose_patch`, and `apply_patch`.
- Memory tools: `recall_memory`, `remember_memory`, and `forget_memory`.
- The agent factory enables these tools by default. `default_registry()` keeps a minimal default set; call `default_registry(include_capability_tools=True)` to get the extended set.
- File mutation and memory mutation tools are marked as requiring approval and flow through the existing agent approval hook.
- Workspace paths are checked for traversal and symbolic links. File reads and writes are capped at 1 MB. Patch application rejects stale content.
- SQLite schema changes are additive; existing memory rows receive category `general` and their original creation timestamp as the update timestamp.

## Context window

`agent.context_tokens` (default 32000, 0 disables) bounds what is sent to the model. When the
estimated size exceeds it, old tool outputs are truncated first, then whole old turns are
dropped. The system prompt and the latest user message are always kept, tool calls stay paired
with their results, and the stored history is not modified.

## Learning

After a run that used at least two tools, the model is asked once for up to three short,
general lessons. `learning.mode` controls what happens next: `off` skips reflection, `ask`
(default) asks for approval before each lesson is saved, `auto` saves without asking. Lessons
are stored as memories in the `lesson` category, filtered for secrets, and de-duplicated.

At the start of each run, up to `learning.lessons_in_prompt` lessons matching the task are added
to the system prompt as hints. Review or remove them with `axiom memory list --category lesson`
and `axiom memory delete ID`. The TUI reads lessons but does not save new ones yet.

## Evaluation

`axiom eval [--task NAME] [--lessons/--no-lessons]` runs built-in coding tasks in temporary
workspaces without the shell tool, scores each by running its tests, and writes a JSON report
to `~/.axiom/evals`. Running once with `--no-lessons` and once with `--lessons` shows whether
saved lessons help.

## Running checks

`run_checks` runs pytest, mypy, ruff, or compileall on the host with a reduced environment and
requires approval. Failures are returned to the model as normal output so it can fix and rerun.

## CLI memory management

```bash
axiom memory add "Keep changes small" --tag workflow --category preference
axiom memory search "changes small"
axiom memory update 1 "Keep changes focused and tested" --category preference
axiom memory delete 1
```

## Verification

Run from the repository root:

```bash
python -m compileall -q src tests
python -m pytest -q
python -m mypy src
ruff check src tests
ruff format --check src tests
```

The unit test suite covers legacy SQLite migration, memory CRUD/search, planner bounds, workspace confinement, symlink rejection, diff preview, stale patch protection, and approval metadata. Native tool availability may differ on Android/Termux; Ruff and MyPy are development tools, not runtime dependencies.
