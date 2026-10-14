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
