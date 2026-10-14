# Multi-file coding workflow

Multi-file workflow extends the existing codebase without replacing the agent loop, SQLite sessions, FTS5 memory, provider interfaces, or tool registry.

## New capabilities

- `index_project`: standard-library AST index of Python files, imports, classes, and functions.
- `preview_patchset`: render a unified diff for a set of file changes without writing files.
- `apply_patchset`: validate and apply up to 20 file patches as one approved tool operation. Existing files require exact `expected_content`; new files use `null`. File deletion is intentionally unsupported.
- `WorkspaceChangeManager`: record agent-owned changes, summarize the changed files, and restore only when the file still matches the agent-written hash. Later user edits are never silently overwritten.
- `run_validation`: run a small allowlist of local test/type/format commands without a shell, with time and output limits.

## Safety and Termux choices

The index uses `ast`, `pathlib`, and other standard-library modules. It skips common generated directories, symbolic links, and files larger than 1 MB. Patch paths must be relative to the configured workspace. Patch application requires one approval for the whole patchset, not a separate approval for each file. Review the final diff and test output afterward.

Git remains optional. No new runtime dependency is required for these primitives. The validation runner only accepts `python -m pytest`, `python -m compileall`, `python -m mypy`, or `ruff check|format` commands; it never invokes a shell.

## Compatibility

`default_registry()` remains the minimal default tool registry by default. `include_capability_tools=True` enables Agent capabilities tools. `include_workflow_tools=True` opts into Multi-file workflow tools. The standard agent factory enables Agent capabilities and Multi-file workflow tools while preserving the existing provider and session APIs.

## Verification

Run from the repository root:

```bash
python -m compileall -q src tests
python -m pytest -q
ruff check .
ruff format --check .
python -m mypy src
```

A patchset is considered safe only if all paths validate before the first write, every existing file matches its expected content, and the resulting diff and validation output are reviewed. Multi-file writes are rollback-attempted on failure; this is not a substitute for an independent backup when working with valuable data.
