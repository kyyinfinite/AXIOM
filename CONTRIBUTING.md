# Contributing

1. Fork the repository and create a branch from `main`.
2. Install with `pip install -e ".[dev]"`.
3. Run `ruff check .`, `ruff format --check .`, `mypy` and `pytest` before opening a pull request.
4. Keep pull requests small and focused, with tests for behavior changes.

## Code rules

- Do not paste or translate code from other projects, including AGPL or GPL projects.
- Code from permissively licensed projects may be used only with its notice recorded in a `NOTICE` file.
- Follow [docs/CLEAN_ROOM.md](docs/CLEAN_ROOM.md).
- Never commit secrets, `.env` files or user data.
