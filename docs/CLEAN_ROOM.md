# Clean-room policy

AXIOM is licensed under MIT. To keep that valid, code is written from scratch.

## Allowed

- Studying the architecture and behavior of other agents, including AGPL projects such as Odysseus
- Writing design notes in your own words under `docs/design/`
- Implementing from those notes

## Not allowed

- Copying, vendoring or line-by-line translating source files
- Copying anything from an AGPL fork archive into this repository
- Reusing function names, prompts, docstrings, test fixtures or data files from other projects
- Pasting code into an AI assistant and asking for a rewrite of it

## Practice

- Write the design note first, implement afterwards, in separate commits
- Record any reused permissively licensed code in `NOTICE` with its license text
- When in doubt, rewrite from the behavior, not from the source
