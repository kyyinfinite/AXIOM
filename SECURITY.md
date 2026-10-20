# Security policy

## Reporting a vulnerability

Do not open a public issue for security problems. Use the GitHub private vulnerability reporting feature on this repository.

Include the affected version, reproduction steps and impact. You will get an acknowledgement within 7 days.

## Scope

Issues of particular interest: shell or filesystem confinement bypass, permission or approval bypass, API key leakage in logs or journals, and prompt injection that leads to unapproved actions.

## Shell execution

Shell commands run through the configured container runtime. Eurus does not fall back to host execution when the runtime is unavailable. The default container has no network, a read-only root filesystem, dropped Linux capabilities, `no-new-privileges`, and CPU, memory, PID, timeout, and output limits. The configured workspace is mounted writable; review and back up workspace contents before approving commands.

Mutating tools require an approval provider. If no provider is configured, the agent denies the action. CLI approval is scoped to the displayed tool call; changing the command requires a new approval. Sandbox isolation depends on the host container runtime and its configuration and is not a substitute for reviewing untrusted code.

## Audit journal

Use `eurus.security.journal.AuditJournal` for newline-delimited JSON audit records. The journal creates files with owner-only permissions where supported. Do not include credentials or raw secrets in event fields.

## Local sessions and memory

Chat history and explicit memory notes are stored in `$EURUS_HOME/sessions.sqlite3` when session persistence or memory commands are used. The database is created with owner-only permissions where supported. Do not put credentials, access tokens, or other secrets into persistent chat history or memory notes. Memory search is explicit; notes are not automatically added to model prompts.
