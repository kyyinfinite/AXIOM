# Providers

## Why a fresh key can still return 403

A 403 does not always mean the key is wrong. Axiom now classifies it:

| Reason | Meaning | Fix |
|---|---|---|
| `edge_blocked` | Cloudflare in front of the API rejected the HTTP client (error code 1010, HTML body) before the key was checked. Default Python user agents (`python-urllib`, `python-requests`) are known triggers. | Axiom sends `axiom-agent/<version>` and retries once with a `Mozilla/5.0 (compatible; ...)` form. Override with `[providers.<name>] user_agent = "..."`, or use the SDK protocol (`groq-sdk`). |
| `model_blocked` | Valid key, but the model is disabled for the organization or project (Groq returns 403 `permissions_error`). | Enable the model in the provider console (Groq: Settings, Limits). |
| `region_blocked` | The provider refused the network or region. | Use another network or provider. |
| `forbidden` | The key was rejected or lacks permission. | Regenerate the key; Axiom already strips quotes, `Bearer ` and whitespace. |

Run `axiom diagnose` to see which one applies.

## Invisible failures fixed in 0.0.8

- HTTP 200 responses that carry `{"error": ...}` (OpenRouter, some gateways) were treated as an empty answer; they now raise the matching error and are retried when appropriate.
- Streams that ended without data were silent; they now raise a retryable error. A 200 response that is not an event stream (login page, proxy page) raises with a readable snippet.
- HTML error bodies are reduced to text instead of 300 raw characters of markup.
- Newer OpenAI models reject `max_tokens` or `temperature`; the request is adapted once and remembered.
- Groq reports streaming usage under `x_groq.usage`; it is now read.
- Keys copied with quotes, a `Bearer ` prefix or line breaks are cleaned; an illegal header value no longer leaks the key into the error text.
- `nvapi-` keys are redacted in logs.
- NVIDIA NIM models with strict chat templates receive a single merged system message.

## Provider reference

| Name | Protocol | Base URL | Key variable |
|---|---|---|---|
| `groq` | `openai` | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` |
| `groq-sdk` | `groq-sdk` | same | `GROQ_API_KEY` |
| `gemini` | `gemini` | `https://generativelanguage.googleapis.com` | `GEMINI_API_KEY` |
| `gemini-sdk` | `genai-sdk` | same | `GEMINI_API_KEY` |
| `nvidia` | `openai` | `https://integrate.api.nvidia.com/v1` | `NVIDIA_API_KEY` |

NVIDIA model ids are namespaced (`meta/llama-3.3-70b-instruct`), so the full reference is `nvidia/meta/llama-3.3-70b-instruct`.

Per-provider overrides:

```toml
[providers.groq]
user_agent = "my-app/1.0"

[providers.groq.headers]
X-Team = "platform"
```
