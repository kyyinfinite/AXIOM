from eurus.catalog import BUILTIN_PROVIDERS, is_ready, resolve_api_key


def test_resolve_api_key_strips_value():
    assert resolve_api_key(BUILTIN_PROVIDERS["groq"], {"GROQ_API_KEY": " key "}) == "key"


def test_missing_key_is_not_ready():
    assert not is_ready(BUILTIN_PROVIDERS["groq"], {})


def test_keyless_provider_is_ready():
    assert is_ready(BUILTIN_PROVIDERS["ollama"], {})
