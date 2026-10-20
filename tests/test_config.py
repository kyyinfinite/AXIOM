import json

import pytest

from eurus.config import load_config
from eurus.errors import ConfigError


def env_for(home, **extra):
    return {"EURUS_HOME": str(home), **extra}


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults(tmp_path):
    config = load_config(cwd=tmp_path, environ=env_for(tmp_path))
    assert config.agent.model == ""
    assert config.agent.max_steps == 40
    assert config.permissions.mode == "ask"
    assert "groq" in config.providers
    assert config.providers["ollama"].api_key_env == ""


def test_layer_precedence(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    write(home / "config.toml", '[agent]\nmodel = "groq/a"\nmax_steps = 10\n')
    write(project / "eurus.toml", '[agent]\nmodel = "openai/b"\n')

    config = load_config(cwd=project, environ=env_for(home))
    assert config.agent.model == "openai/b"
    assert config.agent.max_steps == 10

    config = load_config(cwd=project, environ=env_for(home, EURUS_MODEL="gemini/c"))
    assert config.agent.model == "gemini/c"

    config = load_config(
        cwd=project,
        environ=env_for(home, EURUS_MODEL="gemini/c"),
        overrides={"agent": {"model": "xai/d"}},
    )
    assert config.agent.model == "xai/d"


def test_explicit_path_missing(tmp_path):
    with pytest.raises(ConfigError):
        load_config(config_path=tmp_path / "nope.toml", environ=env_for(tmp_path))


@pytest.mark.parametrize(
    "text",
    [
        "[agent]\nunknown = 1\n",
        "[agent]\nmax_steps = 0\n",
        "[agent]\nmax_steps = true\n",
        "[agent]\ntemperature = 5\n",
        '[agent]\nmodel = "nomodel"\n',
        '[agent]\nmodel = "nope/x"\n',
        '[permissions]\nmode = "root"\n',
        '[logging]\nlevel = "LOUD"\n',
        "[mystery]\nx = 1\n",
        '[providers.x]\nprotocol = "openai"\n',
        '[providers.x]\nprotocol = "soap"\nbase_url = "https://a.b"\n',
        "garbage ===",
    ],
)
def test_invalid_config_rejected(tmp_path, text):
    path = write(tmp_path / "c.toml", text)
    with pytest.raises(ConfigError):
        load_config(config_path=path, environ=env_for(tmp_path))


def test_custom_provider_and_partial_override(tmp_path):
    path = write(
        tmp_path / "c.toml",
        '[providers.myhost]\nprotocol = "openai"\nbase_url = "https://example.com/v1/"\n'
        'api_key_env = "MYHOST_KEY"\n\n[providers.groq]\nbase_url = "https://proxy.local/v1"\n\n'
        '[agent]\nmodel = "myhost/any"\n',
    )
    config = load_config(config_path=path, environ=env_for(tmp_path))
    assert config.providers["myhost"].base_url == "https://example.com/v1"
    assert config.providers["groq"].api_key_env == "GROQ_API_KEY"
    assert config.providers["groq"].base_url == "https://proxy.local/v1"


def test_ollama_base_url_from_env(tmp_path):
    config = load_config(
        cwd=tmp_path, environ=env_for(tmp_path, OLLAMA_BASE_URL="http://10.0.0.5:11434/")
    )
    assert config.providers["ollama"].base_url == "http://10.0.0.5:11434/v1"


def test_env_coercion(tmp_path):
    config = load_config(
        cwd=tmp_path,
        environ=env_for(
            tmp_path,
            EURUS_MAX_STEPS="7",
            EURUS_LOG_JSON="yes",
            EURUS_FALLBACK="groq/a, openai/b",
        ),
    )
    assert config.agent.max_steps == 7
    assert config.logging.json is True
    assert config.agent.fallback == ("groq/a", "openai/b")


def test_env_invalid_int(tmp_path):
    with pytest.raises(ConfigError):
        load_config(cwd=tmp_path, environ=env_for(tmp_path, EURUS_MAX_STEPS="many"))


def test_to_dict_is_json_serialisable(tmp_path):
    config = load_config(cwd=tmp_path, environ=env_for(tmp_path))
    assert json.loads(json.dumps(config.to_dict()))["agent"]["max_steps"] == 40


def test_network_defaults(tmp_path):
    config = load_config(cwd=tmp_path, environ=env_for(tmp_path))
    assert config.network.timeout_seconds == 120.0
    assert config.network.max_retries == 3
    assert config.network.local_only is False


def test_network_from_file_and_env(tmp_path):
    path = write(tmp_path / "c.toml", "[network]\ntimeout_seconds = 30\nmax_retries = 1\n")
    config = load_config(
        config_path=path, environ=env_for(tmp_path, EURUS_LOCAL_ONLY="true", EURUS_MAX_RETRIES="5")
    )
    assert config.network.timeout_seconds == 30.0
    assert config.network.max_retries == 5
    assert config.network.local_only is True


@pytest.mark.parametrize(
    "text",
    [
        "[network]\ntimeout_seconds = 0\n",
        "[network]\nmax_retries = 99\n",
        "[network]\nlocal_only = 1\n",
        "[network]\nunknown = 1\n",
    ],
)
def test_invalid_network_rejected(tmp_path, text):
    path = write(tmp_path / "c.toml", text)
    with pytest.raises(ConfigError):
        load_config(config_path=path, environ=env_for(tmp_path))


def test_tools_and_prompt_defaults(tmp_path):
    config = load_config(cwd=tmp_path, environ=env_for(tmp_path))
    assert config.tools.max_output_chars == 20000
    assert config.tools.timeout_seconds == 120.0
    assert config.agent.system_prompt == ""


def test_tools_and_prompt_from_file(tmp_path):
    path = write(
        tmp_path / "c.toml",
        (
            '[agent]\nsystem_prompt = " Be terse. "\n\n[tools]\n'
            "max_output_chars = 1000\ntimeout_seconds = 5\n"
        ),
    )
    config = load_config(config_path=path, environ=env_for(tmp_path))
    assert config.agent.system_prompt == "Be terse."
    assert config.tools.max_output_chars == 1000
    assert config.tools.timeout_seconds == 5.0


@pytest.mark.parametrize(
    "text",
    [
        "[tools]\nmax_output_chars = 10\n",
        "[tools]\ntimeout_seconds = 0\n",
        "[tools]\nunknown = 1\n",
        "[agent]\nsystem_prompt = 3\n",
    ],
)
def test_invalid_tools_rejected(tmp_path, text):
    path = write(tmp_path / "c.toml", text)
    with pytest.raises(ConfigError):
        load_config(config_path=path, environ=env_for(tmp_path))


def test_tools_parallel_limit_is_configurable(tmp_path):
    path = write(
        tmp_path / "parallel.toml",
        "[tools]\nmax_parallel_tools = 8\n",
    )
    config = load_config(config_path=path, environ={"EURUS_HOME": str(tmp_path)})
    assert config.tools.max_parallel_tools == 8


def test_tools_parallel_limit_is_bounded(tmp_path):
    path = write(
        tmp_path / "parallel.toml",
        "[tools]\nmax_parallel_tools = 0\n",
    )
    with pytest.raises(ConfigError, match="max_parallel_tools"):
        load_config(config_path=path, environ={"EURUS_HOME": str(tmp_path)})


def test_iteration_budget_defaults_to_500_and_legacy_max_steps_remains_supported(tmp_path):
    defaults = load_config(cwd=tmp_path, environ=env_for(tmp_path))
    assert defaults.agent.max_steps == 40
    assert defaults.agent.iteration_budget == 500
    legacy = write(tmp_path / "legacy.toml", "[agent]\nmax_steps = 12\n")
    legacy_config = load_config(config_path=legacy, environ=env_for(tmp_path))
    assert legacy_config.agent.iteration_budget == 12
    explicit = write(
        tmp_path / "budget.toml",
        "[agent]\nmax_steps = 12\niteration_budget = 300\n",
    )
    explicit_config = load_config(config_path=explicit, environ=env_for(tmp_path))
    assert explicit_config.agent.iteration_budget == 300


def test_auxiliary_fallback_chain_is_configurable(tmp_path):
    path = write(
        tmp_path / "auxiliary.toml",
        '[auxiliary]\nmodel = "groq/primary"\nfallback = ["openai/backup"]\n',
    )
    config = load_config(config_path=path, environ=env_for(tmp_path))
    assert config.auxiliary.model == "groq/primary"
    assert config.auxiliary.fallback == ("openai/backup",)


def test_auxiliary_rejects_unknown_model(tmp_path):
    path = write(tmp_path / "auxiliary.toml", '[auxiliary]\nmodel = "unknown/model"\n')
    with pytest.raises(ConfigError, match="auxiliary.model"):
        load_config(config_path=path, environ=env_for(tmp_path))


def test_auxiliary_chain_is_independent_from_primary_fallback(tmp_path):
    from eurus.providers.factory import resolve_auxiliary_chain

    path = write(
        tmp_path / "chains.toml",
        '[agent]\nmodel = "groq/main"\nfallback = ["openai/main-backup"]\n'
        '[auxiliary]\nmodel = "gemini/aux"\nfallback = ["openai/aux-backup"]\n',
    )
    config = load_config(config_path=path, environ=env_for(tmp_path))
    assert [str(item) for item in resolve_auxiliary_chain(config)] == [
        "gemini/aux",
        "openai/aux-backup",
    ]
