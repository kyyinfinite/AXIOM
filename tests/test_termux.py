import asyncio

import pytest
from rich.console import Console

from eurus.config import SAMPLE_CONFIG, load_config, sample_config
from eurus.errors import ToolError
from eurus.execution.host import HostSandbox, scrubbed_env
from eurus.execution.sandbox import ContainerSandbox, ExecutionRequest, ExecutionResult
from eurus.termux import is_termux
from eurus.tools.base import ToolContext
from eurus.tools.shell import shell_tool
from eurus.tui.renderer import TerminalRenderer
from eurus.tui.session import InteractiveSession

PREFIX = "/data/data/com.termux/files/usr"
TERMUX_ENV = {"PREFIX": PREFIX, "TERMUX_VERSION": "0.118"}


def config_in(tmp_path, body="", environ=None):
    path = tmp_path / "eurus.toml"
    path.write_text(body, encoding="utf-8")
    env = {"EURUS_HOME": str(tmp_path), **(environ or {})}
    return load_config(config_path=path, environ=env)


def use_termux(monkeypatch):
    monkeypatch.setenv("PREFIX", PREFIX)
    monkeypatch.setenv("TERMUX_VERSION", "0.118")


def leave_termux(monkeypatch):
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.delenv("LD_PRELOAD", raising=False)


def test_termux_detection():
    assert is_termux({"TERMUX_VERSION": "0.118"})
    assert is_termux({"PREFIX": PREFIX})
    assert not is_termux({"PREFIX": "/usr"})
    assert not is_termux({})


def test_termux_defaults_to_host_backend(tmp_path):
    config = config_in(tmp_path, environ=TERMUX_ENV)
    assert config.sandbox.backend == "host" and config.sandbox.required is False


def test_non_termux_defaults_unchanged(tmp_path):
    config = config_in(tmp_path)
    assert config.sandbox.backend == "container" and config.sandbox.required is True


def test_explicit_choices_are_respected_on_termux(tmp_path):
    config = config_in(tmp_path, '[sandbox]\nbackend = "container"\n', TERMUX_ENV)
    assert config.sandbox.backend == "container" and config.sandbox.required is True
    config = config_in(tmp_path, '[sandbox]\nbackend = "host"\n', TERMUX_ENV)
    assert config.sandbox.backend == "host"


def test_sample_config_for_termux_is_loadable(tmp_path):
    text = sample_config(TERMUX_ENV)
    assert 'backend = "host"' in text and "docker" not in text
    assert sample_config({}) == SAMPLE_CONFIG
    config = config_in(tmp_path, text, TERMUX_ENV)
    assert config.sandbox.backend == "host"


def test_docker_missing_error_explains_termux(tmp_path, monkeypatch):
    use_termux(monkeypatch)
    sandbox = ContainerSandbox(tmp_path, runtime="eurus-runtime-that-does-not-exist")
    request = ExecutionRequest(argv=("/bin/sh", "-c", "true"))
    with pytest.raises(ToolError) as caught:
        asyncio.run(sandbox.execute(request))
    message = str(caught.value)
    assert "host fallback is disabled" in message and "Docker cannot run on Termux" in message
    leave_termux(monkeypatch)
    with pytest.raises(ToolError) as caught:
        asyncio.run(sandbox.execute(request))
    assert 'backend = "host"' in str(caught.value) and "Termux" not in str(caught.value)


def test_scrubbed_env_keeps_termux_runtime_but_not_secrets(monkeypatch):
    use_termux(monkeypatch)
    monkeypatch.setenv("ANDROID_ROOT", "/system")
    monkeypatch.setenv("LD_PRELOAD", PREFIX + "/lib/libtermux-exec.so")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_abcdefghij")
    env = scrubbed_env()
    assert env["ANDROID_ROOT"] == "/system" and env["LD_PRELOAD"].startswith(PREFIX)
    assert "GROQ_API_KEY" not in env
    monkeypatch.setenv("LD_PRELOAD", "/tmp/evil.so")
    assert "LD_PRELOAD" not in scrubbed_env()
    leave_termux(monkeypatch)
    monkeypatch.setenv("LD_PRELOAD", PREFIX + "/lib/libtermux-exec.so")
    assert "LD_PRELOAD" not in scrubbed_env()
    monkeypatch.delenv("LD_PRELOAD", raising=False)


class RecordingSandbox:
    isolated = False

    def __init__(self):
        self.requests = []

    async def execute(self, request):
        self.requests.append(request)
        return ExecutionResult("ok", "", 0)


def run_shell(tmp_path):
    sandbox = RecordingSandbox()
    tool = shell_tool(sandbox)
    context = ToolContext(workspace=tmp_path, max_output=1000, timeout=5.0)
    asyncio.run(tool.handler({"command": "echo hi"}, context))
    return sandbox.requests[0].argv


def test_shell_avoids_login_shell_on_termux(tmp_path, monkeypatch):
    use_termux(monkeypatch)
    assert run_shell(tmp_path)[1] == "-c"
    leave_termux(monkeypatch)
    assert run_shell(tmp_path)[1] == "-lc"


def test_host_sandbox_runs_a_real_command_on_termux_env(tmp_path, monkeypatch):
    use_termux(monkeypatch)
    request = ExecutionRequest(argv=("/bin/sh", "-c", "echo termux-ok"))
    result = asyncio.run(HostSandbox(tmp_path).execute(request))
    assert result.stdout.strip() == "termux-ok" and result.exit_code == 0


def test_tui_warns_when_shell_is_not_isolated(tmp_path):
    config = config_in(tmp_path, '[agent]\nmodel = "groq/llama"\n', TERMUX_ENV)
    console = Console(record=True, width=40, force_terminal=False)
    session = InteractiveSession(config, TerminalRenderer(console))
    session.renderer.startup(config)
    asyncio.run(session.submit("/status"))
    output = " ".join(console.export_text().split())
    assert "host, not isolated" in output
