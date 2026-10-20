import asyncio
import json

from eurus.config import build_config
from eurus.diagnose import probe_model
from eurus.models import ModelRef
from eurus.providers.testing import FakeTransport
from eurus.providers.transport import HttpError

ENV = {"GROQ_API_KEY": "gsk_secretsecret", "NVIDIA_API_KEY": "nvapi-secretsecret"}
OK_BODY = {
    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 4, "completion_tokens": 1},
}


def probe(model, transport):
    config = build_config({}, ENV)
    return asyncio.run(probe_model(config, ModelRef.parse(model), transport=transport, environ=ENV))


def test_probe_success_reports_reply():
    result = probe("groq/llama", FakeTransport(json_responses=[OK_BODY]))
    assert result.ok
    assert "ok" in result.detail


def test_probe_explains_cloudflare_block():
    transport = FakeTransport(json_responses=[HttpError(403, "error code: 1010")])
    result = probe("groq/llama", transport)
    assert not result.ok
    assert result.detail.startswith("edge_blocked:")


def test_probe_never_prints_the_key():
    body = json.dumps({"error": {"message": "bad key nvapi-secretsecret"}})
    result = probe("nvidia/meta/x", FakeTransport(json_responses=[HttpError(401, body)]))
    assert not result.ok
    assert "nvapi-secretsecret" not in result.detail


def test_probe_reports_missing_key_without_network():
    config = build_config({}, {})
    transport = FakeTransport()
    result = asyncio.run(
        probe_model(config, ModelRef.parse("groq/llama"), transport=transport, environ={})
    )
    assert not result.ok
    assert "GROQ_API_KEY" in result.detail
    assert transport.calls == []
