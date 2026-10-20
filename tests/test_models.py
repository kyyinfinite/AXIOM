import pytest

from eurus.errors import ModelRefError
from eurus.models import Message, ModelRef, ToolCall, ToolResult, Usage


def test_message_roundtrip():
    original = Message.assistant("working", (ToolCall("1", "bash", {"cmd": "ls"}),))
    assert Message.from_dict(original.to_dict()) == original


def test_tool_message_requires_call_id():
    with pytest.raises(ValueError):
        Message(role="tool", content="x")


def test_only_assistant_carries_tool_calls():
    with pytest.raises(ValueError):
        Message(role="user", tool_calls=(ToolCall("1", "bash"),))


def test_invalid_role_rejected():
    with pytest.raises(ValueError):
        Message(role="boss")


def test_model_ref_keeps_nested_model_name():
    ref = ModelRef.parse("openrouter/anthropic/some-model")
    assert ref.provider == "openrouter"
    assert ref.model == "anthropic/some-model"
    assert str(ref) == "openrouter/anthropic/some-model"


@pytest.mark.parametrize("text", ["", "nomodel", "/model", "provider/"])
def test_model_ref_rejects_invalid(text):
    with pytest.raises(ModelRefError):
        ModelRef.parse(text)


def test_usage_addition():
    total = Usage(1, 2) + Usage(3, 4)
    assert total.total_tokens == 10


def test_tool_result_to_message():
    message = ToolResult("1", "bash", "ok").to_message()
    assert message.role == "tool"
    assert message.tool_call_id == "1"


def test_tool_call_extra_roundtrip():
    call = ToolCall("1", "bash", {"cmd": "ls"}, {"thoughtSignature": "s"})
    assert ToolCall.from_dict(call.to_dict()) == call
    assert "extra" not in ToolCall("1", "bash").to_dict()
