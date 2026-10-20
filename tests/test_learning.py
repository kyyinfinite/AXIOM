import asyncio
import json
from pathlib import Path

import pytest

from eurus.agent import Agent
from eurus.config import load_config
from eurus.errors import ConfigError
from eurus.learning import LessonStore, Reflector, parse_lessons, transcript_text
from eurus.models import Message, ToolCall
from eurus.providers.testing import openai_message, openai_stream, scripted_router
from eurus.tools import ToolContext, default_registry

LESSON = "Run the tests after every edit to confirm the fix."


def test_store_does_not_create_database_when_reading(tmp_path):
    database = tmp_path / "db.sqlite3"
    assert LessonStore(database).relevant("fix the failing tests") == []
    assert not database.exists()


def test_store_finds_relevant_lessons_by_any_term(tmp_path):
    store = LessonStore(tmp_path / "db.sqlite3")
    assert store.add(LESSON) is not None
    assert store.add("Prefer small patches over rewriting whole files.") is not None
    found = store.relevant("please fix the failing tests in this project")
    assert found == [LESSON]
    assert store.relevant("zzzz qqqq") == []


def test_store_rejects_near_duplicates_and_respects_cap(tmp_path):
    store = LessonStore(tmp_path / "db.sqlite3", max_lessons=2)
    assert store.add(LESSON) is not None
    assert store.add("Run the tests after every edit to confirm the fix!") is None
    assert store.add("Prefer small patches over rewriting files.") is not None
    assert store.add("Check git diff before reporting the final result.") is None


def test_parse_lessons_accepts_fenced_json_and_filters_bad_items():
    text = '```json\n["' + LESSON + '", "short", 7, "' + "x" * 300 + '"]\n```'
    assert parse_lessons(text) == [LESSON]


def test_parse_lessons_rejects_secrets_and_bad_json():
    assert parse_lessons('["Use the key sk-abcdefghijklmnop1234 for the API calls"]') == []
    assert parse_lessons("not json") == []
    assert parse_lessons('{"a": 1}') == []


def test_parse_lessons_caps_count():
    items = [f"Lesson number {index} about running tests first" for index in range(6)]
    assert len(parse_lessons(json.dumps(items))) == 3


def test_transcript_lists_calls_and_errors_not_outputs():
    call_ok = ToolCall("1", "read_file", {"path": "a.py"})
    call_bad = ToolCall("2", "shell", {"command": "pytest"})
    messages = [
        Message.assistant("", (call_ok, call_bad)),
        Message.tool("1", "SECRET FILE CONTENT", "read_file"),
        Message.tool("2", "error: runtime unavailable", "shell"),
        Message.assistant("all done"),
    ]
    text = transcript_text("fix it", messages)
    assert "read_file" in text and "-> ok" in text
    assert "error: runtime unavailable" in text
    assert "SECRET FILE CONTENT" not in text
    assert "Final answer: all done" in text


def tool_transcript(count):
    calls = tuple(ToolCall(str(i), "calculate", {"expression": "1+1"}) for i in range(count))
    messages = [Message.assistant("", calls)] if calls else []
    messages += [Message.tool(str(i), "2", "calculate") for i in range(count)]
    return messages + [Message.assistant("done")]


def reflect(tmp_path, mode, transcript, approver=None, reply=None):
    router, transport = scripted_router(json_responses=[openai_message(reply or f'["{LESSON}"]')])
    store = LessonStore(tmp_path / "db.sqlite3")
    reflector = Reflector(router, store, mode)
    saved = asyncio.run(reflector.learn("task", transcript, approver))
    return saved, store, transport


def test_reflector_off_and_short_runs_make_no_model_call(tmp_path):
    saved, _, transport = reflect(tmp_path, "off", tool_transcript(3))
    assert saved == [] and transport.calls == []
    saved, _, transport = reflect(tmp_path, "auto", tool_transcript(1))
    assert saved == [] and transport.calls == []


def test_reflector_ask_requires_approval(tmp_path):
    async def deny(call, tool):
        return False

    saved, store, _ = reflect(tmp_path, "ask", tool_transcript(3), approver=deny)
    assert saved == [] and store.all() == []
    saved, store, _ = reflect(tmp_path, "ask", tool_transcript(3), approver=None)
    assert saved == [] and store.all() == []


def test_reflector_ask_saves_when_approved(tmp_path):
    seen = []

    async def allow(call, tool):
        seen.append((call.name, call.arguments["lesson"], tool.requires_approval))
        return True

    saved, store, _ = reflect(tmp_path, "ask", tool_transcript(3), approver=allow)
    assert saved == [LESSON] and store.all() == [LESSON]
    assert seen == [("save_lesson", LESSON, True)]


def test_reflector_auto_saves_without_approval(tmp_path):
    saved, store, _ = reflect(tmp_path, "auto", tool_transcript(2))
    assert saved == [LESSON] and store.all() == [LESSON]


def make_agent(router, **kwargs):
    context = ToolContext(workspace=Path("."), max_output=1000, timeout=5.0)
    return Agent(router, default_registry(), context, system_prompt="sys", **kwargs)


class FixedLessons:
    def relevant(self, task):
        return ["Use small patches."]


def test_agent_injects_lessons_without_touching_history():
    router, transport = scripted_router(streams=[openai_stream("ok")])
    agent = make_agent(router, lessons=FixedLessons())

    async def go():
        return [event async for event in agent.run("do it")]

    asyncio.run(go())
    systems = [
        message["content"]
        for message in transport.calls[0].payload["messages"]
        if message["role"] == "system"
    ]
    assert any("Use small patches." in system for system in systems)
    assert agent.history[0].content == "<eurus-tier:stable>\nsys\n</eurus-tier:stable>"


def test_agent_reports_saved_lessons_and_survives_learner_failure(tmp_path):
    calls = [("c1", "calculate", {"expression": "1+1"}), ("c2", "calculate", {"expression": "2+2"})]
    router, _ = scripted_router(
        streams=[openai_stream("", calls), openai_stream("done")],
        json_responses=[openai_message(f'["{LESSON}"]')],
    )
    store = LessonStore(tmp_path / "db.sqlite3")
    agent = make_agent(router, learner=Reflector(router, store, "auto"))

    async def go(agent):
        return [event async for event in agent.run("compute")]

    events = asyncio.run(go(agent))
    assert [e.text for e in events if e.kind == "lesson"] == [LESSON]
    assert events[-1].kind == "lesson" and store.all() == [LESSON]

    class Broken:
        async def learn(self, task, transcript, approver):
            raise RuntimeError("boom")

    router, _ = scripted_router(streams=[openai_stream("fine")])
    events = asyncio.run(go(make_agent(router, learner=Broken())))
    assert events[-1].kind == "done"


def test_learning_config(tmp_path):
    env = {"EURUS_HOME": str(tmp_path)}
    config = load_config(cwd=tmp_path, environ=env)
    assert config.learning.mode == "ask" and config.learning.lessons_in_prompt == 5
    path = tmp_path / "eurus.toml"
    path.write_text('[learning]\nmode = "sometimes"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="learning.mode"):
        load_config(config_path=path, environ=env)
    path.write_text('[learning]\nmode = "auto"\nlessons_in_prompt = 99\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="lessons_in_prompt"):
        load_config(config_path=path, environ=env)


def test_lesson_store_marks_reflections_as_lesson_type(tmp_path):
    store = LessonStore(tmp_path / "db.sqlite3")
    memory_id = store.add(LESSON)
    assert memory_id is not None
    row = store._manager().store.list_memories(category="lesson")[0]
    assert row["memory_type"] == "lesson"
    assert row["source"] == "reflector"
