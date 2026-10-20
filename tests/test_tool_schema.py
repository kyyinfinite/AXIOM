import pytest

from eurus.tools.schema import validate_arguments

SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "count": {"type": "integer", "minimum": 1, "maximum": 5},
        "mode": {"type": "string", "enum": ["a", "b"]},
        "tags": {"type": "array", "items": {"type": "string"}},
        "ratio": {"type": "number"},
        "flag": {"type": "boolean"},
    },
    "required": ["path"],
    "additionalProperties": False,
}


def test_valid_arguments():
    value = {"path": "x", "count": 2, "mode": "a", "tags": ["t"], "ratio": 1, "flag": True}
    assert validate_arguments(SCHEMA, value) == []


@pytest.mark.parametrize(
    ("value", "fragment"),
    [
        ({}, "arguments.path is required"),
        ({"path": 1}, "arguments.path must be string"),
        ({"path": "x", "count": 0}, "at least 1"),
        ({"path": "x", "count": 9}, "at most 5"),
        ({"path": "x", "count": True}, "arguments.count must be integer"),
        ({"path": "x", "count": 1.5}, "arguments.count must be integer"),
        ({"path": "x", "mode": "z"}, "must be one of"),
        ({"path": "x", "tags": ["a", 2]}, "arguments.tags[1] must be string"),
        ({"path": "x", "extra": 1}, "arguments.extra is not allowed"),
        ({"path": "x", "flag": "yes"}, "arguments.flag must be boolean"),
        ("text", "arguments must be object"),
    ],
)
def test_invalid_arguments(value, fragment):
    errors = validate_arguments(SCHEMA, value)
    assert any(fragment in error for error in errors), errors


def test_union_types_and_unknown_types():
    assert validate_arguments({"type": ["string", "null"]}, None) == []
    assert validate_arguments({"type": ["string", "null"]}, 3) != []
    assert validate_arguments({"type": "mystery"}, 3) == []
