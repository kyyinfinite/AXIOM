from __future__ import annotations

from collections.abc import Callable
from typing import Any

TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, int | float) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "array": lambda value: isinstance(value, list),
    "object": lambda value: isinstance(value, dict),
    "null": lambda value: value is None,
}


def _type_matches(value: Any, expected: Any) -> bool:
    names = expected if isinstance(expected, list) else [expected]
    return any(TYPE_CHECKS.get(str(name), lambda _value: True)(value) for name in names)


def validate_arguments(schema: dict[str, Any], value: Any, path: str = "arguments") -> list[str]:
    expected = schema.get("type")
    if expected is not None and not _type_matches(value, expected):
        label = " or ".join(expected) if isinstance(expected, list) else str(expected)
        return [f"{path} must be {label}"]
    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} must be one of {schema['enum']}")
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                errors.append(f"{path}.{name} is required")
        for name, item in value.items():
            if name in properties:
                errors.extend(validate_arguments(properties[name], item, f"{path}.{name}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{name} is not allowed")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            errors.extend(validate_arguments(schema["items"], item, f"{path}[{index}]"))
    if isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path} must be at least {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path} must be at most {schema['maximum']}")
    return errors
