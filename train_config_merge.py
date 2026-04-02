from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict


def to_plain_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_plain_data(val) for key, val in dict(value).items()}
    if isinstance(value, tuple):
        return tuple(to_plain_data(item) for item in value)
    return value


def diff_explicit_config_overrides(parsed: Any, base: Any) -> Dict[str, Any]:
    parsed_data = to_plain_data(parsed)
    base_data = to_plain_data(base)

    if not isinstance(parsed_data, dict):
        raise TypeError("parsed must resolve to a mapping-like object")
    if not isinstance(base_data, dict):
        raise TypeError("base must resolve to a mapping-like object")

    diff: Dict[str, Any] = {}
    for key, parsed_value in parsed_data.items():
        base_value = base_data.get(key)
        if isinstance(parsed_value, dict) and isinstance(base_value, dict):
            nested_diff = diff_explicit_config_overrides(parsed_value, base_value)
            if nested_diff:
                diff[key] = nested_diff
        elif parsed_value != base_value:
            diff[key] = parsed_value
    return diff
