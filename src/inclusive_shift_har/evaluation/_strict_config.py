"""Small strict-YAML helpers for immutable evaluation configurations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode
from yaml.tokens import AliasToken, AnchorToken


class StrictConfigError(ValueError):
    """Raised when an evaluation configuration is ambiguous or malformed."""


def _reject_duplicate_keys(node: Node, *, location: str = "$") -> None:
    if isinstance(node, MappingNode):
        seen: set[str] = set()
        for key_node, value_node in node.value:
            if not isinstance(key_node, ScalarNode) or not isinstance(key_node.value, str):
                raise StrictConfigError(f"{location} contains a non-scalar mapping key")
            key = key_node.value
            if key in seen:
                raise StrictConfigError(f"{location} contains duplicate key {key!r}")
            seen.add(key)
            _reject_duplicate_keys(value_node, location=f"{location}.{key}")
    elif isinstance(node, SequenceNode):
        for index, child in enumerate(node.value):
            _reject_duplicate_keys(child, location=f"{location}[{index}]")


def load_strict_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load one UTF-8 YAML mapping while rejecting aliases and duplicate keys."""

    source = Path(path)
    if source.is_symlink():
        raise StrictConfigError(f"refusing to load configuration through symlink: {source}")
    try:
        text = source.read_text(encoding="utf-8")
        tokens = yaml.scan(text, Loader=yaml.SafeLoader)
        if any(isinstance(token, (AliasToken, AnchorToken)) for token in tokens):
            raise StrictConfigError("YAML anchors and aliases are forbidden")
        node = yaml.compose(text, Loader=yaml.SafeLoader)
    except StrictConfigError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise StrictConfigError(f"cannot parse configuration {source}: {exc}") from exc
    if node is None:
        raise StrictConfigError(f"configuration is empty: {source}")
    _reject_duplicate_keys(node)
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise StrictConfigError(f"cannot parse configuration {source}: {exc}") from exc
    if not isinstance(parsed, Mapping) or any(not isinstance(key, str) for key in parsed):
        raise StrictConfigError("configuration root must be a string-keyed mapping")
    return dict(parsed)


def require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, location: str
) -> None:
    """Reject missing and unexpected keys instead of silently applying defaults."""

    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise StrictConfigError(
            f"{location} keys disagree with schema; missing={missing}, unexpected={unexpected}"
        )


def require_mapping(value: Any, *, location: str) -> dict[str, Any]:
    """Return a plain string-keyed mapping or fail with a contextual error."""

    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise StrictConfigError(f"{location} must be a string-keyed mapping")
    return dict(value)
