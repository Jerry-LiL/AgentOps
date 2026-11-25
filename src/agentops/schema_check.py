"""Validate tool arguments against the JSON schema advertised by the MCP server."""

from __future__ import annotations

import jsonschema


class SchemaError(ValueError):
    """Arguments do not match the tool schema. These are not retried."""


def validate_arguments(schema: dict, arguments: dict) -> None:
    if not isinstance(arguments, dict):
        raise SchemaError("arguments must be an object")
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise SchemaError("tool schema is missing")
    try:
        jsonschema.validate(instance=arguments, schema=schema)
    except jsonschema.ValidationError as exc:
        raise SchemaError(exc.message) from exc
