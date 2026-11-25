"""In-memory tool server for graph tests."""

from __future__ import annotations

import asyncio

from agentops.mcp_registry import ToolResult, ToolSpec


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required}


def _string(name: str) -> dict:
    return {"type": "string", "title": name}


class FakeRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.transient_remaining = 0
        self.slow = False
        self.specs = [
            ToolSpec(
                "search",
                "search_documents",
                "Search documents.",
                _schema({"query": _string("Query"), "limit": {"type": "integer"}}, ["query"]),
                True,
            ),
            ToolSpec(
                "spreadsheet",
                "read_range",
                "Read a range.",
                _schema(
                    {"sheet": _string("Sheet"), "cell_range": _string("Range")},
                    ["sheet", "cell_range"],
                ),
                True,
            ),
            ToolSpec(
                "spreadsheet",
                "append_row",
                "Append a row.",
                _schema(
                    {"sheet": _string("Sheet"), "values": {"type": "array"}},
                    ["sheet", "values"],
                ),
                False,
            ),
            ToolSpec(
                "spreadsheet",
                "write_range",
                "Write a range.",
                _schema(
                    {
                        "sheet": _string("Sheet"),
                        "cell_range": _string("Range"),
                        "values": {"type": "array"},
                    },
                    ["sheet", "cell_range", "values"],
                ),
                False,
            ),
            ToolSpec(
                "email_draft",
                "list_drafts",
                "List drafts.",
                _schema({}, []),
                True,
            ),
            ToolSpec(
                "email_draft",
                "create_draft",
                "Create a draft.",
                _schema(
                    {"to": _string("To"), "subject": _string("Subject"), "body": _string("Body")},
                    ["to", "subject", "body"],
                ),
                False,
            ),
            ToolSpec(
                "email_draft",
                "update_draft",
                "Update a draft.",
                _schema(
                    {
                        "draft_id": _string("Id"),
                        "subject": _string("Subject"),
                        "body": _string("Body"),
                    },
                    ["draft_id"],
                ),
                False,
            ),
        ]

    async def list_all(self) -> list[ToolSpec]:
        return list(self.specs)

    async def list_tools(self, server: str) -> list[ToolSpec]:
        return [spec for spec in self.specs if spec.server == server]

    async def call_tool(self, server: str, name: str, arguments: dict) -> ToolResult:
        self.calls.append((server, name, arguments))
        if self.slow:
            await asyncio.sleep(1)
        if self.transient_remaining and name in {"append_row", "write_range"}:
            self.transient_remaining -= 1
            return ToolResult(
                ok=False,
                output=None,
                error="transient spreadsheet backend error",
                transient=True,
            )
        return ToolResult(
            ok=True,
            output={"tool": name, "arguments": arguments},
            error=None,
            transient=False,
        )
