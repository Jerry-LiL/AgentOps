"""Keeps a stdio session open to each MCP server."""

from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agentops.config import Settings

SERVERS = {
    "search": "agentops.servers.search",
    "spreadsheet": "agentops.servers.spreadsheet",
    "email_draft": "agentops.servers.email_draft",
}

# The child process does not need the model key.
_HIDDEN_ENV = {"OPENAI_API_KEY"}


@dataclass
class ToolSpec:
    server: str
    name: str
    description: str
    input_schema: dict
    read_only: bool


@dataclass
class ToolResult:
    ok: bool
    output: dict | None
    error: str | None
    transient: bool


class MCPRegistry:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._stack: AsyncExitStack | None = None
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[str, list[ToolSpec]] = {}

    async def start(self) -> None:
        self._settings.data_dir.mkdir(parents=True, exist_ok=True)
        stack = AsyncExitStack()
        await stack.__aenter__()
        self._stack = stack
        try:
            for name, module in SERVERS.items():
                params = StdioServerParameters(
                    command=sys.executable,
                    args=["-m", module],
                    env=_child_env(self._settings, name),
                )
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._sessions[name] = session
            await self._load_tools()
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        stack = self._stack
        self._stack = None
        self._sessions.clear()
        if stack is not None:
            await stack.aclose()

    async def list_tools(self, server: str) -> list[ToolSpec]:
        self._require(server)
        return list(self._tools.get(server, []))

    async def list_all(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for name in SERVERS:
            specs.extend(await self.list_tools(name))
        return specs

    async def call_tool(self, server: str, name: str, arguments: dict) -> ToolResult:
        session = self._require(server)
        try:
            result = await session.call_tool(name, arguments)
        except Exception as exc:
            return ToolResult(ok=False, output=None, error=str(exc), transient=False)
        text = ""
        if result.content:
            text = getattr(result.content[0], "text", "") or ""
        if result.isError:
            transient = "transient" in text.lower()
            return ToolResult(
                ok=False,
                output=None,
                error=text or "tool error",
                transient=transient,
            )
        return ToolResult(ok=True, output=_parse_output(text), error=None, transient=False)

    async def _load_tools(self) -> None:
        for server, session in self._sessions.items():
            listed = await session.list_tools()
            specs = []
            for tool in listed.tools:
                read_only = bool(tool.annotations and tool.annotations.readOnlyHint)
                specs.append(
                    ToolSpec(
                        server=server,
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=dict(tool.inputSchema),
                        read_only=read_only,
                    )
                )
            self._tools[server] = specs

    def _require(self, server: str) -> ClientSession:
        try:
            return self._sessions[server]
        except KeyError as exc:
            raise KeyError(f"unknown MCP server {server}") from exc


def _child_env(settings: Settings, server: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key not in _HIDDEN_ENV}
    env["AGENTOPS_DATA_DIR"] = str(settings.data_dir)
    src = str(Path(__file__).resolve().parents[1])
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not current else os.pathsep.join((src, current))
    if server == "spreadsheet":
        env["AGENTOPS_SPREADSHEET_FAIL_FIRST"] = str(settings.spreadsheet_fail_first)
    else:
        env.pop("AGENTOPS_SPREADSHEET_FAIL_FIRST", None)
    return env


def _parse_output(text: str) -> dict:
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}
