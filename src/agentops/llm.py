"""Function calling against OpenAI, or the local stand-in when no key is configured."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from agentops.config import Settings
from agentops.local_plan import plan_task, resolve_worker_call, task_from_messages

PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": "Submit the ordered MCP tool calls for this task.",
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "server": {
                                "type": "string",
                                "enum": ["search", "spreadsheet", "email_draft"],
                            },
                            "tool": {"type": "string"},
                            "arguments": {"type": "object"},
                            "objective": {"type": "string"},
                        },
                        "required": ["server", "tool", "arguments", "objective"],
                    },
                }
            },
            "required": ["steps"],
        },
    },
}


class ModelError(RuntimeError):
    """The model response was missing a usable function call."""


@dataclass
class FunctionCall:
    name: str
    arguments: dict


class LanguageModel(Protocol):
    async def function_call(self, messages: list[dict], tools: list[dict]) -> FunctionCall:
        """Return one function call. Both implementations share this shape."""


class LocalLanguageModel:
    """Same method as the OpenAI client. Rules live in local_plan.py."""

    async def function_call(self, messages: list[dict], tools: list[dict]) -> FunctionCall:
        names = _tool_names(tools)
        if "submit_plan" in names:
            return FunctionCall("submit_plan", {"steps": plan_task(task_from_messages(messages))})
        payload = _json_object(messages[-1]["content"])
        try:
            name, arguments = resolve_worker_call(payload, names)
        except ValueError as exc:
            raise ModelError(str(exc)) from exc
        return FunctionCall(name, arguments)


class OpenAILanguageModel:
    def __init__(self, api_key: str, model: str, client=None) -> None:
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key)
        self._client = client
        self._model = model

    async def function_call(self, messages: list[dict], tools: list[dict]) -> FunctionCall:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools,
                tool_choice="required",
            )
        except Exception as exc:
            raise ModelError(f"openai request failed: {exc}") from exc
        choice = response.choices[0].message
        tool_calls = getattr(choice, "tool_calls", None) or []
        if not tool_calls:
            raise ModelError("model returned no function call")
        call = tool_calls[0]
        name = call.function.name
        raw_arguments = call.function.arguments
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ModelError("function arguments were not json") from exc
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            raise ModelError("function arguments were not an object")
        if not isinstance(arguments, dict):
            raise ModelError("function arguments were not an object")
        return FunctionCall(name, arguments)


def build_llm(settings: Settings) -> LanguageModel:
    if settings.llm_mode == "local":
        return LocalLanguageModel()
    if settings.llm_mode == "openai":
        if not settings.openai_api_key:
            raise ModelError("OPENAI_API_KEY is required when AGENTOPS_LLM_MODE=openai")
        return OpenAILanguageModel(settings.openai_api_key, settings.openai_model)
    raise ModelError(f"unknown AGENTOPS_LLM_MODE: {settings.llm_mode}")


def normalize_steps(raw: object, *, limit: int = 16) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        raise ModelError("submit_plan returned no steps")
    steps = []
    for item in raw[:limit]:
        if not isinstance(item, dict):
            raise ModelError("a plan step was not an object")
        server = item.get("server")
        tool = item.get("tool")
        arguments = item.get("arguments", {})
        objective = item.get("objective") or ""
        if server not in {"search", "spreadsheet", "email_draft"}:
            raise ModelError(f"unknown server {server}")
        if not isinstance(tool, str) or not tool:
            raise ModelError("a plan step is missing a tool name")
        if not isinstance(arguments, dict):
            raise ModelError("tool arguments must be an object")
        steps.append(
            {
                "server": server,
                "tool": tool,
                "arguments": arguments,
                "objective": str(objective) or f"Call {tool}",
            }
        )
    return steps


def _tool_names(tools: list[dict]) -> set[str]:
    names = set()
    for tool in tools:
        function = tool.get("function") or {}
        if function.get("name"):
            names.add(function["name"])
    return names


def _json_object(content: str) -> dict:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ModelError("worker prompt was not json") from exc
    if not isinstance(parsed, dict):
        raise ModelError("worker prompt was not an object")
    return parsed
