"""Planner / worker graph.

The planner submits a plan with a function call. The worker turns each step
into another function call, then a separate node performs the MCP call. Write
tools stop at an interrupt before that call happens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agentops.llm import PLAN_TOOL, LanguageModel, ModelError, normalize_steps
from agentops.mcp_registry import MCPRegistry, ToolResult
from agentops.retry import RetryLimit, call_with_retry
from agentops.schema_check import SchemaError, validate_arguments
from agentops.trace import Tracer

PLANNER_SYSTEM = (
    "Break the task into ordered tool calls on the search, spreadsheet, and "
    "email_draft MCP servers. Call submit_plan once. Use only tools from the "
    "catalog. Read before you write. Arguments must match the tool schema."
)

WORKER_SYSTEM = (
    "Call exactly one tool for this step. Prefer the suggested tool and "
    "arguments when they match the schema. When drafting an email, use the "
    "prior search results in the body."
)


class GraphState(TypedDict, total=False):
    task: str
    run_id: str
    max_steps: int
    steps: list[dict]
    cursor: int
    executed: int
    tool_calls: int
    approved: bool
    results: list[dict]
    status: str
    error: str | None


@dataclass
class GraphDeps:
    llm: LanguageModel
    registry: MCPRegistry
    tracer: Tracer
    timeout: float
    retries: int
    backoff: float


def build_graph(deps: GraphDeps):
    async def planner(state: GraphState) -> dict:
        catalog = [
            {
                "server": spec.server,
                "name": spec.name,
                "description": spec.description,
                "read_only": spec.read_only,
                "parameters": spec.input_schema,
            }
            for spec in await deps.registry.list_all()
        ]
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {
                "role": "user",
                "content": json.dumps({"task": state["task"], "catalog": catalog}),
            },
        ]
        try:
            call = await deps.llm.function_call(messages, [PLAN_TOOL])
            if call.name != "submit_plan":
                raise ModelError(f"expected submit_plan, got {call.name}")
            steps = normalize_steps(call.arguments.get("steps"))
        except ModelError as exc:
            deps.tracer.record(
                state["run_id"],
                "plan_failed",
                detail={"error": str(exc)},
            )
            return {"steps": [], "status": "failed", "error": str(exc)}
        deps.tracer.record(
            state["run_id"],
            "plan_created",
            detail={"steps": steps},
        )
        return {"steps": steps, "status": "running", "cursor": 0}

    async def worker(state: GraphState) -> dict:
        step = state["steps"][state["cursor"]]
        specs = await deps.registry.list_tools(step["server"])
        messages = [
            {"role": "system", "content": WORKER_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "objective": step["objective"],
                        "suggested_tool": step["tool"],
                        "suggested_arguments": step["arguments"],
                        "prior_results": _public_results(state.get("results", [])),
                    }
                ),
            },
        ]
        try:
            call = await deps.llm.function_call(messages, [_openai_tool(spec) for spec in specs])
        except ModelError as exc:
            return _failed(state, step, str(exc))
        spec = next((item for item in specs if item.name == call.name), None)
        if spec is None:
            return _failed(state, step, f"unknown tool {call.name}", tool=call.name)
        try:
            validate_arguments(spec.input_schema, call.arguments)
        except SchemaError as exc:
            deps.tracer.record(
                state["run_id"],
                "validation_failed",
                server=step["server"],
                tool=call.name,
                detail={"error": str(exc), "arguments": call.arguments},
            )
            return _failed(
                state,
                step,
                str(exc),
                tool=call.name,
                arguments=call.arguments,
            )
        steps = [dict(item) for item in state["steps"]]
        updated = dict(steps[state["cursor"]])
        updated["resolved_tool"] = call.name
        updated["resolved_arguments"] = call.arguments
        updated["read_only"] = spec.read_only
        steps[state["cursor"]] = updated
        return {"steps": steps}

    async def approval(state: GraphState) -> dict:
        step = state["steps"][state["cursor"]]
        # On resume this node starts over, and interrupt() returns the decision
        # instead of pausing. Keep the trace write below that call.
        decision = interrupt(
            {
                "run_id": state["run_id"],
                "step_index": state["cursor"],
                "server": step["server"],
                "tool": step["resolved_tool"],
                "arguments": step["resolved_arguments"],
                "objective": step["objective"],
            }
        )
        approved = isinstance(decision, dict) and bool(decision.get("approved"))
        comment = ""
        if isinstance(decision, dict):
            comment = str(decision.get("comment") or "")
        deps.tracer.record(
            state["run_id"],
            "approval_decision",
            server=step["server"],
            tool=step["resolved_tool"],
            detail={"approved": approved, "comment": comment},
        )
        if approved:
            return {"approved": True}
        entry = _entry(
            step,
            tool=step["resolved_tool"],
            arguments=step["resolved_arguments"],
            status="rejected",
            error=comment or "rejected by reviewer",
            attempts=0,
        )
        return {
            "approved": False,
            "cursor": state["cursor"] + 1,
            "executed": state["executed"] + 1,
            "results": [*state.get("results", []), entry],
        }

    async def call_tool(state: GraphState) -> dict:
        step = state["steps"][state["cursor"]]
        server = step["server"]
        tool = step["resolved_tool"]
        arguments = step["resolved_arguments"]
        attempt_count = {"n": 0}

        async def invoke() -> ToolResult:
            attempt_count["n"] += 1
            deps.tracer.record(
                state["run_id"],
                "tool_call",
                server=server,
                tool=tool,
                detail={"attempt": attempt_count["n"], "arguments": arguments},
            )
            return await deps.registry.call_tool(server, tool, arguments)

        async def on_retry(attempt: int, reason: str) -> None:
            deps.tracer.record(
                state["run_id"],
                "tool_retry",
                server=server,
                tool=tool,
                detail={"attempt": attempt, "reason": reason},
            )

        try:
            result, attempts = await call_with_retry(
                invoke,
                timeout=deps.timeout,
                retries=deps.retries,
                backoff=deps.backoff,
                on_retry=on_retry,
            )
        except RetryLimit as exc:
            result = ToolResult(ok=False, output=None, error=exc.reason, transient=True)
            attempts = exc.attempts
        except Exception as exc:
            result = ToolResult(ok=False, output=None, error=str(exc), transient=False)
            attempts = max(attempt_count["n"], 1)

        entry = _entry(
            step,
            tool=tool,
            arguments=arguments,
            status="succeeded" if result.ok else "failed",
            output=result.output,
            error=result.error,
            attempts=attempts,
        )
        deps.tracer.record(
            state["run_id"],
            "tool_result",
            server=server,
            tool=tool,
            detail={"ok": result.ok, "attempts": attempts, "error": result.error},
        )
        update = {
            "results": [*state.get("results", []), entry],
            "cursor": state["cursor"] + 1,
            "executed": state["executed"] + 1,
            "tool_calls": state.get("tool_calls", 0) + attempts,
            "approved": False,
        }
        if not result.ok:
            update["status"] = "failed"
            update["error"] = result.error
        return update

    def finish(state: GraphState) -> dict:
        if state.get("status") == "failed":
            status = "failed"
        elif state.get("cursor", 0) < len(state.get("steps", [])):
            status = "stopped"
        else:
            status = "completed"
        if status == "stopped":
            deps.tracer.record(
                state["run_id"],
                "step_limit",
                detail={
                    "max_steps": state.get("max_steps"),
                    "executed": state.get("executed", 0),
                },
            )
        deps.tracer.record(
            state["run_id"],
            "run_finished",
            detail={"status": status, "tool_calls": state.get("tool_calls", 0)},
        )
        return {"status": status}

    builder = StateGraph(GraphState)
    builder.add_node("planner", planner)
    builder.add_node("worker", worker)
    builder.add_node("approval", approval)
    builder.add_node("call_tool", call_tool)
    builder.add_node("finish", finish)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        _after_planner,
        {"worker": "worker", "finish": "finish"},
    )
    builder.add_conditional_edges(
        "worker",
        _after_worker,
        {"approval": "approval", "call_tool": "call_tool", "finish": "finish"},
    )
    builder.add_conditional_edges(
        "approval",
        _after_approval,
        {"call_tool": "call_tool", "worker": "worker", "finish": "finish"},
    )
    builder.add_conditional_edges(
        "call_tool",
        _after_call,
        {"worker": "worker", "finish": "finish"},
    )
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=MemorySaver())


def initial_state(task: str, run_id: str, max_steps: int) -> GraphState:
    return {
        "task": task,
        "run_id": run_id,
        "max_steps": max_steps,
        "steps": [],
        "cursor": 0,
        "executed": 0,
        "tool_calls": 0,
        "approved": False,
        "results": [],
        "status": "planning",
        "error": None,
    }


def _next(state: GraphState) -> str:
    if state.get("status") == "failed":
        return "finish"
    steps = state.get("steps", [])
    if state.get("cursor", 0) >= len(steps):
        return "finish"
    if state.get("executed", 0) >= state.get("max_steps", 1):
        return "finish"
    return "worker"


def _after_planner(state: GraphState) -> str:
    if state.get("status") == "failed" or not state.get("steps"):
        return "finish"
    hop = _next(state)
    return "finish" if hop == "finish" else "worker"


def _after_worker(state: GraphState) -> str:
    if state.get("status") == "failed":
        return "finish"
    step = state["steps"][state["cursor"]]
    if step.get("read_only") or state.get("approved"):
        return "call_tool"
    return "approval"


def _after_approval(state: GraphState) -> str:
    if state.get("approved"):
        return "call_tool"
    return _next(state)


def _after_call(state: GraphState) -> str:
    if state.get("status") == "failed":
        return "finish"
    return _next(state)


def _failed(
    state: GraphState,
    step: dict,
    error: str,
    *,
    tool: str | None = None,
    arguments: dict | None = None,
) -> dict:
    entry = _entry(
        step,
        tool=tool or step.get("tool", ""),
        arguments=arguments if arguments is not None else step.get("arguments", {}),
        status="failed",
        error=error,
        attempts=0,
    )
    return {
        "status": "failed",
        "error": error,
        "results": [*state.get("results", []), entry],
    }


def _entry(
    step: dict,
    *,
    tool: str,
    arguments: dict,
    status: str,
    output: dict | None = None,
    error: str | None = None,
    attempts: int,
) -> dict:
    return {
        "server": step["server"],
        "tool": tool,
        "arguments": arguments,
        "status": status,
        "output": output,
        "error": error,
        "attempts": attempts,
    }


def _public_results(results: list[dict]) -> list[dict]:
    return [
        {
            "server": item.get("server"),
            "tool": item.get("tool"),
            "status": item.get("status"),
            "output": item.get("output"),
            "error": item.get("error"),
        }
        for item in results
    ]


def _openai_tool(spec) -> dict:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description or spec.name,
            "parameters": spec.input_schema,
        },
    }
