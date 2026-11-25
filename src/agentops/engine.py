"""Runs one task on the graph and resumes it after a person answers the gate."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from langgraph.types import Command

from agentops.config import Settings
from agentops.graph import GraphDeps, build_graph, initial_state
from agentops.llm import LanguageModel
from agentops.trace import Tracer


class UnknownRun(KeyError):
    pass


class NotWaiting(RuntimeError):
    pass


@dataclass
class RunView:
    run_id: str
    task: str
    status: str
    tool_calls: int
    results: list[dict]
    steps: list[dict]
    pending_approval: dict | None
    error: str | None

    def succeeded_tools(self) -> list[str]:
        return [item["tool"] for item in self.results if item["status"] == "succeeded"]


class AgentEngine:
    def __init__(self, settings: Settings, llm: LanguageModel, registry, tracer: Tracer) -> None:
        self.settings = settings
        self.tracer = tracer
        self.registry = registry
        self.graph = build_graph(
            GraphDeps(
                llm=llm,
                registry=registry,
                tracer=tracer,
                timeout=settings.tool_timeout_seconds,
                retries=settings.tool_retries,
                backoff=settings.retry_backoff_seconds,
            )
        )

    async def start(self, task: str, *, max_steps: int | None = None) -> RunView:
        run_id = uuid.uuid4().hex
        limit = self.settings.max_steps if max_steps is None else max_steps
        self.tracer.record(
            run_id,
            "run_started",
            detail={"task": task, "max_steps": limit},
        )
        result = await self.graph.ainvoke(
            initial_state(task, run_id, limit),
            _config(run_id),
        )
        return self._from_invoke(run_id, result)

    async def resume(self, run_id: str, *, approved: bool, comment: str = "") -> RunView:
        current = await self.get(run_id)
        if current.status != "awaiting_approval":
            raise NotWaiting(run_id)
        result = await self.graph.ainvoke(
            Command(resume={"approved": approved, "comment": comment}),
            _config(run_id),
        )
        return self._from_invoke(run_id, result)

    async def get(self, run_id: str) -> RunView:
        snapshot = await self.graph.aget_state(_config(run_id))
        if not snapshot.values:
            raise UnknownRun(run_id)
        pending = _pending(snapshot.interrupts)
        values = dict(snapshot.values)
        if pending is not None:
            values["status"] = "awaiting_approval"
        return _view(run_id, values, pending)

    def _from_invoke(self, run_id: str, result: dict) -> RunView:
        pending = _pending(result.get("__interrupt__"))
        if pending is not None:
            self.tracer.record(
                run_id,
                "approval_required",
                server=pending.get("server"),
                tool=pending.get("tool"),
                detail=pending,
            )
        values = dict(result)
        values.pop("__interrupt__", None)
        if pending is not None:
            values["status"] = "awaiting_approval"
        return _view(run_id, values, pending)


def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _pending(interrupts) -> dict | None:
    if not interrupts:
        return None
    first = interrupts[0]
    value = first.value if hasattr(first, "value") else first
    if isinstance(value, dict):
        return value
    return {"value": value}


def _view(run_id: str, values: dict, pending: dict | None) -> RunView:
    return RunView(
        run_id=run_id,
        task=values.get("task", ""),
        status=values.get("status", "running"),
        tool_calls=int(values.get("tool_calls") or 0),
        results=list(values.get("results") or []),
        steps=list(values.get("steps") or []),
        pending_approval=pending,
        error=values.get("error"),
    )
