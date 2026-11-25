import json

from agentops.config import Settings
from agentops.engine import AgentEngine
from agentops.llm import FunctionCall
from agentops.trace import Tracer
from tests.fakes import FakeRegistry


class ScriptedModel:
    def __init__(self, steps: list[dict]) -> None:
        self.steps = steps

    async def function_call(self, messages, tools) -> FunctionCall:
        names = [tool["function"]["name"] for tool in tools]
        if "submit_plan" in names:
            return FunctionCall("submit_plan", {"steps": self.steps})
        payload = json.loads(messages[-1]["content"])
        return FunctionCall(payload["suggested_tool"], dict(payload["suggested_arguments"]))


def _settings(tmp_path, **overrides) -> Settings:
    values = {
        "llm_mode": "local",
        "db_path": tmp_path / "trace.sqlite",
        "data_dir": tmp_path,
        "max_steps": 8,
        "tool_timeout_seconds": 1,
        "tool_retries": 1,
        "retry_backoff_seconds": 0,
    }
    values.update(overrides)
    return Settings(**values)


def _step(server: str, tool: str, arguments: dict) -> dict:
    return {
        "server": server,
        "tool": tool,
        "arguments": arguments,
        "objective": tool,
    }


def _engine(tmp_path, steps, registry=None, **settings):
    registry = registry or FakeRegistry()
    tracer = Tracer(tmp_path / "trace.sqlite")
    engine = AgentEngine(_settings(tmp_path, **settings), ScriptedModel(steps), registry, tracer)
    return engine, registry, tracer


async def test_write_does_not_run_until_approved(tmp_path):
    draft = {
        "to": "alex@example.com",
        "subject": "Hello",
        "body": "A short note.",
    }
    engine, registry, tracer = _engine(
        tmp_path,
        [_step("email_draft", "create_draft", draft)],
    )
    view = await engine.start("draft a note")
    assert view.status == "awaiting_approval"
    assert registry.calls == []
    assert view.pending_approval["tool"] == "create_draft"
    assert "approval_required" in [event["event"] for event in tracer.list_events(view.run_id)]

    finished = await engine.resume(view.run_id, approved=True, comment="ok")
    assert finished.status == "completed"
    assert registry.calls == [("email_draft", "create_draft", draft)]
    assert "approval_decision" in [event["event"] for event in tracer.list_events(view.run_id)]
    tracer.close()


async def test_rejected_write_is_skipped(tmp_path):
    engine, registry, tracer = _engine(
        tmp_path,
        [
            _step("spreadsheet", "read_range", {"sheet": "budget", "cell_range": "A1:C4"}),
            _step("spreadsheet", "append_row", {"sheet": "budget", "values": ["Q4", "1", "2"]}),
        ],
    )
    view = await engine.start("update the sheet")
    assert view.status == "awaiting_approval"
    assert [call[1] for call in registry.calls] == ["read_range"]

    finished = await engine.resume(view.run_id, approved=False, comment="no")
    assert finished.status == "completed"
    assert [call[1] for call in registry.calls] == ["read_range"]
    assert finished.results[-1]["status"] == "rejected"
    tracer.close()


async def test_read_only_plan_does_not_ask(tmp_path):
    engine, registry, tracer = _engine(
        tmp_path,
        [_step("search", "search_documents", {"query": "onboarding", "limit": 3})],
    )
    view = await engine.start("search")
    assert view.status == "completed"
    assert view.pending_approval is None
    assert registry.calls[0][1] == "search_documents"
    tracer.close()


async def test_invalid_arguments_never_call_the_tool(tmp_path):
    engine, registry, tracer = _engine(
        tmp_path,
        [_step("spreadsheet", "append_row", {"sheet": "budget"})],
    )
    view = await engine.start("append a broken row")
    assert view.status == "failed"
    assert registry.calls == []
    events = [event["event"] for event in tracer.list_events(view.run_id)]
    assert "validation_failed" in events
    assert "tool_call" not in events
    tracer.close()


async def test_step_limit_stops_a_long_plan(tmp_path):
    steps = [_step("search", "search_documents", {"query": f"item {index}"}) for index in range(5)]
    engine, registry, tracer = _engine(tmp_path, steps, max_steps=2)
    view = await engine.start("loop", max_steps=2)
    assert view.status == "stopped"
    assert len(registry.calls) == 2
    assert view.tool_calls == 2
    assert "step_limit" in [event["event"] for event in tracer.list_events(view.run_id)]
    tracer.close()


async def test_transient_tool_error_is_retried(tmp_path):
    registry = FakeRegistry()
    registry.transient_remaining = 1
    engine, registry, tracer = _engine(
        tmp_path,
        [_step("spreadsheet", "append_row", {"sheet": "budget", "values": ["Q4"]})],
        registry=registry,
        tool_retries=1,
    )
    view = await engine.start("append")
    finished = await engine.resume(view.run_id, approved=True)
    assert finished.status == "completed"
    assert [call[1] for call in registry.calls] == ["append_row", "append_row"]
    assert finished.results[-1]["attempts"] == 2
    assert "tool_retry" in [event["event"] for event in tracer.list_events(view.run_id)]
    tracer.close()


async def test_timeout_fails_the_step(tmp_path):
    registry = FakeRegistry()
    registry.slow = True
    engine, registry, tracer = _engine(
        tmp_path,
        [_step("search", "search_documents", {"query": "slow"})],
        registry=registry,
        tool_timeout_seconds=0.05,
        tool_retries=0,
    )
    view = await engine.start("search slowly")
    assert view.status == "failed"
    assert view.error == "timed out"
    assert len(registry.calls) == 1
    tracer.close()
