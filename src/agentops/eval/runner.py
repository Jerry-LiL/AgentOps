"""Run the fixed cases and write a report with success rate and tool-call counts."""

from __future__ import annotations

import json
from pathlib import Path

from agentops.config import Settings
from agentops.engine import AgentEngine
from agentops.eval.cases import CASES
from agentops.llm import LocalLanguageModel
from agentops.mcp_registry import MCPRegistry
from agentops.trace import Tracer


async def run_suite(report_path: Path | None = None, *, root: Path | None = None) -> dict:
    base = root or Path("eval-data")
    base.mkdir(parents=True, exist_ok=True)
    case_reports = []
    for case in CASES:
        case_reports.append(await _run_case(case, base / case["name"]))
    passed = sum(1 for item in case_reports if item["success"])
    total = len(case_reports)
    report = {
        "cases": case_reports,
        "summary": {
            "total": total,
            "passed": passed,
            "success_rate": round(passed / total, 4) if total else 0.0,
            "tool_calls": sum(item["tool_calls"] for item in case_reports),
        },
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


async def _run_case(case: dict, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        llm_mode="local",
        db_path=directory / "trace.sqlite",
        data_dir=directory,
        max_steps=case.get("max_steps", 8),
        tool_timeout_seconds=10,
        tool_retries=2,
        retry_backoff_seconds=0.01,
        spreadsheet_fail_first=case.get("fail_first", 0),
    )
    tracer = Tracer(settings.db_path)
    registry = MCPRegistry(settings)
    try:
        await registry.start()
        engine = AgentEngine(settings, LocalLanguageModel(), registry, tracer)
        view = await engine.start(case["task"])
        approvals = list(case.get("approvals") or [])
        while view.status == "awaiting_approval":
            if not approvals:
                raise RuntimeError("case ended while still waiting for approval")
            view = await engine.resume(
                view.run_id,
                approved=bool(approvals.pop(0)),
                comment="eval",
            )
        events = tracer.list_events(view.run_id)
        error = _expect(case, view, events, directory)
        return {
            "name": case["name"],
            "success": error is None,
            "status": view.status,
            "tool_calls": view.tool_calls,
            "tools": view.succeeded_tools(),
            "error": error,
        }
    except Exception as exc:
        return {
            "name": case["name"],
            "success": False,
            "status": "error",
            "tool_calls": 0,
            "tools": [],
            "error": str(exc),
        }
    finally:
        await registry.close()
        tracer.close()


def _expect(case: dict, view, events: list[dict], directory: Path) -> str | None:
    if view.status != case["expect_status"]:
        return f"status {view.status}, expected {case['expect_status']}: {view.error}"
    tools = view.succeeded_tools()
    if tools != case["expect_tools"]:
        return f"tools {tools}, expected {case['expect_tools']}"
    check = _CHECKS.get(case.get("check"))
    if check is None:
        return None
    return check(view, events, directory)


def _draft_has_onboarding_detail(view, events, directory: Path) -> str | None:
    del view, events
    drafts = _read(directory / "drafts.json", {"drafts": []})["drafts"]
    if not drafts:
        return "no draft was saved"
    draft = drafts[-1]
    if draft["to"] != "alex@example.com":
        return f"draft recipient was {draft['to']}"
    if "government-issued photo id" not in draft["body"]:
        return "draft body did not include the onboarding sentence"
    return None


def _budget_has_q4(view, events, directory: Path) -> str | None:
    del view, events
    rows = _read(directory / "workbook.json", {}).get("budget", [])
    if not rows or rows[-1][:3] != ["Q4", "180", "100"]:
        return f"budget ended with {rows[-1] if rows else None}"
    return None


def _budget_unchanged(view, events, directory: Path) -> str | None:
    del view, events
    rows = _read(directory / "workbook.json", {}).get("budget", [])
    if len(rows) != 4 or rows[-1][0] != "Q3":
        return f"budget changed: {rows}"
    return None


def _no_approval(view, events, directory: Path) -> str | None:
    del view, directory
    names = [event["event"] for event in events]
    if "approval_required" in names:
        return "a read-only run asked for approval"
    return None


def _stopped_before_draft(view, events, directory: Path) -> str | None:
    del view
    names = [event["event"] for event in events]
    if "step_limit" not in names:
        return "step limit was not recorded"
    drafts = _read(directory / "drafts.json", {"drafts": []})["drafts"]
    if drafts:
        return "a draft was written after the step limit"
    return None


def _retried_write(view, events, directory: Path) -> str | None:
    if not any(event["event"] == "tool_retry" for event in events):
        return "no retry was recorded"
    appends = [item for item in view.results if item["tool"] == "append_row"]
    if not appends or appends[-1]["attempts"] < 2:
        return "append did not record a second attempt"
    rows = _read(directory / "workbook.json", {}).get("budget", [])
    if not rows or rows[-1][:3] != ["Q4", "10", "5"]:
        return f"budget ended with {rows[-1] if rows else None}"
    return None


def _read(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


_CHECKS = {
    "draft_has_onboarding_detail": _draft_has_onboarding_detail,
    "budget_has_q4": _budget_has_q4,
    "budget_unchanged": _budget_unchanged,
    "no_approval": _no_approval,
    "stopped_before_draft": _stopped_before_draft,
    "retried_write": _retried_write,
}
