"""Deterministic stand-in for the function-calling model.

The phrases below are the ones the service and the eval suite send. Anything
unrecognised becomes a single search step, which is still a real tool call.
"""

from __future__ import annotations

import json
import re

SERVERS = ("search", "spreadsheet", "email_draft")
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
_RANGE = re.compile(r"\b([A-Za-z]{1,3}\d+:[A-Za-z]{1,3}\d+)\b")


def task_from_messages(messages: list[dict]) -> str:
    content = messages[-1]["content"]
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return str(content)
    if isinstance(parsed, dict) and isinstance(parsed.get("task"), str):
        return parsed["task"]
    return str(content)


def plan_task(task: str) -> list[dict]:
    text = " ".join(task.split())
    lowered = text.lower()
    steps: list[dict] = []

    if re.search(r"\b(search|find|look up|docs|documents)\b", lowered):
        query = _search_query(text)
        steps.append(
            _step(
                "search",
                "search_documents",
                {"query": query, "limit": 3},
                f"Search documents for {query}",
            )
        )

    if any(word in lowered for word in ("sheet", "spreadsheet", "budget", "tracker")):
        steps.extend(_sheet_steps(text, lowered))

    if "draft" in lowered or "email" in lowered:
        steps.append(_email_step(text, lowered))

    if not steps:
        steps.append(
            _step(
                "search",
                "search_documents",
                {"query": text, "limit": 3},
                f"Search documents for {text}",
            )
        )
    return steps


def resolve_worker_call(payload: dict, tool_names: set[str]) -> tuple[str, dict]:
    """Pick the suggested tool, and fold earlier search hits into a draft body."""
    tool = payload.get("suggested_tool")
    if tool not in tool_names:
        raise ValueError(f"suggested tool {tool} is not on this server")
    arguments = dict(payload.get("suggested_arguments") or {})
    if tool == "create_draft":
        lines = []
        for prior in payload.get("prior_results") or []:
            output = prior.get("output") or {}
            for hit in output.get("results") or []:
                title = hit.get("title") or hit.get("id") or "document"
                snippet = hit.get("snippet") or ""
                lines.append(f"{title}: {snippet}".strip())
        if lines:
            arguments["body"] = "Summary of what I found:\n" + "\n".join(lines)
    return str(tool), arguments


def _sheet_steps(text: str, lowered: str) -> list[dict]:
    if "budget" in lowered:
        sheet = "budget"
    elif "tracker" in lowered:
        sheet = "tracker"
    else:
        sheet = "sheet1"
    cell_range = _cell_range(text) or "A1:C4"
    steps = []
    if re.search(r"\b(read|show|open)\b", lowered):
        steps.append(
            _step(
                "spreadsheet",
                "read_range",
                {"sheet": sheet, "cell_range": cell_range},
                f"Read {sheet} {cell_range}",
            )
        )
    if "append" in lowered and "row" in lowered:
        steps.append(
            _step(
                "spreadsheet",
                "append_row",
                {"sheet": sheet, "values": _row_values(text)},
                f"Append a row to {sheet}",
            )
        )
    elif re.search(r"\b(write|update)\b", lowered) and "draft" not in lowered:
        steps.append(
            _step(
                "spreadsheet",
                "write_range",
                {"sheet": sheet, "cell_range": cell_range, "values": _grid_values(text)},
                f"Write {cell_range} on {sheet}",
            )
        )
    return steps


def _email_step(text: str, lowered: str) -> dict:
    if "list" in lowered:
        return _step("email_draft", "list_drafts", {}, "List email drafts")
    if "update" in lowered or "revise" in lowered:
        return _step(
            "email_draft",
            "update_draft",
            _draft_update(text),
            "Update an email draft",
        )
    return _step(
        "email_draft",
        "create_draft",
        _draft_create(text),
        "Draft an email",
    )


def _step(server: str, tool: str, arguments: dict, objective: str) -> dict:
    return {
        "server": server,
        "tool": tool,
        "arguments": arguments,
        "objective": objective,
    }


def _search_query(text: str) -> str:
    match = re.search(
        r"\b(?:for|about)\s+(?:the\s+)?(.+?)(?:\s+and\s+|\s+then\s+|$)",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip(" .")
    return text.strip()


def _cell_range(text: str) -> str | None:
    match = _RANGE.search(text)
    if match:
        return match.group(1).upper()
    return None


def _row_values(text: str) -> list[str]:
    match = re.search(r"append a row\s+(.+?)(?:\s+to\b|$)", text, re.IGNORECASE)
    if not match:
        return ["", "", ""]
    return _cells(match.group(1))


def _grid_values(text: str) -> list[list[str]]:
    match = re.search(r"\bwith\s+(.+)$", text, re.IGNORECASE)
    if not match:
        return [[""]]
    rows = []
    for line in match.group(1).split(";"):
        cells = _cells(line)
        if cells:
            rows.append(cells)
    return rows or [[""]]


def _cells(chunk: str) -> list[str]:
    return [part.strip(" .") for part in chunk.split(",") if part.strip(" .")]


def _draft_create(text: str) -> dict:
    found = _EMAIL.search(text)
    to = found.group(0) if found else "team@example.com"
    query = _search_query(text)
    if query == text.strip():
        subject = "Notes"
    else:
        subject = query[0].upper() + query[1:]
    if len(subject) > 80:
        subject = subject[:77] + "..."
    return {"to": to, "subject": subject, "body": f"Summary of {query}."}


def _draft_update(text: str) -> dict:
    found_id = re.search(r"draft_\d+", text.lower())
    subject = re.search(r"subject to\s+(.+?)(?:\s+and\s+body\b|$)", text, re.IGNORECASE)
    body = re.search(r"body to\s+(.+)$", text, re.IGNORECASE)
    arguments: dict[str, str] = {
        "draft_id": found_id.group(0) if found_id else "draft_1",
    }
    if subject:
        arguments["subject"] = subject.group(1).strip(" .")
    if body:
        arguments["body"] = body.group(1).strip()
    return arguments
