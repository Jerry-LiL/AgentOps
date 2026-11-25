"""Email drafts stored as JSON. Nothing is sent; create and update only save a draft."""

from __future__ import annotations

import re

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from agentops.servers.store import load_json, save_json

mcp = FastMCP(
    "email-draft",
    instructions="List, create, and update email drafts. This server never sends mail.",
    log_level="ERROR",
)

_READ = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
_WRITE = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)

_DRAFT_ID = re.compile(r"^draft_\d+$")


def _drafts() -> list[dict]:
    payload = load_json("drafts.json", {"drafts": []})
    return list(payload["drafts"])


def _save(drafts: list[dict]) -> None:
    save_json("drafts.json", {"drafts": drafts})


def _check_recipient(to: str) -> None:
    if len(to) > 200 or "@" not in to or to.startswith("@") or to.endswith("@"):
        raise ValueError("invalid recipient")


def _check_subject(subject: str) -> None:
    if not subject.strip() or len(subject) > 200:
        raise ValueError("invalid subject")


def _check_body(body: str) -> None:
    if not body.strip() or len(body) > 8000:
        raise ValueError("invalid body")


def _next_id(drafts: list[dict]) -> str:
    taken = {draft["id"] for draft in drafts}
    number = 1
    while f"draft_{number}" in taken:
        number += 1
    return f"draft_{number}"


@mcp.tool(annotations=_READ)
def list_drafts() -> dict:
    """Return every saved draft."""
    return {"drafts": _drafts()}


@mcp.tool(annotations=_WRITE)
def create_draft(to: str, subject: str, body: str) -> dict:
    """Save a new draft and return its id. This does not send the email."""
    _check_recipient(to)
    _check_subject(subject)
    _check_body(body)
    drafts = _drafts()
    draft = {
        "id": _next_id(drafts),
        "to": to,
        "subject": subject.strip(),
        "body": body.strip(),
    }
    drafts.append(draft)
    _save(drafts)
    return {"id": draft["id"], "status": "draft", "to": draft["to"], "subject": draft["subject"]}


@mcp.tool(annotations=_WRITE)
def update_draft(
    draft_id: str,
    subject: str | None = None,
    body: str | None = None,
) -> dict:
    """Replace the subject, the body, or both on an existing draft."""
    if not _DRAFT_ID.fullmatch(draft_id):
        raise ValueError("invalid draft id")
    if subject is None and body is None:
        raise ValueError("nothing to update")
    if subject is not None:
        _check_subject(subject)
    if body is not None:
        _check_body(body)
    drafts = _drafts()
    for draft in drafts:
        if draft["id"] != draft_id:
            continue
        if subject is not None:
            draft["subject"] = subject.strip()
        if body is not None:
            draft["body"] = body.strip()
        _save(drafts)
        return {"id": draft_id, "status": "draft", "subject": draft["subject"]}
    raise ValueError(f"unknown draft {draft_id}")


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
