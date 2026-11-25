"""Read-only search over a small in-repo document corpus."""

from __future__ import annotations

import re

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP(
    "search",
    instructions="Search the internal document corpus. This server is read-only.",
    log_level="ERROR",
)

DOCUMENTS = [
    {
        "id": "onboarding-guide",
        "title": "Onboarding guide",
        "body": (
            "Welcome to the company. Your buddy will meet you in the lobby. "
            "Bring a government-issued photo id on day one. "
            "Laptop login uses your employee number. "
            "Accounts for email and chat are provisioned before you arrive."
        ),
    },
    {
        "id": "pricing",
        "title": "Pricing overview",
        "body": (
            "Standard plan is billed monthly. Enterprise plan adds a dedicated "
            "support channel and a custom data retention schedule."
        ),
    },
    {
        "id": "incident-response",
        "title": "Incident response",
        "body": (
            "Severity 1 pages the on-call lead. Severity 2 is handled in business "
            "hours. Write the timeline in the incident channel before closing."
        ),
    },
]

_READ = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)


def _terms(query: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2]


def _snippet(body: str) -> str:
    return body if len(body) <= 220 else body[:217] + "..."


@mcp.tool(annotations=_READ)
def search_documents(query: str, limit: int = 5) -> dict:
    """Search titles and bodies. Returns the closest documents, highest score first."""
    if not query.strip():
        raise ValueError("query is empty")
    if len(query) > 300:
        raise ValueError("query is too long")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    limit = min(limit, 10)

    terms = _terms(query)
    scored: list[tuple[int, dict]] = []
    needle = query.lower().strip()
    for document in DOCUMENTS:
        haystack = f"{document['title']} {document['body']}".lower()
        score = sum(haystack.count(term) for term in terms)
        if needle and needle in haystack:
            score += 3
        if score:
            scored.append((score, document))
    scored.sort(key=lambda item: (-item[0], item[1]["id"]))

    results = [
        {
            "id": document["id"],
            "title": document["title"],
            "snippet": _snippet(document["body"]),
            "score": score,
        }
        for score, document in scored[:limit]
    ]
    return {"query": query, "results": results}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
