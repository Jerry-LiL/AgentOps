import json

import pytest

from agentops.config import Settings
from agentops.mcp_registry import MCPRegistry
from agentops.schema_check import SchemaError, validate_arguments
from agentops.servers import email_draft, search, spreadsheet


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTOPS_DATA_DIR", str(tmp_path))
    spreadsheet._failures_left = None
    return tmp_path


def test_search_ranks_the_onboarding_guide():
    found = search.search_documents("onboarding guide", limit=2)
    assert found["results"][0]["id"] == "onboarding-guide"
    assert "photo id" in found["results"][0]["snippet"]


def test_search_rejects_an_empty_query():
    with pytest.raises(ValueError):
        search.search_documents("   ")


def test_spreadsheet_read_write_and_append(data_dir):
    values = spreadsheet.read_range("budget", "A1:C4")["values"]
    assert values[0] == ["quarter", "revenue", "cost"]
    assert values[3][0] == "Q3"

    written = spreadsheet.write_range("budget", "B2:B2", [["200"]])
    assert written["updated"] == 1
    assert spreadsheet.read_range("budget", "B2")["values"] == [["200"]]

    appended = spreadsheet.append_row("budget", ["Q4", "180", "100"])
    assert appended["values"] == ["Q4", "180", "100"]
    book = json.loads((data_dir / "workbook.json").read_text())
    assert book["budget"][-1] == ["Q4", "180", "100"]


def test_spreadsheet_flake_does_not_write(data_dir, monkeypatch):
    monkeypatch.setenv("AGENTOPS_SPREADSHEET_FAIL_FIRST", "1")
    spreadsheet._failures_left = None
    spreadsheet.read_range("budget", "A1:A1")
    with pytest.raises(RuntimeError, match="transient"):
        spreadsheet.append_row("budget", ["Q4", "1", "2"])
    book = json.loads((data_dir / "workbook.json").read_text())
    assert book["budget"][-1][0] == "Q3"
    spreadsheet.append_row("budget", ["Q4", "1", "2"])
    book = json.loads((data_dir / "workbook.json").read_text())
    assert book["budget"][-1] == ["Q4", "1", "2"]


def test_email_draft_create_and_update(data_dir):
    created = email_draft.create_draft("alex@example.com", "Hello", "First version")
    updated = email_draft.update_draft(created["id"], subject="Hello again", body="Second version")
    assert updated["subject"] == "Hello again"
    listed = email_draft.list_drafts()["drafts"]
    assert listed[0]["body"] == "Second version"
    with pytest.raises(ValueError, match="unknown draft"):
        email_draft.update_draft("draft_9", body="nope")
    saved = json.loads((data_dir / "drafts.json").read_text())
    assert saved["drafts"][0]["body"] == "Second version"


async def test_stdio_servers_expose_schemas_and_writes(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        db_path=tmp_path / "trace.sqlite",
        spreadsheet_fail_first=0,
    )
    registry = MCPRegistry(settings)
    await registry.start()
    try:
        specs = await registry.list_all()
        by_name = {spec.name: spec for spec in specs}
        assert set(by_name) == {
            "search_documents",
            "read_range",
            "write_range",
            "append_row",
            "list_drafts",
            "create_draft",
            "update_draft",
        }
        for name in ("search_documents", "read_range", "list_drafts"):
            assert by_name[name].read_only
        for name in ("write_range", "append_row", "create_draft", "update_draft"):
            assert not by_name[name].read_only

        append_schema = by_name["append_row"].input_schema
        with pytest.raises(SchemaError):
            validate_arguments(append_schema, {"sheet": "budget"})
        validate_arguments(append_schema, {"sheet": "budget", "values": ["Q4", "1", "2"]})

        found = await registry.call_tool(
            "search", "search_documents", {"query": "retention schedule", "limit": 1}
        )
        assert found.ok
        assert found.output["results"][0]["id"] == "pricing"

        written = await registry.call_tool(
            "spreadsheet",
            "write_range",
            {"sheet": "budget", "cell_range": "C2:C2", "values": [["77"]]},
        )
        assert written.ok
        read_back = await registry.call_tool(
            "spreadsheet", "read_range", {"sheet": "budget", "cell_range": "C2"}
        )
        assert read_back.output["values"] == [["77"]]

        drafted = await registry.call_tool(
            "email_draft",
            "create_draft",
            {"to": "alex@example.com", "subject": "Hello", "body": "A draft."},
        )
        assert drafted.ok
        revised = await registry.call_tool(
            "email_draft",
            "update_draft",
            {"draft_id": drafted.output["id"], "body": "A revised draft."},
        )
        assert revised.ok
        listed = await registry.call_tool("email_draft", "list_drafts", {})
        assert listed.output["drafts"][0]["body"] == "A revised draft."
    finally:
        await registry.close()
