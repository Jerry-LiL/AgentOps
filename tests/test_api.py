import json

import pytest
from fastapi.testclient import TestClient

from agentops.api import create_app
from agentops.config import Settings


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    data = tmp_path_factory.mktemp("api")
    settings = Settings(
        llm_mode="local",
        db_path=data / "trace.sqlite",
        data_dir=data,
        max_steps=8,
        tool_timeout_seconds=10,
        tool_retries=1,
        retry_backoff_seconds=0,
    )
    application = create_app(settings)
    with TestClient(application) as client:
        yield client, data


def test_health_and_tool_catalog(api):
    client, _data = api
    assert client.get("/health").json() == {"status": "ok"}
    tools = client.get("/v1/tools").json()["tools"]
    names = {tool["name"] for tool in tools}
    assert "search_documents" in names
    assert "append_row" in names
    assert "create_draft" in names
    write_tools = {tool["name"] for tool in tools if not tool["read_only"]}
    assert write_tools == {"write_range", "append_row", "create_draft", "update_draft"}


def test_search_and_draft_requires_approval(api):
    client, data = api
    started = client.post(
        "/v1/tasks",
        json={
            "task": (
                "Search the docs for the onboarding guide and draft an email to "
                "alex@example.com summarizing what you find."
            )
        },
    )
    assert started.status_code == 200
    body = started.json()
    assert body["status"] == "awaiting_approval"
    assert body["pending_approval"]["tool"] == "create_draft"
    assert body["results"][0]["tool"] == "search_documents"

    fetched = client.get(f"/v1/runs/{body['run_id']}")
    assert fetched.json()["status"] == "awaiting_approval"

    approved = client.post(
        f"/v1/runs/{body['run_id']}/approval",
        json={"approved": True, "comment": "send the draft"},
    )
    assert approved.status_code == 200
    finished = approved.json()
    assert finished["status"] == "completed"
    assert finished["pending_approval"] is None

    trace = client.get(f"/v1/runs/{body['run_id']}/trace").json()["events"]
    event_names = [event["event"] for event in trace]
    for name in (
        "run_started",
        "plan_created",
        "tool_call",
        "tool_result",
        "approval_required",
        "approval_decision",
        "run_finished",
    ):
        assert name in event_names

    drafts = (data / "drafts.json").read_text()
    assert "alex@example.com" in drafts
    assert "government-issued photo id" in drafts

    again = client.post(
        f"/v1/runs/{body['run_id']}/approval",
        json={"approved": True},
    )
    assert again.status_code == 409


def test_rejected_append_leaves_the_tracker(api):
    client, data = api
    started = client.post(
        "/v1/tasks",
        json={
            "task": (
                "Read tracker sheet A1:C1 and append a row gloves, sam, ordered "
                "to the tracker sheet."
            )
        },
    )
    body = started.json()
    assert body["status"] == "awaiting_approval"
    assert body["pending_approval"]["tool"] == "append_row"
    rejected = client.post(
        f"/v1/runs/{body['run_id']}/approval",
        json={"approved": False, "comment": "not yet"},
    )
    assert rejected.json()["status"] == "completed"
    book = json.loads((data / "workbook.json").read_text())
    assert book["tracker"] == [["item", "owner", "status"]]


def test_missing_run_and_blank_task(api):
    client, _data = api
    assert client.get("/v1/runs/missing").status_code == 404
    assert client.get("/v1/runs/missing/trace").status_code == 404
    assert client.post("/v1/runs/missing/approval", json={"approved": True}).status_code == 404
    blank = client.post("/v1/tasks", json={"task": "   "})
    assert blank.status_code == 422


def test_openai_mode_without_a_key_does_not_start(tmp_path):
    settings = Settings(
        llm_mode="openai",
        openai_api_key="",
        db_path=tmp_path / "trace.sqlite",
        data_dir=tmp_path,
    )
    application = create_app(settings)
    with pytest.raises(Exception):
        with TestClient(application):
            pass
