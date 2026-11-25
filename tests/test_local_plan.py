import json

import pytest

from agentops.config import Settings
from agentops.llm import (
    FunctionCall,
    LocalLanguageModel,
    ModelError,
    OpenAILanguageModel,
    build_llm,
)
from agentops.local_plan import plan_task, resolve_worker_call


def test_search_and_draft_plan():
    steps = plan_task(
        "Search the docs for the onboarding guide and draft an email to "
        "alex@example.com summarizing what you find."
    )
    assert [step["tool"] for step in steps] == ["search_documents", "create_draft"]
    assert steps[0]["arguments"]["query"] == "onboarding guide"
    assert steps[1]["arguments"]["to"] == "alex@example.com"
    assert steps[1]["arguments"]["subject"] == "Onboarding guide"


def test_sheet_read_and_append_plan():
    steps = plan_task("Read budget sheet A1:C4 and append a row Q4, 180, 100 to the budget sheet.")
    assert [step["tool"] for step in steps] == ["read_range", "append_row"]
    assert steps[0]["arguments"] == {"sheet": "budget", "cell_range": "A1:C4"}
    assert steps[1]["arguments"]["values"] == ["Q4", "180", "100"]


def test_list_drafts_is_only_a_read():
    steps = plan_task("List the current email drafts.")
    assert steps == [
        {
            "server": "email_draft",
            "tool": "list_drafts",
            "arguments": {},
            "objective": "List email drafts",
        }
    ]


def test_write_range_and_update_draft_plans():
    write_steps = plan_task("Update the budget sheet range B2:B2 with 200.")
    assert write_steps[0]["tool"] == "write_range"
    assert write_steps[0]["arguments"]["values"] == [["200"]]

    update_steps = plan_task(
        "Revise email draft draft_1 subject to Hello and body to Thanks for the help."
    )
    assert update_steps[0]["tool"] == "update_draft"
    assert update_steps[0]["arguments"]["draft_id"] == "draft_1"
    assert update_steps[0]["arguments"]["subject"] == "Hello"
    assert update_steps[0]["arguments"]["body"] == "Thanks for the help."


def test_unknown_task_falls_back_to_search():
    steps = plan_task("what is the retention schedule")
    assert steps[0]["tool"] == "search_documents"
    assert steps[0]["arguments"]["query"] == "what is the retention schedule"


def test_worker_copies_search_hits_into_the_draft():
    name, arguments = resolve_worker_call(
        {
            "suggested_tool": "create_draft",
            "suggested_arguments": {
                "to": "alex@example.com",
                "subject": "Onboarding guide",
                "body": "Summary of onboarding guide.",
            },
            "prior_results": [
                {
                    "output": {
                        "results": [
                            {
                                "title": "Onboarding guide",
                                "snippet": "Bring a government-issued photo id on day one.",
                            }
                        ]
                    }
                }
            ],
        },
        {"create_draft", "list_drafts"},
    )
    assert name == "create_draft"
    assert "government-issued photo id" in arguments["body"]


def test_settings_and_llm_factory(monkeypatch):
    monkeypatch.setenv("AGENTOPS_LLM_MODE", "local")
    monkeypatch.setenv("AGENTOPS_MAX_STEPS", "4")
    settings = Settings.from_env()
    assert settings.max_steps == 4
    assert settings.llm_mode == "local"
    model = build_llm(settings)
    assert isinstance(model, LocalLanguageModel)

    openai_settings = Settings(llm_mode="openai", openai_api_key="")
    with pytest.raises(ModelError):
        build_llm(openai_settings)


def test_openai_client_reads_the_function_call():
    class _Function:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class _Call:
        def __init__(self, name, arguments):
            self.function = _Function(name, arguments)

    class _Message:
        def __init__(self, arguments):
            self.tool_calls = [_Call("submit_plan", arguments)]

    class _Completions:
        def __init__(self, arguments):
            self.arguments = arguments
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            message = _Message(self.arguments)
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    completions = _Completions('{"steps": []}')
    client = type("Client", (), {})()
    client.chat = type("Chat", (), {"completions": completions})()
    model = OpenAILanguageModel("test-key", "gpt-4o-mini", client=client)

    async def run():
        return await model.function_call(
            [{"role": "user", "content": "plan this"}],
            [{"type": "function", "function": {"name": "submit_plan"}}],
        )

    import asyncio

    call = asyncio.run(run())
    assert call == FunctionCall("submit_plan", {"steps": []})
    assert completions.kwargs["tool_choice"] == "required"
    assert completions.kwargs["model"] == "gpt-4o-mini"

    completions.arguments = "not-json"

    with pytest.raises(ModelError):
        asyncio.run(run())


def test_local_model_submits_a_plan():
    import asyncio

    model = LocalLanguageModel()

    async def run():
        return await model.function_call(
            [{"role": "user", "content": json.dumps({"task": "List the current email drafts."})}],
            [{"type": "function", "function": {"name": "submit_plan", "parameters": {}}}],
        )

    call = asyncio.run(run())
    assert call.name == "submit_plan"
    assert call.arguments["steps"][0]["tool"] == "list_drafts"
