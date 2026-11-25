"""Tasks the evaluation run sends through the local model and the real MCP servers."""

CASES = [
    {
        "name": "search_and_draft",
        "task": (
            "Search the docs for the onboarding guide and draft an email to "
            "alex@example.com summarizing what you find."
        ),
        "approvals": [True],
        "expect_status": "completed",
        "expect_tools": ["search_documents", "create_draft"],
        "check": "draft_has_onboarding_detail",
    },
    {
        "name": "budget_append",
        "task": "Read budget sheet A1:C4 and append a row Q4, 180, 100 to the budget sheet.",
        "approvals": [True],
        "expect_status": "completed",
        "expect_tools": ["read_range", "append_row"],
        "check": "budget_has_q4",
    },
    {
        "name": "budget_append_rejected",
        "task": "Read budget sheet A1:C4 and append a row Q4, 180, 100 to the budget sheet.",
        "approvals": [False],
        "expect_status": "completed",
        "expect_tools": ["read_range"],
        "check": "budget_unchanged",
    },
    {
        "name": "list_drafts",
        "task": "List the current email drafts.",
        "approvals": [],
        "expect_status": "completed",
        "expect_tools": ["list_drafts"],
        "check": "no_approval",
    },
    {
        "name": "step_limit",
        "task": (
            "Search the docs for the onboarding guide and draft an email to "
            "alex@example.com summarizing what you find."
        ),
        "approvals": [],
        "max_steps": 1,
        "expect_status": "stopped",
        "expect_tools": ["search_documents"],
        "check": "stopped_before_draft",
    },
    {
        "name": "spreadsheet_retry",
        "task": "Read budget sheet A1:C2 and append a row Q4, 10, 5 to the budget sheet.",
        "approvals": [True],
        "fail_first": 1,
        "expect_status": "completed",
        "expect_tools": ["read_range", "append_row"],
        "check": "retried_write",
    },
]
