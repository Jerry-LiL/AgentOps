# AgentOps

AgentOps takes a short task, splits it into steps, and runs those steps against three MCP servers: search, spreadsheet, and email draft. A person has to approve every write before it happens. The run leaves a SQLite trace of the plan, each tool call, and the approval decision.

The model behind the planner and the worker is swappable. Without a key, a local function-calling stand-in maps a few plain phrases onto the tools. With `AGENTOPS_LLM_MODE=openai` the same calls go to the OpenAI chat completions API.

## Features

- A LangGraph planner submits a plan as a function call. A worker then makes one function call per step, scoped to that step's MCP server.
- Three stdio MCP servers, built with the MCP Python SDK: document search, a JSON workbook, and email drafts. Drafts are saved, not sent.
- Write tools (`write_range`, `append_row`, `create_draft`, `update_draft`) pause on a LangGraph interrupt. Read tools run straight through. A missing `readOnlyHint` is treated as a write.
- Tool arguments are checked with JSON Schema against the schema from `tools/list` before the call. A schema failure is not retried.
- Each call has a timeout. Timeouts and transient tool errors are retried with backoff. A step counter ends the run when the plan would keep going.
- Every stage is appended to SQLite: the plan, validation failures, approval requests and decisions, each attempt, and the final status.
- The loop is served by FastAPI. `POST /v1/tasks` starts a run, `POST /v1/runs/{id}/approval` resumes it.
- GitHub Actions runs the unit tests and an evaluation suite. The suite writes a report with a success rate and a tool-call count per case.

## Architecture

```mermaid
flowchart TD
  api[FastAPI]
  planner[Planner function call]
  worker[Worker function call]
  gate[Approval interrupt]
  call[MCP call with timeout and retry]
  search[search server]
  sheet[spreadsheet server]
  mail[email draft server]
  db[(SQLite trace)]

  api --> planner
  planner --> worker
  worker -->|read tool| call
  worker -->|write tool| gate
  gate -->|approved| call
  gate -->|rejected| worker
  call --> search
  call --> sheet
  call --> mail
  planner --> db
  worker --> db
  gate --> db
  call --> db
```

The planner only produces steps. The worker chooses the tool and the arguments. A third node is the one that actually calls MCP, so a rejected approval never reaches the server. The LangGraph checkpointer holds the paused run in memory for the life of the process. The SQLite file is the audit log and is what you read afterwards.

## Folder structure

```
src/agentops/
  api.py                 HTTP routes
  engine.py              start a run, resume after approval
  graph.py               planner, worker, approval, tool call
  llm.py                 OpenAI function calling and the local stand-in
  local_plan.py          phrase rules used when no API key is set
  mcp_registry.py        stdio sessions for the three servers
  schema_check.py        JSON Schema check
  retry.py               timeout and retry
  trace.py               SQLite event log
  config.py              environment settings
  servers/
    search.py            document search
    spreadsheet.py       workbook read, write, append
    email_draft.py       list, create, update drafts
  eval/
    cases.py             fixed tasks
    runner.py            runs them and writes the report
tests/                   unit tests and one end-to-end pass per server
.github/workflows/eval.yml
Dockerfile
docker-compose.yml
```

## Running locally

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
python -m agentops
```

The service listens on port 8000 and uses the local model. Workbook and draft files land in `./data`.

```bash
curl -s localhost:8000/v1/tasks \
  -H 'content-type: application/json' \
  -d '{"task":"Search the docs for the onboarding guide and draft an email to alex@example.com summarizing what you find."}'
```

The response status is `awaiting_approval` once search has finished and the draft is about to be saved. Copy `run_id` and `pending_approval`, then:

```bash
curl -s localhost:8000/v1/runs/$RUN_ID/approval \
  -H 'content-type: application/json' \
  -d '{"approved": true, "comment": "looks right"}'
```

`GET /v1/runs/$RUN_ID` returns the same view. `GET /v1/runs/$RUN_ID/trace` returns the event log. `GET /v1/tools` lists the tools and which ones are read-only. `GET /health` is the process check.

To call OpenAI instead, set `AGENTOPS_LLM_MODE=openai` and `OPENAI_API_KEY` in the environment. The key is not written to the trace and is not passed into the MCP server processes. The process exits at startup if the mode is `openai` and the key is missing.

### Docker

```bash
docker compose up --build
```

The container uses the local model. The SQLite file and the workbook live in the `agentops-data` volume, mounted at `/data`.

## Tests

```bash
ruff check src tests
ruff format --check src tests
pytest
python -m agentops.eval --report eval-report.json
```

`pytest` covers schema checks, retries, the approval gate, the step limit, the three MCP servers over stdio, and the HTTP API. The eval command runs six scripted tasks against the real servers and writes `eval-report.json`. Each case records whether it passed, the tools that actually ran, and how many MCP calls were made, including retries. The workflow in `.github/workflows/eval.yml` runs the same commands and uploads that report.

The spreadsheet server has a test hook, `AGENTOPS_SPREADSHEET_FAIL_FIRST`. When it is set to `1`, the next write fails once with a transient error so the retry path can be exercised. Leave it at `0` otherwise.

## Design decisions

The planner and the worker both go through one method, `function_call(messages, tools)`. The OpenAI client sends those tools to chat completions with `tool_choice=required` and reads the first tool call. The local client implements the same method with a small set of phrase rules, so the rest of the loop does not grow a second code path. The local rules are there for development and for the eval suite. They are not a model.

Write versus read comes from the MCP `readOnlyHint` annotation on each tool. The gate interrupts before `call_tool`. On resume the approval node starts again, `interrupt()` returns the decision, and only then is the decision written to SQLite. That keeps the pause from recording twice.

Validation uses the schema returned by the server, not a copy kept in the client. Invalid arguments fail the step. Transient errors (the spreadsheet hook says `transient`) and timeouts go through the retry budget. Anything else returns immediately.

The step limit is checked before every step, including steps that would have asked for approval. A plan that is longer than `AGENTOPS_MAX_STEPS` stops with status `stopped` and a `step_limit` event. The counter is what stops a loop if a future change forgets to advance the cursor.

The checkpointer is in-memory. A restart drops paused runs. The trace file does not. That split is deliberate: the interrupt only has to survive until the approval request, and the trace is the record you keep. Swapping in a SQLite checkpointer would be the change if a pause had to outlive the process.

Draft tools never send mail. Search reads a fixed three-document corpus compiled into the server. The workbook starts with a small budget sheet and a tracker sheet the first time it is opened.

## Configuration

See `.env.example`. Nothing in the repo is a secret. `OPENAI_API_KEY` stays empty unless you set it yourself.
