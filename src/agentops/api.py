"""HTTP API for starting a task and approving write steps."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator

from agentops.config import Settings
from agentops.engine import AgentEngine, NotWaiting, RunView, UnknownRun
from agentops.llm import ModelError, build_llm
from agentops.mcp_registry import MCPRegistry
from agentops.trace import Tracer


class TaskRequest(BaseModel):
    task: str

    @field_validator("task")
    @classmethod
    def task_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("task is empty")
        if len(text) > 2000:
            raise ValueError("task is too long")
        return text


class ApprovalRequest(BaseModel):
    approved: bool
    comment: str = ""

    @field_validator("comment")
    @classmethod
    def short_comment(cls, value: str) -> str:
        return value.strip()[:500]


class RunResponse(BaseModel):
    run_id: str
    task: str
    status: str
    tool_calls: int
    results: list[dict]
    steps: list[dict]
    pending_approval: dict | None = None
    error: str | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg: Settings = app.state.settings
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        tracer = Tracer(cfg.db_path)
        registry = MCPRegistry(cfg)
        try:
            llm = build_llm(cfg)
            await registry.start()
            app.state.tracer = tracer
            app.state.registry = registry
            app.state.engine = AgentEngine(cfg, llm, registry, tracer)
            yield
        finally:
            await registry.close()
            tracer.close()

    app = FastAPI(title="AgentOps", lifespan=lifespan)
    app.state.settings = settings

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/tools")
    async def tools(request: Request) -> dict:
        specs = await request.app.state.registry.list_all()
        return {"tools": [asdict(spec) for spec in specs]}

    @app.post("/v1/tasks", response_model=RunResponse)
    async def create_task(body: TaskRequest, request: Request) -> RunResponse:
        view = await request.app.state.engine.start(body.task)
        return _response(view)

    @app.get("/v1/runs/{run_id}", response_model=RunResponse)
    async def get_run(run_id: str, request: Request) -> RunResponse:
        view = await _load(request, run_id)
        return _response(view)

    @app.post("/v1/runs/{run_id}/approval", response_model=RunResponse)
    async def approve(run_id: str, body: ApprovalRequest, request: Request) -> RunResponse:
        engine: AgentEngine = request.app.state.engine
        try:
            view = await engine.resume(run_id, approved=body.approved, comment=body.comment)
        except UnknownRun:
            raise HTTPException(status_code=404, detail="run not found") from None
        except NotWaiting:
            raise HTTPException(status_code=409, detail="run is not waiting for approval") from None
        return _response(view)

    @app.get("/v1/runs/{run_id}/trace")
    async def get_trace(run_id: str, request: Request) -> dict:
        await _load(request, run_id)
        events = request.app.state.tracer.list_events(run_id)
        return {"run_id": run_id, "events": events}

    return app


async def _load(request: Request, run_id: str) -> RunView:
    try:
        return await request.app.state.engine.get(run_id)
    except UnknownRun:
        raise HTTPException(status_code=404, detail="run not found") from None


def _response(view: RunView) -> RunResponse:
    return RunResponse(
        run_id=view.run_id,
        task=view.task,
        status=view.status,
        tool_calls=view.tool_calls,
        results=view.results,
        steps=view.steps,
        pending_approval=view.pending_approval,
        error=view.error,
    )


app = create_app()


def main() -> None:
    settings = Settings.from_env()
    try:
        build_llm(settings)
    except ModelError as exc:
        raise SystemExit(str(exc)) from exc
    uvicorn.run(
        "agentops.api:app",
        host=settings.host,
        port=settings.port,
        factory=False,
    )
