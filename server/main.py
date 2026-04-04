"""
main.py — FastAPI server exposing the BugTriageEnv as an OpenEnv-compliant API.

Endpoints
─────────
POST /reset          initialise or restart an episode
POST /step           send one action, get observation + reward
GET  /state          inspect full internal environment state
GET  /tasks          list all available tasks with descriptions
GET  /health         liveness probe (used by HF Spaces + validator)
GET  /               API info and quick-start guide

All request/response bodies are typed Pydantic models from models.py.
The single BugTriageEnv instance is module-level — one episode at a time.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .env import BugTriageEnv
from .models import BugAction, EnvState, ResetRequest, StepResult
from .tasks import task_easy, task_medium, task_hard


# ---------------------------------------------------------------------------
# Environment singleton
# ---------------------------------------------------------------------------

env = BugTriageEnv()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run a silent reset on startup so /state never returns 'not initialised'."""
    env.reset(task_id="easy")
    yield


app = FastAPI(
    title        = "Bug Triage OpenEnv",
    description  = (
        "A real-world software-engineering environment where an AI agent "
        "triages bug reports and reviews pull requests. "
        "Implements the full OpenEnv step()/reset()/state() interface."
    ),
    version      = "1.0.0",
    lifespan     = lifespan,
)

# Allow all origins — needed for HuggingFace Spaces iframe + external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_error(exc: Exception, context: str) -> JSONResponse:
    """Return a structured 500 with the error message rather than crashing."""
    return JSONResponse(
        status_code = 500,
        content     = {"error": context, "detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# OpenEnv core endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/reset",
    response_model = StepResult,
    summary        = "Reset the environment and start a new episode",
    tags           = ["OpenEnv"],
)
async def reset(body: ResetRequest = ResetRequest()) -> StepResult:
    """
    Initialise a fresh episode.

    - Send `{}` or omit body to default to the **easy** task.
    - Send `{"task_id": "medium"}` or `{"task_id": "hard"}` to select a task.

    Returns the first observation.
    """
    try:
        result = env.reset(task_id=body.task_id)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/step",
    response_model = StepResult,
    summary        = "Send one action and receive the next observation + reward",
    tags           = ["OpenEnv"],
)
async def step(action: BugAction) -> StepResult:
    """
    Advance the environment by one step.

    The `message` field must be a JSON string encoding the action.

    **Easy task example**
    ```json
    {"message": "{\"action\": \"classify\", \"severity\": \"high\"}"}
    ```

    **Medium task example**
    ```json
    {"message": "{\"action\": \"triage\", \"severity\": \"critical\", \"team\": \"security\"}"}
    ```

    **Hard task — bug example**
    ```json
    {
      "message": "{\"action\": \"triage\", \"severity\": \"critical\",
                   \"team\": \"security\", \"comment\": \"auth bypass vulnerability\",
                   \"security_issue\": true}"
    }
    ```

    **Hard task — PR example**
    ```json
    {
      "message": "{\"action\": \"review_pr\", \"verdict\": \"request_changes\",
                   \"comment\": \"SQL injection via f-string\",
                   \"security_issue\": true}"
    }
    ```

    **Hard task — final prioritization**
    ```json
    {
      "message": "{\"action\": \"prioritize\",
                   \"order\": [\"BUG-005\", \"BUG-001\", \"BUG-003\"]}"
    }
    ```
    """
    try:
        result = env.step(action)
        return result
    except RuntimeError as exc:
        # reset() not called yet
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/state",
    response_model = EnvState,
    summary        = "Return the full internal environment state",
    tags           = ["OpenEnv"],
)
async def state() -> EnvState:
    """
    Returns the complete internal state of the environment.

    Intended for debugging, logging, and grader inspection.
    Not shown to the agent during normal interaction.
    """
    try:
        return env.get_state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Discovery endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/tasks",
    summary = "List all available tasks with descriptions and scoring info",
    tags    = ["Discovery"],
)
async def list_tasks() -> dict[str, Any]:
    """
    Returns metadata for all three tasks: easy, medium, hard.

    Includes action schemas, scoring weights, difficulty, and tips.
    """
    return {
        "tasks": [
            task_easy.describe(),
            task_medium.describe(),
            task_hard.describe(),
        ]
    }


@app.get(
    "/tasks/{task_id}",
    summary = "Get description and schema for a specific task",
    tags    = ["Discovery"],
)
async def get_task(task_id: str) -> dict[str, Any]:
    """Return metadata for a single task by ID (easy / medium / hard)."""
    mapping = {
        "easy":   task_easy.describe,
        "medium": task_medium.describe,
        "hard":   task_hard.describe,
    }
    if task_id not in mapping:
        raise HTTPException(
            status_code = 404,
            detail      = f"Task '{task_id}' not found. Valid: easy, medium, hard",
        )
    return mapping[task_id]()


# ---------------------------------------------------------------------------
# Health + info
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    summary = "Liveness probe",
    tags    = ["System"],
)
async def health() -> dict[str, str]:
    """
    Returns `{"status": "ok"}`.

    Called by the HF Spaces health-check and the pre-submission validator.
    """
    return {"status": "ok"}


@app.get(
    "/",
    summary = "API overview and quick-start",
    tags    = ["System"],
)
async def root() -> dict[str, Any]:
    """Quick-start guide and endpoint map."""
    return {
        "name":        "Bug Triage OpenEnv",
        "version":     "1.0.0",
        "description": (
            "Real-world software engineering environment for AI agent training. "
            "Agents triage bug reports and review pull requests."
        ),
        "quickstart": [
            "1. POST /reset                     — start easy task",
            "2. POST /reset {task_id: 'medium'} — start medium task",
            "3. POST /step  {message: '<json>'}  — send action",
            "4. GET  /state                     — inspect state",
            "5. GET  /tasks                     — see all task schemas",
        ],
        "endpoints": {
            "POST /reset":         "Start a new episode",
            "POST /step":          "Send action, get observation + reward",
            "GET  /state":         "Full internal state",
            "GET  /tasks":         "All task descriptions",
            "GET  /tasks/{id}":    "Single task description",
            "GET  /health":        "Liveness probe",
        },
        "tasks": {
            "easy":   "Single bug severity classification (max_steps=3)",
            "medium": "5-bug triage queue — severity + team (max_steps=15)",
            "hard":   "8 bugs + 3 PRs + prioritization (max_steps=25)",
        },
        "docs": "/docs",
    }