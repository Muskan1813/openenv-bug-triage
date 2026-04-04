"""
task_medium.py — Metadata and description for the Medium task.

Bug queue triage — agent must classify severity AND assign
each of 5 bug reports to the correct engineering team.
"""

from __future__ import annotations

TASK_ID      = "medium"
TASK_NAME    = "Bug Queue Triage"
DIFFICULTY   = "medium"
MAX_STEPS    = 15
SUCCESS_THRESHOLD = 0.5

DESCRIPTION = (
    "Process a queue of 5 bug reports. For each bug you must "
    "classify its severity (critical/high/medium/low) AND assign it "
    "to the correct team (frontend/backend/infra/security). "
    "You have 15 steps for 5 items — use them wisely."
)

ACTION_SCHEMA = {
    "action":   "triage",
    "severity": "critical | high | medium | low",
    "team":     "frontend | backend | infra | security",
    "comment":  "<optional, not graded>",
}

SCORING = {
    "severity_weight":   0.50,
    "team_weight":       0.50,
    "severity_exact":    1.0,
    "severity_adjacent": 0.5,
    "severity_wrong":    0.0,
    "team_exact":        1.0,
    "team_wrong":        0.0,
    "skip_penalty":     -0.2,
    "invalid_penalty":  -0.3,
}

TIPS = [
    "frontend = UI, charts, browser issues.",
    "backend = APIs, databases, business logic.",
    "infra = Kubernetes, deployments, memory, CI/CD.",
    "security = auth, rate limiting, injection, access control.",
    "A bug can be high severity but belong to frontend — severity and team are independent.",
]


def describe() -> dict:
    return {
        "task_id":           TASK_ID,
        "name":              TASK_NAME,
        "difficulty":        DIFFICULTY,
        "max_steps":         MAX_STEPS,
        "success_threshold": SUCCESS_THRESHOLD,
        "description":       DESCRIPTION,
        "action_schema":     ACTION_SCHEMA,
        "scoring":           SCORING,
        "tips":              TIPS,
    }