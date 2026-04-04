"""
task_easy.py — Metadata and description for the Easy task.

Single bug classification — agent must identify the severity
of one bug report. No team assignment, no comments required.
"""

from __future__ import annotations

TASK_ID      = "easy"
TASK_NAME    = "Single Bug Classification"
DIFFICULTY   = "easy"
MAX_STEPS    = 3
SUCCESS_THRESHOLD = 0.6

DESCRIPTION = (
    "Classify the severity of a single bug report. "
    "Choose from: critical, high, medium, low. "
    "Partial credit is given for adjacent severity levels."
)

ACTION_SCHEMA = {
    "action":   "classify",
    "severity": "critical | high | medium | low",
}

SCORING = {
    "severity_exact":    1.0,
    "severity_adjacent": 0.5,
    "severity_wrong":    0.0,
    "skip_penalty":     -0.2,
    "invalid_penalty":  -0.3,
}

TIPS = [
    "Read the bug body carefully — crash/data loss = critical, security = critical.",
    "Labels like 'crash' or 'authentication' are strong severity hints.",
    "A typo or cosmetic issue is almost always low severity.",
]


def describe() -> dict:
    """Return a clean dict for the /tasks endpoint."""
    return {
        "task_id":          TASK_ID,
        "name":             TASK_NAME,
        "difficulty":       DIFFICULTY,
        "max_steps":        MAX_STEPS,
        "success_threshold": SUCCESS_THRESHOLD,
        "description":      DESCRIPTION,
        "action_schema":    ACTION_SCHEMA,
        "scoring":          SCORING,
        "tips":             TIPS,
    }