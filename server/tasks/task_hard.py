"""
task_hard.py — Metadata and description for the Hard task.

Full code-review pipeline — 8 bugs + 3 PRs + final prioritization.
Agent must triage bugs, review PRs, detect security vulnerabilities,
and submit a prioritized fix order. Genuinely challenges frontier models.
"""

from __future__ import annotations

TASK_ID      = "hard"
TASK_NAME    = "Full Code Review Pipeline"
DIFFICULTY   = "hard"
MAX_STEPS    = 25
SUCCESS_THRESHOLD = 0.4

DESCRIPTION = (
    "Handle a mixed queue of 8 bug reports and 3 pull requests. "
    "For bugs: classify severity, assign team, write a comment, flag security issues. "
    "For PRs: give a verdict (approve/request_changes/comment), write a review, "
    "flag security vulnerabilities. "
    "After all items are processed, submit a prioritized fix order. "
    "Warning: at least one PR contains a critical security vulnerability."
)

BUG_ACTION_SCHEMA = {
    "action":         "triage",
    "severity":       "critical | high | medium | low",
    "team":           "frontend | backend | infra | security",
    "comment":        "<review comment — mention relevant technical keywords>",
    "security_issue": "true | false",
}

PR_ACTION_SCHEMA = {
    "action":         "review_pr",
    "verdict":        "approve | request_changes | comment",
    "comment":        "<review comment — call out specific code issues>",
    "security_issue": "true | false",
}

PRIORITIZE_ACTION_SCHEMA = {
    "action": "prioritize",
    "order":  ["BUG-XXX", "PR-XXX", "..."],
}

SCORING = {
    "bug": {
        "severity_weight":  0.30,
        "team_weight":      0.25,
        "comment_weight":   0.25,
        "security_weight":  0.20,
    },
    "pr": {
        "verdict_weight":   0.40,
        "security_weight":  0.40,
        "comment_weight":   0.20,
    },
    "prioritization": {
        "rank_accuracy_weight": 0.70,
        "coverage_weight":      0.30,
    },
    "penalties": {
        "skip":             -0.2,
        "invalid":          -0.3,
        "repeat":           -0.1,
        "false_negative_security": 0.0,
        "false_positive_security": 0.3,
    },
}

TIPS = [
    "Always read PR code snippets carefully — look for SQL injection, "
    "hardcoded secrets, missing input validation.",
    "f-string interpolation in SQL queries is always a critical security issue.",
    "Prioritize: critical security > critical crash > high production > "
    "high infra > PRs with security issues > medium > low.",
    "Missing a security issue costs you more than a false positive.",
    "You have 25 steps for 11 items + 1 prioritize = budget ~2 steps per item.",
]


def describe() -> dict:
    return {
        "task_id":              TASK_ID,
        "name":                 TASK_NAME,
        "difficulty":           DIFFICULTY,
        "max_steps":            MAX_STEPS,
        "success_threshold":    SUCCESS_THRESHOLD,
        "description":          DESCRIPTION,
        "bug_action_schema":    BUG_ACTION_SCHEMA,
        "pr_action_schema":     PR_ACTION_SCHEMA,
        "prioritize_schema":    PRIORITIZE_ACTION_SCHEMA,
        "scoring":              SCORING,
        "tips":                 TIPS,
    }