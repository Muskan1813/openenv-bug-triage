"""
grader_medium.py — Grader for Task: medium

Task goal   : Triage a queue of 5 bug reports — classify severity AND assign team.
Agent action: {
    "action":   "triage",
    "severity": "critical|high|medium|low",
    "team":     "frontend|backend|infra|security",
    "comment":  "<optional>"
}
Scoring per bug:
    severity  : 0.50 weight  (1.0 exact / 0.5 adjacent / 0.0 wrong)
    team      : 0.50 weight  (1.0 exact / 0.0 wrong)
Max reward  : 1.0 per bug
"""

from __future__ import annotations

from ..models import BugReport, RewardBreakdown

_SEVERITY_ORDER = ["critical", "high", "medium", "low"]
_VALID_TEAMS    = {"frontend", "backend", "infra", "security"}

_W_SEVERITY = 0.50
_W_TEAM     = 0.50


def _score_severity(agent: str, truth: str) -> float:
    if agent not in _SEVERITY_ORDER or truth not in _SEVERITY_ORDER:
        return 0.0
    dist = abs(_SEVERITY_ORDER.index(agent) - _SEVERITY_ORDER.index(truth))
    return 1.0 if dist == 0 else (0.5 if dist == 1 else 0.0)


def _score_team(agent: str, truth: str) -> float:
    return 1.0 if agent.strip().lower() == truth.strip().lower() else 0.0


def grade_bug(action: dict, bug: BugReport) -> tuple[float, RewardBreakdown, str]:
    agent_severity = str(action.get("severity", "")).strip().lower()
    agent_team     = str(action.get("team", "")).strip().lower()

    true_severity  = bug.ground_truth.severity.value
    true_team      = bug.ground_truth.team.value

    sev_score  = _score_severity(agent_severity, true_severity)
    team_score = _score_team(agent_team, true_team)

    reward = _W_SEVERITY * sev_score + _W_TEAM * team_score

    breakdown = RewardBreakdown(
        severity_score=sev_score,
        team_score=team_score,
    )

    sev_icon  = "✓" if sev_score == 1.0 else ("~" if sev_score == 0.5 else "✗")
    team_icon = "✓" if team_score == 1.0 else "✗"

    feedback_parts = [
        f"[{bug.id}]",
        f"Severity {sev_icon}: got '{agent_severity}' | expected '{true_severity}'",
        f"Team {team_icon}: got '{agent_team}' | expected '{true_team}'",
        f"Step reward: {reward:.2f}",
    ]

    if team_score == 0.0 and agent_team not in _VALID_TEAMS:
        feedback_parts.append(
            f"Hint: '{agent_team}' is not a valid team. "
            f"Valid teams: frontend, backend, infra, security."
        )

    return reward, breakdown, " | ".join(feedback_parts)