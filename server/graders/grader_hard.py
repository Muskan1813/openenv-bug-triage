"""
grader_hard.py — Grader for Task: hard

Task goal   : Full code-review pipeline over 8 bugs + 3 PRs, then prioritize.

Three sub-graders:
─────────────────────────────────────────────────────────────────────────────
1. grade_bug(action, bug)
   Action  : {"action":"triage", "severity":..., "team":...,
               "comment":..., "security_issue": true|false}
   Scoring :
     severity       0.30  (1.0/0.5/0.0)
     team           0.25  (1.0/0.0)
     comment        0.25  (keyword overlap with ground_truth.keywords)
     security_flag  0.20  (must correctly flag security bugs as security=true)

2. grade_pr(action, pr)
   Action  : {"action":"review_pr", "verdict":...,
               "comment":..., "security_issue": true|false}
   Scoring :
     verdict        0.40  (1.0 exact / 0.0 wrong)
     security_flag  0.40  (critical — PR-103 has SQL injection)
     comment        0.20  (keyword overlap)

3. grade_prioritization(action, task_config)
   Action  : {"action":"prioritize", "order": ["BUG-005", "BUG-001", ...]}
   Scoring :
     rank_accuracy  0.70  (normalised mean absolute rank error)
     coverage       0.30  (fraction of expected items included)
─────────────────────────────────────────────────────────────────────────────

Max reward per item: 1.0
"""

from __future__ import annotations

from ..models import BugReport, PullRequest, RewardBreakdown

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_SEVERITY_ORDER = ["critical", "high", "medium", "low"]
_VALID_TEAMS    = {"frontend", "backend", "infra", "security"}
_VALID_VERDICTS = {"approve", "request_changes", "comment"}

# Weights for grade_bug
_WB_SEVERITY  = 0.30
_WB_TEAM      = 0.25
_WB_COMMENT   = 0.25
_WB_SECURITY  = 0.20

# Weights for grade_pr
_WP_VERDICT   = 0.40
_WP_SECURITY  = 0.40
_WP_COMMENT   = 0.20

# Weights for grade_prioritization
_WPR_RANK     = 0.70
_WPR_COVERAGE = 0.30


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _score_severity(agent: str, truth: str) -> float:
    if agent not in _SEVERITY_ORDER or truth not in _SEVERITY_ORDER:
        return 0.0
    dist = abs(_SEVERITY_ORDER.index(agent) - _SEVERITY_ORDER.index(truth))
    return 1.0 if dist == 0 else (0.5 if dist == 1 else 0.0)


def _score_team(agent: str, truth: str) -> float:
    return 1.0 if agent.strip().lower() == truth.strip().lower() else 0.0


def _score_comment(comment: str, keywords: list[str]) -> float:
    """
    Keyword-overlap comment quality score.

    Score = hits / threshold  capped at 1.0
    Threshold = 40% of ground_truth keywords (so agent doesn't need every keyword)

    Empty comment always returns 0.0.
    """
    if not comment or not keywords:
        return 0.0

    comment_lower = comment.lower()
    hits = sum(1 for kw in keywords if kw.lower() in comment_lower)
    threshold = max(1, len(keywords) * 0.4)  # need 40% coverage for full score
    return min(1.0, hits / threshold)


def _score_security_flag(agent_flag: bool, truly_has_security: bool) -> float:
    """
    Binary security detection score.

    Correct detection (true positive or true negative) → 1.0
    False negative (missed a security issue)           → 0.0  (worst outcome)
    False positive (flagged when there is none)        → 0.3  (minor penalty)
    """
    if agent_flag == truly_has_security:
        return 1.0
    if truly_has_security and not agent_flag:
        return 0.0   # missed a real security issue
    return 0.3       # false alarm — less bad than missing one


def _parse_bool_flag(value) -> bool:
    """Safely parse security_issue field from agent action."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


# ---------------------------------------------------------------------------
# 1. grade_bug
# ---------------------------------------------------------------------------

def grade_bug(action: dict, bug: BugReport) -> tuple[float, RewardBreakdown, str]:
    """
    Grade a 'triage' action for the hard task.

    Extra vs medium: comment quality + security flag detection.
    A bug is a 'security bug' when its ground_truth team == 'security'.
    """
    agent_severity  = str(action.get("severity", "")).strip().lower()
    agent_team      = str(action.get("team", "")).strip().lower()
    agent_comment   = str(action.get("comment", "")).strip()
    agent_sec_flag  = _parse_bool_flag(action.get("security_issue", False))

    true_severity   = bug.ground_truth.severity.value
    true_team       = bug.ground_truth.team.value
    true_keywords   = bug.ground_truth.keywords
    # A bug is a security issue iff it belongs to the security team
    truly_security  = (true_team == "security")

    # -- Component scores -----------------------------------------------------
    sev_score  = _score_severity(agent_severity, true_severity)
    team_score = _score_team(agent_team, true_team)
    com_score  = _score_comment(agent_comment, true_keywords)
    sec_score  = _score_security_flag(agent_sec_flag, truly_security)

    # -- Weighted reward -------------------------------------------------------
    reward = (
        _WB_SEVERITY * sev_score
        + _WB_TEAM    * team_score
        + _WB_COMMENT * com_score
        + _WB_SECURITY * sec_score
    )

    breakdown = RewardBreakdown(
        severity_score=sev_score,
        team_score=team_score,
        comment_score=com_score,
        security_score=sec_score,
    )

    # -- Feedback -------------------------------------------------------------
    icons = {1.0: "✓", 0.5: "~", 0.0: "✗"}

    feedback_parts = [
        f"[{bug.id}]",
        f"Severity {icons.get(sev_score, '?')}: '{agent_severity}' vs '{true_severity}'",
        f"Team {icons[team_score]}: '{agent_team}' vs '{true_team}'",
        f"Comment {icons.get(round(com_score), '?')}: {com_score:.0%} keyword coverage",
        f"Security flag {'✓' if sec_score == 1.0 else ('⚠' if sec_score == 0.3 else '✗')}: "
        f"agent={agent_sec_flag} truth={truly_security}",
        f"Step reward: {reward:.2f}",
    ]

    if truly_security and not agent_sec_flag:
        feedback_parts.append(
            "⚠ MISSED SECURITY ISSUE — This bug should be flagged as a security concern."
        )

    return reward, breakdown, " | ".join(feedback_parts)


# ---------------------------------------------------------------------------
# 2. grade_pr
# ---------------------------------------------------------------------------

def grade_pr(action: dict, pr: PullRequest) -> tuple[float, RewardBreakdown, str]:
    """
    Grade a 'review_pr' action.

    The hardest signal: PR-103 contains a SQL injection via f-string interpolation.
    The agent must both request_changes AND flag security_issue=true to get full marks.
    """
    agent_verdict   = str(action.get("verdict", "")).strip().lower()
    agent_comment   = str(action.get("comment", "")).strip()
    agent_sec_flag  = _parse_bool_flag(action.get("security_issue", False))

    true_verdict    = pr.ground_truth.verdict.value
    true_keywords   = pr.ground_truth.review_keywords
    truly_security  = pr.ground_truth.has_security_issue

    # -- Component scores -----------------------------------------------------
    verdict_score = 1.0 if agent_verdict == true_verdict else 0.0
    sec_score     = _score_security_flag(agent_sec_flag, truly_security)
    com_score     = _score_comment(agent_comment, true_keywords)

    # -- Weighted reward -------------------------------------------------------
    reward = (
        _WP_VERDICT  * verdict_score
        + _WP_SECURITY * sec_score
        + _WP_COMMENT  * com_score
    )

    breakdown = RewardBreakdown(
        comment_score=com_score,
        security_score=sec_score,
    )

    # -- Feedback -------------------------------------------------------------
    verdict_icon  = "✓" if verdict_score == 1.0 else "✗"
    sec_icon      = "✓" if sec_score == 1.0 else ("⚠" if sec_score == 0.3 else "✗")

    feedback_parts = [
        f"[{pr.id}]",
        f"Verdict {verdict_icon}: '{agent_verdict}' vs '{true_verdict}'",
        f"Security {sec_icon}: agent={agent_sec_flag} truth={truly_security}",
        f"Comment: {com_score:.0%} keyword coverage",
        f"Step reward: {reward:.2f}",
    ]

    if truly_security and not agent_sec_flag:
        feedback_parts.append(
            "✗ CRITICAL MISS — This PR has a security vulnerability (SQL injection). "
            "Always check for f-string/string-format SQL patterns."
        )
    if agent_verdict != true_verdict and true_verdict == "request_changes":
        feedback_parts.append(
            "Hint: This PR should NOT be approved — check the code carefully for bugs."
        )

    return reward, breakdown, " | ".join(feedback_parts)


# ---------------------------------------------------------------------------
# 3. grade_prioritization
# ---------------------------------------------------------------------------

def grade_prioritization(
    action: dict,
    task_config: dict,
) -> tuple[float, RewardBreakdown, str]:
    """
    Grade the final 'prioritize' action for the hard task.

    Expected order is in task_config["expected_priority_order"].
    The ground truth ordering is: critical security issues first → critical bugs →
    high-severity infra → PRs with security issues → other high → medium → low.

    Score components:
      rank_accuracy (0.70) : 1 - normalised mean absolute rank error
      coverage      (0.30) : fraction of expected items present in agent's list

    Both are scaled to [0.0, 1.0] and combined via weighted sum.
    """
    expected: list[str]     = task_config.get("expected_priority_order", [])
    agent_order: list[str]  = action.get("order", [])

    # -- Edge cases -----------------------------------------------------------
    if not expected:
        return 0.0, RewardBreakdown(), "No expected order defined for this task."

    if not agent_order or not isinstance(agent_order, list):
        return 0.0, RewardBreakdown(invalid_penalty=-0.3), (
            "✗ No 'order' list provided. Send: {\"action\": \"prioritize\", \"order\": [\"BUG-005\", ...]}"
        )

    n = len(expected)

    # Build rank maps (0 = highest priority)
    expected_rank: dict[str, int] = {item: i for i, item in enumerate(expected)}
    agent_rank:    dict[str, int] = {item: i for i, item in enumerate(agent_order)}

    # Items present in both lists
    common = [item for item in expected if item in agent_rank]

    # -- Coverage score -------------------------------------------------------
    coverage = len(common) / n  # fraction of expected items the agent included

    # -- Rank accuracy score --------------------------------------------------
    if not common:
        rank_score = 0.0
    else:
        total_error = sum(
            abs(expected_rank[item] - agent_rank[item])
            for item in common
        )
        # Max possible error if every item is maximally displaced
        max_error = (n - 1) * len(common)
        rank_score = 1.0 - (total_error / max(1, max_error))
        rank_score = max(0.0, rank_score)

    # -- Final weighted score -------------------------------------------------
    final = _WPR_RANK * rank_score + _WPR_COVERAGE * coverage

    breakdown = RewardBreakdown(
        severity_score=rank_score,  # re-use severity_score slot for rank accuracy
        team_score=coverage,         # re-use team_score slot for coverage
    )

    # -- Feedback -------------------------------------------------------------
    missing = [item for item in expected if item not in agent_rank]
    extra   = [item for item in agent_order if item not in expected_rank]

    fb_parts = [
        f"Prioritization score: {final:.2f}",
        f"Rank accuracy: {rank_score:.2%}",
        f"Coverage: {len(common)}/{n} items",
    ]
    if missing:
        fb_parts.append(f"Missing items: {missing}")
    if extra:
        fb_parts.append(f"Unknown items ignored: {extra}")

    top5_expected = expected[:5]
    top5_agent    = agent_order[:5]
    fb_parts.append(f"Expected top-5: {top5_expected}")
    fb_parts.append(f"Your top-5:     {top5_agent}")

    return final, breakdown, " | ".join(fb_parts)