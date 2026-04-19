"""
consensus.py — Joint reward logic for two-agent bug triage.

Agent A (Triager)  : proposes severity + team + comment
Agent B (Reviewer) : sees A's proposal, agrees or disputes

Consensus reward matrix:
  Both correct + agree      → 1.0 + 0.2 bonus each
  Both correct + disagree   → 1.0 each (no bonus, but no penalty)
  A correct, B wrong        → A gets 0.8, B gets 0.0
  A wrong, B correct        → A gets 0.2, B gets 0.8 (B caught the mistake)
  Both wrong + agree        → -0.1 each (dangerous blind consensus)
  Both wrong + disagree     → 0.2 each (at least B questioned A)
"""

from __future__ import annotations

from .models import RewardBreakdown


# ---------------------------------------------------------------------------
# Severity adjacency helper (same as graders)
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = ["critical", "high", "medium", "low"]


def _severity_correct(agent_val: str, truth: str) -> float:
    """Returns 1.0 exact, 0.5 adjacent, 0.0 wrong."""
    if agent_val not in _SEVERITY_ORDER or truth not in _SEVERITY_ORDER:
        return 0.0
    dist = abs(_SEVERITY_ORDER.index(agent_val) - _SEVERITY_ORDER.index(truth))
    return 1.0 if dist == 0 else (0.5 if dist == 1 else 0.0)


def _team_correct(agent_val: str, truth: str) -> float:
    return 1.0 if agent_val.strip().lower() == truth.strip().lower() else 0.0


# ---------------------------------------------------------------------------
# Individual action accuracy scorer
# ---------------------------------------------------------------------------

def score_action(action: dict, ground_truth: dict) -> float:
    """
    Score a single agent's action against ground truth.
    Returns a float between 0.0 and 1.0.

    ground_truth keys: severity, team (optional for easy task)
    """
    task_id = ground_truth.get("task_id", "easy")

    agent_severity = str(action.get("severity", "")).strip().lower()
    true_severity  = str(ground_truth.get("severity", "")).strip().lower()

    sev_score = _severity_correct(agent_severity, true_severity)

    if task_id == "easy":
        return sev_score

    agent_team = str(action.get("team", "")).strip().lower()
    true_team  = str(ground_truth.get("team", "")).strip().lower()
    team_score = _team_correct(agent_team, true_team)

    return 0.5 * sev_score + 0.5 * team_score


# ---------------------------------------------------------------------------
# Agreement checker
# ---------------------------------------------------------------------------

def _actions_agree(action_a: dict, action_b: dict) -> bool:
    """
    Two actions agree if they have the same severity AND team (if present).
    For PR review, they agree if they have the same verdict.
    """
    action_type = action_a.get("action", "")

    if action_type == "review_pr":
        return (
            str(action_a.get("verdict", "")).lower()
            == str(action_b.get("verdict", "")).lower()
        )

    same_severity = (
        str(action_a.get("severity", "")).lower()
        == str(action_b.get("severity", "")).lower()
    )
    same_team = (
        str(action_a.get("team", "")).lower()
        == str(action_b.get("team", "")).lower()
    )

    # For easy task (no team), only check severity
    if not action_a.get("team") and not action_b.get("team"):
        return same_severity

    return same_severity and same_team


# ---------------------------------------------------------------------------
# Main consensus grader
# ---------------------------------------------------------------------------

def compute_joint_reward(
    action_a:     dict,
    action_b:     dict,
    ground_truth: dict,
) -> tuple[float, float, RewardBreakdown, RewardBreakdown, str]:
    """
    Compute joint rewards for Agent A and Agent B.

    Parameters
    ----------
    action_a     : parsed action dict from Agent A (Triager)
    action_b     : parsed action dict from Agent B (Reviewer)
    ground_truth : dict with keys: severity, team, task_id

    Returns
    -------
    (reward_a, reward_b, breakdown_a, breakdown_b, feedback)
    """
    score_a  = score_action(action_a, ground_truth)
    score_b  = score_action(action_b, ground_truth)
    agree    = _actions_agree(action_a, action_b)

    a_correct = score_a >= 0.75   # threshold for "correct"
    b_correct = score_b >= 0.75

    # -- Consensus reward matrix ------------------------------------------
    if a_correct and b_correct and agree:
        reward_a = min(1.0, score_a + 0.2)   # consensus bonus
        reward_b = min(1.0, score_b + 0.2)
        outcome  = "both_correct_agree"

    elif a_correct and b_correct and not agree:
        reward_a = score_a                    # both right, no bonus for disagreeing
        reward_b = score_b
        outcome  = "both_correct_disagree"

    elif a_correct and not b_correct:
        reward_a = score_a * 0.8              # slight penalty — B didn't validate A
        reward_b = 0.0                        # B was wrong
        outcome  = "a_correct_b_wrong"

    elif not a_correct and b_correct:
        reward_a = 0.2                        # A was wrong but B caught it
        reward_b = score_b * 0.8             # B gets most credit for catching error
        outcome  = "a_wrong_b_correct"

    elif not a_correct and not b_correct and agree:
        reward_a = -0.1                       # dangerous blind consensus
        reward_b = -0.1
        outcome  = "both_wrong_agree"

    else:
        reward_a = 0.2                        # both wrong but B questioned A
        reward_b = 0.2
        outcome  = "both_wrong_disagree"

    # -- Breakdowns -------------------------------------------------------
    breakdown_a = RewardBreakdown(
        severity_score = _severity_correct(
            str(action_a.get("severity", "")).lower(),
            str(ground_truth.get("severity", "")).lower()
        ),
        team_score     = _team_correct(
            str(action_a.get("team", "")).lower(),
            str(ground_truth.get("team", "")).lower()
        ) if ground_truth.get("task_id") != "easy" else 0.0,
    )

    breakdown_b = RewardBreakdown(
        severity_score = _severity_correct(
            str(action_b.get("severity", "")).lower(),
            str(ground_truth.get("severity", "")).lower()
        ),
        team_score     = _team_correct(
            str(action_b.get("team", "")).lower(),
            str(ground_truth.get("team", "")).lower()
        ) if ground_truth.get("task_id") != "easy" else 0.0,
    )

    # -- Human-readable feedback ------------------------------------------
    outcome_messages = {
        "both_correct_agree":     "✓✓ Both agents correct and agree — consensus bonus applied!",
        "both_correct_disagree":  "✓? Both agents correct but disagree — no bonus.",
        "a_correct_b_wrong":      "✓✗ Agent A correct, Agent B wrong — B failed to validate.",
        "a_wrong_b_correct":      "✗✓ Agent A wrong, Agent B caught the mistake — B rewarded.",
        "both_wrong_agree":       "✗✗ Both agents wrong and agreed — dangerous consensus penalty.",
        "both_wrong_disagree":    "✗✗ Both agents wrong but disagreed — minor credit for skepticism.",
    }

    agree_str  = "agreed" if agree else "disagreed"
    feedback   = (
        f"{outcome_messages[outcome]} | "
        f"Agents {agree_str}. | "
        f"A score: {score_a:.2f} → reward {reward_a:.2f} | "
        f"B score: {score_b:.2f} → reward {reward_b:.2f}"
    )

    return reward_a, reward_b, breakdown_a, breakdown_b, feedback