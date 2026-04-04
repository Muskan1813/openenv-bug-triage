"""
models.py — All Pydantic typed models for the Bug Triage OpenEnv environment.

Covers:
  - BugReport / PullRequest  — data models for queue items
  - BugObservation           — what the agent sees each step
  - BugAction                — what the agent sends
  - BugReward                — reward breakdown
  - StepResult               — full return from step()
  - EnvState                 — internal state (returned by state())
  - ResetRequest             — body accepted by POST /reset
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"


class Team(str, Enum):
    FRONTEND = "frontend"
    BACKEND  = "backend"
    INFRA    = "infra"
    SECURITY = "security"


class PRVerdict(str, Enum):
    APPROVE          = "approve"
    REQUEST_CHANGES  = "request_changes"
    COMMENT          = "comment"


class ItemType(str, Enum):
    BUG_REPORT   = "bug_report"
    PULL_REQUEST = "pull_request"


class TaskDifficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ---------------------------------------------------------------------------
# Data models — what lives in bug_reports.json
# ---------------------------------------------------------------------------

class GroundTruthBug(BaseModel):
    """Expected correct answer for a bug report (used by graders, not shown to agent)."""
    severity: Severity
    team: Team
    keywords: list[str] = Field(default_factory=list)


class GroundTruthPR(BaseModel):
    """Expected correct answer for a pull request (used by graders, not shown to agent)."""
    verdict: PRVerdict
    has_security_issue: bool = False
    security_issue_type: str | None = None
    review_keywords: list[str] = Field(default_factory=list)
    quality_score: float = Field(ge=0.0, le=1.0, default=0.5)


class BugReport(BaseModel):
    """A single bug report item in the queue."""
    id: str
    title: str
    body: str
    labels: list[str] = Field(default_factory=list)
    reporter: str
    created_at: str
    ground_truth: GroundTruthBug

    def to_agent_view(self) -> dict[str, Any]:
        """Returns only the fields the agent is allowed to see (no ground_truth)."""
        return {
            "id":         self.id,
            "type":       ItemType.BUG_REPORT,
            "title":      self.title,
            "body":       self.body,
            "labels":     self.labels,
            "reporter":   self.reporter,
            "created_at": self.created_at,
        }


class PullRequest(BaseModel):
    """A pull request item in the queue."""
    id: str
    title: str
    body: str
    labels: list[str] = Field(default_factory=list)
    author: str
    base_branch: str
    files_changed: int
    additions: int
    deletions: int
    ground_truth: GroundTruthPR

    def to_agent_view(self) -> dict[str, Any]:
        """Returns only the fields the agent is allowed to see (no ground_truth)."""
        return {
            "id":            self.id,
            "type":          ItemType.PULL_REQUEST,
            "title":         self.title,
            "body":          self.body,
            "labels":        self.labels,
            "author":        self.author,
            "base_branch":   self.base_branch,
            "files_changed": self.files_changed,
            "additions":     self.additions,
            "deletions":     self.deletions,
        }


# ---------------------------------------------------------------------------
# Observation — what the agent receives each step
# ---------------------------------------------------------------------------

class BugObservation(BaseModel):
    """
    Full observation returned after reset() and each step().

    Fields
    ------
    task_id        : Which task is active (easy / medium / hard).
    episode_id     : Unique identifier for this episode run.
    item_index     : 0-based index of the current queue item.
    total_items    : Total items in this episode's queue.
    item_type      : 'bug_report' or 'pull_request'.
    item_id        : The BUG-xxx or PR-xxx identifier.
    title          : Title of the current item.
    body           : Full description / PR body of the current item.
    labels         : Existing labels on the item.
    reporter       : Who filed the bug / who opened the PR.
    available_actions : Valid action names the agent may take this step.
    step_number    : Current step index (1-based).
    max_steps      : Episode terminates after this many steps.
    current_score  : Cumulative normalised score so far [0.0, 1.0].
    feedback       : Human-readable result of the previous action.
    echoed_message : Mirrors `feedback` — required by the inference script.
    """

    task_id:           str
    episode_id:        str
    item_index:        int  = Field(ge=0)
    total_items:       int  = Field(ge=1)
    item_type:         ItemType
    item_id:           str
    title:             str
    body:              str
    labels:            list[str]          = Field(default_factory=list)
    reporter:          str                = ""
    available_actions: list[str]          = Field(default_factory=list)
    step_number:       int                = Field(ge=1)
    max_steps:         int                = Field(ge=1)
    current_score:     float              = Field(ge=0.0, le=1.0, default=0.0)
    feedback:          str                = ""
    echoed_message:    str                = ""   # mirrors feedback for inference compat


# ---------------------------------------------------------------------------
# Action — what the agent sends
# ---------------------------------------------------------------------------

class BugAction(BaseModel):
    """
    Action sent by the agent.

    The entire action is encoded as a JSON string in `message`.
    The environment parses the JSON and routes to the appropriate handler.

    Valid JSON schemas inside `message`:

    Easy task:
        {"action": "classify", "severity": "critical|high|medium|low"}

    Medium task:
        {"action": "triage",
         "severity": "critical|high|medium|low",
         "team":     "frontend|backend|infra|security",
         "comment":  "<optional short comment>"}

    Hard task (bugs):
        {"action": "triage",
         "severity":       "critical|high|medium|low",
         "team":           "frontend|backend|infra|security",
         "comment":        "<review comment>",
         "security_issue": true|false}

    Hard task (PRs):
        {"action":         "review_pr",
         "verdict":        "approve|request_changes|comment",
         "comment":        "<review comment>",
         "security_issue": true|false}

    Hard task (final step):
        {"action": "prioritize",
         "order": ["BUG-005", "BUG-001", ...]}

    Any task:
        {"action": "skip"}   — skips the current item (penalty applied)
    """

    message: str = Field(
        description="JSON string containing the action and its parameters."
    )


# ---------------------------------------------------------------------------
# Reward — breakdown returned with each step
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """Itemised components that sum to the total reward for this step."""
    severity_score:      float = Field(ge=0.0, le=1.0, default=0.0,
                                       description="0.0/0.5/1.0 — wrong/adjacent/correct severity")
    team_score:          float = Field(ge=0.0, le=1.0, default=0.0,
                                       description="1.0 correct team, 0.0 wrong (N/A for easy)")
    comment_score:       float = Field(ge=0.0, le=1.0, default=0.0,
                                       description="Quality of review comment (keyword overlap)")
    security_score:      float = Field(ge=0.0, le=1.0, default=0.0,
                                       description="Correct detection of security issues in hard task")
    efficiency_bonus:    float = Field(ge=0.0, le=0.2,  default=0.0,
                                       description="+0.1 bonus if solved in fewer than half the allowed steps")
    skip_penalty:        float = Field(ge=-1.0, le=0.0, default=0.0,
                                       description="-0.2 for each skip action used")
    invalid_penalty:     float = Field(ge=-1.0, le=0.0, default=0.0,
                                       description="-0.3 for malformed / unrecognised actions")
    repeat_penalty:      float = Field(ge=-1.0, le=0.0, default=0.0,
                                       description="-0.1 for repeating the exact same action twice in a row")


class BugReward(BaseModel):
    """Full reward signal returned alongside an observation."""
    value:          float           = Field(description="Net reward for this step (may be negative)")
    breakdown:      RewardBreakdown = Field(description="Itemised component scores")
    partial_credit: float           = Field(ge=0.0, le=1.0,
                                            description="Normalised partial progress [0.0, 1.0]")


# ---------------------------------------------------------------------------
# StepResult — the envelope returned by POST /step and POST /reset
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """
    Complete result returned by step() and reset().

    Compatible with the openenv-core client which reads:
      result.observation, result.reward, result.done, result.info
    """
    observation: BugObservation
    reward:      float                  = Field(default=0.0)
    done:        bool                   = Field(default=False)
    info:        dict[str, Any]         = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# EnvState — full internal state returned by GET /state
# ---------------------------------------------------------------------------

class EnvState(BaseModel):
    """
    Full internal environment state.

    Returned verbatim by GET /state.
    Not shown to the agent during normal interaction — intended for
    debugging, logging, and the grader.
    """
    task_id:           str
    episode_id:        str
    step_count:        int                  = Field(ge=0)
    max_steps:         int                  = Field(ge=1)
    item_index:        int                  = Field(ge=0)
    total_items:       int                  = Field(ge=1)
    is_done:           bool                 = False
    cumulative_reward: float                = 0.0
    normalised_score:  float                = Field(ge=0.0, le=1.0, default=0.0)
    actions_taken:     list[dict[str, Any]] = Field(default_factory=list)
    rewards_history:   list[float]          = Field(default_factory=list)
    last_feedback:     str                  = ""
    last_action_valid: bool                 = True


# ---------------------------------------------------------------------------
# ResetRequest — body accepted by POST /reset
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """
    Optional body for POST /reset.

    `task_id` selects which task to initialise.
    Omitting it (or sending `{}`) defaults to "easy".
    """
    task_id: str | None = Field(
        default=None,
        description="Task to load: 'easy', 'medium', or 'hard'. Defaults to 'easy'."
    )
