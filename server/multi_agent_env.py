"""
multi_agent_env.py — Two-agent Bug Triage environment.

Extends BugTriageEnv with:
  multi_reset(task_id)              → StepResult   start fresh episode
  multi_step(action_a, action_b)    → MultiStepResult  both agents act together
  get_multi_state()                 → MultiAgentState   full state for both agents

Episode flow
────────────
  multi_reset()
      ↓
  Agent A sees observation → sends action_a (triage proposal)
  Agent B sees same observation + action_a → sends action_b (review)
      ↓
  consensus.compute_joint_reward() → reward_a, reward_b
      ↓
  Both rewards logged, episode advances to next item
      ↓
  Curriculum check: if both agents avg score > 0.75 → escalate task difficulty

Curriculum escalation
─────────────────────
  easy   → medium  : if rolling avg reward >= 0.75 over last 3 episodes
  medium → hard    : if rolling avg reward >= 0.70 over last 3 episodes
  hard             : stays at hard (ceiling)
"""

from __future__ import annotations

import json
import uuid
from collections import deque
from typing import Any

from pydantic import BaseModel, Field

from .env import BugTriageEnv
from .models import (
    BugAction,
    BugObservation,
    EnvState,
    ItemType,
    RewardBreakdown,
    StepResult,
)
from .consensus import compute_joint_reward


# ---------------------------------------------------------------------------
# Extended models for multi-agent
# ---------------------------------------------------------------------------

class AgentStepInfo(BaseModel):
    """Per-agent reward info returned in MultiStepResult."""
    agent_id:  str   = Field(description="'agent_a' or 'agent_b'")
    action:    dict  = Field(description="Parsed action the agent took")
    reward:    float = Field(description="Individual reward for this step")
    score:     float = Field(description="Raw accuracy score before consensus")


class MultiStepResult(BaseModel):
    """
    Result returned by POST /multi_step.

    Contains separate observations and rewards for both agents,
    plus the consensus outcome and joint feedback.
    """
    observation_a:  BugObservation
    observation_b:  BugObservation
    reward_a:       float
    reward_b:       float
    joint_reward:   float                  = Field(description="Average of reward_a and reward_b")
    done:           bool
    consensus:      bool                   = Field(description="Whether agents agreed on this item")
    agents:         list[AgentStepInfo]    = Field(default_factory=list)
    info:           dict[str, Any]         = Field(default_factory=dict)


class MultiAgentState(BaseModel):
    """Full internal state for the multi-agent environment."""
    task_id:              str
    episode_id:           str
    current_difficulty:   str              = "easy"
    step_count:           int              = 0
    max_steps:            int              = 0
    item_index:           int              = 0
    total_items:          int              = 0
    is_done:              bool             = False
    rewards_a:            list[float]      = Field(default_factory=list)
    rewards_b:            list[float]      = Field(default_factory=list)
    joint_rewards:        list[float]      = Field(default_factory=list)
    consensus_rate:       float            = 0.0
    normalised_score_a:   float            = 0.0
    normalised_score_b:   float            = 0.0
    joint_score:          float            = 0.0
    episode_history:      list[float]      = Field(default_factory=list)
    curriculum_level:     int              = 0
    last_feedback:        str              = ""


# ---------------------------------------------------------------------------
# Curriculum manager
# ---------------------------------------------------------------------------

class CurriculumManager:
    """
    Tracks episode scores and decides when to escalate difficulty.

    Escalation thresholds:
      easy   → medium : rolling avg >= 0.75 over last 3 episodes
      medium → hard   : rolling avg >= 0.70 over last 3 episodes
    """

    TASK_ORDER  = ["easy", "medium", "hard"]
    THRESHOLDS  = {"easy": 0.75, "medium": 0.70, "hard": float("inf")}
    WINDOW_SIZE = 3

    def __init__(self) -> None:
        self._level:   int              = 0
        self._history: deque[float]     = deque(maxlen=self.WINDOW_SIZE)

    @property
    def current_task(self) -> str:
        return self.TASK_ORDER[self._level]

    @property
    def level(self) -> int:
        return self._level

    def record_episode(self, joint_score: float) -> bool:
        """
        Record an episode's joint score. Returns True if difficulty escalated.
        """
        self._history.append(joint_score)

        if self._level >= len(self.TASK_ORDER) - 1:
            return False  # already at hardest

        if len(self._history) < self.WINDOW_SIZE:
            return False  # not enough history yet

        avg   = sum(self._history) / len(self._history)
        threshold = self.THRESHOLDS[self.current_task]

        if avg >= threshold:
            self._level += 1
            self._history.clear()   # reset history for new difficulty
            return True

        return False

    def reset(self) -> None:
        """Reset curriculum to start — used when starting a completely new training run."""
        self._level   = 0
        self._history = deque(maxlen=self.WINDOW_SIZE)

    @property
    def rolling_avg(self) -> float:
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)


# ---------------------------------------------------------------------------
# Multi-agent environment
# ---------------------------------------------------------------------------

class MultiAgentBugTriageEnv:
    """
    Two-agent extension of BugTriageEnv.

    Agent A (Triager)  — proposes triage/review action
    Agent B (Reviewer) — sees Agent A's proposal, agrees or disputes

    The environment maintains a single item queue (same as BugTriageEnv)
    but processes each item twice — once per agent — before advancing.

    State is fully independent from the single-agent env — both can
    run on the same server simultaneously without interference.
    """

    def __init__(self) -> None:
        self._base_env    = BugTriageEnv()
        self._curriculum  = CurriculumManager()
        self._state:      MultiAgentState | None = None
        self._ep_rewards_a: list[float] = []
        self._ep_rewards_b: list[float] = []

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def multi_reset(self, task_id: str | None = None) -> MultiStepResult:
        """
        Start a new multi-agent episode.

        If task_id is None, uses the curriculum's current difficulty.
        Passing task_id explicitly overrides the curriculum (useful for testing).
        """
        # Record previous episode score before resetting
        if self._ep_rewards_a and self._ep_rewards_b:
            avg_joint = sum(
                (a + b) / 2
                for a, b in zip(self._ep_rewards_a, self._ep_rewards_b)
            ) / max(1, len(self._ep_rewards_a))

            escalated = self._curriculum.record_episode(avg_joint)
            if escalated:
                print(
                    f"[CURRICULUM] Escalated to '{self._curriculum.current_task}' "
                    f"(rolling avg was {self._curriculum.rolling_avg:.2f})",
                    flush=True,
                )

        # Pick task
        effective_task = task_id or self._curriculum.current_task

        # Reset base env
        base_result = self._base_env.reset(task_id=effective_task)

        self._ep_rewards_a = []
        self._ep_rewards_b = []
        ep_id = str(uuid.uuid4())[:8]

        self._state = MultiAgentState(
            task_id            = effective_task,
            episode_id         = ep_id,
            current_difficulty = effective_task,
            step_count         = 0,
            max_steps          = base_result.observation.max_steps,
            item_index         = 0,
            total_items        = base_result.observation.total_items,
            is_done            = False,
            curriculum_level   = self._curriculum.level,
            last_feedback      = "Multi-agent episode started. Both agents should act.",
        )

        obs_a = self._build_agent_obs(base_result.observation, "agent_a", None)
        obs_b = self._build_agent_obs(base_result.observation, "agent_b", None)

        return MultiStepResult(
            observation_a = obs_a,
            observation_b = obs_b,
            reward_a      = 0.0,
            reward_b      = 0.0,
            joint_reward  = 0.0,
            done          = False,
            consensus     = False,
            agents        = [],
            info          = {
                "episode_id":        ep_id,
                "task_id":           effective_task,
                "curriculum_level":  self._curriculum.level,
                "rolling_avg":       round(self._curriculum.rolling_avg, 3),
            },
        )

    def multi_step(
        self,
        action_a: BugAction,
        action_b: BugAction,
    ) -> MultiStepResult:
        """
        Both agents act on the current item simultaneously.

        action_a : Agent A's triage proposal
        action_b : Agent B's review response (has seen action_a)
        """
        if self._state is None:
            raise RuntimeError("Call multi_reset() before multi_step().")

        if self._state.is_done:
            return self._terminal_result("Episode already finished.")

        s = self._state
        s.step_count += 1

        # -- Parse both actions -------------------------------------------
        parsed_a, err_a = self._parse_json(action_a.message)
        parsed_b, err_b = self._parse_json(action_b.message)

        # Handle parse errors gracefully
        if err_a:
            parsed_a = {"action": "skip", "severity": "medium", "team": "backend"}
        if err_b:
            parsed_b = {"action": "skip", "severity": "medium", "team": "backend"}

        # -- Get ground truth from base env state -------------------------
        base_state = self._base_env.get_state()
        ground_truth = self._get_ground_truth(base_state)

        # -- Compute joint reward -----------------------------------------
        reward_a, reward_b, bd_a, bd_b, feedback = compute_joint_reward(
            action_a     = parsed_a,
            action_b     = parsed_b,
            ground_truth = ground_truth,
        )

        # -- Apply penalty for parse errors -------------------------------
        if err_a:
            reward_a = -0.3
        if err_b:
            reward_b = -0.3

        joint_reward = (reward_a + reward_b) / 2.0

        # -- Record rewards -----------------------------------------------
        self._ep_rewards_a.append(reward_a)
        self._ep_rewards_b.append(reward_b)
        s.rewards_a.append(reward_a)
        s.rewards_b.append(reward_b)
        s.joint_rewards.append(joint_reward)

        # -- Update scores ------------------------------------------------
        n = len(s.joint_rewards)
        s.normalised_score_a = max(0.0, min(1.0, sum(s.rewards_a) / max(1, s.total_items)))
        s.normalised_score_b = max(0.0, min(1.0, sum(s.rewards_b) / max(1, s.total_items)))
        s.joint_score        = max(0.0, min(1.0, sum(s.joint_rewards) / max(1, s.total_items)))
        s.last_feedback      = feedback

        # -- Consensus check ----------------------------------------------
        from .consensus import _actions_agree
        agreed = _actions_agree(parsed_a, parsed_b)
        if n > 0:
            agree_count       = sum(
                1 for i in range(len(s.rewards_a))
                if abs(s.rewards_a[i] - s.rewards_b[i]) < 0.3
            )
            s.consensus_rate  = agree_count / n

        # -- Advance base env (step with agent A's action as primary) -----
        base_result = self._base_env.step(BugAction(message=action_a.message))
        s.item_index = base_result.observation.item_index

        # -- Check episode done -------------------------------------------
        if base_result.done or s.step_count >= s.max_steps:
            s.is_done = True

        # -- Build observations for next step -----------------------------
        obs_a = self._build_agent_obs(base_result.observation, "agent_a", None)
        obs_b = self._build_agent_obs(base_result.observation, "agent_b", parsed_a)

        return MultiStepResult(
            observation_a = obs_a,
            observation_b = obs_b,
            reward_a      = reward_a,
            reward_b      = reward_b,
            joint_reward  = joint_reward,
            done          = s.is_done,
            consensus     = agreed,
            agents        = [
                AgentStepInfo(agent_id="agent_a", action=parsed_a,
                              reward=reward_a, score=round(reward_a, 3)),
                AgentStepInfo(agent_id="agent_b", action=parsed_b,
                              reward=reward_b, score=round(reward_b, 3)),
            ],
            info = {
                "step":              s.step_count,
                "joint_reward":      round(joint_reward, 4),
                "joint_score":       round(s.joint_score, 4),
                "score_a":           round(s.normalised_score_a, 4),
                "score_b":           round(s.normalised_score_b, 4),
                "consensus_rate":    round(s.consensus_rate, 4),
                "curriculum_level":  self._curriculum.level,
                "current_task":      self._curriculum.current_task,
                "rolling_avg":       round(self._curriculum.rolling_avg, 3),
                "feedback":          feedback,
            },
        )

    def get_multi_state(self) -> MultiAgentState:
        if self._state is None:
            raise RuntimeError("Call multi_reset() before get_multi_state().")
        return self._state

    def get_curriculum(self) -> dict:
        return {
            "current_task":  self._curriculum.current_task,
            "level":         self._curriculum.level,
            "rolling_avg":   round(self._curriculum.rolling_avg, 3),
            "window_size":   CurriculumManager.WINDOW_SIZE,
            "thresholds":    CurriculumManager.THRESHOLDS,
        }

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _build_agent_obs(
        self,
        base_obs:        BugObservation,
        agent_id:        str,
        partner_action:  dict | None,
    ) -> BugObservation:
        """
        Build an agent-specific observation.

        Agent B's observation includes Agent A's proposal in the body
        so it can make an informed review decision.
        """
        s = self._state

        if agent_id == "agent_b" and partner_action:
            partner_hint = (
                f"\n\n--- Agent A proposed ---\n"
                f"Severity: {partner_action.get('severity', 'unknown')}\n"
                f"Team: {partner_action.get('team', 'unknown')}\n"
                f"Comment: {partner_action.get('comment', 'none')}\n"
                f"Security flag: {partner_action.get('security_issue', False)}\n"
                f"---\n"
                f"Do you agree? Submit your own triage action."
            )
            body = base_obs.body + partner_hint
        else:
            body = base_obs.body

        role_hint = (
            "[Agent A — Triager] You go first. Propose your triage."
            if agent_id == "agent_a"
            else "[Agent B — Reviewer] Review Agent A's proposal above. Agree or correct it."
        )

        return BugObservation(
            task_id           = base_obs.task_id,
            episode_id        = s.episode_id if s else base_obs.episode_id,
            item_index        = base_obs.item_index,
            total_items       = base_obs.total_items,
            item_type         = base_obs.item_type,
            item_id           = base_obs.item_id,
            title             = base_obs.title,
            body              = f"{role_hint}\n\n{body}",
            labels            = base_obs.labels,
            reporter          = base_obs.reporter,
            available_actions = base_obs.available_actions,
            step_number       = base_obs.step_number,
            max_steps         = base_obs.max_steps,
            current_score     = s.joint_score if s else 0.0,
            feedback          = base_obs.echoed_message,
            echoed_message    = base_obs.echoed_message,
        )

    def _get_ground_truth(self, base_state: EnvState) -> dict:
        """Extract ground truth from current item in base env queue."""
        try:
            raw   = self._base_env._raw
            idx   = base_state.item_index
            task  = base_state.task_id
            cfg   = raw["task_configs"][task]
            bug_ids = cfg.get("bug_ids", [])

            if idx < len(bug_ids):
                bug_id   = bug_ids[idx]
                bugs_idx = {b["id"]: b for b in raw["bugs"]}
                bug      = bugs_idx.get(bug_id, {})
                gt       = bug.get("ground_truth", {})
                return {
                    "severity": gt.get("severity", "medium"),
                    "team":     gt.get("team", "backend"),
                    "task_id":  task,
                }
        except Exception:
            pass

        return {"severity": "medium", "team": "backend", "task_id": "easy"}

    def _terminal_result(self, message: str) -> MultiStepResult:
        s = self._state
        terminal_obs = BugObservation(
            task_id           = s.task_id if s else "easy",
            episode_id        = s.episode_id if s else "DONE",
            item_index        = s.item_index if s else 0,
            total_items       = s.total_items if s else 1,
            item_type         = ItemType.BUG_REPORT,
            item_id           = "DONE",
            title             = "Episode Complete",
            body              = message,
            labels            = [],
            reporter          = "",
            available_actions = [],
            step_number       = max(1, s.step_count if s else 1),
            max_steps         = s.max_steps if s else 1,
            current_score     = s.joint_score if s else 0.0,
            feedback          = message,
            echoed_message    = message,
        )
        return MultiStepResult(
            observation_a = terminal_obs,
            observation_b = terminal_obs,
            reward_a      = 0.0,
            reward_b      = 0.0,
            joint_reward  = 0.0,
            done          = True,
            consensus     = False,
            agents        = [],
            info          = {"warning": "episode_already_done"},
        )

    @staticmethod
    def _parse_json(message: str) -> tuple[dict, str | None]:
        try:
            obj = json.loads(message)
            if not isinstance(obj, dict):
                return {}, "Action must be a JSON object"
            if "action" not in obj:
                return {}, "Missing required key 'action'"
            return obj, None
        except json.JSONDecodeError as exc:
            return {}, f"JSON error: {exc}"