"""
env.py — Core BugTriageEnv environment.

Implements the OpenEnv interface:
  reset(task_id)  → StepResult   initialise a fresh episode
  step(action)    → StepResult   advance one step
  get_state()     → EnvState     return full internal state (for /state endpoint)

Episode state machine
─────────────────────
  IDLE  ──reset()──►  PROCESSING  ──step()×N──►  [PRIORITIZING]  ──►  DONE
                           │                            │
                      (easy/medium)               (hard only)
                        auto-DONE                 one final
                       when queue                "prioritize"
                       exhausted                   action

Reward summary (per step)
──────────────────────────
  Valid action on a bug/PR  :  grader returns 0.0 – 1.0
  Skip                      : -0.20
  Malformed / unknown action: -0.30
  Repeat same action twice  : -0.10 (stacked on top of the above)
  Max per-step reward       :  1.0
  Normalised episode score  :  cumulative_reward / total_items  (clamped 0–1)
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .models import (
    BugObservation,
    BugAction,
    BugReport,
    EnvState,
    ItemType,
    PullRequest,
    RewardBreakdown,
    StepResult,
)

_DATA_PATH = Path(__file__).parent.parent / "data" / "bug_reports.json"

_VALID_ACTIONS: dict[str, set[str]] = {
    "easy":   {"classify", "skip"},
    "medium": {"triage", "skip"},
    "hard":   {"triage", "review_pr", "prioritize", "skip"},
}


class BugTriageEnv:
    """
    Stateful environment instance. One instance per server process.
    Not thread-safe; the FastAPI server must serialise concurrent requests
    (acceptable for single-agent evaluation runs).
    """

    def __init__(self) -> None:
        self._raw:              dict[str, Any]               = self._load_data()
        self._state:            EnvState | None              = None
        self._queue:            list[BugReport | PullRequest] = []
        self._task_cfg:         dict[str, Any]               = {}
        self._prioritize_phase: bool                         = False

    # -----------------------------------------------------------------------
    # Public OpenEnv interface
    # -----------------------------------------------------------------------

    def reset(self, task_id: str | None = None) -> StepResult:
        if task_id not in ("easy", "medium", "hard"):
            task_id = "easy"

        cfg                    = self._raw["task_configs"][task_id]
        self._task_cfg         = cfg
        self._prioritize_phase = False

        bugs_idx = {b["id"]: b for b in self._raw["bugs"]}
        prs_idx  = {p["id"]: p for p in self._raw["pull_requests"]}

        self._queue = []
        for bid in cfg.get("bug_ids", []):
            self._queue.append(BugReport(**bugs_idx[bid]))
        for pid in cfg.get("pr_ids", []):
            self._queue.append(PullRequest(**prs_idx[pid]))

        ep_id = str(uuid.uuid4())[:8]

        self._state = EnvState(
            task_id           = task_id,
            episode_id        = ep_id,
            step_count        = 0,
            max_steps         = cfg["max_steps"],
            item_index        = 0,
            total_items       = len(self._queue),
            is_done           = False,
            cumulative_reward = 0.0,
            normalised_score  = 0.0,
            actions_taken     = [],
            rewards_history   = [],
            last_feedback     = "Episode started. Process the first item.",
            last_action_valid = True,
        )

        return StepResult(
            observation = self._build_obs("Episode started. Process the first item."),
            reward      = 0.0,
            done        = False,
            info        = {"episode_id": ep_id, "task_id": task_id,
                           "total_items": len(self._queue)},
        )

    def step(self, action: BugAction) -> StepResult:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        if self._state.is_done:
            return StepResult(
                observation = self._build_obs("Episode already finished."),
                reward      = 0.0,
                done        = True,
                info        = {"warning": "episode_already_done"},
            )

        self._state.step_count += 1
        s = self._state

        parsed, parse_err = self._parse_json(action.message)

        if parse_err:
            reward_val = -0.3
            breakdown  = RewardBreakdown(invalid_penalty=-0.3)
            feedback   = (
                f"✗ Could not parse action: {parse_err}. "
                f"Send valid JSON, e.g. {{\"action\": \"classify\", \"severity\": \"high\"}}"
            )
            s.last_action_valid = False
            self._record(s, "INVALID", reward_val)
            return self._finalise(reward_val, breakdown, feedback)

        action_name         = str(parsed.get("action", "")).strip().lower()
        s.last_action_valid = True

        valid_set = _VALID_ACTIONS.get(s.task_id, set())
        if action_name not in valid_set:
            reward_val = -0.3
            breakdown  = RewardBreakdown(invalid_penalty=-0.3)
            feedback   = (
                f"✗ Action '{action_name}' not valid for task '{s.task_id}'. "
                f"Valid actions: {sorted(valid_set)}"
            )
            self._record(s, action_name, reward_val)
            return self._finalise(reward_val, breakdown, feedback)

        repeat_pen = 0.0
        if s.actions_taken:
            prev = s.actions_taken[-1]
            if prev.get("action") == action_name and prev.get("item_id") == self._item_id():
                repeat_pen = -0.1

        if action_name == "skip":
            reward_val, breakdown, feedback = self._handle_skip(repeat_pen)
        elif action_name in ("classify", "triage"):
            reward_val, breakdown, feedback = self._handle_bug_action(parsed, repeat_pen)
        elif action_name == "review_pr":
            reward_val, breakdown, feedback = self._handle_pr_action(parsed, repeat_pen)
        elif action_name == "prioritize":
            reward_val, breakdown, feedback = self._handle_prioritize(parsed)
        else:
            reward_val = -0.3
            breakdown  = RewardBreakdown(invalid_penalty=-0.3)
            feedback   = f"✗ Unhandled action '{action_name}'."

        self._record(s, action_name, reward_val)
        return self._finalise(reward_val, breakdown, feedback)

    def get_state(self) -> EnvState:
        if self._state is None:
            raise RuntimeError("Call reset() before state().")
        return self._state

    # -----------------------------------------------------------------------
    # Action handlers
    # -----------------------------------------------------------------------

    def _handle_skip(self, repeat_pen: float) -> tuple[float, RewardBreakdown, str]:
        self._advance()
        val = -0.2 + repeat_pen
        bd  = RewardBreakdown(skip_penalty=-0.2, repeat_penalty=repeat_pen)
        return val, bd, f"Item '{self._prev_item_id()}' skipped (−0.2). Moving on."

    def _handle_bug_action(
        self, parsed: dict, repeat_pen: float
    ) -> tuple[float, RewardBreakdown, str]:
        from .graders import grader_easy, grader_medium, grader_hard

        item = self._current_item()
        if item is None:
            return self._no_item_error()

        if not isinstance(item, BugReport):
            bd = RewardBreakdown(invalid_penalty=-0.3)
            return -0.3, bd, (
                f"✗ Current item '{item.id}' is a Pull Request — use 'review_pr', not 'triage'."
            )

        task = self._state.task_id
        if task == "easy":
            val, bd, fb = grader_easy.grade(parsed, item)
        elif task == "medium":
            val, bd, fb = grader_medium.grade_bug(parsed, item)
        else:
            val, bd, fb = grader_hard.grade_bug(parsed, item)

        val            += repeat_pen
        bd.repeat_penalty = repeat_pen
        self._advance()
        return val, bd, fb

    def _handle_pr_action(
        self, parsed: dict, repeat_pen: float
    ) -> tuple[float, RewardBreakdown, str]:
        from .graders import grader_hard

        item = self._current_item()
        if item is None:
            return self._no_item_error()

        if not isinstance(item, PullRequest):
            bd = RewardBreakdown(invalid_penalty=-0.3)
            return -0.3, bd, (
                f"✗ Current item '{item.id}' is a Bug Report — use 'triage', not 'review_pr'."
            )

        val, bd, fb    = grader_hard.grade_pr(parsed, item)
        val            += repeat_pen
        bd.repeat_penalty = repeat_pen
        self._advance()
        return val, bd, fb

    def _handle_prioritize(
        self, parsed: dict
    ) -> tuple[float, RewardBreakdown, str]:
        from .graders import grader_hard

        if self._state.task_id != "hard":
            bd = RewardBreakdown(invalid_penalty=-0.3)
            return -0.3, bd, "✗ 'prioritize' is only valid in the hard task."

        val, bd, fb       = grader_hard.grade_prioritization(parsed, self._task_cfg)
        self._state.is_done = True
        return val, bd, fb

    # -----------------------------------------------------------------------
    # Finalise step
    # -----------------------------------------------------------------------

    def _finalise(
        self,
        reward_val: float,
        breakdown:  RewardBreakdown,
        feedback:   str,
    ) -> StepResult:
        s = self._state

        s.cumulative_reward += reward_val
        s.rewards_history.append(reward_val)
        s.normalised_score = float(
            max(0.0, min(1.0, s.cumulative_reward / max(1, s.total_items)))
        )
        s.last_feedback = feedback

        if not s.is_done:
            if s.item_index >= s.total_items:
                if s.task_id == "hard" and not self._prioritize_phase:
                    self._prioritize_phase = True
                    all_ids = [i.id for i in self._queue]
                    feedback = (
                        feedback
                        + f" ✅ All items processed! Now submit: "
                        f'{{\"action\": \"prioritize\", \"order\": {all_ids}}}'
                    )
                    s.last_feedback = feedback
                elif s.task_id == "hard" and self._prioritize_phase:
                    # ← ADD THIS: waiting for prioritize action — hold episode open
                    pass
                else:
                    s.is_done = True

            if s.step_count >= s.max_steps:
                s.is_done       = True
                feedback        = feedback + " ⏱ Max steps reached — episode over."
                s.last_feedback = feedback

        obs = self._build_obs(feedback)
        return StepResult(
            observation = obs,
            reward      = reward_val,
            done        = s.is_done,
            info        = {
                "step":             s.step_count,
                "breakdown":        breakdown.model_dump(),
                "normalised_score": s.normalised_score,
                "item_index":       s.item_index,
                "total_items":      s.total_items,
            },
        )

    # -----------------------------------------------------------------------
    # Observation builder
    # -----------------------------------------------------------------------

    def _build_obs(self, feedback: str) -> BugObservation:
        s    = self._state
        item = self._current_item()

        if self._prioritize_phase and s.task_id == "hard":
            all_ids = [i.id for i in self._queue]
            return BugObservation(
                task_id           = s.task_id,
                episode_id        = s.episode_id,
                item_index        = s.item_index,
                total_items       = s.total_items,
                item_type         = ItemType.BUG_REPORT,
                item_id           = "PRIORITIZE",
                title             = "Final Step: Prioritize Fix Order",
                body              = (
                    "All items processed. Submit a 'prioritize' action with IDs ordered "
                    "highest to lowest priority.\n"
                    f"All item IDs: {all_ids}"
                ),
                labels            = [],
                reporter          = "system",
                available_actions = ["prioritize"],
                #step_number       = s.step_count + 1,
                step_number       = max(1, s.step_count),  # guarantee >= 1
                max_steps         = s.max_steps,
                current_score     = s.normalised_score,
                feedback          = feedback,
                echoed_message    = feedback,
            )
            
            
        if item is None or s.is_done:
            return BugObservation(
                task_id           = s.task_id,
                episode_id        = s.episode_id,
                item_index        = s.item_index,
                total_items       = s.total_items,
                item_type         = ItemType.BUG_REPORT,
                item_id           = "DONE",
                title             = "Episode Complete",
                body              = (
                    f"Final score: {s.normalised_score:.3f}. "
                    f"Steps used: {s.step_count}/{s.max_steps}."
                ),
                labels            = [],
                reporter          = "",
                available_actions = [],
                step_number       = s.step_count,
                max_steps         = s.max_steps,
                current_score     = s.normalised_score,
                feedback          = feedback,
                echoed_message    = feedback,
            )


        if isinstance(item, BugReport):
            item_type = ItemType.BUG_REPORT
            reporter  = item.reporter
        else:
            item_type = ItemType.PULL_REQUEST
            reporter  = item.author

        available  = self._available_actions(item_type)
        task_hint  = self._task_hint()

        return BugObservation(
            task_id           = s.task_id,
            episode_id        = s.episode_id,
            item_index        = s.item_index,
            total_items       = s.total_items,
            item_type         = item_type,
            item_id           = item.id,
            title             = item.title,
            body              = f"{task_hint}\n\n---\n{item.body}",
            labels            = item.labels,
            reporter          = reporter,
            available_actions = available,
            step_number       = s.step_count + 1,
            max_steps         = s.max_steps,
            current_score     = s.normalised_score,
            feedback          = feedback,
            echoed_message    = feedback,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _load_data(self) -> dict[str, Any]:
        with open(_DATA_PATH, encoding="utf-8") as fh:
            return json.load(fh)

    def _current_item(self) -> BugReport | PullRequest | None:
        s = self._state
        if s is None or s.item_index >= len(self._queue):
            return None
        return self._queue[s.item_index]

    def _item_id(self) -> str:
        item = self._current_item()
        return item.id if item else "NONE"

    def _prev_item_id(self) -> str:
        idx = self._state.item_index - 1
        if 0 <= idx < len(self._queue):
            return self._queue[idx].id
        return "NONE"

    def _advance(self) -> None:
        self._state.item_index += 1

    def _record(self, s: EnvState, action_name: str, reward: float) -> None:
        s.actions_taken.append({
            "step":    s.step_count,
            "item_id": self._item_id(),
            "action":  action_name,
            "reward":  round(reward, 4),
        })

    def _available_actions(self, item_type: ItemType) -> list[str]:
        task = self._state.task_id
        if task == "easy":
            return ["classify", "skip"]
        if task == "medium":
            return ["triage", "skip"]
        if item_type == ItemType.PULL_REQUEST:
            return ["review_pr", "skip"]
        return ["triage", "skip"]

    def _task_hint(self) -> str:
        s    = self._state
        task = s.task_id
        base = f"[Item {s.item_index + 1}/{s.total_items}]  Task: {task}"
        if task == "easy":
            return base + '  →  Classify severity. Reply: {"action":"classify","severity":"..."}'
        if task == "medium":
            return (
                base
                + '  →  Triage: severity + team. '
                'Reply: {"action":"triage","severity":"...","team":"..."}'
            )
        return (
            base
            + '  →  For bugs: {"action":"triage","severity":"...","team":"...",'
            '"comment":"...","security_issue":false}  '
            'For PRs: {"action":"review_pr","verdict":"...","comment":"...","security_issue":false}'
        )

    @staticmethod
    def _no_item_error() -> tuple[float, RewardBreakdown, str]:
        bd = RewardBreakdown(invalid_penalty=-0.3)
        return -0.3, bd, "✗ No current item — queue may be exhausted."

    @staticmethod
    def _parse_json(message: str) -> tuple[dict, str | None]:
        try:
            obj = json.loads(message)
            if not isinstance(obj, dict):
                return {}, "Action must be a JSON object {…}"
            if "action" not in obj:
                return {}, "Missing required key 'action'"
            return obj, None
        except json.JSONDecodeError as exc:
            return {}, f"JSON error: {exc}"