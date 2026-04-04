"""
inference.py — Baseline inference script for Bug Triage OpenEnv.

Runs an LLM agent against all three tasks (easy, medium, hard) and
produces reproducible baseline scores.

Mandatory stdout format (DO NOT change):
  [START] task=<n> env=<n> model=<n>
  [STEP]  step=<n> action=<str> reward=<float> done=<bool> error=<val>
  [END]   success=<bool> steps=<n> score=<float> rewards=<list>

Environment variables (set in .env or HF Space secrets):
  API_BASE_URL  — LLM API base URL  (e.g. https://api.openai.com/v1)
  MODEL_NAME    — model identifier   (e.g. gpt-4o-mini)
  HF_TOKEN      — API key / HF token
  ENV_URL       — running env URL    (default: http://localhost:7860)
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, List

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — read from environment variables
# ---------------------------------------------------------------------------

# API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
# API_KEY:      str = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
# MODEL_NAME:   str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
# ENV_URL:      str = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")

from dotenv import load_dotenv
load_dotenv()

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY:      str = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
MODEL_NAME:   str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
ENV_URL:      str = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")

BENCHMARK = "bug-triage-env"

# Per-task config
TASK_CONFIGS: dict[str, dict] = {
    "easy": {
        "max_steps":          3,
        "max_total_reward":   1.0,
        "success_threshold":  0.6,
    },
    "medium": {
        "max_steps":          15,
        "max_total_reward":   5.0,
        "success_threshold":  0.5,
    },
    "hard": {
        "max_steps":          25,
        "max_total_reward":   11.0,   # 8 bugs + 3 PRs
        "success_threshold":  0.4,
    },
}

# ---------------------------------------------------------------------------
# Structured logging — EXACT format required by evaluator (do not change)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step:   int,
    action: str,
    reward: float,
    done:   bool,
    error:  Any,
) -> None:
    print(
        f"[STEP] step={step} action={action!r} "
        f"reward={reward:.4f} done={done} error={error}",
        flush=True,
    )


def log_end(
    success: bool,
    steps:   int,
    score:   float,
    rewards: List[float],
) -> None:
    print(
        f"[END] success={success} steps={steps} "
        f"score={score:.4f} rewards={rewards}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Lightweight environment client (hits the running HTTP server)
# ---------------------------------------------------------------------------

class EnvClient:
    """
    Thin async HTTP wrapper around the Bug Triage OpenEnv server.
    Mirrors the openenv-core client interface used in the sample script:
      await env.reset()
      await env.step(message)
      await env.close()
    """

    def __init__(self) -> None:
        self._http: httpx.AsyncClient | None = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=30.0)
        return self._http

    async def reset(self, task_id: str = "easy") -> "EnvResult":
        http = await self._get_http()
        resp = await http.post(f"{ENV_URL}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        return EnvResult(resp.json())

    async def step(self, message: str) -> "EnvResult":
        http = await self._get_http()
        resp = await http.post(f"{ENV_URL}/step", json={"message": message})
        resp.raise_for_status()
        return EnvResult(resp.json())

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None


class EnvResult:
    """
    Wraps the JSON response so fields can be accessed as attributes —
    matching the sample pattern:
      result.observation.echoed_message
      result.done
      result.reward
    """

    def __init__(self, data: dict) -> None:
        self._data            = data
        self.reward: float    = float(data.get("reward") or 0.0)
        self.done:   bool     = bool(data.get("done", False))
        self.info:   dict     = data.get("info", {})
        self.observation      = EnvObservation(data.get("observation", {}))


class EnvObservation:
    """Wraps observation dict as object attributes."""

    def __init__(self, data: dict) -> None:
        self._data                      = data
        self.echoed_message: str        = data.get("echoed_message") or data.get("feedback", "")
        self.item_id:        str        = data.get("item_id", "UNKNOWN")
        self.item_type:      str        = data.get("item_type", "bug_report")
        self.title:          str        = data.get("title", "")
        self.body:           str        = data.get("body", "")
        self.available_actions: list    = data.get("available_actions", [])
        self.current_score:  float      = float(data.get("current_score", 0.0))
        self.task_id:        str        = data.get("task_id", "")
        self.step_number:    int        = int(data.get("step_number", 1))
        self.max_steps:      int        = int(data.get("max_steps", 1))


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

def build_system_prompt(task_id: str) -> str:
    """System prompt tailored to the task difficulty."""
    base = (
        "You are an expert software engineer performing bug triage and code review. "
        "You will receive bug reports and pull requests one at a time. "
        "You must respond with ONLY a valid JSON object — no explanation, no markdown, "
        "no code fences. Just the raw JSON.\n\n"
        "Teams available: frontend, backend, infra, security\n"
        "Severities available: critical, high, medium, low\n\n"
        "Severity guide:\n"
        "  critical — crashes, security vulnerabilities, data loss, auth bypass\n"
        "  high     — production degradation, significant functionality broken\n"
        "  medium   — noticeable bugs, incorrect behavior, moderate impact\n"
        "  low      — typos, cosmetic issues, minor inconveniences\n\n"
        "Team guide:\n"
        "  frontend — UI, charts, browser, CSS, images, rendering\n"
        "  backend  — APIs, databases, business logic, exports, queries\n"
        "  infra    — Kubernetes, Docker, CI/CD, memory, deployments, batch jobs\n"
        "  security — auth, rate limiting, SQL injection, access control, tokens\n"
    )

    if task_id == "easy":
        return base + (
            "\nYour ONLY job: classify severity.\n"
            "Always respond with exactly:\n"
            '{"action": "classify", "severity": "<critical|high|medium|low>"}'
        )

    if task_id == "medium":
        return base + (
            "\nYour job: classify severity AND assign team for each bug.\n"
            "Always respond with exactly:\n"
            '{"action": "triage", "severity": "<level>", "team": "<team>"}'
        )

    # hard
    return base + (
        "\nYour job: full pipeline — triage bugs, review PRs, then prioritize.\n\n"
        "For BUG REPORTS respond with:\n"
        '{"action": "triage", "severity": "<level>", "team": "<team>", '
        '"comment": "<technical comment>", "security_issue": <true|false>}\n\n'
        "For PULL REQUESTS respond with:\n"
        '{"action": "review_pr", "verdict": "<approve|request_changes|comment>", '
        '"comment": "<review comment>", "security_issue": <true|false>}\n\n'
        "For the FINAL PRIORITIZATION step respond with:\n"
        '{"action": "prioritize", "order": ["ID1", "ID2", ...]}\n\n'
        "SECURITY WARNING: f-string SQL interpolation is ALWAYS a critical security "
        "issue — set security_issue=true and verdict=request_changes.\n\n"
        "Priority order rule: critical security > critical crash > "
        "high production > high infra > PRs with security > medium > low."
    )


def get_model_message(
    client:      OpenAI,
    step:        int,
    last_echoed: str,
    last_reward: float,
    history:     List[str],
    obs:         EnvObservation | None = None,
) -> str:
    """
    Call the LLM with current context and return its action JSON string.
    Signature matches sample: get_model_message(client, step, last_echoed, last_reward, history)
    Falls back to a safe default action if the call fails.
    """
    task_id   = obs.task_id            if obs else "easy"
    item_id   = obs.item_id            if obs else "UNKNOWN"
    item_type = obs.item_type          if obs else "bug_report"
    title     = obs.title              if obs else ""
    body      = obs.body               if obs else ""
    avail     = obs.available_actions  if obs else []

    user_content = (
        f"Step {step} | Item: {item_id} | Type: {item_type}\n"
        f"Available actions: {avail}\n"
        f"Last reward: {last_reward:+.2f}\n"
    )

    if last_echoed:
        user_content += f"Feedback from last action: {last_echoed}\n"

    user_content += f"\n--- TITLE ---\n{title}\n\n--- BODY ---\n{body}\n"

    if history:
        recent = history[-3:]   # last 3 steps for context, keep prompt short
        user_content += "\n--- Recent history ---\n" + "\n".join(recent)

    user_content += "\n\nRespond with ONLY a JSON object."

    try:
        completion = client.chat.completions.create(
            model       = MODEL_NAME,
            max_tokens  = 300,
            temperature = 0.0,   # deterministic — required for reproducibility
            messages    = [
                {"role": "system", "content": build_system_prompt(task_id)},
                {"role": "user",   "content": user_content},
            ],
        )
        text = (completion.choices[0].message.content or "").strip()

        # Strip accidental markdown fences
        if text.startswith("```"):
            text = "\n".join(
                line for line in text.split("\n")
                if not line.startswith("```")
            ).strip()

        # Validate it parses before returning
        json.loads(text)
        return text

    except json.JSONDecodeError:
        print(f"[DEBUG] Model returned non-JSON, using fallback", flush=True)
        return _fallback_action(task_id, item_type, item_id)

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return _fallback_action(task_id, item_type, item_id)


def _fallback_action(task_id: str, item_type: str, item_id: str) -> str:
    """Safe fallback when model fails — never crashes the episode."""
    if item_id in ("DONE", "PRIORITIZE"):
        return '{"action": "prioritize", "order": []}'
    if task_id == "easy":
        return '{"action": "classify", "severity": "medium"}'
    if task_id == "medium":
        return '{"action": "triage", "severity": "medium", "team": "backend"}'
    if item_type == "pull_request":
        return ('{"action": "review_pr", "verdict": "comment", '
                '"comment": "Needs review", "security_issue": false}')
    return ('{"action": "triage", "severity": "medium", "team": "backend", '
            '"comment": "Needs investigation", "security_issue": false}')


# ---------------------------------------------------------------------------
# Single task runner — structure mirrors sample script exactly
# ---------------------------------------------------------------------------

async def run_task(task_id: str) -> dict:
    """Run one full episode. Returns score, success, steps, rewards."""
    cfg               = TASK_CONFIGS[task_id]
    max_steps         = cfg["max_steps"]
    max_total_reward  = cfg["max_total_reward"]
    success_threshold = cfg["success_threshold"]

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env    = EnvClient()

    history:     List[str]   = []
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # -- Reset — matches sample: result = await env.reset() ---------------
        result      = await env.reset(task_id)
        last_echoed = result.observation.echoed_message  # matches sample
        last_reward = 0.0

        # -- Step loop — matches sample structure exactly ---------------------
        for step in range(1, max_steps + 1):
            if result.done:          # check BEFORE step — matches sample
                break

            error = None
            try:
                message = get_model_message(
                    client, step, last_echoed, last_reward, history,
                    obs=result.observation,
                )
                result  = await env.step(message)
                obs     = result.observation             # matches sample
                reward  = result.reward or 0.0           # matches sample
                done    = result.done                    # matches sample

            except Exception as exc:
                error   = str(exc)
                reward  = 0.0
                done    = False
                message = _fallback_action(task_id, "bug_report", "UNKNOWN")
                print(f"[DEBUG] step {step} error: {exc}", flush=True)
                obs = result.observation

            rewards.append(reward)
            steps_taken  = step
            last_echoed  = obs.echoed_message            # matches sample
            last_reward  = reward

            log_step(step=step, action=message, reward=reward,
                     done=done, error=error)

            # EXACT history format from sample spec
            history.append(f"Step {step}: {message!r} -> reward {reward:+.2f}")

            if done:
                break

        # -- Score — matches sample exactly -----------------------------------
        score   = sum(rewards) / max_total_reward if max_total_reward > 0 else 0.0
        score   = min(max(score, 0.0), 1.0)
        success = score >= success_threshold

    finally:
        # matches sample: finally: await env.close()
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {
        "score":   score,
        "success": success,
        "steps":   steps_taken,
        "rewards": rewards,
    }


# ---------------------------------------------------------------------------
# Main — run all three tasks sequentially
# ---------------------------------------------------------------------------

async def main() -> None:
    print("\n" + "=" * 60, flush=True)
    print("  Bug Triage OpenEnv — Baseline Inference", flush=True)
    print(f"  Model : {MODEL_NAME}", flush=True)
    print(f"  Env   : {ENV_URL}", flush=True)
    print("=" * 60 + "\n", flush=True)

    all_results: dict[str, dict] = {}

    for task_id in ("easy", "medium", "hard"):
        print(f"\n{'─' * 40}", flush=True)
        print(f"  Running task: {task_id.upper()}", flush=True)
        print(f"{'─' * 40}", flush=True)

        result = await run_task(task_id)
        all_results[task_id] = result

        print(
            f"\n  Task '{task_id}' complete → "
            f"score={result['score']:.4f} "
            f"success={result['success']} "
            f"steps={result['steps']}",
            flush=True,
        )

    # -- Summary --------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("  BASELINE RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)

    total_score = 0.0
    for task_id, res in all_results.items():
        status = "✓ PASS" if res["success"] else "✗ FAIL"
        print(
            f"  {status}  {task_id:<8} "
            f"score={res['score']:.4f}  steps={res['steps']}",
            flush=True,
        )
        total_score += res["score"]

    overall = total_score / len(all_results)
    print(f"\n  Overall average score : {overall:.4f}", flush=True)
    print("=" * 60 + "\n", flush=True)


if __name__ == "__main__":
    asyncio.run(main())