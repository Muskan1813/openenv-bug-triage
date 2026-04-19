"""
inference_multi.py — Multi-agent baseline inference script.

Two agents collaborate on bug triage:
  Agent A (Triager)  — proposes severity + team + comment
  Agent B (Reviewer) — sees Agent A's proposal, agrees or corrects

Stdout log format (same spec as inference.py):
  [START] task=<n> env=<n> model=<n>
  [STEP]  step=<n> action=<str> reward=<float> done=<bool> error=<val>
  [END]   success=<bool> steps=<n> score=<float> rewards=<list>

Environment variables:
  API_BASE_URL  — LLM API base URL
  MODEL_NAME    — model identifier
  HF_TOKEN      — API key
  ENV_URL       — running env URL (default: http://localhost:7860)
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, List

import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY:      str = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
MODEL_NAME:   str = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_URL:      str = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")

BENCHMARK = "bug-triage-env-multi-agent"

TASK_CONFIGS: dict[str, dict] = {
    "easy":   {"max_steps": 3,  "max_total_reward": 1.0,  "success_threshold": 0.6},
    "medium": {"max_steps": 15, "max_total_reward": 5.0,  "success_threshold": 0.5},
    "hard":   {"max_steps": 25, "max_total_reward": 11.0, "success_threshold": 0.4},
}

# ---------------------------------------------------------------------------
# Structured logging — exact format required by evaluator
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Any) -> None:
    print(
        f"[STEP] step={step} action={action!r} "
        f"reward={reward:.4f} done={done} error={error}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={success} steps={steps} "
        f"score={score:.4f} rewards={rewards}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment HTTP client
# ---------------------------------------------------------------------------

class MultiEnvClient:
    """Async HTTP client for the multi-agent environment."""

    def __init__(self) -> None:
        self._http: httpx.AsyncClient | None = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=60.0)
        return self._http

    async def reset(self, task_id: str = "easy") -> "MultiEnvResult":
        http = await self._get_http()
        resp = await http.post(f"{ENV_URL}/multi_reset", json={"task_id": task_id})
        resp.raise_for_status()
        return MultiEnvResult(resp.json())

    async def step(self, message_a: str, message_b: str) -> "MultiEnvResult":
        http = await self._get_http()
        resp = await http.post(
            f"{ENV_URL}/multi_step",
            json={"message_a": message_a, "message_b": message_b},
        )
        resp.raise_for_status()
        return MultiEnvResult(resp.json())

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None


class MultiEnvObservation:
    """Wraps observation dict as attributes."""
    def __init__(self, data: dict) -> None:
        self.echoed_message:    str   = data.get("echoed_message") or data.get("feedback", "")
        self.item_id:           str   = data.get("item_id", "UNKNOWN")
        self.item_type:         str   = data.get("item_type", "bug_report")
        self.title:             str   = data.get("title", "")
        self.body:              str   = data.get("body", "")
        self.available_actions: list  = data.get("available_actions", [])
        self.task_id:           str   = data.get("task_id", "easy")
        self.step_number:       int   = int(data.get("step_number", 1))
        self.max_steps:         int   = int(data.get("max_steps", 1))
        self.current_score:     float = float(data.get("current_score", 0.0))


class MultiEnvResult:
    """Wraps full multi-step response."""
    def __init__(self, data: dict) -> None:
        self._data          = data
        self.reward_a:      float = float(data.get("reward_a") or 0.0)
        self.reward_b:      float = float(data.get("reward_b") or 0.0)
        self.joint_reward:  float = float(data.get("joint_reward") or 0.0)
        self.done:          bool  = bool(data.get("done", False))
        self.consensus:     bool  = bool(data.get("consensus", False))
        self.info:          dict  = data.get("info", {})
        self.observation_a        = MultiEnvObservation(data.get("observation_a", {}))
        self.observation_b        = MultiEnvObservation(data.get("observation_b", {}))


# ---------------------------------------------------------------------------
# Agent system prompts
# ---------------------------------------------------------------------------

AGENT_A_PROMPT = """You are Agent A — the Triager. You are an expert software engineer.
Your job is to be the FIRST to analyze each bug report or pull request and propose a triage.

You must respond with ONLY a valid JSON object. No explanation, no markdown.

For BUG REPORTS:
{"action": "triage", "severity": "critical|high|medium|low", "team": "frontend|backend|infra|security", "comment": "<brief technical note>", "security_issue": false}

For the EASY task (severity only):
{"action": "classify", "severity": "critical|high|medium|low"}

Severity guide:
  critical — crashes, auth bypass, data loss, security vulnerabilities
  high     — production degradation, significant bugs
  medium   — noticeable bugs, moderate impact
  low      — typos, cosmetic issues

Team guide:
  frontend — UI, charts, browser, images
  backend  — APIs, databases, business logic, exports
  infra    — Kubernetes, memory, deployments, batch jobs
  security — auth, rate limiting, SQL injection, access control"""


AGENT_B_PROMPT = """You are Agent B — the Reviewer. You are a senior software engineer.
Your job is to REVIEW Agent A's triage proposal and either agree or correct it.

You will see Agent A's proposal in the observation. Read it carefully.
If you agree, submit the SAME triage. If you disagree, submit your CORRECTED version.

You must respond with ONLY a valid JSON object. No explanation, no markdown.

For BUG REPORTS:
{"action": "triage", "severity": "critical|high|medium|low", "team": "frontend|backend|infra|security", "comment": "<your review note>", "security_issue": false}

For the EASY task:
{"action": "classify", "severity": "critical|high|medium|low"}

IMPORTANT: You get a BONUS reward when you and Agent A BOTH give the correct answer.
You get PENALIZED for agreeing blindly when Agent A is wrong.
So: agree when you genuinely think A is correct, correct when you spot an error.

SECURITY: Always check for SQL injection (f-string queries), missing input validation,
auth bypass. These must be severity=critical, team=security, security_issue=true."""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_agent_action(
    client:       OpenAI,
    agent_id:     str,
    step:         int,
    obs:          MultiEnvObservation,
    last_echoed:  str,
    last_reward:  float,
    history:      List[str],
) -> str:
    """Call LLM for one agent and return its action JSON string."""

    system_prompt = AGENT_A_PROMPT if agent_id == "agent_a" else AGENT_B_PROMPT

    user_content = (
        f"Step {step} | Item: {obs.item_id} | Type: {obs.item_type}\n"
        f"Available actions: {obs.available_actions}\n"
        f"Last reward: {last_reward:+.2f}\n"
    )

    if last_echoed:
        user_content += f"Feedback: {last_echoed}\n"

    user_content += f"\n--- TITLE ---\n{obs.title}\n\n--- BODY ---\n{obs.body}\n"

    if history:
        recent = history[-3:]
        user_content += "\n--- Recent history ---\n" + "\n".join(recent)

    user_content += "\n\nRespond with ONLY a JSON object."

    try:
        completion = client.chat.completions.create(
            model       = MODEL_NAME,
            max_tokens  = 300,
            temperature = 0.0,
            messages    = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
        )
        text = (completion.choices[0].message.content or "").strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            text = "\n".join(
                line for line in text.split("\n")
                if not line.startswith("```")
            ).strip()

        json.loads(text)   # validate
        return text

    except json.JSONDecodeError:
        print(f"[DEBUG] {agent_id} returned non-JSON, using fallback", flush=True)
        return _fallback_action(obs.task_id, obs.item_type, obs.item_id)

    except Exception as exc:
        print(f"[DEBUG] {agent_id} model call failed: {exc}", flush=True)
        return _fallback_action(obs.task_id, obs.item_type, obs.item_id)


def _fallback_action(task_id: str, item_type: str, item_id: str) -> str:
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
# Single task runner
# ---------------------------------------------------------------------------

async def run_task(task_id: str) -> dict:
    """Run one full multi-agent episode."""
    cfg               = TASK_CONFIGS[task_id]
    max_steps         = cfg["max_steps"]
    max_total_reward  = cfg["max_total_reward"]
    success_threshold = cfg["success_threshold"]

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env    = MultiEnvClient()

    history_a:   List[str]   = []
    history_b:   List[str]   = []
    rewards:     List[float] = []   # joint rewards for scoring
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # -- Reset --------------------------------------------------------
        result       = await env.reset(task_id)
        last_echoed  = result.observation_a.echoed_message
        last_reward_a = 0.0
        last_reward_b = 0.0

        # -- Step loop ----------------------------------------------------
        for step in range(1, max_steps + 1):
            if result.done:
                break

            error = None
            try:
                # Agent A acts first
                message_a = get_agent_action(
                    client, "agent_a", step,
                    result.observation_a, last_echoed,
                    last_reward_a, history_a,
                )

                # Agent B sees the observation + Agent A's action embedded in obs_b body
                message_b = get_agent_action(
                    client, "agent_b", step,
                    result.observation_b, last_echoed,
                    last_reward_b, history_b,
                )

                result        = await env.step(message_a, message_b)
                joint_reward  = result.joint_reward
                done          = result.done
                consensus     = result.consensus

            except Exception as exc:
                error        = str(exc)
                joint_reward = 0.0
                done         = False
                consensus    = False
                message_a    = _fallback_action(task_id, "bug_report", "UNKNOWN")
                message_b    = _fallback_action(task_id, "bug_report", "UNKNOWN")
                print(f"[DEBUG] step {step} error: {exc}", flush=True)

            rewards.append(joint_reward)
            steps_taken   = step
            last_reward_a = result.reward_a
            last_reward_b = result.reward_b
            last_echoed   = result.observation_a.echoed_message

            # Log combined action for [STEP] format
            combined_action = json.dumps({
                "agent_a": message_a,
                "agent_b": message_b,
                "consensus": consensus,
            })

            log_step(
                step   = step,
                action = combined_action,
                reward = joint_reward,
                done   = done,
                error  = error,
            )

            # History for both agents
            history_a.append(f"Step {step}: {message_a!r} -> reward {last_reward_a:+.2f}")
            history_b.append(f"Step {step}: {message_b!r} -> reward {last_reward_b:+.2f}")

            if done:
                break

        # -- Score --------------------------------------------------------
        score   = sum(rewards) / max_total_reward if max_total_reward > 0 else 0.0
        score   = min(max(score, 0.001), 0.999)
        success = score >= success_threshold

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {
        "score":   score,
        "success": success,
        "steps":   steps_taken,
        "rewards": rewards,
    }


# ---------------------------------------------------------------------------
# Main — run all three tasks
# ---------------------------------------------------------------------------

async def main() -> None:
    print("\n" + "=" * 60, flush=True)
    print("  Bug Triage OpenEnv — Multi-Agent Baseline", flush=True)
    print(f"  Model  : {MODEL_NAME}", flush=True)
    print(f"  Env    : {ENV_URL}", flush=True)
    print(f"  Agents : Agent A (Triager) + Agent B (Reviewer)", flush=True)
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

    # -- Summary ----------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("  MULTI-AGENT BASELINE RESULTS", flush=True)
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