"""
demo_curriculum.py — Curriculum Escalation Demo

Demonstrates the self-learning curriculum mechanic:
  - Agents start on EASY task
  - After 3 consecutive episodes with avg reward >= 0.75 → escalates to MEDIUM
  - After 3 consecutive episodes with avg reward >= 0.70 → escalates to HARD

This script is completely separate from inference_multi.py.
It only calls the /multi_reset and /multi_step endpoints.

Usage:
  python demo_curriculum.py

Environment variables (reads from .env):
  ENV_URL      — environment URL (default: http://localhost:7860)
  API_BASE_URL — LLM API base URL
  MODEL_NAME   — model identifier
  HF_TOKEN     — API key
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import List

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

MAX_EPISODES  = 15   # run up to 15 episodes to show full escalation
MAX_STEPS     = 25   # max steps per episode

# ---------------------------------------------------------------------------
# Agent prompts
# ---------------------------------------------------------------------------

AGENT_A_PROMPT = """You are Agent A (Triager). Analyze the bug and propose triage.
Reply with ONLY a valid JSON object. No explanation, no markdown.

For easy task: {"action": "classify", "severity": "critical|high|medium|low"}
For other tasks: {"action": "triage", "severity": "critical|high|medium|low", "team": "frontend|backend|infra|security", "comment": "<brief note>", "security_issue": false}

Severity: critical=crashes/security, high=production, medium=bugs, low=cosmetic
Teams: frontend=UI/browser, backend=API/DB, infra=k8s/memory, security=auth/injection"""

AGENT_B_PROMPT = """You are Agent B (Reviewer). Review Agent A's proposal and agree or correct it.
Reply with ONLY a valid JSON object. No explanation, no markdown.

For easy task: {"action": "classify", "severity": "critical|high|medium|low"}
For other tasks: {"action": "triage", "severity": "critical|high|medium|low", "team": "frontend|backend|infra|security", "comment": "<your note>", "security_issue": false}

Bonus reward when both you and Agent A are correct and agree.
Penalty for blind agreement when A is wrong."""

# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class EnvClient:
    def __init__(self) -> None:
        self._http: httpx.AsyncClient | None = None

    async def _get(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=30.0)
        return self._http

    async def reset(self, task_id: str | None = None) -> dict:
        http = await self._get()
        body = {"task_id": task_id} if task_id else {}
        resp = await http.post(f"{ENV_URL}/multi_reset", json=body)
        resp.raise_for_status()
        return resp.json()

    async def step(self, message_a: str, message_b: str) -> dict:
        http = await self._get()
        resp = await http.post(
            f"{ENV_URL}/multi_step",
            json={"message_a": message_a, "message_b": message_b},
        )
        resp.raise_for_status()
        return resp.json()

    async def get_curriculum(self) -> dict:
        http = await self._get()
        resp = await http.get(f"{ENV_URL}/multi_state")
        resp.raise_for_status()
        data = resp.json()
        return data.get("curriculum", {})

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def get_action(
    client:    OpenAI,
    agent_id:  str,
    obs:       dict,
    task_id:   str,
) -> str:
    system = AGENT_A_PROMPT if agent_id == "agent_a" else AGENT_B_PROMPT
    title  = obs.get("title", "")
    body   = obs.get("body", "")[:300]
    item_id = obs.get("item_id", "")
    avail  = obs.get("available_actions", [])

    user = (
        f"Task: {task_id}\n"
        f"Item: {item_id}\n"
        f"Title: {title}\n"
        f"Body: {body}\n"
        f"Available actions: {avail}\n"
        f"Reply with ONLY a JSON object."
    )

    try:
        resp = client.chat.completions.create(
            model       = MODEL_NAME,
            max_tokens  = 150,
            temperature = 0.0,
            messages    = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        if "```" in text:
            text = "\n".join(l for l in text.split("\n") if not l.startswith("```")).strip()
        json.loads(text)
        return text
    except Exception as exc:
        print(f"    [DEBUG] {agent_id} failed: {exc}", flush=True)
        return _fallback(task_id, obs.get("item_type", "bug_report"))


def _fallback(task_id: str, item_type: str) -> str:
    if task_id == "easy":
        return '{"action": "classify", "severity": "medium"}'
    if item_type == "pull_request":
        return '{"action": "review_pr", "verdict": "comment", "comment": "needs review", "security_issue": false}'
    return '{"action": "triage", "severity": "medium", "team": "backend", "comment": "needs investigation", "security_issue": false}'

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

async def main() -> None:
    print("\n" + "=" * 60, flush=True)
    print("  Bug Triage — Curriculum Self-Learning Demo", flush=True)
    print(f"  Model  : {MODEL_NAME}", flush=True)
    print(f"  Env    : {ENV_URL}", flush=True)
    print("=" * 60, flush=True)
    print("\nWatching agents improve and curriculum escalate...\n", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env    = EnvClient()

    prev_task        = None
    episode_rewards  = []

    try:
        for ep in range(1, MAX_EPISODES + 1):

            # Reset — let curriculum decide task (no task_id override)
            result   = await env.reset()
            obs_a    = result["observation_a"]
            task_id  = obs_a.get("task_id", "easy")
            ep_id    = obs_a.get("episode_id", "")

            # Detect curriculum escalation
            if prev_task and prev_task != task_id:
                print(f"\n{'★' * 50}", flush=True)
                print(f"  CURRICULUM ESCALATED: {prev_task.upper()} → {task_id.upper()}", flush=True)
                print(f"  Agents mastered {prev_task} task — moving to harder challenge!", flush=True)
                print(f"{'★' * 50}\n", flush=True)

            prev_task   = task_id
            ep_rewards  = []

            print(f"Episode {ep:2d} | task={task_id:<6} | episode_id={ep_id}", flush=True)

            # Step loop
            for step in range(1, MAX_STEPS + 1):
                if result.get("done", False):
                    break

                obs_a    = result["observation_a"]
                obs_b    = result["observation_b"]
                item_id  = obs_a.get("item_id", "")
                item_type = obs_a.get("item_type", "bug_report")

                if item_id in ("DONE", "PRIORITIZE"):
                    break

                action_a = get_action(client, "agent_a", obs_a, task_id)
                action_b = get_action(client, "agent_b", obs_b, task_id)

                result      = await env.step(action_a, action_b)
                reward_a    = result.get("reward_a", 0.0)
                reward_b    = result.get("reward_b", 0.0)
                joint       = result.get("joint_reward", 0.0)
                consensus   = result.get("consensus", False)
                done        = result.get("done", False)

                ep_rewards.append(joint)

                consensus_str = "✓ agree" if consensus else "✗ disagree"
                print(
                    f"  Step {step:2d} | {item_id:<8} | "
                    f"A={reward_a:.2f} B={reward_b:.2f} joint={joint:.2f} | "
                    f"{consensus_str}",
                    flush=True,
                )

                if done:
                    break

            # Episode summary
            ep_avg = sum(ep_rewards) / len(ep_rewards) if ep_rewards else 0.0
            episode_rewards.append(ep_avg)

            # Get curriculum state
            try:
                curriculum = await env.get_curriculum()
                level      = curriculum.get("level", 0)
                rolling    = curriculum.get("rolling_avg", 0.0)
                curr_task  = curriculum.get("current_task", task_id)
                threshold  = {0: 0.75, 1: 0.70, 2: float("inf")}.get(level, 0.75)
                print(
                    f"  → Episode avg: {ep_avg:.3f} | "
                    f"Rolling avg: {rolling:.3f} | "
                    f"Threshold to escalate: {threshold} | "
                    f"Next task: {curr_task}",
                    flush=True,
                )
            except Exception:
                print(f"  → Episode avg: {ep_avg:.3f}", flush=True)

            print(flush=True)

            # Stop if we've seen all three difficulty levels
            if len(set(t for t in [task_id])) >= 1 and ep_avg > 0 and task_id == "hard" and ep >= 6:
                print("All difficulty levels demonstrated. Demo complete!", flush=True)
                break

    finally:
        await env.close()

    # Final summary
    print("\n" + "=" * 60, flush=True)
    print("  CURRICULUM DEMO SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  Total episodes run : {len(episode_rewards)}", flush=True)
    if episode_rewards:
        print(f"  Avg reward overall : {sum(episode_rewards)/len(episode_rewards):.3f}", flush=True)
    print("\n  What this demonstrates:", flush=True)
    print("  1. Two agents (A+B) collaborating on each bug", flush=True)
    print("  2. Consensus rewards — agents learn to agree correctly", flush=True)
    print("  3. Curriculum escalation — harder tasks as agents improve", flush=True)
    print("=" * 60 + "\n", flush=True)


if __name__ == "__main__":
    asyncio.run(main())