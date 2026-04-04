---
title: Bug Triage OpenEnv
emoji: 🐛
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
---

# 🐛 Bug Triage & Code Review — OpenEnv Environment

An AI agent environment where the agent acts as a software engineer —
reading bug reports, triaging issues, reviewing pull requests, and
prioritizing what to fix first. This is something real engineering teams
do every single day.

---

## Why This Exists

Most RL environments teach agents to play games. This one teaches agents
to do actual software engineering work. Bug triage is a perfect fit because:

- Every software team does it daily
- It has clear right and wrong answers (severity, team, security issues)
- It naturally scales from easy to hard
- A bad agent decision has real-world consequences (e.g. missing a security vulnerability)

---

## What the Agent Does

The agent receives bug reports and pull requests one at a time and must:

1. **Classify** how severe the bug is (`critical`, `high`, `medium`, `low`)
2. **Route** it to the right team (`frontend`, `backend`, `infra`, `security`)
3. **Review** pull requests and decide whether to approve or request changes
4. **Detect** security vulnerabilities in code (e.g. SQL injection)
5. **Prioritize** a fix order across all items

---

## The Three Tasks

### 🟢 Easy — Single Bug Classification
The agent gets one bug report and must classify its severity.

- **Action required:** `{"action": "classify", "severity": "critical|high|medium|low"}`
- **Max steps:** 3
- **Scoring:** Exact match = 1.0, adjacent level = 0.5, wrong = 0.0
- **Expected score for a decent LLM:** ~0.75

**Example bug:** Login endpoint crashes with 500 on empty password → should be `critical`

---

### 🟡 Medium — Bug Queue Triage
The agent gets a queue of 5 bug reports and must classify severity AND assign each to the right team.

- **Action required:** `{"action": "triage", "severity": "...", "team": "..."}`
- **Max steps:** 15
- **Scoring:** Severity (50%) + Team (50%) per bug, averaged across all 5
- **Expected score for a decent LLM:** ~0.55

**Example bug:** DB connection pool exhausted in production → `high` severity, `backend` team

---

### 🔴 Hard — Full Code Review Pipeline
The agent handles 8 bug reports + 3 pull requests, then submits a prioritized fix order. One of the PRs contains a **SQL injection vulnerability** that the agent must catch.

- **Actions required:**
  - For bugs: `{"action": "triage", "severity": "...", "team": "...", "comment": "...", "security_issue": false}`
  - For PRs: `{"action": "review_pr", "verdict": "approve|request_changes|comment", "comment": "...", "security_issue": false}`
  - Final step: `{"action": "prioritize", "order": ["BUG-005", "BUG-001", ...]}`
- **Max steps:** 25
- **Scoring:** Severity + Team + Comment quality + Security detection + Prioritization accuracy
- **Expected score for a decent LLM:** ~0.30 (genuinely hard)

**The trap:** PR-103 uses `f"SELECT * FROM users WHERE name LIKE '%{query}%'"` — a classic SQL injection. Missing this costs the agent 0.40 of the PR reward.

---

## Observation Space

What the agent sees at each step:

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Which task is running (easy/medium/hard) |
| `item_id` | string | ID of current item e.g. BUG-003, PR-101 |
| `item_type` | string | `bug_report` or `pull_request` |
| `title` | string | Bug title or PR title |
| `body` | string | Full bug description or PR diff/summary |
| `labels` | list | Existing labels on the item |
| `available_actions` | list | What actions are valid right now |
| `step_number` | int | Current step (1-based) |
| `max_steps` | int | Maximum steps before episode ends |
| `current_score` | float | Running score so far (0.0–1.0) |
| `feedback` | string | Result of the previous action |
| `echoed_message` | string | Same as feedback (for inference compatibility) |

---

## Action Space

The agent always sends a JSON string in the `message` field:

```json
// Easy task
{"action": "classify", "severity": "critical"}

// Medium task
{"action": "triage", "severity": "high", "team": "backend"}

// Hard task — bug
{"action": "triage", "severity": "critical", "team": "security",
 "comment": "Auth bypass vulnerability", "security_issue": true}

// Hard task — PR
{"action": "review_pr", "verdict": "request_changes",
 "comment": "SQL injection via f-string interpolation", "security_issue": true}

// Hard task — final step
{"action": "prioritize", "order": ["BUG-005", "BUG-001", "BUG-003"]}

// Any task — skip current item (penalty: -0.2)
{"action": "skip"}
```

---

## Reward Function

The reward is dense — the agent gets signal at **every single step**, not just at the end.

### Per-step reward breakdown

| Component | Weight | Description |
|---|---|---|
| Severity score | 0.30–0.50 | 1.0 exact, 0.5 adjacent, 0.0 wrong |
| Team score | 0.25–0.50 | 1.0 correct team, 0.0 wrong |
| Comment quality | 0.25 | Keyword overlap with expected review terms |
| Security detection | 0.20–0.40 | Missing a real security issue = 0.0 |
| Verdict (PR) | 0.40 | Approve vs request_changes vs comment |

### Penalties

| Penalty | Amount |
|---|---|
| Skip action | −0.20 |
| Invalid/malformed action | −0.30 |
| Repeating same action twice | −0.10 |
| Missing a security issue | 0.0 (no credit) |

### Episode score

```
score = sum(step_rewards) / total_items
score = clamp(score, 0.0, 1.0)
```

---

## API Endpoints

Once deployed, the environment exposes these endpoints:

| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset` | Start a new episode. Send `{"task_id": "easy"}` |
| POST | `/step` | Send an action. Body: `{"message": "<json>"}` |
| GET | `/state` | See full internal state (for debugging) |
| GET | `/tasks` | List all tasks with schemas and scoring info |
| GET | `/health` | Returns `{"status": "ok"}` |
| GET | `/docs` | Interactive API docs (Swagger UI) |

---

## Baseline Scores

Scores achieved by `gpt-4o-mini` running `inference.py`:

| Task | Score | Success | Steps Used |
|---|---|---|---|
| Easy | ~0.75 | ✅ | 1–3 |
| Medium | ~0.55 | ✅ | 10–15 |
| Hard | ~0.30 | ❌ | 25 |

> Hard task is intentionally difficult — even frontier models struggle with
> the security detection and correct prioritization components.

---

## Setup & Running Locally

### Requirements
- Python 3.11+
- Docker Desktop

### Option 1 — Run with Docker (recommended)

```bash
git clone <your-repo-url>
cd openenv-bug-triage

docker build -t bug-triage-env .
docker run -p 7860:7860 bug-triage-env
```

Server is now running at `http://localhost:7860`

### Option 2 — Run without Docker

```bash
pip install -r requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 7860
```

### Test it's working

```bash
# Health check
curl http://localhost:7860/health

# Start an episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# Send an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"message": "{\"action\": \"classify\", \"severity\": \"low\"}"}'
```

---

## Running the Baseline Inference Script

### 1. Set environment variables

Create a `.env` file in the project root:
```
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
HF_TOKEN=your_api_key_here
ENV_URL=http://localhost:7860
```

### 2. Start the environment (in one terminal)
```bash
docker run -p 7860:7860 bug-triage-env
```

### 3. Run inference (in another terminal)
```bash
python inference.py
```

You'll see structured logs like:
```
[START] task=easy env=bug-triage-env model=gpt-4o-mini
[STEP] step=1 action='{"action": "classify", "severity": "low"}' reward=1.0000 done=True error=None
[END] success=True steps=1 score=1.0000 rewards=[1.0]
```

---

## Project Structure

```
openenv-bug-triage/
├── Dockerfile                  # Container definition
├── openenv.yaml                # OpenEnv spec metadata
├── requirements.txt            # Python dependencies
├── inference.py                # Baseline agent script
├── README.md                   # This file
├── data/
│   └── bug_reports.json        # Fixed dataset (8 bugs + 3 PRs)
└── server/
    ├── main.py                 # FastAPI app + endpoints
    ├── env.py                  # Core environment logic
    ├── models.py               # Pydantic typed models
    ├── tasks/
    │   ├── task_easy.py        # Easy task definition
    │   ├── task_medium.py      # Medium task definition
    │   └── task_hard.py        # Hard task definition
    └── graders/
        ├── grader_easy.py      # Easy task grader
        ├── grader_medium.py    # Medium task grader
        └── grader_hard.py      # Hard task grader
```

---

## Environment Details

- **Runtime:** Python 3.11, FastAPI, Uvicorn
- **Port:** 7860 (HuggingFace Spaces default)
- **State:** Single stateful instance — one episode at a time
- **Determinism:** Fixed dataset + `temperature=0.0` = fully reproducible scores
- **Hardware:** Runs comfortably on 2 vCPU / 8GB RAM
- **Inference time:** All 3 tasks complete in under 20 minutes