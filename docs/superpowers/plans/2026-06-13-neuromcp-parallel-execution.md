# NeuroMCP — Parallel Execution Workflow

This document describes how the implementation will be coordinated across parallel subagents.
Read this alongside `2026-06-13-neuromcp-implementation.md` (the task-level plan).

---

## Roles

| Role | Responsibility |
|---|---|
| **Coordinator (me)** | Dispatches agents, reviews diffs, merges branches, runs tests between rounds |
| **Subagents** | Execute one task each in an isolated git worktree — write tests, implement, commit, report back |
| **You** | Approve progression between rounds; handle anything requiring human action (cloud training, demo recording) |

---

## Dependency Graph

```
Round 1:  [Task 1: Setup] ─────────────────────────── (sequential, foundation for everything)
               │
     ┌─────────┼─────────┐
     ▼         ▼         ▼
Round 2:  [Task 2]    [Task 3]    [Task 6]              (parallel)
          EpochBuf   pipeline.py  EEGNet
     └─────────┬─────────┘         │
               ▼                   ▼
Round 3:  [Task 4]              [Task 7]                (parallel)
          Streaming             MC Dropout
     └─────────┬───────────────────┘
               ▼
Round 4:  [Task 5]    [Task 8]    [Task 11]             (parallel)
          NB-01       train.py    schemas.py
               │
               ▼
Round 5:  [Task 9: Cloud Training] ─────────────── (YOU run this — ~1 hour wall time)
               │
     ┌─────────┘
     ▼         ▼
Round 6:  [Task 10]   [Task 12]                         (parallel)
          NB-02+03    server.py
               │
     ┌─────────┘
     ▼         ▼
Round 7:  [Task 13]   [Task 14]                         (parallel)
          MCP Insp.   screening
               │
               ▼
Round 8:  [Task 15: Demo Agent] ─────────────────── (sequential)
               │
               ▼
Round 9:  [Task 16: Demo Recording] ──────────────── (YOU do this)
               │
     ┌─────────┘
     ▼         ▼
Round 10: [Task 17]   [Task 18]                         (parallel)
          README      NB cleanup
```

**Wall-clock bottleneck:** Round 5 (cloud training). Everything before it takes ~2–3 hours of agent work; Round 5 adds ~1 hour of human-supervised waiting regardless of parallelism.

---

## Isolation Strategy

Each subagent gets its own **git worktree** on a dedicated branch. This prevents file conflicts when agents write to the same `src/` or `tests/` directories simultaneously.

**Per-round workflow:**
1. I create one worktree per agent in the round
2. Agents work in isolation, commit to their branch
3. After all agents in a round finish, I merge their branches into `main` in dependency order
4. I run the full test suite on `main` before starting the next round
5. You approve before I dispatch the next round (or I proceed automatically if you grant it)

**Branch naming:** `task-<N>-<short-name>` (e.g. `task-2-epoch-buffer`, `task-6-eegnet`)

---

## Round-by-Round Breakdown

### Round 1 — Setup (sequential, ~10 min)
**I execute this myself** — no subagent needed, it's just directory creation and a toml file.

Tasks: `Task 1`
Output: repo scaffold, `pyproject.toml`, all `__init__.py` files, deps installed

**You approve → Round 2 starts**

---

### Round 2 — Independent implementations (parallel, ~20–30 min)

Three agents dispatched simultaneously in three worktrees:

| Agent | Task | Worktree branch | Files produced |
|---|---|---|---|
| A | Task 2 — EpochBuffer | `task-2-epoch-buffer` | `src/preprocessing/buffer.py`, `tests/test_buffer.py` |
| B | Task 3 — Pipeline batch | `task-3-pipeline` | `src/preprocessing/pipeline.py`, `tests/test_preprocessing.py` (batch tests only) |
| C | Task 6 — EEGNet | `task-6-eegnet` | `src/models/eegnet.py`, `tests/test_decoder.py` (architecture tests only) |

No file conflicts — each agent touches a disjoint set of files.

**After all three finish:**
- I merge A → main, B → main, C → main (no conflicts expected)
- Run `pytest tests/test_buffer.py tests/test_preprocessing.py tests/test_decoder.py`
- You approve → Round 3

---

### Round 3 — Dependent implementations (parallel, ~20–30 min)

Two agents — each depends on Round 2 output now merged into `main`:

| Agent | Task | Worktree branch | Depends on |
|---|---|---|---|
| A | Task 4 — Streaming | `task-4-streaming` | `buffer.py` (Task 2) |
| B | Task 7 — MC Dropout | `task-7-mc-dropout` | `eegnet.py` (Task 6) |

Files produced:
- Agent A: `src/preprocessing/streaming.py`, additional tests in `tests/test_preprocessing.py`
- Agent B: `src/models/mc_dropout.py`, additional tests in `tests/test_decoder.py`

Potential conflict: both agents append to existing test files. I instruct each agent to only add new test functions — no modifications to existing ones.

**After both finish:**
- Merge A → main, B → main
- Run full test suite
- You approve → Round 4

---

### Round 4 — Training prep + schemas (parallel, ~20 min)

Three agents:

| Agent | Task | Worktree branch | Notes |
|---|---|---|---|
| A | Task 5 — Notebook 01 | `task-5-nb01` | Validation notebook only, no code changes |
| B | Task 8 — train.py | `task-8-train` | `src/models/train.py` — new file, no conflicts |
| C | Task 11 — Schemas | `task-11-schemas` | `src/server/schemas.py`, schema tests in `tests/test_server.py` |

**After all three finish:**
- Merge all → main
- Smoke-test train.py locally: `python -m src.models.train --subjects 1 2 --test-subjects 3 --epochs 3 --out checkpoints/`
- You approve → Round 5

---

### Round 5 — Cloud Training (YOU)

**No agents dispatched.** This is a human-executed step.

```bash
# You run this on Lambda Labs / Vast.ai
python -m src.models.train \
  --subjects $(seq -s' ' 1 80) \
  --test-subjects $(seq -s' ' 81 109) \
  --epochs 200 --out checkpoints/

# Then download: rsync ubuntu@<IP>:~/neuromcp/checkpoints/ ./checkpoints/
```

Expected: ~1 hour. Downloads: `eegnet.pt`, `test_probs_baseline.npy`, `test_labels.npy`, `X_test.npy`

**You confirm checkpoint is local → Round 6 starts**

---

### Round 6 — Calibration + Server (parallel, ~30 min)

Two agents:

| Agent | Task | Worktree branch | Notes |
|---|---|---|---|
| A | Task 10 — Calibration notebooks | `task-10-calibration` | Notebooks 02 + 03, reads from `checkpoints/` |
| B | Task 12 — FastMCP server | `task-12-server` | `src/server/server.py`, server tests in `tests/test_server.py` |

No file conflicts.

**After both finish:**
- Merge → main
- Run `pytest tests/test_server.py`
- You approve → Round 7

---

### Round 7 — Validation + Screening (parallel, ~20 min)

Two agents:

| Agent | Task | Worktree branch | Notes |
|---|---|---|---|
| A | Task 13 — MCP Inspector | `task-13-mcp-inspector` | **Manual** — I provide the server start command, you run it and report results back |
| B | Task 14 — Screening script | `task-14-screening` | `scripts/screen_subjects.py` |

Task 13 requires you to run the server and MCP Inspector. I'll give you the exact commands.

**After Task 14 finishes and you report MCP Inspector results:**
- Merge Task 14 → main
- Note best demo subject from screening output
- You approve → Round 8

---

### Round 8 — Demo Agent (sequential, ~20 min)

Single agent — depends on Task 12 (server) and Task 14 (best subject known):

Task 15: `src/demo/agent.py`

**After agent finishes:**
- Run demo end-to-end: `python -m src.demo.agent --checkpoint checkpoints/eegnet.pt --subject <N> --run 4`
- You approve → Round 9

---

### Round 9 — Demo Recording (YOU)

**No agents.** You record the 90–120 second demo video.

Setup: two terminal panels — one for agent output, one showing tool calls.
Record 2–3 takes, keep the best one showing all three behavioral modes.

**You confirm video saved → Round 10**

---

### Round 10 — Polish (parallel, ~15 min)

Two agents:

| Agent | Task | Worktree branch | Notes |
|---|---|---|---|
| A | Task 17 — README | `task-17-readme` | Fill in ECE numbers from notebook 03, add demo video link |
| B | Task 18 — NB cleanup | `task-18-nb-cleanup` | Re-run all notebooks, clean outputs |

**After both finish:**
- Final merge → main
- Final `pytest` — all tests green
- Done

---

## What I Do Between Rounds

After each parallel round completes:
1. Read each agent's summary report
2. Verify no file conflicts (check `git diff` between branches before merge)
3. Merge branches into `main` in sequence
4. Run `pytest` — if failures, I fix inline before declaring the round done
5. Report to you: what was built, test results, any issues found
6. Ask for your go-ahead before starting the next round

---

## Your Checkpoints (where I need you)

| After Round | What you do |
|---|---|
| 1 | Approve scaffold, confirm deps installed correctly |
| 2 | Glance at test output, approve |
| 3 | Glance at test output, approve |
| 4 | Run local smoke test of train.py, approve |
| **5** | **Run cloud training (~1 hr), download checkpoint, confirm** |
| 6 | Approve server tests |
| 7 | Run MCP Inspector manually, report results back; note best demo subject |
| 8 | Run demo agent, confirm it works end-to-end |
| **9** | **Record demo video** |
| 10 | Final review, confirm done |

---

## If an Agent Fails

If a subagent returns with failing tests or an error:
1. I read its report and diagnose the issue
2. I fix it inline in this session (simple issues) or dispatch a second agent to the same worktree
3. The other parallel agents in that round are not affected
4. The round isn't closed until all tasks in it pass

---

## Estimated Timeline

| Phase | Wall time |
|---|---|
| Rounds 1–4 (coding) | ~1.5–2 hours |
| Round 5 (cloud training) | ~1 hour (you're waiting) |
| Rounds 6–8 (server + demo) | ~1 hour |
| Round 9 (recording) | ~30 min |
| Round 10 (polish) | ~15 min |
| **Total** | **~4–5 hours** |

Without parallelism, Rounds 2–4 and 6–7 would each be sequential, adding ~1–1.5 hours.
