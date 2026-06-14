# NeuroMCP Demo Script

Walk-through for recording the focus-gated code review demo.
Recommended subject: **86**, run 4. Expected duration: 90–120 s.

---

## Pre-Run Checklist

```
[ ] checkpoints/eegnet.pt exists
[ ] .env contains ANTHROPIC_API_KEY
[ ] pip install -e ".[dev]"  completed
[ ] terminal window ~120 cols wide, font >= 14pt (legible on recording)
[ ] screen recorder running (e.g. QuickTime > New Screen Recording)
```

---

## Step 1 — Start the Demo

```bash
python -m src.demo.agent \
  --checkpoint checkpoints/eegnet.pt \
  --subject 86 \
  --run 4
```

**You should see:**

```
Starting EEG pipeline...
Waiting 3s for buffer to fill...

============================================================
NeuroMCP Demo — Focus-Gated Code Review
Model : EEGNet + MC Dropout  (N=50 passes)
Subject: 86  |  Run: 4  |  Classes: REST / LEFT / RIGHT
------------------------------------------------------------
Brain states
  REST           -> full finding detail
  LEFT/RIGHT     -> one-sentence summary (operator is focused)
  LOW_CONFIDENCE -> agent rechecks before surfacing
============================================================
```

**Talking point:**
> "Before each code review finding the agent reads the operator's brain state
> from an EEG stream. It adapts how much detail it surfaces based on whether
> the operator is at rest or in an active imagery state."

---

## Step 2 — Finding 1: JWT Expiration Bug

The agent calls `get_signal_quality()` then `get_brain_state()` before surfacing
the first finding.

**What to watch for:**

| Brain state returned | Agent behaviour |
|---|---|
| `REST` | Full multi-sentence explanation of the JWT auth bug |
| `LEFT_IMAGERY` / `RIGHT_IMAGERY` | "You seem focused on something — one thing: expired JWTs are accepted indefinitely." |
| `LOW_CONFIDENCE` | Agent says it will recheck, calls `get_brain_state()` again |

**Typical terminal output:**

```
  [tool] get_signal_quality({})
  [result] {"snr": 7.2, "artifact_ratio": 0.12, "epoch_count": 14}

  [tool] get_brain_state({"confidence_threshold": 0.65})
  [result] {"state": "REST", "confidence": 0.71, "timestamp": "2026-06-14T...Z"}

Agent: The authentication middleware doesn't validate JWT expiration. An attacker
who captures a valid token retains access indefinitely — even after a password
change or logout. Recommendation: check `exp` claim on every request, and
enforce a short max-lifetime (e.g. 15 min) with refresh-token rotation.
```

**Talking point:**
> "SNR 7.2, artifact ratio 12% — clean signal. State is REST at 71% confidence,
> above our 65% threshold, so the agent surfaces the full finding."

---

## Step 3 — Finding 2: SQL Injection

Same gate pattern. If the model returns `LEFT_IMAGERY` here it's a good demo
moment — the agent withholds detail deliberately.

**Ideal narration:**
> "The operator's brain state shifted — motor imagery detected. The agent gives
> just the headline so we don't break focus, and will expand when the state
> returns to rest."

---

## Step 4 — Finding 3: Cache TTL

Third finding. By this point the buffer has the most epochs (highest confidence).

**Look for `epoch_count` going up** across the three rounds — it shows the pipeline
accumulating real-time data.

---

## Step 5 — End of Session

The agent prints its final turn and exits cleanly (no Ctrl-C needed).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `LOW_CONFIDENCE` every turn | Normal if subject 86's run 4 is ambiguous; the agent rechecks once then surfaces anyway |
| `snr < 3` warning | Rare with recorded data; if it happens the agent says "Signal noisy — take a breath" and skips that finding round |
| `ModuleNotFoundError: dotenv` | `pip install python-dotenv` |
| `AuthenticationError` | Check `ANTHROPIC_API_KEY` in `.env` is valid |
| Hangs at "Waiting 3s" | EEG data download in progress (first run); wait ~30 s |

---

## Recording Tips

- Run once dry before recording so MNE data is cached.
- Zoom terminal to show both `[tool]` lines and `Agent:` text in frame.
- No need to narrate every tool call line — pause there, let it render, then narrate the `Agent:` output.
- Trim to the first complete finding + one state-adaptation moment for a punchy 60 s cut.
