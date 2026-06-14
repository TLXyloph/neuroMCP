# NeuroMCP Demo Script

Word-for-word narration and actions for recording the focus-gated code review demo.
Everything in **SAY:** is spoken out loud. Everything in **[ACTION]** is a physical step.
Expected duration: 90-120 seconds.

---

## Before You Hit Record

**[ACTION]** Open two terminal windows side by side. Left is the main demo. Right is the thinking log.

**[ACTION]** In the RIGHT terminal, run:
```bash
tail -f /tmp/neuromcp_thinking.log
```
It will say "no such file or directory" — that is fine. It starts showing output once the demo runs.

**[ACTION]** In the LEFT terminal, navigate to the project root.

**[ACTION]** Run once without recording to warm the cache:
```bash
python -m src.demo.agent --checkpoint checkpoints/eegnet.pt --subject 86 --run 4
```
Type anything at the `You:` prompt, wait for it to finish. This pre-downloads EEG data so the recording has no 30-second loading pause.

**[ACTION]** Start your screen recorder. Make sure both terminal windows are in frame.

---

## The Recording

### Opening (while the command is still being typed)

**[ACTION]** Type the command slowly, visibly:
```bash
python -m src.demo.agent --checkpoint checkpoints/eegnet.pt --subject 86 --run 4
```

**SAY:**
> "This is NeuroMCP — an MCP server that gives AI agents a real-time read on your cognitive state. The idea is simple: before the agent says anything, it checks whether you're focused or at rest, and adapts what it surfaces accordingly."

**[ACTION]** Press Enter.

---

### Pipeline Startup

The terminal prints:
```
Starting EEG pipeline...
Waiting 3s for buffer to fill...
```

**SAY:**
> "Under the hood, a streaming EEG pipeline is running — filtering the signal to 8 to 30 Hz, windowing it into one-second epochs, and feeding those into a calibrated EEGNet decoder. It takes a few seconds to fill the buffer."

**[ACTION]** Let the banner print. Do not rush.

```
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

**SAY:**
> "Three possible brain states: rest, left imagery, right imagery. When you're at rest, the agent gives the full finding. When you're in an active motor imagery state — meaning you're mentally focused on something — it backs off and gives you a one-liner instead."

---

### The Findings and the You: Prompt

The terminal prints the three findings, then:
```
You: _
```

**[ACTION]** Pause for one beat, then type:
```
let's work through these
```
**[ACTION]** Press Enter.

**SAY:** *(say this as you type, naturally)*
> "Now I'm sending the opening message. The findings are passed along with it — the agent knows what it needs to review."

---

### Claude Starts Streaming

The terminal prints:
```

Claude: 
```
and text begins appearing token by token.

**SAY:**
> "Claude is streaming its response now. Watch the tool calls that come next."

**[ACTION]** Point at (or gesture toward) the tool blocks as they appear.

---

### First Tool Call Block

```
── tool: get_signal_quality ──────────────────────────────
  result: {"snr": 7.2, "artifact_ratio": 0.12, "epoch_count": 14}

── tool: get_brain_state ─────────────────────────────────
  input:  {"confidence_threshold": 0.65}
  result: {"state": "REST", "confidence": 0.71, "timestamp": "...Z"}
```

**SAY:**
> "Signal-to-noise ratio 7.2, artifact ratio 12 percent — clean signal. Brain state decoded as REST at 71 percent confidence, which is above our 65 percent threshold. So the agent will surface the full finding."

**[ACTION]** Glance briefly at the RIGHT terminal (thinking log) to show it's active.

**SAY:**
> "Over on the right you can see Claude's extended thinking — the raw reasoning before it decides what to say."

---

### Claude Surfaces Finding 1

```
────────────────────────────────────────────────────────────

Claude: 
```
Text streams in.

**SAY:** *(while the text streams, speak naturally, do not read it aloud)*
> "Full detail. The agent is treating this as a teachable moment because the signal says you're available to absorb it."

**[ACTION]** Wait for Claude to finish. Let the text render completely.

---

### Finding 2 (the adaptation moment — adapt your narration to what the tool returns)

**If `get_brain_state` returns `REST`:**

**SAY:**
> "Still at rest. Full finding again."

**If `get_brain_state` returns `LEFT_IMAGERY` or `RIGHT_IMAGERY`:**

**SAY:**
> "State shifted — motor imagery. The agent detected that and pulled back. One sentence instead of a paragraph. That is the adaptation in action."

**[ACTION]** Pause after Claude's response so the viewer can read it.

**If `get_brain_state` returns `LOW_CONFIDENCE`:**

**SAY:**
> "Confidence was below threshold — the agent is rechecking before it decides. That is the uncertainty gate working."

---

### Finding 3

**SAY:**
> "Third finding. Notice the `epoch_count` in the signal quality result — it has been climbing since we started. The pipeline has been running the whole time, continuously updating the buffer."

**[ACTION]** Point at the epoch_count value in the tool result.

---

### Close

The agent finishes and the shell prompt returns.

**SAY:**
> "And that's it. The whole thing runs locally — EEG pipeline, decoder, and agent loop. The two tool calls, `get_signal_quality` and `get_brain_state`, are standard MCP tool calls. Any compatible agent can use them. The code reviewer is just the simplest demo."

**[ACTION]** Stop recording.

---

## If Something Goes Wrong Mid-Recording

| What you see | What to say | What to do |
|---|---|---|
| `LOW_CONFIDENCE` on every finding | "The decoder is uncertain — confidence is under threshold each time. It rechecks and surfaces anyway." | Nothing. Let it run. |
| Thinking log stays empty | Nothing — don't mention it. | The model fell back to standard mode. The demo still works. |
| `Signal noisy — take a breath, I'll wait.` | "SNR dropped below the noise floor — the agent is waiting rather than surfacing a finding on a bad signal." | Wait; it will re-check automatically next turn. |
| Long pause between turns | Pause your narration naturally. | API latency. Do not comment on it unless it exceeds 15 seconds. |
| Python error on startup | Stop recording. | Check `.env` has `ANTHROPIC_API_KEY` and `checkpoints/eegnet.pt` exists. |

---

## Timing Guide

| Segment | Target time |
|---|---|
| Opening + pipeline startup | 15-20 s |
| Finding 1 (tool calls + response) | 25-35 s |
| Finding 2 (the adaptation moment) | 20-25 s |
| Finding 3 + close | 20-25 s |
| **Total** | **80-105 s** |

If you want a shorter cut, trim after Finding 2. The adaptation moment (Finding 2 with imagery state) is the most important beat — that is the one to keep.
