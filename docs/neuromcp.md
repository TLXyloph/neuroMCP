# NeuroMCP — Project Description

## What This Is

NeuroMCP is an MCP (Model Context Protocol) server that exposes decoded EEG brain signals as structured tool calls for AI agents. It bridges neural signal processing and the AI agent tooling ecosystem — enabling any MCP-compatible agent (Claude, custom agents) to query a decoded human mental state with calibrated confidence and act on it.

The project has three layers that build on each other:
1. **Signal processing** — preprocess raw EEG into clean, classifiable epochs
2. **Neural decoding** — classify motor imagery with calibrated uncertainty (not just a label, a confidence you can trust)
3. **MCP server** — expose the decoder as two clean tool calls an agent can compose

The distinguishing technical contribution over "EEGNet tutorial + FastMCP hello world": Monte Carlo dropout for calibrated confidence, streaming windowing with artifact rejection, and an agent demo where the agent's behavior *meaningfully changes* based on signal quality and decoded state.

---

## Target Audience (for resume/interview framing)

**Primary:** Roles at the intersection of AI agent tooling and novel interfaces — AI platform engineers, ML infrastructure, agent SDK teams, companies building with MCP.

**Secondary:** ML engineering roles that will ask "what's your technical contribution?" — answered by the calibration artifact and the signal pipeline.

**Framing:** "I designed the MCP interface for biosignal decoding — the layer that lets any agent consume neural state without knowing anything about neuroscience." Lead with the interface design story, back it up with the ML rigor.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        MCP Server                           │
│   get_brain_state() → {state, confidence, timestamp}        │
│   get_signal_quality() → {snr, artifact_ratio, epoch_count} │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                    Neural Decoder                           │
│   EEGNet + Monte Carlo Dropout → calibrated probabilities   │
│   States: LEFT_IMAGERY | RIGHT_IMAGERY | REST               │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  Signal Pipeline                            │
│   Streaming windowing → bandpass filter (8–30 Hz)          │
│   → artifact rejection → epoch buffer                       │
│   (runs as background thread; agent queries read from it)   │
└─────────────────────────────────────────────────────────────┘
```

---

## Technical Contribution (the defensible result)

**Monte Carlo Dropout for calibrated confidence.** A softmax score is only useful if it's a *trustworthy* probability — a 90% confidence from a miscalibrated classifier is meaningless. MC dropout runs N stochastic forward passes at inference time and averages the softmax distributions to produce a per-prediction uncertainty estimate that gates the `LOW_CONFIDENCE` response.

What you produce as an artifact (measured result, cross-subject 80→29 split, 1695 test epochs):
- Reliability diagram (calibration curve) comparing baseline EEGNet vs. MC-EEGNet on a held-out test set
- ECE (Expected Calibration Error): **baseline 2.84%, MC-dropout 3.02%** — the EEGNet trained with dropout=0.5 is *already well-calibrated* (mean confidence 0.591 vs accuracy 0.576), so MC dropout does not further reduce ECE here. The defensible result is a verified-calibrated decoder, not a calibration *fix*; MC dropout supplies the per-prediction uncertainty that powers confidence gating.
- A `confidence_threshold` parameter in `get_brain_state()` that the agent can use to gate on quality

Why this matters for the agent story: the agent can call `get_signal_quality()` first, check the SNR, then call `get_brain_state()` and inspect the confidence before acting. This is the uncertainty-aware interface the council said nobody has built. That's the design decision you defend in interviews.

---

## API Design Rationale (your interview armor)

**Why two tools instead of one?**

`get_signal_quality()` is cheap — it reads from the running epoch buffer and returns noise metrics without running inference. `get_brain_state()` is expensive — it runs N forward passes through EEGNet.

Separating them lets an agent implement a quality gate:
```
quality = get_signal_quality()
if quality.snr > threshold:
    state = get_brain_state()
    act_on(state)
else:
    ask_user_to_focus()
```

If they were one tool, every call would pay the inference cost even on a noisy signal. Separation also makes each tool independently testable and composable with other MCP tools.

This is the design question every technical interviewer will ask. Have this answer cold.

---

## Stack

| Layer | Library | Why |
|---|---|---|
| EEG data | MNE-Python | Standard in neuroscience, handles PhysioNet download natively |
| Signal processing | MNE-Python + NumPy | Bandpass filter, epoching, artifact rejection |
| Neural decoder | PyTorch | EEGNet implementation + MC dropout |
| MCP server | FastMCP | Clean Python MCP server, minimal boilerplate |
| Demo agent | Anthropic Python SDK | Claude with tool use |
| Dataset | PhysioNet EEGBCI (via MNE) | 109 subjects, 14 runs each, free download |
| Notebooks | Jupyter | Calibration curves, training results |

---

## Directory Structure

```
neuromcp/
├── src/
│   ├── preprocessing/
│   │   ├── pipeline.py        # bandpass filter, epoching, artifact rejection
│   │   └── streaming.py       # background thread, epoch buffer, SNR computation
│   ├── models/
│   │   ├── eegnet.py          # EEGNet architecture in PyTorch
│   │   ├── mc_dropout.py      # MC dropout wrapper, N-pass inference
│   │   └── train.py           # training script, saves checkpoint + calibration data
│   ├── server/
│   │   ├── server.py          # FastMCP server, tool definitions
│   │   └── schemas.py         # typed return schemas for both tools
│   └── demo/
│       └── agent.py           # Claude agent demo with tool use
├── notebooks/
│   ├── 01_preprocessing.ipynb # explore PhysioNet data, validate pipeline
│   ├── 02_training.ipynb      # train EEGNet, generate calibration curves
│   └── 03_calibration.ipynb   # reliability diagram, ECE computation
├── tests/
│   ├── test_preprocessing.py
│   ├── test_decoder.py
│   └── test_server.py
├── docs/
│   └── design.md              # API design rationale (expanded version of above)
├── data/                      # gitignored — PhysioNet downloads here
├── checkpoints/               # gitignored — trained model weights
├── pyproject.toml
└── README.md
```

---

## Build Order

Build in this sequence. Each step is independently testable before moving to the next.

### Phase 1 — Signal Pipeline (2–3 days)
1. Set up repo, `pyproject.toml`, dev environment
2. Download PhysioNet EEGBCI via MNE (`mne.datasets.eegbci.load_data`)
3. Implement `pipeline.py`: bandpass filter 8–30 Hz (mu + beta bands), epoch around task events, basic artifact rejection (peak-to-peak amplitude threshold)
4. Validate in `01_preprocessing.ipynb`: plot power spectral density, confirm motor imagery epochs look as expected
5. Implement `streaming.py`: circular epoch buffer, background thread that continuously processes new windows, SNR computation

**Gate:** you can run the pipeline on a PhysioNet subject and get clean epochs with an SNR metric.

### Phase 2 — Neural Decoder (3–4 days)
1. Implement `eegnet.py` — the EEGNet architecture (depthwise + separable convolutions, see Lawhern et al. 2018 for architecture details)
2. Implement `mc_dropout.py`: wrap EEGNet to keep dropout active at inference, run N=50 forward passes, return mean + variance
3. Write `train.py`: train on subjects 1–80, test on subjects 81–109 (held-out), save checkpoint
4. Generate calibration data in `02_training.ipynb`
5. Plot reliability diagram and compute ECE in `03_calibration.ipynb` — this is your artifact

**Gate:** you have a trained model, a calibration curve image, and an ECE number to cite.

### Phase 3 — MCP Server (1–2 days)
1. Implement `schemas.py`: typed Pydantic schemas for both tool return values
2. Implement `server.py`:
   - `get_signal_quality()`: reads from the epoch buffer, returns SNR + artifact ratio + epoch count
   - `get_brain_state()`: runs MC dropout inference on the latest clean epoch, returns state + confidence + timestamp
3. Wire the streaming pipeline to start on server startup
4. Test both tools with MCP Inspector

**Gate:** you can run the server and call both tools from MCP Inspector and get structured responses.

### Phase 4 — Demo Agent (1–2 days)
1. Design a demo scenario where agent behavior meaningfully changes based on brain state (see Demo Script below)
2. Implement `agent.py` using the Anthropic Python SDK with tool use
3. Record the demo video

**Gate:** the demo video shows the agent checking signal quality, querying brain state, and doing something visibly different based on the decoded state.

### Phase 5 — Polish (1–2 days)
1. Write `README.md` (see structure below)
2. Write `docs/design.md` (full API design rationale)
3. Add architecture diagram to README (can use Mermaid or a simple ASCII version)
4. Clean up notebooks so they render nicely on GitHub

---

## Demo Script

The weak demo: Claude prints "LEFT_IMAGERY detected" to terminal. Don't do this.

A compelling demo has a real interaction loop where the agent's behavior visibly changes. Suggested scenario:

**"Focus-gated code review"**

The agent is performing a code review. Before surfacing each finding, it checks signal quality. If quality is high AND brain state is REST, it surfaces the finding verbosely. If quality is high AND brain state is LEFT or RIGHT imagery, it surfaces a short version ("you seem focused on something else — one thing: X"). If signal quality is low, it pauses and says "signal noisy — take a breath, I'll wait."

This demonstrates:
- The quality gate (two-tool composition)
- Confidence-gated behavior (agent acts differently at high vs. low confidence)
- A real use case (not just classification output)

The demo video should be 90–120 seconds. Show the terminal, the EEG epoch visualization, and the agent response side by side.

---

## Resume Line

**Before (weak):** "Built NeuroMCP — a neural decoding pipeline as an MCP server"

**After (what to use):** "Designed and implemented NeuroMCP: an MCP server exposing calibrated EEG motor imagery decoding as agent tool calls. EEGNet with Monte Carlo dropout, verified well-calibrated (2.8% ECE on a cross-subject held-out test set) via reliability-diagram analysis, supplies per-prediction uncertainty that gates agent actions; streaming signal pipeline with artifact rejection produces decoded state on demand. Enables any MCP-compatible agent to query decoded human mental state with calibrated uncertainty."

Note on the calibration result: the measured outcome is a *verified-calibrated* decoder (baseline ECE 2.84%, MC-dropout 3.02% — statistically equivalent), not an ECE reduction. The dropout=0.5 EEGNet is already well-calibrated, so the honest, defensible claim is trustworthy confidence + uncertainty-gated tool calls. Do not cite an "X% ECE reduction" — it didn't happen on this model, and the verified-calibration framing is stronger because it's true.

---

## What This Is Not

- Not a claim to be reading live thoughts (the dataset is recorded signals used as a proxy for live input — state this clearly in the README)
- Not production BCI software — it's a research-grade demonstration of the interface pattern
- Not a research paper — you're not claiming novel neuroscience, you're demonstrating novel software architecture

The README should say this plainly. Honesty about scope reads as engineering maturity, not weakness.

---

## Confirmed Intent

- **Outcome:** Polished GitHub repo + 2-minute demo video showing Claude doing something meaningfully different based on decoded brain state
- **User:** Builder, targeting AI agent tooling roles with ML credibility
- **Why now:** Building from scratch, excited about agent tooling + neural interface intersection
- **Success:** Resume link + demo video that survives a technical screening
- **Constraint:** Must have one defensible technical artifact (calibration curve + ECE number)
- **Out of scope:** Live hardware, research publication, production deployment, real-time from an actual human brain

*Confirmed by user 2026-06-12*
