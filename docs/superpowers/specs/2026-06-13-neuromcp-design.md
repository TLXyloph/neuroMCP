# NeuroMCP — Design Spec

*Confirmed: 2026-06-13*

---

## What We're Building

An MCP server that exposes decoded EEG brain signals as structured tool calls for AI agents. Three layers build on each other: signal processing → neural decoding → MCP interface. The distinguishing contribution is Monte Carlo dropout for calibrated uncertainty, a streaming pipeline with real-time playback, and a demo where agent behavior meaningfully changes based on decoded state and signal quality.

**Target:** Polished GitHub repo + 2-minute demo video. Resume artifact: calibration curve image + ECE number.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        MCP Server                           │
│   get_brain_state() → {state, confidence, timestamp}        │
│   get_signal_quality() → {snr, artifact_ratio, epoch_count} │
└────────────────────┬────────────────────────────────────────┘
                     │ reads from
┌────────────────────▼────────────────────────────────────────┐
│                    EpochBuffer                              │
│   deque(maxlen=10) + threading.Lock                         │
│   .push(epoch)  .latest()  .stats()                         │
└────────────────────┬────────────────────────────────────────┘
                     │ writes to
┌────────────────────▼────────────────────────────────────────┐
│                  Signal Pipeline                            │
│   PlaybackThread → sample buffer → bandpass (8–30 Hz)      │
│   → artifact rejection → EpochBuffer                        │
└─────────────────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                    Neural Decoder                           │
│   EEGNet + Monte Carlo Dropout (N=50) → mean + variance     │
│   States: LEFT_IMAGERY | RIGHT_IMAGERY | REST               │
└─────────────────────────────────────────────────────────────┘
```

---

## Shared State Architecture

The pipeline and server run in separate threads within the same process. They share state through `EpochBuffer` — a small class in `src/preprocessing/buffer.py`:

```python
class EpochBuffer:
    def __init__(self, maxlen=10):
        self._buf = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, epoch): ...     # pipeline thread writes
    def latest(self): ...          # get_brain_state() reads newest epoch
    def stats(self): ...           # get_signal_quality() computes over last N epochs
```

`get_brain_state()` calls `latest()` — always the freshest clean epoch, no stale accumulation. `get_signal_quality()` calls `stats()` — rolling SNR/artifact ratio over the buffer window.

A queue was considered and rejected: it accumulates items between reads and would serve the oldest buffered epoch rather than the newest. A global singleton was considered and rejected: no explicit thread-safety boundary. The circular buffer is the right abstraction.

---

## Streaming Simulation

The data source is recorded PhysioNet EEGBCI signals (not live hardware). Real-time streaming is simulated by a `PlaybackThread` that reads the raw recording as a numpy array and pushes sample chunks into the pipeline at the original sample rate (160 Hz):

```python
# PlaybackThread core loop
chunk_size = 16  # samples
sleep_interval = chunk_size / sfreq  # ~0.1s
for chunk in iter_chunks(recording, chunk_size):
    sample_queue.put(chunk)
    time.sleep(sleep_interval)
```

The pipeline thread reads from the sample queue, applies windowing, filtering, and artifact rejection, and writes clean epochs to `EpochBuffer`.

**Fallback:** If playback timing drifts noticeably during demo recording, switch to pre-segmented mode: load epochs offline, push them to `EpochBuffer` at a fixed cadence (one epoch per 2s). Nothing downstream changes — `EpochBuffer`'s interface is identical in both modes.

---

## Neural Decoder

**EEGNet** (Lawhern et al. 2018): depthwise + separable convolutions. Input shape: `(64 channels, 160 samples)` per epoch. ~2,500 parameters.

**Monte Carlo Dropout:** keep dropout active at inference, run N=50 stochastic forward passes, return `mean` (predicted state) and `variance` (uncertainty) across passes. The variance is the calibrated confidence the agent uses.

Standard EEGNet softmax scores are not calibrated probabilities — a 90% score from a miscalibrated model is meaningless. MC dropout variance is an honest uncertainty estimate that degrades gracefully with signal noise.

---

## MCP API

### `get_signal_quality()`
Reads from `EpochBuffer.stats()`. No inference. Fast.

```json
{
  "snr": 12.4,
  "artifact_ratio": 0.08,
  "epoch_count": 7
}
```

### `get_brain_state(confidence_threshold: float = 0.7)`
Runs N=50 MC dropout passes on `EpochBuffer.latest()`. Returns decoded state only if confidence exceeds threshold; otherwise returns `"LOW_CONFIDENCE"`.

```json
{
  "state": "LEFT_IMAGERY",
  "confidence": 0.84,
  "timestamp": "2026-06-13T20:14:00Z"
}
```

**Why two tools:** `get_signal_quality()` is cheap — no inference cost. `get_brain_state()` is expensive — N forward passes. Separation lets the agent implement a quality gate without paying inference cost on noisy signals. Each tool is independently testable and composable.

---

## Inference Environment

Training: cloud GPU (Lambda Labs A10G or Vast.ai RTX 3090). Estimated cost: **$2–5 total** for a 30–90 minute training run + MC dropout inference over the test set.

Inference (MCP server): M3 Pro with PyTorch MPS backend. EEGNet is small enough that 50 forward passes run in **20–40ms** on MPS — well under the 200ms target. No cloud GPU needed at serve time.

Implementation note: save/load only `state_dict` (not the full model object) so the cloud-trained checkpoint loads cleanly on MPS:
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.load_state_dict(torch.load("checkpoints/eegnet.pt", map_location=device))
```

---

## Calibration Artifact

The resume-defensible result. After training:

1. Run baseline EEGNet (softmax, no MC dropout) over held-out test subjects
2. Run MC-EEGNet (N=50 passes) over the same subjects
3. Plot reliability diagrams for both — confidence buckets vs. actual accuracy
4. Compute ECE (Expected Calibration Error) for both — cite the delta on the resume

Typical MC dropout ECE improvement on EEG data: 15–35% reduction over baseline softmax. Fill in the actual number once you have results.

---

## Demo

**Scenario:** Focus-gated code review. The agent reviews findings from a codebase. Before surfacing each finding it checks signal quality, then checks brain state.

| Condition | Agent behavior |
|---|---|
| SNR low | "Signal noisy — take a breath, I'll wait." |
| SNR ok, state = REST, confidence high | Surface finding verbosely |
| SNR ok, state = LEFT or RIGHT imagery, confidence high | "You seem focused on something — one thing: X" |
| Confidence below threshold | Skip or re-check |

**Demo segment selection (explicit Phase 4 step):**
After training, evaluate the model over all held-out test subjects. Rank by per-subject accuracy. Manually screen the top 3 subjects: load their recordings, run the decoder, identify 2-minute segments with clear REST → IMAGERY transitions at high confidence. Hardcode the best subject + run as the demo playback file. This prevents the need to record many takes and ensures the demo reliably shows all three agent behaviors.

PhysioNet EEGBCI runs naturally alternate rest and task periods (~15s rest, ~4s task), so any good subject's run will contain multiple state transitions within a 2-minute window.

**Demo video:** 90–120 seconds. Show terminal (agent output), EEG epoch visualization, and decoded state side by side.

---

## Directory Structure

```
neuromcp/
├── src/
│   ├── preprocessing/
│   │   ├── buffer.py          # EpochBuffer — thread-safe circular buffer (NEW)
│   │   ├── pipeline.py        # bandpass filter, epoching, artifact rejection
│   │   └── streaming.py       # PlaybackThread, sample queue, SNR computation
│   ├── models/
│   │   ├── eegnet.py          # EEGNet architecture in PyTorch
│   │   ├── mc_dropout.py      # MC dropout wrapper, N-pass inference
│   │   └── train.py           # training script, saves checkpoint + calibration data
│   ├── server/
│   │   ├── server.py          # FastMCP server, tool definitions
│   │   └── schemas.py         # typed Pydantic schemas for both tools
│   └── demo/
│       └── agent.py           # Claude agent demo with tool use
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_training.ipynb
│   └── 03_calibration.ipynb
├── tests/
│   ├── test_buffer.py
│   ├── test_preprocessing.py
│   ├── test_decoder.py
│   └── test_server.py
├── docs/
│   ├── neuromcp.md            # original project description
│   └── superpowers/specs/     # this file
├── data/                      # gitignored
├── checkpoints/               # gitignored
├── pyproject.toml
└── README.md
```

---

## Build Order

### Phase 1 — Signal Pipeline (2–3 days)
1. Set up repo, `pyproject.toml`, dev environment (Python 3.11+, MNE, PyTorch, FastMCP)
2. Download PhysioNet EEGBCI via `mne.datasets.eegbci.load_data`
3. Implement `buffer.py`: `EpochBuffer` with `push`, `latest`, `stats`
4. Implement `pipeline.py`: bandpass filter 8–30 Hz, epoch around task events, artifact rejection (peak-to-peak amplitude threshold)
5. Implement `streaming.py`: `PlaybackThread` at 160 Hz, sample queue, SNR computation
6. Validate in `01_preprocessing.ipynb`: PSD plots, confirm motor imagery epochs look correct

**Gate:** pipeline runs on a PhysioNet subject, produces clean epochs in `EpochBuffer`, `stats()` returns sensible SNR.

### Phase 2 — Neural Decoder (3–4 days)
1. Implement `eegnet.py` (Lawhern et al. 2018 architecture)
2. Implement `mc_dropout.py`: keep dropout active at inference, N=50 passes, return mean + variance
3. Write `train.py`: train on subjects 1–80, test on subjects 81–109, save `state_dict` checkpoint
4. Generate calibration data in `02_training.ipynb`
5. Plot reliability diagram and compute ECE in `03_calibration.ipynb`

**Gate:** trained checkpoint, calibration curve image, ECE number to cite.

### Phase 3 — MCP Server (1–2 days)
1. Implement `schemas.py`: Pydantic schemas for both tool return values
2. Implement `server.py`: wire `EpochBuffer` to both tools, start `PlaybackThread` on server startup, load model with MPS device map
3. Verify 50-pass inference latency on M3 Pro — target < 200ms; reduce N if needed
4. Test both tools with MCP Inspector

**Gate:** both tools callable from MCP Inspector with structured responses under 200ms.

### Phase 4 — Demo Agent (1–2 days)
1. Evaluate model over all test subjects, rank by accuracy, screen top 3 for demo segment
2. Hardcode best subject + run as demo playback file
3. Implement `agent.py` using Anthropic Python SDK with tool use
4. Record demo video (90–120 seconds)

**Gate:** demo video shows agent checking signal quality, querying brain state, and behaving visibly differently across at least two distinct states.

### Phase 5 — Polish (1–2 days)
1. Write `README.md`
2. Add architecture diagram (Mermaid or ASCII)
3. Clean up notebooks for GitHub rendering

---

## Stack

| Layer | Library | Why |
|---|---|---|
| EEG data | MNE-Python | Standard in neuroscience, handles PhysioNet download natively |
| Signal processing | MNE-Python + NumPy | Bandpass filter, epoching, artifact rejection |
| Neural decoder | PyTorch (MPS) | EEGNet + MC dropout; MPS for M3 Pro inference |
| MCP server | FastMCP | Clean Python MCP server, minimal boilerplate |
| Demo agent | Anthropic Python SDK | Claude with tool use |
| Dataset | PhysioNet EEGBCI (via MNE) | 109 subjects, 14 runs each, free download |
| Notebooks | Jupyter | Calibration curves, training results |

---

## What This Is Not

- Not a claim to read live thoughts — recorded signals used as a proxy for live input
- Not production BCI software — research-grade demonstration of the interface pattern
- Not a research paper — novel software architecture, not novel neuroscience

State this plainly in the README. Honesty about scope reads as engineering maturity.
