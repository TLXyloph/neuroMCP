# NeuroMCP

An MCP server exposing calibrated EEG motor-imagery decoding as agent tool calls.
EEGNet with Monte Carlo dropout — verified well-calibrated (2.8% ECE on a cross-subject
held-out test set) — supplies per-prediction uncertainty that gates agent actions.
A streaming signal pipeline with artifact rejection produces decoded state on demand at
~27 ms per inference. Any MCP-compatible agent can query decoded human mental state with
calibrated uncertainty.

> **Note:** Uses recorded PhysioNet EEGBCI signals as a proxy for live input.
> This is a research-grade demonstration of the interface pattern, not production BCI software.

## Demo

_[Demo video link — record per `docs/.../neuromcp-implementation.md` Task 16, recommended subject 86.]_

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

A `PlaybackThread` feeds recorded EEG at the original sample rate into a `PipelineThread`
that filters, windows (1 s / 160 samples, 50% overlap), and rejects artifacts, writing
clean epochs into a thread-safe `EpochBuffer`. The MCP server reads from the buffer on each
tool call; `get_brain_state()` runs N=50 MC-dropout passes through EEGNet.

## API

### `get_signal_quality()`
Returns `{snr, artifact_ratio, epoch_count}` — reads from the epoch buffer, no inference cost.

### `get_brain_state(confidence_threshold=0.7)`
Returns `{state, confidence, timestamp}`. Runs N=50 MC-dropout passes through EEGNet.
State is one of: `LEFT_IMAGERY`, `RIGHT_IMAGERY`, `REST`, `LOW_CONFIDENCE`.
Returns `LOW_CONFIDENCE` when the winning-class confidence is below the threshold or no
epoch is available yet.

The two-tool design lets agents implement a quality gate — call `get_signal_quality()` first,
only call `get_brain_state()` when SNR is acceptable.

## Results

Measured on a cross-subject split (train subjects 1–80, test 81–109; 1695 test epochs):

| Metric | Value |
|---|---|
| Test accuracy (3-class, chance 33%) | 57.6% |
| ECE — baseline EEGNet | 2.84% |
| ECE — MC dropout (N=50) | 3.02% |
| `get_brain_state` latency (MPS, N=50) | ~27 ms |

**On calibration:** the EEGNet trained with dropout=0.5 is *already well-calibrated*
(mean confidence 0.591 vs accuracy 0.576), so MC dropout does not further reduce ECE here —
the two are statistically equivalent. The defensible technical contribution is a
*verified-calibrated* decoder plus the per-prediction uncertainty MC dropout provides for
confidence gating, not a calibration fix. See `notebooks/03_calibration.ipynb` for the
reliability diagrams.

## Setup

```bash
pip install -e ".[dev]"
```

PhysioNet EEGBCI data downloads automatically via MNE on first run.

## Running the Server

```bash
python -m src.server.server \
  --checkpoint checkpoints/eegnet.pt \
  --subject 86 \
  --run 4
```

## Running the Demo

Requires `ANTHROPIC_API_KEY` in the environment (the agent loop calls the Claude API).

```bash
python -m src.demo.agent \
  --checkpoint checkpoints/eegnet.pt \
  --subject 86 \
  --run 4
```

## Training

EEGNet is small (~16K params); training runs locally on Apple Silicon (MPS) in minutes —
no cloud GPU required. The main cost is the one-time PhysioNet download.

```bash
python -m src.models.train \
  --subjects $(seq 1 80) \
  --test-subjects $(seq 81 109) \
  --epochs 200 \
  --out checkpoints/
```

Pick the best demo subject with:

```bash
python scripts/screen_subjects.py --checkpoint checkpoints/eegnet.pt
```

## Tests

```bash
pytest
```
