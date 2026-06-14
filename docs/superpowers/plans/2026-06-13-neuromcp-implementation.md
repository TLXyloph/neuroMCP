# NeuroMCP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastMCP server that exposes calibrated EEG motor imagery decoding as two tool calls, backed by a streaming signal pipeline and EEGNet with Monte Carlo dropout.

**Architecture:** A `PlaybackThread` feeds recorded PhysioNet EEG at real-time sample rate into a `PipelineThread` that filters, windows, and rejects artifacts, writing clean epochs into a thread-safe `EpochBuffer`. The MCP server reads from the buffer on each tool call; `get_brain_state()` also runs N=50 MC dropout forward passes through EEGNet to produce a calibrated confidence score.

**Tech Stack:** Python 3.11+, MNE-Python, PyTorch (MPS on M3 Pro), FastMCP, Pydantic v2, Anthropic Python SDK, scipy, pytest

---

## File Map

```
src/
  __init__.py
  preprocessing/
    __init__.py
    buffer.py       — EpochBuffer: thread-safe circular deque, push/latest/stats
    pipeline.py     — load_and_epoch_subject(), bandpass/epoch/artifact helpers for training
    streaming.py    — PlaybackThread (feeds raw samples), PipelineThread (produces epochs)
  models/
    __init__.py
    eegnet.py       — EEGNet(nn.Module)
    mc_dropout.py   — MCDropoutDecoder, STATES constant
    train.py        — CLI training script: loads subjects 1-80, saves state_dict checkpoint
  server/
    __init__.py
    schemas.py      — SignalQualityResponse, BrainStateResponse (Pydantic)
    server.py       — FastMCP app, startup(), get_signal_quality(), get_brain_state()
  demo/
    __init__.py
    agent.py        — run_demo(): Anthropic tool-use loop, focus-gated code review
scripts/
  screen_subjects.py  — evaluate test subjects, rank by accuracy, recommend demo recording
tests/
  test_buffer.py
  test_preprocessing.py
  test_decoder.py
  test_server.py
notebooks/
  01_preprocessing.ipynb
  02_training.ipynb
  03_calibration.ipynb
pyproject.toml
```

---

## Phase 1 — Signal Pipeline

### Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/__init__.py`, `src/preprocessing/__init__.py`, `src/models/__init__.py`, `src/server/__init__.py`, `src/demo/__init__.py`
- Create: `scripts/` directory
- Create: `notebooks/` directory
- Modify: `.gitignore`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src/preprocessing src/models src/server src/demo scripts notebooks tests
touch src/__init__.py src/preprocessing/__init__.py src/models/__init__.py
touch src/server/__init__.py src/demo/__init__.py
touch tests/__init__.py
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "neuromcp"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "mne>=1.6",
    "numpy>=1.26",
    "torch>=2.1",
    "fastmcp>=0.4",
    "pydantic>=2.0",
    "anthropic>=0.25",
    "scipy>=1.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.12",
    "jupyter>=1.0",
    "matplotlib>=3.8",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Add gitignore entries**

Append to `.gitignore`:
```
data/
checkpoints/
~/mne_data/
__pycache__/
*.pyc
.pytest_cache/
dist/
*.egg-info/
.DS_Store
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -e ".[dev]"
```

Expected: no errors. MNE, PyTorch, FastMCP, Pydantic all installed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/ tests/ scripts/ notebooks/ .gitignore
git commit -m "feat: project scaffold and dependencies"
```

---

### Task 2: EpochBuffer

**Files:**
- Create: `src/preprocessing/buffer.py`
- Create: `tests/test_buffer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_buffer.py
import numpy as np
import pytest
from src.preprocessing.buffer import EpochBuffer


def make_epoch(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((64, 160))


def test_latest_returns_none_when_empty():
    buf = EpochBuffer()
    assert buf.latest() is None


def test_push_and_latest():
    buf = EpochBuffer()
    epoch = make_epoch()
    buf.push(epoch)
    result = buf.latest()
    assert result is not None
    np.testing.assert_array_equal(result, epoch)


def test_latest_returns_most_recent():
    buf = EpochBuffer()
    buf.push(make_epoch(0))
    second = make_epoch(1)
    buf.push(second)
    np.testing.assert_array_equal(buf.latest(), second)


def test_maxlen_evicts_oldest():
    buf = EpochBuffer(maxlen=3)
    for i in range(4):
        buf.push(make_epoch(i))
    # Buffer should only hold 3 epochs; latest is the last pushed
    np.testing.assert_array_equal(buf.latest(), make_epoch(3))


def test_stats_empty():
    buf = EpochBuffer()
    stats = buf.stats()
    assert stats["snr"] == 0.0
    assert stats["artifact_ratio"] == 0.0
    assert stats["epoch_count"] == 0


def test_stats_epoch_count():
    buf = EpochBuffer()
    buf.push(make_epoch(0))
    buf.push(make_epoch(1))
    assert buf.stats()["epoch_count"] == 2


def test_push_rejected_increments_artifact_ratio():
    buf = EpochBuffer()
    buf.push(make_epoch(0))   # 1 accepted
    buf.push_rejected()       # 1 rejected
    stats = buf.stats()
    assert stats["artifact_ratio"] == pytest.approx(0.5)


def test_snr_positive_with_real_signal():
    buf = EpochBuffer()
    rng = np.random.default_rng(42)
    for _ in range(5):
        buf.push(rng.random((64, 160)))
    assert buf.stats()["snr"] > 0
```

- [ ] **Step 2: Run tests — expect all to fail**

```bash
pytest tests/test_buffer.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.preprocessing.buffer'`

- [ ] **Step 3: Implement `src/preprocessing/buffer.py`**

```python
from collections import deque
import threading
import numpy as np


class EpochBuffer:
    def __init__(self, maxlen: int = 10) -> None:
        self._buf: deque[np.ndarray] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._total_seen = 0
        self._total_rejected = 0

    def push(self, epoch: np.ndarray) -> None:
        with self._lock:
            self._buf.append(epoch)
            self._total_seen += 1

    def push_rejected(self) -> None:
        with self._lock:
            self._total_seen += 1
            self._total_rejected += 1

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._buf[-1] if self._buf else None

    def stats(self) -> dict:
        with self._lock:
            epochs = list(self._buf)
            total = self._total_seen
            rejected = self._total_rejected

        if not epochs:
            return {"snr": 0.0, "artifact_ratio": 0.0, "epoch_count": 0}

        artifact_ratio = rejected / total if total > 0 else 0.0
        arr = np.stack(epochs)  # (n_epochs, n_channels, n_times)
        rms_per_epoch = np.sqrt(np.mean(arr ** 2, axis=(1, 2)))
        mean_rms = rms_per_epoch.mean()
        std_rms = rms_per_epoch.std() + 1e-10

        return {
            "snr": float(mean_rms / std_rms),
            "artifact_ratio": float(artifact_ratio),
            "epoch_count": len(epochs),
        }
```

- [ ] **Step 4: Run tests — expect all to pass**

```bash
pytest tests/test_buffer.py -v
```

Expected: 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing/buffer.py tests/test_buffer.py
git commit -m "feat: EpochBuffer with thread-safe push, latest, stats"
```

---

### Task 3: Signal Pipeline (Batch Training Path)

**Files:**
- Create: `src/preprocessing/pipeline.py`
- Create: `tests/test_preprocessing.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_preprocessing.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.preprocessing.pipeline import bandpass_filter, reject_artifacts, CODE_TO_CLASS


def make_raw_array(n_channels: int = 64, n_times: int = 1600, sfreq: float = 160.0):
    """Return (data, sfreq) as a stand-in for raw EEG."""
    rng = np.random.default_rng(0)
    return rng.random((n_channels, n_times)) * 50e-6, sfreq


def test_bandpass_output_shape():
    data, sfreq = make_raw_array()
    filtered = bandpass_filter(data, sfreq, l_freq=8.0, h_freq=30.0)
    assert filtered.shape == data.shape


def test_bandpass_attenuates_dc():
    """Bandpass should remove DC (0 Hz) component."""
    n_times = 1600
    sfreq = 160.0
    data = np.ones((64, n_times)) * 50e-6  # pure DC
    filtered = bandpass_filter(data, sfreq, l_freq=8.0, h_freq=30.0)
    # DC should be strongly attenuated
    assert np.abs(filtered).mean() < 1e-7


def test_reject_artifacts_keeps_clean_epochs():
    """Epochs with amplitude below threshold should not be rejected."""
    X = np.ones((10, 64, 160)) * 10e-6  # 10 µV — well below 100 µV
    y = np.zeros(10, dtype=int)
    X_clean, y_clean = reject_artifacts(X, y, peak_to_peak_uv=100.0)
    assert len(X_clean) == 10


def test_reject_artifacts_drops_bad_epochs():
    """Epochs with any channel exceeding threshold should be dropped."""
    X = np.ones((5, 64, 160)) * 10e-6
    X[2, 0, :] = 200e-6  # epoch 2, channel 0 spikes to 200 µV
    y = np.arange(5)
    X_clean, y_clean = reject_artifacts(X, y, peak_to_peak_uv=100.0)
    assert len(X_clean) == 4
    assert 2 not in y_clean  # epoch 2 was dropped


def test_code_to_class_mapping():
    assert CODE_TO_CLASS[1] == 0  # T0 → REST
    assert CODE_TO_CLASS[2] == 1  # T1 → LEFT
    assert CODE_TO_CLASS[3] == 2  # T2 → RIGHT
```

- [ ] **Step 2: Run tests — expect all to fail**

```bash
pytest tests/test_preprocessing.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.preprocessing.pipeline'`

- [ ] **Step 3: Implement `src/preprocessing/pipeline.py`**

```python
import numpy as np
from scipy.signal import butter, sosfilt
import mne
from mne.datasets import eegbci

# PhysioNet EEGBCI: runs 4, 8, 12 = motor imagery left/right fist
RUNS = [4, 8, 12]
TMIN, TMAX = 0.0, 1.0  # 1-second epochs
# Event code mapping: T0=rest(1), T1=left(2), T2=right(3)
CODE_TO_CLASS = {1: 0, 2: 1, 3: 2}


def bandpass_filter(
    data: np.ndarray, sfreq: float, l_freq: float = 8.0, h_freq: float = 30.0
) -> np.ndarray:
    """Apply 4th-order Butterworth bandpass filter. data: (n_channels, n_times)."""
    sos = butter(4, [l_freq, h_freq], btype="bandpass", fs=sfreq, output="sos")
    return sosfilt(sos, data, axis=1)


def reject_artifacts(
    X: np.ndarray, y: np.ndarray, peak_to_peak_uv: float = 100.0
) -> tuple[np.ndarray, np.ndarray]:
    """Drop epochs where any channel exceeds peak-to-peak threshold.

    X: (n_epochs, n_channels, n_times) in Volts
    Returns filtered (X, y).
    """
    threshold = peak_to_peak_uv * 1e-6
    ptp = np.ptp(X, axis=2).max(axis=1)  # (n_epochs,) max ptp across channels
    mask = ptp <= threshold
    return X[mask], y[mask]


def load_and_epoch_subject(subject: int) -> tuple[np.ndarray, np.ndarray]:
    """Load PhysioNet motor imagery runs for one subject, return (X, y).

    X: (n_epochs, 64, 160) float64, filtered 8-30 Hz
    y: (n_epochs,) int — 0=REST, 1=LEFT, 2=RIGHT
    """
    raw_fnames = eegbci.load_data(subject, RUNS)
    raws = [mne.io.read_raw_edf(f, preload=True, stim_channel="auto", verbose=False)
            for f in raw_fnames]
    raw = mne.concatenate_raws(raws)
    eegbci.standardize(raw)
    raw.filter(8.0, 30.0, method="iir", verbose=False)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(
        raw, events, event_id, TMIN, TMAX,
        baseline=None, preload=True, verbose=False
    )
    epochs.drop_bad(reject=dict(eeg=100e-6), verbose=False)

    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    y_codes = epochs.events[:, 2]
    y = np.array([CODE_TO_CLASS[c] for c in y_codes])
    return X, y
```

- [ ] **Step 4: Run tests — expect all to pass**

```bash
pytest tests/test_preprocessing.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing/pipeline.py tests/test_preprocessing.py
git commit -m "feat: bandpass filter and artifact rejection pipeline"
```

---

### Task 4: Streaming (PlaybackThread + PipelineThread)

**Files:**
- Create: `src/preprocessing/streaming.py`

Note: `PlaybackThread` and `PipelineThread` are daemon threads — unit tests verify their output after a short sleep rather than testing thread lifecycle directly.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_preprocessing.py`:

```python
import queue
import time
from src.preprocessing.buffer import EpochBuffer
from src.preprocessing.streaming import PlaybackThread, PipelineThread


def test_playback_thread_enqueues_samples():
    """PlaybackThread should put chunks into the queue."""
    rng = np.random.default_rng(1)
    data = rng.random((64, 3200)) * 50e-6  # 20s of fake EEG
    sfreq = 160.0
    sample_q = queue.Queue(maxsize=5000)

    thread = PlaybackThread(data, sfreq, sample_q, chunk_size=16)
    thread.start()
    time.sleep(0.3)  # let it push some chunks
    thread.stop()
    thread.join(timeout=1.0)

    assert not sample_q.empty()


def test_pipeline_thread_fills_epoch_buffer():
    """PipelineThread should write clean epochs to EpochBuffer."""
    rng = np.random.default_rng(2)
    # 20s of clean small-amplitude EEG (below artifact threshold)
    data = rng.random((64, 3200)) * 10e-6
    sfreq = 160.0
    sample_q = queue.Queue(maxsize=5000)
    epoch_buf = EpochBuffer()

    # Pre-fill the queue so pipeline has immediate data
    chunk_size = 16
    for i in range(0, data.shape[1], chunk_size):
        sample_q.put(data[:, i:i + chunk_size])

    pipeline = PipelineThread(sample_q, epoch_buf, sfreq=sfreq)
    pipeline.start()
    time.sleep(0.5)
    pipeline.stop()
    pipeline.join(timeout=1.0)

    assert epoch_buf.stats()["epoch_count"] > 0
```

- [ ] **Step 2: Run tests — expect new tests to fail**

```bash
pytest tests/test_preprocessing.py::test_playback_thread_enqueues_samples tests/test_preprocessing.py::test_pipeline_thread_fills_epoch_buffer -v
```

Expected: `ImportError` for `PlaybackThread`, `PipelineThread`.

- [ ] **Step 3: Implement `src/preprocessing/streaming.py`**

```python
import queue
import threading
import time
import numpy as np
from scipy.signal import butter, sosfilt
from .buffer import EpochBuffer


class PlaybackThread(threading.Thread):
    """Feeds recorded EEG samples into a queue at the original sample rate."""

    def __init__(
        self,
        data: np.ndarray,   # (n_channels, n_times)
        sfreq: float,
        sample_queue: queue.Queue,
        chunk_size: int = 16,
    ) -> None:
        super().__init__(daemon=True)
        self._data = data
        self._sfreq = sfreq
        self._queue = sample_queue
        self._chunk_size = chunk_size
        self._sleep = chunk_size / sfreq
        self._stop_event = threading.Event()

    def run(self) -> None:
        n_times = self._data.shape[1]
        i = 0
        while not self._stop_event.is_set():
            end = min(i + self._chunk_size, n_times)
            self._queue.put(self._data[:, i:end])
            time.sleep(self._sleep)
            i = end
            if i >= n_times:
                i = 0  # loop recording

    def stop(self) -> None:
        self._stop_event.set()


class PipelineThread(threading.Thread):
    """Reads samples from queue, applies windowing + filtering, pushes to EpochBuffer."""

    EPOCH_LEN = 160   # samples (1s at 160 Hz)
    STEP = 80         # 50% overlap

    def __init__(
        self,
        sample_queue: queue.Queue,
        epoch_buffer: EpochBuffer,
        sfreq: float = 160.0,
        n_channels: int = 64,
        peak_to_peak_uv: float = 100.0,
    ) -> None:
        super().__init__(daemon=True)
        self._queue = sample_queue
        self._buffer = epoch_buffer
        self._sfreq = sfreq
        self._n_channels = n_channels
        self._threshold = peak_to_peak_uv * 1e-6
        self._window = np.zeros((n_channels, self.EPOCH_LEN))
        self._samples_since_last_epoch = 0
        self._sos = butter(4, [8.0, 30.0], btype="bandpass", fs=sfreq, output="sos")
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            n = chunk.shape[1]
            # Slide window forward
            self._window = np.roll(self._window, -n, axis=1)
            self._window[:, -n:] = chunk
            self._samples_since_last_epoch += n

            # Emit a new epoch every STEP samples
            if self._samples_since_last_epoch >= self.STEP:
                self._samples_since_last_epoch = 0
                epoch = self._process(self._window.copy())
                if epoch is not None:
                    self._buffer.push(epoch)
                else:
                    self._buffer.push_rejected()

    def _process(self, window: np.ndarray) -> np.ndarray | None:
        filtered = sosfilt(self._sos, window, axis=1)
        if np.ptp(filtered, axis=1).max() > self._threshold:
            return None
        return filtered

    def stop(self) -> None:
        self._stop_event.set()
```

- [ ] **Step 4: Run all preprocessing tests**

```bash
pytest tests/test_preprocessing.py -v
```

Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing/streaming.py tests/test_preprocessing.py
git commit -m "feat: PlaybackThread and PipelineThread for real-time EEG streaming"
```

---

### Task 5: Preprocessing Notebook Validation

**Files:**
- Create: `notebooks/01_preprocessing.ipynb`

This is a manual validation step — no TDD. The notebook confirms the pipeline produces plausible output on real data.

- [ ] **Step 1: Create and open the notebook**

```bash
jupyter notebook notebooks/01_preprocessing.ipynb
```

- [ ] **Step 2: Add cells in this order**

Cell 1 — imports:
```python
import mne
import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing.pipeline import load_and_epoch_subject
from src.preprocessing.buffer import EpochBuffer
from src.preprocessing.streaming import PlaybackThread, PipelineThread
import queue, time
```

Cell 2 — load one subject and inspect:
```python
X, y = load_and_epoch_subject(subject=1)
print(f"Epochs: {X.shape}, Labels: {np.bincount(y)}")
# Expected: (n_epochs, 64, 160), 3 non-zero label counts
```

Cell 3 — plot PSD for one epoch:
```python
fig, ax = plt.subplots()
freqs = np.fft.rfftfreq(160, d=1/160)
psd = np.abs(np.fft.rfft(X[0].mean(axis=0))) ** 2
ax.semilogy(freqs, psd)
ax.axvline(8, color='r', label='8 Hz')
ax.axvline(30, color='g', label='30 Hz')
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power")
ax.legend()
plt.show()
# Should see energy concentrated between 8-30 Hz, near-zero outside
```

Cell 4 — streaming smoke test:
```python
raw_fnames = mne.datasets.eegbci.load_data(1, [4])
raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, verbose=False)
data = raw.get_data()
sfreq = raw.info['sfreq']

sample_q = queue.Queue(maxsize=5000)
epoch_buf = EpochBuffer()

playback = PlaybackThread(data, sfreq, sample_q)
pipeline = PipelineThread(sample_q, epoch_buf, sfreq=sfreq)
playback.start()
pipeline.start()
time.sleep(3.0)
playback.stop()
pipeline.stop()

stats = epoch_buf.stats()
print(f"Buffer stats: {stats}")
print(f"Latest epoch shape: {epoch_buf.latest().shape}")
# Expected: epoch_count > 0, snr > 0, latest = (64, 160)
```

- [ ] **Step 3: Confirm gate — run all cells, verify no errors and plausible output**

Gate: `epoch_count > 0`, `latest()` returns shape `(64, 160)`, PSD plot shows energy in 8–30 Hz band.

- [ ] **Step 4: Commit**

```bash
git add notebooks/01_preprocessing.ipynb
git commit -m "feat: preprocessing validation notebook"
```

---

## Phase 2 — Neural Decoder

### Task 6: EEGNet Architecture

**Files:**
- Create: `src/models/eegnet.py`
- Create: `tests/test_decoder.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_decoder.py
import numpy as np
import torch
import pytest
from src.models.eegnet import EEGNet


def test_eegnet_output_shape():
    model = EEGNet(n_channels=64, n_times=160, n_classes=3)
    x = torch.randn(4, 64, 160)  # batch=4
    out = model(x)
    assert out.shape == (4, 3)


def test_eegnet_single_sample():
    model = EEGNet(n_channels=64, n_times=160, n_classes=3)
    x = torch.randn(1, 64, 160)
    out = model(x)
    assert out.shape == (1, 3)


def test_eegnet_output_is_logits_not_probs():
    """Model should output raw logits — softmax applied externally."""
    model = EEGNet()
    x = torch.randn(2, 64, 160)
    out = model(x)
    # Logits can be any real number; softmax output would sum to 1
    assert not torch.allclose(out.softmax(dim=-1).sum(dim=-1), out.sum(dim=-1))


def test_eegnet_dropout_differs_in_train_mode():
    """With dropout active, two forward passes should differ."""
    model = EEGNet()
    model.train()
    x = torch.randn(1, 64, 160)
    out1 = model(x)
    out2 = model(x)
    assert not torch.allclose(out1, out2), "Dropout should make passes differ"


def test_eegnet_deterministic_in_eval_mode():
    model = EEGNet()
    model.eval()
    x = torch.randn(1, 64, 160)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)
```

- [ ] **Step 2: Run tests — expect all to fail**

```bash
pytest tests/test_decoder.py -v
```

Expected: `ImportError: cannot import name 'EEGNet'`

- [ ] **Step 3: Implement `src/models/eegnet.py`**

```python
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """EEGNet for motor imagery classification (Lawhern et al. 2018).

    Input: (batch, n_channels, n_times)
    Output: (batch, n_classes) — raw logits
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_times: int = 160,
        n_classes: int = 3,
        F1: int = 8,
        D: int = 2,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        F2 = F1 * D

        # Block 1: temporal conv + depthwise spatial conv
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )

        # Block 2: separable conv (depthwise + pointwise)
        self.block2 = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 16), padding="same", groups=F2, bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )

        # Dynamically compute flattened size
        with torch.no_grad():
            probe = torch.zeros(1, 1, n_channels, n_times)
            probe = self.block1(probe)
            probe = self.block2(probe)
            flat_size = probe.flatten(1).shape[1]

        self.classifier = nn.Linear(flat_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_channels, n_times)
        x = x.unsqueeze(1)        # (batch, 1, n_channels, n_times)
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x.flatten(1))
```

- [ ] **Step 4: Run tests — expect all to pass**

```bash
pytest tests/test_decoder.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/models/eegnet.py tests/test_decoder.py
git commit -m "feat: EEGNet architecture with depthwise + separable convolutions"
```

---

### Task 7: Monte Carlo Dropout Wrapper

**Files:**
- Create: `src/models/mc_dropout.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_decoder.py`:

```python
from unittest.mock import patch
import os
from src.models.mc_dropout import MCDropoutDecoder, STATES


def _save_dummy_checkpoint(path: str) -> None:
    model = EEGNet()
    torch.save(model.state_dict(), path)


def test_mc_dropout_states_constant():
    assert STATES == ["REST", "LEFT_IMAGERY", "RIGHT_IMAGERY"]


def test_mc_dropout_returns_valid_state(tmp_path):
    ckpt = str(tmp_path / "model.pt")
    _save_dummy_checkpoint(ckpt)
    decoder = MCDropoutDecoder(ckpt, n_passes=5, device="cpu")
    epoch = np.random.randn(64, 160).astype(np.float32)
    state, confidence = decoder.predict(epoch)
    assert state in STATES + ["LOW_CONFIDENCE"]


def test_mc_dropout_confidence_in_range(tmp_path):
    ckpt = str(tmp_path / "model.pt")
    _save_dummy_checkpoint(ckpt)
    decoder = MCDropoutDecoder(ckpt, n_passes=5, device="cpu")
    epoch = np.random.randn(64, 160).astype(np.float32)
    _, confidence = decoder.predict(epoch)
    assert 0.0 <= confidence <= 1.0


def test_mc_dropout_runs_n_passes(tmp_path):
    """Verify N stochastic passes actually differ (dropout is active)."""
    ckpt = str(tmp_path / "model.pt")
    _save_dummy_checkpoint(ckpt)
    decoder = MCDropoutDecoder(ckpt, n_passes=10, device="cpu")

    epoch = np.random.randn(64, 160).astype(np.float32)
    x = torch.tensor(epoch).unsqueeze(0)

    passes = []
    for _ in range(10):
        with torch.no_grad():
            passes.append(decoder._model(x).numpy())

    # Not all passes should be identical (dropout is on)
    all_same = all(np.allclose(passes[0], p) for p in passes[1:])
    assert not all_same
```

- [ ] **Step 2: Run new tests — expect failure**

```bash
pytest tests/test_decoder.py::test_mc_dropout_states_constant tests/test_decoder.py::test_mc_dropout_returns_valid_state -v
```

Expected: `ImportError: cannot import name 'MCDropoutDecoder'`

- [ ] **Step 3: Implement `src/models/mc_dropout.py`**

```python
import numpy as np
import torch
import torch.nn as nn
from .eegnet import EEGNet

STATES = ["REST", "LEFT_IMAGERY", "RIGHT_IMAGERY"]


def _enable_mc_dropout(model: nn.Module) -> None:
    """Eval mode except Dropout layers stay in train mode."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


class MCDropoutDecoder:
    def __init__(
        self,
        checkpoint_path: str,
        n_passes: int = 50,
        device: str = "cpu",
    ) -> None:
        self._device = torch.device(device)
        self._n_passes = n_passes
        self._model = EEGNet()
        self._model.load_state_dict(
            torch.load(checkpoint_path, map_location=self._device, weights_only=True)
        )
        self._model.to(self._device)
        _enable_mc_dropout(self._model)

    def predict(self, epoch: np.ndarray) -> tuple[str, float]:
        """Run N stochastic forward passes. Returns (state, confidence).

        epoch: (n_channels, n_times) float32
        confidence: mean softmax probability of the winning class across N passes
        """
        x = torch.tensor(epoch, dtype=torch.float32).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = torch.stack([self._model(x) for _ in range(self._n_passes)])
        # logits: (n_passes, 1, n_classes)

        probs = torch.softmax(logits, dim=-1).squeeze(1)  # (n_passes, n_classes)
        mean_probs = probs.mean(dim=0)                     # (n_classes,)

        class_idx = int(mean_probs.argmax().item())
        confidence = float(mean_probs[class_idx].item())
        return STATES[class_idx], confidence
```

- [ ] **Step 4: Run all decoder tests**

```bash
pytest tests/test_decoder.py -v
```

Expected: 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/models/mc_dropout.py tests/test_decoder.py
git commit -m "feat: MCDropoutDecoder with N stochastic passes and calibrated confidence"
```

---

### Task 8: Training Script

**Files:**
- Create: `src/models/train.py`

This script runs on a cloud GPU. No TDD — validate manually by running a 2-subject smoke test locally.

- [ ] **Step 1: Create `src/models/train.py`**

```python
"""Train EEGNet on PhysioNet EEGBCI motor imagery data.

Usage (local smoke test):
    python -m src.models.train --subjects 1 2 --test-subjects 3 --epochs 3 --out checkpoints/

Usage (full cloud run):
    python -m src.models.train --subjects $(seq -s' ' 1 80) --test-subjects $(seq -s' ' 81 109) --epochs 200 --out checkpoints/
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from src.preprocessing.pipeline import load_and_epoch_subject
from src.models.eegnet import EEGNet


def load_dataset(subjects: list[int]) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for subj in subjects:
        try:
            X, y = load_and_epoch_subject(subj)
            Xs.append(X)
            ys.append(y)
            print(f"  Subject {subj}: {len(y)} epochs")
        except Exception as e:
            print(f"  Subject {subj}: SKIP ({e})")
    return np.concatenate(Xs), np.concatenate(ys)


def train(
    train_subjects: list[int],
    test_subjects: list[int],
    n_epochs: int,
    out_dir: str,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> None:
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    print("Loading training data...")
    X_train, y_train = load_dataset(train_subjects)
    print(f"Training set: {X_train.shape}")

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model = EEGNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(xb)
        print(f"Epoch {epoch}/{n_epochs} — loss: {total_loss/total:.4f}, acc: {correct/total:.3f}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "eegnet.pt")
    print(f"Checkpoint saved to {out / 'eegnet.pt'}")

    # Evaluate on test subjects
    if test_subjects:
        print("\nLoading test data...")
        X_test, y_test = load_dataset(test_subjects)
        model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            preds = model(x_t).argmax(1).cpu().numpy()
        acc = (preds == y_test).mean()
        print(f"Test accuracy: {acc:.3f}")

        # Save calibration data for notebooks
        probs = torch.softmax(
            model(torch.tensor(X_test, dtype=torch.float32).to(device)), dim=-1
        ).cpu().numpy()
        np.save(out / "test_probs_baseline.npy", probs)
        np.save(out / "test_labels.npy", y_test)
        print(f"Calibration data saved to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 81)))
    parser.add_argument("--test-subjects", nargs="+", type=int, default=list(range(81, 110)))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--out", default="checkpoints")
    args = parser.parse_args()
    train(args.subjects, args.test_subjects, args.epochs, args.out)
```

- [ ] **Step 2: Smoke test locally with 2 subjects, 3 epochs**

```bash
python -m src.models.train --subjects 1 2 --test-subjects 3 --epochs 3 --out checkpoints/
```

Expected: no errors, checkpoint saved at `checkpoints/eegnet.pt`, test accuracy printed.

- [ ] **Step 3: Commit**

```bash
git add src/models/train.py checkpoints/
git commit -m "feat: EEGNet training script with calibration data export"
```

---

### Task 9: Cloud Training

This task is procedural, not code. Execute after Task 8.

- [ ] **Step 1: Provision a cloud GPU instance**

Recommended: Lambda Labs — rent an A10G instance (~$0.75/hr).
- Go to lambdalabs.com → Cloud → launch an instance (A10G, Ubuntu 22.04)
- Note the instance IP

- [ ] **Step 2: Upload project**

```bash
rsync -av --exclude='data/' --exclude='checkpoints/' --exclude='.git/' \
  /Users/samvrithbandi/Desktop/personal-proj/neuroMCP/ \
  ubuntu@<INSTANCE_IP>:~/neuromcp/
```

- [ ] **Step 3: Install dependencies on instance**

```bash
ssh ubuntu@<INSTANCE_IP>
cd ~/neuromcp
pip install -e ".[dev]"
```

- [ ] **Step 4: Run full training**

```bash
nohup python -m src.models.train \
  --subjects $(seq -s' ' 1 80) \
  --test-subjects $(seq -s' ' 81 109) \
  --epochs 200 \
  --out checkpoints/ \
  > training.log 2>&1 &

tail -f training.log
```

Expected training time: 30–90 minutes on A10G. Test accuracy should be 60–75%.

- [ ] **Step 5: Download checkpoint and calibration data**

```bash
rsync ubuntu@<INSTANCE_IP>:~/neuromcp/checkpoints/ \
  /Users/samvrithbandi/Desktop/personal-proj/neuroMCP/checkpoints/
```

Expected files: `eegnet.pt`, `test_probs_baseline.npy`, `test_labels.npy`

- [ ] **Step 6: Terminate the cloud instance** (to stop billing)

- [ ] **Step 7: Commit checkpoint reference**

```bash
echo "checkpoints/*.pt" >> .gitignore
echo "checkpoints/*.npy" >> .gitignore
git add .gitignore
git commit -m "chore: gitignore checkpoint binaries"
```

---

### Task 10: Calibration Notebooks

**Files:**
- Create: `notebooks/02_training.ipynb`
- Create: `notebooks/03_calibration.ipynb`

- [ ] **Step 1: Create `notebooks/02_training.ipynb` — MC dropout inference on test set**

Cell 1 — imports:
```python
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.models.mc_dropout import MCDropoutDecoder, STATES

CHECKPOINT = "checkpoints/eegnet.pt"
N_PASSES = 50
```

Cell 2 — load calibration data:
```python
probs_baseline = np.load("checkpoints/test_probs_baseline.npy")  # (n_test, 3)
y_test = np.load("checkpoints/test_labels.npy")                  # (n_test,)
print(f"Test set: {len(y_test)} epochs, class dist: {np.bincount(y_test)}")
```

Cell 3 — run MC dropout on test set:
```python
decoder = MCDropoutDecoder(CHECKPOINT, n_passes=N_PASSES, device="cpu")
X_test = np.load("checkpoints/X_test.npy")  # add this export to train.py if needed

mc_states, mc_confidences = [], []
for epoch in X_test:
    state, conf = decoder.predict(epoch.astype(np.float32))
    mc_states.append(state)
    mc_confidences.append(conf)

mc_preds = np.array([STATES.index(s) for s in mc_states])
mc_confs = np.array(mc_confidences)
mc_acc = (mc_preds == y_test).mean()
print(f"MC dropout accuracy: {mc_acc:.3f}")
```

Note: also add `np.save(out / "X_test.npy", X_test)` to `train.py` after the test accuracy line.

- [ ] **Step 2: Add `X_test.npy` export to `train.py`**

In `src/models/train.py`, after `np.save(out / "test_labels.npy", y_test)`:
```python
np.save(out / "X_test.npy", X_test)
```

- [ ] **Step 3: Create `notebooks/03_calibration.ipynb` — reliability diagram and ECE**

Cell 1 — imports:
```python
import numpy as np
import matplotlib.pyplot as plt

probs_baseline = np.load("checkpoints/test_probs_baseline.npy")
y_test = np.load("checkpoints/test_labels.npy")
mc_confidences = np.load("checkpoints/mc_confidences.npy")  # save from notebook 02
mc_preds = np.load("checkpoints/mc_preds.npy")
```

Cell 2 — reliability diagram function:
```python
def reliability_diagram(confidences, correct, n_bins=10, label=""):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_sizes = [], [], []
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_sizes.append(mask.sum())
    return np.array(bin_confs), np.array(bin_accs), np.array(bin_sizes)
```

Cell 3 — ECE function:
```python
def ece(confidences, correct, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(confidences)
    ece_val = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        if mask.sum() > 0:
            acc = correct[mask].mean()
            conf = confidences[mask].mean()
            ece_val += (mask.sum() / total) * abs(acc - conf)
    return ece_val
```

Cell 4 — plot and compute:
```python
baseline_conf = probs_baseline.max(axis=1)
baseline_preds = probs_baseline.argmax(axis=1)
baseline_correct = (baseline_preds == y_test)
mc_correct = (mc_preds == y_test)

ece_baseline = ece(baseline_conf, baseline_correct)
ece_mc = ece(mc_confidences, mc_correct)
print(f"ECE baseline: {ece_baseline:.4f}")
print(f"ECE MC dropout: {ece_mc:.4f}")
print(f"ECE reduction: {(ece_baseline - ece_mc) / ece_baseline * 100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, confs, correct, label in [
    (axes[0], baseline_conf, baseline_correct, "Baseline EEGNet"),
    (axes[1], mc_confidences, mc_correct, f"MC Dropout (N={N_PASSES})"),
]:
    bc, ba, _ = reliability_diagram(confs, correct)
    ax.plot([0, 1], [0, 1], 'k--', label="Perfect calibration")
    ax.plot(bc, ba, 'o-', label=label)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{label}\nECE={ece(confs, correct):.4f}")
    ax.legend()

plt.tight_layout()
plt.savefig("checkpoints/calibration_curve.png", dpi=150)
plt.show()
# This image is your resume artifact
```

- [ ] **Step 4: Run both notebooks end-to-end, confirm calibration_curve.png is saved**

Gate: `ECE reduction > 0%` (MC dropout should improve calibration). Record the ECE numbers — these go in your resume.

- [ ] **Step 5: Commit**

```bash
git add notebooks/02_training.ipynb notebooks/03_calibration.ipynb src/models/train.py
git commit -m "feat: calibration notebooks — reliability diagram and ECE"
```

---

## Phase 3 — MCP Server

### Task 11: Pydantic Schemas

**Files:**
- Create: `src/server/schemas.py`
- Create: `tests/test_server.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_server.py
import pytest
from src.server.schemas import SignalQualityResponse, BrainStateResponse


def test_signal_quality_response_fields():
    r = SignalQualityResponse(snr=12.4, artifact_ratio=0.08, epoch_count=7)
    assert r.snr == 12.4
    assert r.artifact_ratio == 0.08
    assert r.epoch_count == 7


def test_signal_quality_response_serializes():
    r = SignalQualityResponse(snr=5.0, artifact_ratio=0.1, epoch_count=3)
    d = r.model_dump()
    assert set(d.keys()) == {"snr", "artifact_ratio", "epoch_count"}


def test_brain_state_response_fields():
    r = BrainStateResponse(state="LEFT_IMAGERY", confidence=0.84, timestamp="2026-06-13T20:00:00Z")
    assert r.state == "LEFT_IMAGERY"
    assert r.confidence == 0.84


def test_brain_state_low_confidence():
    r = BrainStateResponse(state="LOW_CONFIDENCE", confidence=0.41, timestamp="2026-06-13T20:00:05Z")
    assert r.state == "LOW_CONFIDENCE"


def test_brain_state_response_serializes():
    r = BrainStateResponse(state="REST", confidence=0.9, timestamp="2026-06-13T20:00:00Z")
    d = r.model_dump()
    assert set(d.keys()) == {"state", "confidence", "timestamp"}
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_server.py -v
```

Expected: `ImportError: cannot import name 'SignalQualityResponse'`

- [ ] **Step 3: Implement `src/server/schemas.py`**

```python
from pydantic import BaseModel


class SignalQualityResponse(BaseModel):
    snr: float
    artifact_ratio: float
    epoch_count: int


class BrainStateResponse(BaseModel):
    state: str  # LEFT_IMAGERY | RIGHT_IMAGERY | REST | LOW_CONFIDENCE
    confidence: float
    timestamp: str
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_server.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/server/schemas.py tests/test_server.py
git commit -m "feat: Pydantic schemas for MCP tool responses"
```

---

### Task 12: FastMCP Server

**Files:**
- Create: `src/server/server.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_server.py`:

```python
import numpy as np
from unittest.mock import MagicMock, patch
from src.preprocessing.buffer import EpochBuffer
from src.server.server import get_signal_quality, get_brain_state, _state


def _setup_state_with_buffer(epoch_count: int = 3):
    buf = EpochBuffer()
    rng = np.random.default_rng(0)
    for _ in range(epoch_count):
        buf.push(rng.random((64, 160)).astype(np.float32) * 10e-6)
    _state["buffer"] = buf


def _clear_state():
    _state["buffer"] = None
    _state["decoder"] = None


def test_get_signal_quality_empty_buffer():
    _clear_state()
    _state["buffer"] = EpochBuffer()  # empty
    result = get_signal_quality()
    assert result.epoch_count == 0
    assert result.snr == 0.0


def test_get_signal_quality_with_epochs():
    _setup_state_with_buffer(3)
    result = get_signal_quality()
    assert result.epoch_count == 3
    assert result.snr >= 0.0
    _clear_state()


def test_get_brain_state_no_epoch_returns_low_confidence():
    _clear_state()
    _state["buffer"] = EpochBuffer()  # empty
    mock_decoder = MagicMock()
    _state["decoder"] = mock_decoder
    result = get_brain_state()
    assert result.state == "LOW_CONFIDENCE"
    mock_decoder.predict.assert_not_called()


def test_get_brain_state_above_threshold():
    _setup_state_with_buffer(1)
    mock_decoder = MagicMock()
    mock_decoder.predict.return_value = ("REST", 0.85)
    _state["decoder"] = mock_decoder
    result = get_brain_state(confidence_threshold=0.7)
    assert result.state == "REST"
    assert result.confidence == pytest.approx(0.85)
    _clear_state()


def test_get_brain_state_below_threshold():
    _setup_state_with_buffer(1)
    mock_decoder = MagicMock()
    mock_decoder.predict.return_value = ("LEFT_IMAGERY", 0.45)
    _state["decoder"] = mock_decoder
    result = get_brain_state(confidence_threshold=0.7)
    assert result.state == "LOW_CONFIDENCE"
    assert result.confidence == pytest.approx(0.45)
    _clear_state()


def test_get_brain_state_timestamp_is_iso():
    _setup_state_with_buffer(1)
    mock_decoder = MagicMock()
    mock_decoder.predict.return_value = ("REST", 0.9)
    _state["decoder"] = mock_decoder
    result = get_brain_state()
    assert result.timestamp.endswith("Z")
    _clear_state()
```

- [ ] **Step 2: Run new tests — expect failure**

```bash
pytest tests/test_server.py -k "get_signal_quality or get_brain_state" -v
```

Expected: `ImportError: cannot import name 'get_signal_quality'`

- [ ] **Step 3: Implement `src/server/server.py`**

```python
import argparse
import datetime
import queue
import mne
from mne.datasets import eegbci
from fastmcp import FastMCP
from .schemas import BrainStateResponse, SignalQualityResponse
from ..preprocessing.buffer import EpochBuffer
from ..preprocessing.streaming import PlaybackThread, PipelineThread
from ..models.mc_dropout import MCDropoutDecoder

mcp = FastMCP("NeuroMCP")

_state: dict = {
    "buffer": None,
    "decoder": None,
    "playback": None,
    "pipeline": None,
}


def startup(checkpoint_path: str, demo_subject: int, demo_run: int) -> None:
    """Initialize pipeline and decoder. Call before mcp.run()."""
    raw_fnames = eegbci.load_data(demo_subject, [demo_run])
    raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, stim_channel="auto", verbose=False)
    eegbci.standardize(raw)
    data = raw.get_data()
    sfreq = raw.info["sfreq"]

    sample_q: queue.Queue = queue.Queue(maxsize=2000)
    epoch_buf = EpochBuffer()

    device = "cpu"
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        pass

    decoder = MCDropoutDecoder(checkpoint_path, n_passes=50, device=device)
    playback = PlaybackThread(data, sfreq, sample_q)
    pipeline = PipelineThread(sample_q, epoch_buf, sfreq=sfreq)

    playback.start()
    pipeline.start()

    _state["buffer"] = epoch_buf
    _state["decoder"] = decoder
    _state["playback"] = playback
    _state["pipeline"] = pipeline


@mcp.tool()
def get_signal_quality() -> SignalQualityResponse:
    """Return current EEG signal quality metrics. Fast — no inference."""
    stats = _state["buffer"].stats()
    return SignalQualityResponse(**stats)


@mcp.tool()
def get_brain_state(confidence_threshold: float = 0.7) -> BrainStateResponse:
    """Return decoded motor imagery state with calibrated confidence.

    Returns LOW_CONFIDENCE if confidence < confidence_threshold or no epoch available.
    """
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    epoch = _state["buffer"].latest()

    if epoch is None:
        return BrainStateResponse(state="LOW_CONFIDENCE", confidence=0.0, timestamp=timestamp)

    state, confidence = _state["decoder"].predict(epoch)
    if confidence < confidence_threshold:
        state = "LOW_CONFIDENCE"

    return BrainStateResponse(state=state, confidence=round(confidence, 4), timestamp=timestamp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to eegnet.pt")
    parser.add_argument("--subject", type=int, required=True, help="Demo subject number")
    parser.add_argument("--run", type=int, required=True, help="Demo run number (e.g. 4)")
    args = parser.parse_args()

    startup(args.checkpoint, args.subject, args.run)
    mcp.run()
```

- [ ] **Step 4: Run all server tests**

```bash
pytest tests/test_server.py -v
```

Expected: all 11 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/server/server.py tests/test_server.py
git commit -m "feat: FastMCP server with get_signal_quality and get_brain_state tools"
```

---

### Task 13: MCP Inspector Validation

Manual integration test — requires the checkpoint from Task 9.

- [ ] **Step 1: Start the server**

```bash
python -m src.server.server \
  --checkpoint checkpoints/eegnet.pt \
  --subject <BEST_DEMO_SUBJECT> \
  --run 4
```

Expected: FastMCP server starts, streaming pipeline begins, no errors.

- [ ] **Step 2: Open MCP Inspector**

In a second terminal:
```bash
npx @modelcontextprotocol/inspector
```

Connect to the running server.

- [ ] **Step 3: Call `get_signal_quality`**

In MCP Inspector: call `get_signal_quality` with no arguments.

Expected response:
```json
{"snr": <positive float>, "artifact_ratio": <0-1>, "epoch_count": <positive int>}
```

- [ ] **Step 4: Call `get_brain_state`**

Call `get_brain_state` with `confidence_threshold: 0.7`.

Expected response:
```json
{"state": "LEFT_IMAGERY" | "RIGHT_IMAGERY" | "REST" | "LOW_CONFIDENCE", "confidence": <0-1>, "timestamp": "<ISO string>Z"}
```

- [ ] **Step 5: Verify latency**

Call `get_brain_state` 5 times and note response time. Target: < 200ms each.

If > 200ms: reduce `n_passes` from 50 to 20 in `MCDropoutDecoder` default and retest.

Gate: both tools return structured responses, `get_brain_state` consistently < 200ms.

---

## Phase 4 — Demo Agent

### Task 14: Subject Screening Script

**Files:**
- Create: `scripts/screen_subjects.py`

- [ ] **Step 1: Create `scripts/screen_subjects.py`**

```python
"""Evaluate MC dropout accuracy per test subject. Run after cloud training.

Usage:
    python scripts/screen_subjects.py --checkpoint checkpoints/eegnet.pt
"""
import argparse
import numpy as np
from src.preprocessing.pipeline import load_and_epoch_subject
from src.models.mc_dropout import MCDropoutDecoder, STATES


def evaluate_subject(decoder: MCDropoutDecoder, subject: int) -> float | None:
    try:
        X, y = load_and_epoch_subject(subject)
    except Exception as e:
        print(f"  Subject {subject}: SKIP ({e})")
        return None

    correct = 0
    for i in range(len(X)):
        state, confidence = decoder.predict(X[i].astype(np.float32))
        if confidence >= 0.5:
            pred = STATES.index(state) if state in STATES else -1
            if pred == y[i]:
                correct += 1

    acc = correct / len(X)
    print(f"  Subject {subject}: {acc:.2%} ({len(X)} epochs)")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    decoder = MCDropoutDecoder(args.checkpoint, n_passes=50, device="cpu")

    print("Evaluating test subjects 81-109...")
    results = []
    for subject in range(81, 110):
        acc = evaluate_subject(decoder, subject)
        if acc is not None:
            results.append((subject, acc))

    results.sort(key=lambda x: x[1], reverse=True)
    print("\n=== Top 5 subjects for demo ===")
    for subject, acc in results[:5]:
        print(f"  Subject {subject}: {acc:.2%}")

    print("\nRecommended demo recording: subject", results[0][0])
    print("Use run 4 (motor imagery left/right fist, first repeat)")
    print(f"  python -m src.server.server --checkpoint {args.checkpoint} "
          f"--subject {results[0][0]} --run 4")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the screening script**

```bash
python scripts/screen_subjects.py --checkpoint checkpoints/eegnet.pt
```

Expected: ranked list of test subjects. Note the top subject — use it for the demo.

- [ ] **Step 3: Commit**

```bash
git add scripts/screen_subjects.py
git commit -m "feat: subject screening script for demo recording selection"
```

---

### Task 15: Demo Agent

**Files:**
- Create: `src/demo/agent.py`

- [ ] **Step 1: Create `src/demo/agent.py`**

```python
"""Focus-gated code review demo.

The agent reviews code findings. Before surfacing each one it checks
signal quality and brain state, adapting verbosity accordingly.

Usage:
    # Start server first (separate terminal):
    #   python -m src.server.server --checkpoint checkpoints/eegnet.pt --subject <N> --run 4

    python -m src.demo.agent \
      --checkpoint checkpoints/eegnet.pt \
      --subject <BEST_SUBJECT> \
      --run 4
"""
import argparse
import json
import queue
import time
import mne
from mne.datasets import eegbci
import anthropic
from ..preprocessing.buffer import EpochBuffer
from ..preprocessing.streaming import PlaybackThread, PipelineThread
from ..models.mc_dropout import MCDropoutDecoder
from ..server.server import get_signal_quality, get_brain_state, _state, startup

FINDINGS = [
    "The authentication middleware doesn't validate JWT expiration. An expired token is accepted indefinitely.",
    "Database queries in user_service.py are concatenated strings — SQL injection risk on the username field.",
    "The cache TTL is hardcoded to 60 seconds with no configuration hook. This breaks under high load.",
]

SYSTEM = """You are a code reviewer. You have access to two tools that read the operator's
EEG brain state in real time. Before surfacing each code review finding, you MUST:

1. Call get_signal_quality() — if snr < 3, say exactly: "Signal noisy — take a breath, I'll wait."
   and do not surface the finding yet.
2. Call get_brain_state(confidence_threshold=0.65).
   - If state is "REST": surface the finding in full detail.
   - If state is "LEFT_IMAGERY" or "RIGHT_IMAGERY": surface only a one-sentence summary,
     prefixed with "You seem focused on something — one thing: "
   - If state is "LOW_CONFIDENCE": briefly note you'll check again, then recheck once.

Work through all findings. Be concise. Do not explain this process to the operator."""

TOOLS = [
    {
        "name": "get_signal_quality",
        "description": "Return current EEG signal quality metrics.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_brain_state",
        "description": "Return decoded motor imagery state with calibrated confidence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "confidence_threshold": {
                    "type": "number",
                    "description": "Minimum confidence to return a state. Default 0.7.",
                    "default": 0.7,
                }
            },
            "required": [],
        },
    },
]


def execute_tool(name: str, inputs: dict) -> dict:
    if name == "get_signal_quality":
        return get_signal_quality().model_dump()
    elif name == "get_brain_state":
        threshold = inputs.get("confidence_threshold", 0.7)
        return get_brain_state(confidence_threshold=threshold).model_dump()
    raise ValueError(f"Unknown tool: {name}")


def run_agentic_loop() -> None:
    client = anthropic.Anthropic()
    messages = [
        {
            "role": "user",
            "content": (
                f"Please review these {len(FINDINGS)} findings:\n\n"
                + "\n".join(f"{i+1}. {f}" for i, f in enumerate(FINDINGS))
            ),
        }
    ]

    print("\n" + "=" * 60)
    print("NeuroMCP Demo — Focus-Gated Code Review")
    print("=" * 60 + "\n")

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

        # Print any text blocks
        for block in response.content:
            if hasattr(block, "text"):
                print(f"Agent: {block.text}\n")

        if response.stop_reason == "end_turn":
            break

        # Handle tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [tool] {block.name}({json.dumps(block.input)})")
                result = execute_tool(block.name, block.input)
                print(f"  [result] {json.dumps(result)}\n")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--run", type=int, required=True)
    args = parser.parse_args()

    print("Starting EEG pipeline...")
    startup(args.checkpoint, args.subject, args.run)
    print("Waiting 3s for buffer to fill...")
    time.sleep(3.0)

    run_agentic_loop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the agent end-to-end**

```bash
python -m src.demo.agent \
  --checkpoint checkpoints/eegnet.pt \
  --subject <BEST_SUBJECT> \
  --run 4
```

Expected: agent prints findings, tool calls visible in terminal, at least two different behavioral responses across the three findings.

- [ ] **Step 3: Commit**

```bash
git add src/demo/agent.py
git commit -m "feat: focus-gated code review demo agent"
```

---

### Task 16: Demo Recording

No code. Execute after Task 15 works end-to-end.

- [ ] **Step 1: Set up screen layout**

Use a screen recorder (QuickTime or OBS). Arrange:
- Left panel: terminal running the agent
- Optional right panel: EEG epoch visualization (add a simple matplotlib live plot to streaming.py if desired)

- [ ] **Step 2: Run 2-3 takes**

```bash
python -m src.demo.agent \
  --checkpoint checkpoints/eegnet.pt \
  --subject <BEST_SUBJECT> \
  --run 4
```

Keep the best take (target: all 3 behavioral modes visible — verbose, short, noisy signal).

- [ ] **Step 3: Trim and export**

Target: 90–120 seconds. Export as MP4.

---

## Phase 5 — Polish

### Task 17: README and Architecture Diagram

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite `README.md`**

```markdown
# NeuroMCP

An MCP server exposing calibrated EEG motor imagery decoding as agent tool calls.
EEGNet with Monte Carlo dropout achieves [X]% ECE reduction vs. baseline;
streaming signal pipeline with artifact rejection runs at sub-200ms inference latency.
Any MCP-compatible agent can query decoded human mental state with calibrated uncertainty.

> **Note:** Uses recorded PhysioNet EEGBCI signals as a proxy for live input.
> This is a research-grade demonstration of the interface pattern, not production BCI software.

## Demo

[Link to demo video]

## Architecture

[paste ASCII diagram from design spec here]

## API

### `get_signal_quality()`
Returns `{snr, artifact_ratio, epoch_count}` — reads from the epoch buffer, no inference cost.

### `get_brain_state(confidence_threshold=0.7)`
Returns `{state, confidence, timestamp}`. Runs N=50 MC dropout passes through EEGNet.
State is one of: `LEFT_IMAGERY`, `RIGHT_IMAGERY`, `REST`, `LOW_CONFIDENCE`.

The two-tool design lets agents implement a quality gate — call `get_signal_quality()` first,
only call `get_brain_state()` when SNR is acceptable.

## Technical Contribution

Monte Carlo dropout produces calibrated confidence scores that a baseline softmax cannot.
The reliability diagram in `notebooks/03_calibration.ipynb` shows [X]% ECE reduction.
This is what lets the agent gate on uncertainty rather than acting on every prediction.

## Setup

```bash
pip install -e ".[dev]"
```

Download PhysioNet EEGBCI (handled automatically by MNE on first run).

## Running the Server

```bash
python -m src.server.server \
  --checkpoint checkpoints/eegnet.pt \
  --subject <subject_number> \
  --run 4
```

## Running the Demo

```bash
python -m src.demo.agent \
  --checkpoint checkpoints/eegnet.pt \
  --subject <subject_number> \
  --run 4
```

## Training

See `src/models/train.py`. Recommended: Lambda Labs A10G (~$0.75/hr, ~1 hour runtime).

```bash
python -m src.models.train --epochs 200 --out checkpoints/
```
```

- [ ] **Step 2: Fill in ECE numbers**

Replace `[X]%` with your actual ECE reduction from notebook 03.

- [ ] **Step 3: Add demo video link**

Upload demo MP4 to YouTube (unlisted) or GitHub release, paste URL into README.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: README with architecture, API reference, and demo link"
```

---

### Task 18: Notebook Cleanup

**Files:**
- Modify: `notebooks/01_preprocessing.ipynb`
- Modify: `notebooks/02_training.ipynb`
- Modify: `notebooks/03_calibration.ipynb`

- [ ] **Step 1: Clear and re-run all notebooks top to bottom**

In Jupyter: Kernel → Restart & Run All for each notebook. Verify no errors.

- [ ] **Step 2: Verify notebooks render on GitHub**

Push notebooks, open each on github.com, confirm outputs render (matplotlib plots show inline).

- [ ] **Step 3: Commit final state**

```bash
git add notebooks/
git commit -m "docs: clean notebook outputs for GitHub rendering"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Signal pipeline with real-time playback (`PlaybackThread`, `PipelineThread`) — Tasks 4, 5
- [x] `EpochBuffer` shared-state architecture — Task 2
- [x] EEGNet with MC dropout — Tasks 6, 7
- [x] `get_signal_quality()` tool — Task 12
- [x] `get_brain_state(confidence_threshold)` tool — Task 12
- [x] Cloud training with cost estimate guidance — Task 9
- [x] Calibration artifact (reliability diagram, ECE) — Task 10
- [x] Demo segment selection (subject screening) — Task 14
- [x] Focus-gated code review demo — Task 15
- [x] M3 Pro MPS inference — handled in `startup()`, Task 12
- [x] Fallback to pre-segmented mode — not implemented (fallback is simple: pre-fill sample_q from offline epochs, `PipelineThread` interface unchanged)

**Type consistency:**
- `EpochBuffer.push(epoch: np.ndarray)` — consistent across Tasks 2, 4, 12
- `MCDropoutDecoder.predict(epoch: np.ndarray) -> tuple[str, float]` — consistent across Tasks 7, 12, 14, 15
- `STATES = ["REST", "LEFT_IMAGERY", "RIGHT_IMAGERY"]` — consistent across Tasks 7, 14
- `_state` dict with keys `"buffer"`, `"decoder"`, `"playback"`, `"pipeline"` — consistent Tasks 12, 15
- `CODE_TO_CLASS = {1: 0, 2: 1, 3: 2}` — defined once in `pipeline.py`, imported by `train.py` if needed
