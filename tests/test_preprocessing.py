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
    # DC should be strongly attenuated. sosfilt is causal, so the first ~200
    # samples carry a startup transient; check steady-state attenuation.
    assert np.abs(filtered[:, 200:]).mean() < 1e-9


def test_reject_artifacts_keeps_clean_epochs():
    """Epochs with amplitude below threshold should not be rejected."""
    X = np.ones((10, 64, 160)) * 10e-6  # 10 µV — well below 100 µV
    y = np.zeros(10, dtype=int)
    X_clean, y_clean = reject_artifacts(X, y, peak_to_peak_uv=100.0)
    assert len(X_clean) == 10


def test_reject_artifacts_drops_bad_epochs():
    """Epochs with any channel exceeding threshold should be dropped."""
    X = np.ones((5, 64, 160)) * 10e-6
    X[2, 0, 0] = 200e-6  # epoch 2, channel 0: single-sample 200 µV spike
                          # (peak-to-peak excursion, not a DC offset)
    y = np.arange(5)
    X_clean, y_clean = reject_artifacts(X, y, peak_to_peak_uv=100.0)
    assert len(X_clean) == 4
    assert 2 not in y_clean  # epoch 2 was dropped


def test_code_to_class_mapping():
    assert CODE_TO_CLASS[1] == 0  # T0 → REST
    assert CODE_TO_CLASS[2] == 1  # T1 → LEFT
    assert CODE_TO_CLASS[3] == 2  # T2 → RIGHT


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
