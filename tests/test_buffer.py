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
