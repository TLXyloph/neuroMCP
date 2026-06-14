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
