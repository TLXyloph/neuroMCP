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
