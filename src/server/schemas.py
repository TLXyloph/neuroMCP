from pydantic import BaseModel


class SignalQualityResponse(BaseModel):
    snr: float
    artifact_ratio: float
    epoch_count: int


class BrainStateResponse(BaseModel):
    state: str  # LEFT_IMAGERY | RIGHT_IMAGERY | REST | LOW_CONFIDENCE
    confidence: float
    timestamp: str
