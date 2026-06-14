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


def _utc_timestamp() -> str:
    """ISO-8601 UTC timestamp with a trailing 'Z' (e.g. 2026-06-13T20:00:00.000000Z)."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


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
    timestamp = _utc_timestamp()
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
