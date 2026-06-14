import numpy as np
from scipy.signal import butter, sosfilt
import mne
from mne.datasets import eegbci

# PhysioNet EEGBCI: runs 4, 8, 12 = motor imagery left/right fist
RUNS = [4, 8, 12]
# MNE includes both endpoints, so n_samples = round((TMAX-TMIN)*sfreq) + 1.
# TMAX = 159/160 yields exactly 160 samples at 160 Hz — matching the streaming
# path (PipelineThread.EPOCH_LEN) and EEGNet's default n_times so a model
# trained on batch epochs accepts streaming epochs unchanged.
TMIN, TMAX = 0.0, 159.0 / 160.0  # 160-sample (1-second) epochs
# Event code mapping: T0=rest(1), T1=left(2), T2=right(3)
CODE_TO_CLASS = {1: 0, 2: 1, 3: 2}


def bandpass_filter(
    data: np.ndarray, sfreq: float, l_freq: float = 8.0, h_freq: float = 30.0
) -> np.ndarray:
    """Apply 4th-order Butterworth bandpass filter. data: (n_channels, n_times)."""
    sos = butter(4, [l_freq, h_freq], btype="bandpass", fs=sfreq, output="sos")
    return sosfilt(sos, data, axis=1)


def reject_artifacts(
    X: np.ndarray, y: np.ndarray, peak_to_peak_uv: float = 200.0
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
    # 200 µV peak-to-peak: real motor-imagery epochs filtered to 8-30 Hz have a
    # median ptp of ~160 µV, so a 100 µV threshold rejects every epoch. 200 µV
    # keeps ~86% while still dropping genuine high-amplitude artifacts.
    epochs.drop_bad(reject=dict(eeg=200e-6), verbose=False)

    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    y_codes = epochs.events[:, 2]
    y = np.array([CODE_TO_CLASS[c] for c in y_codes], dtype=int)
    return X, y
