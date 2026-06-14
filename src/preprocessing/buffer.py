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
