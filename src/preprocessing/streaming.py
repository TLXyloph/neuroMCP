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
