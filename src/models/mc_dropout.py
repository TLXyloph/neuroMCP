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
