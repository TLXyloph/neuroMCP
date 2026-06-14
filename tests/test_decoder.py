import numpy as np
import torch
import pytest
from src.models.eegnet import EEGNet


def test_eegnet_output_shape():
    model = EEGNet(n_channels=64, n_times=160, n_classes=3)
    x = torch.randn(4, 64, 160)  # batch=4
    out = model(x)
    assert out.shape == (4, 3)


def test_eegnet_single_sample():
    model = EEGNet(n_channels=64, n_times=160, n_classes=3)
    x = torch.randn(1, 64, 160)
    out = model(x)
    assert out.shape == (1, 3)


def test_eegnet_output_is_logits_not_probs():
    """Model should output raw logits — softmax applied externally."""
    model = EEGNet()
    x = torch.randn(2, 64, 160)
    out = model(x)
    # Logits can be any real number; softmax output would sum to 1
    assert not torch.allclose(out.softmax(dim=-1).sum(dim=-1), out.sum(dim=-1))


def test_eegnet_dropout_differs_in_train_mode():
    """With dropout active, two forward passes should differ."""
    model = EEGNet()
    model.train()
    x = torch.randn(1, 64, 160)
    out1 = model(x)
    out2 = model(x)
    assert not torch.allclose(out1, out2), "Dropout should make passes differ"


def test_eegnet_deterministic_in_eval_mode():
    model = EEGNet()
    model.eval()
    x = torch.randn(1, 64, 160)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)


from unittest.mock import patch
import os
from src.models.mc_dropout import MCDropoutDecoder, STATES


def _save_dummy_checkpoint(path: str) -> None:
    model = EEGNet()
    torch.save(model.state_dict(), path)


def test_mc_dropout_states_constant():
    assert STATES == ["REST", "LEFT_IMAGERY", "RIGHT_IMAGERY"]


def test_mc_dropout_returns_valid_state(tmp_path):
    ckpt = str(tmp_path / "model.pt")
    _save_dummy_checkpoint(ckpt)
    decoder = MCDropoutDecoder(ckpt, n_passes=5, device="cpu")
    epoch = np.random.randn(64, 160).astype(np.float32)
    state, confidence = decoder.predict(epoch)
    assert state in STATES + ["LOW_CONFIDENCE"]


def test_mc_dropout_confidence_in_range(tmp_path):
    ckpt = str(tmp_path / "model.pt")
    _save_dummy_checkpoint(ckpt)
    decoder = MCDropoutDecoder(ckpt, n_passes=5, device="cpu")
    epoch = np.random.randn(64, 160).astype(np.float32)
    _, confidence = decoder.predict(epoch)
    assert 0.0 <= confidence <= 1.0


def test_mc_dropout_runs_n_passes(tmp_path):
    """Verify N stochastic passes actually differ (dropout is active)."""
    ckpt = str(tmp_path / "model.pt")
    _save_dummy_checkpoint(ckpt)
    decoder = MCDropoutDecoder(ckpt, n_passes=10, device="cpu")

    epoch = np.random.randn(64, 160).astype(np.float32)
    x = torch.tensor(epoch).unsqueeze(0)

    passes = []
    for _ in range(10):
        with torch.no_grad():
            passes.append(decoder._model(x).numpy())

    # Not all passes should be identical (dropout is on)
    all_same = all(np.allclose(passes[0], p) for p in passes[1:])
    assert not all_same
