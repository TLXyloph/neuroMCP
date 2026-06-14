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
