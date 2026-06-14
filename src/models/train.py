"""Train EEGNet on PhysioNet EEGBCI motor imagery data.

Usage (local smoke test):
    python -m src.models.train --subjects 1 2 --test-subjects 3 --epochs 3 --out checkpoints/

Usage (full cloud run):
    python -m src.models.train --subjects $(seq -s' ' 1 80) --test-subjects $(seq -s' ' 81 109) --epochs 200 --out checkpoints/
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from src.preprocessing.pipeline import load_and_epoch_subject
from src.models.eegnet import EEGNet


def load_dataset(subjects: list[int]) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for subj in subjects:
        try:
            X, y = load_and_epoch_subject(subj)
            if len(y) == 0:
                print(f"  Subject {subj}: SKIP (no epochs survived rejection)")
                continue
            Xs.append(X)
            ys.append(y)
            print(f"  Subject {subj}: {len(y)} epochs")
        except Exception as e:
            print(f"  Subject {subj}: SKIP ({e})")
    return np.concatenate(Xs), np.concatenate(ys)


def train(
    train_subjects: list[int],
    test_subjects: list[int],
    n_epochs: int,
    out_dir: str,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> None:
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    print("Loading training data...")
    X_train, y_train = load_dataset(train_subjects)
    print(f"Training set: {X_train.shape}")

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model = EEGNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(xb)
        print(f"Epoch {epoch}/{n_epochs} — loss: {total_loss/total:.4f}, acc: {correct/total:.3f}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "eegnet.pt")
    print(f"Checkpoint saved to {out / 'eegnet.pt'}")

    # Evaluate on test subjects
    if test_subjects:
        print("\nLoading test data...")
        X_test, y_test = load_dataset(test_subjects)
        model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            logits = model(x_t)
            preds = logits.argmax(1).cpu().numpy()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        acc = (preds == y_test).mean()
        print(f"Test accuracy: {acc:.3f}")

        # Save calibration data for notebooks
        np.save(out / "test_probs_baseline.npy", probs)
        np.save(out / "test_labels.npy", y_test)
        np.save(out / "X_test.npy", X_test)
        print(f"Calibration data saved to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 81)))
    parser.add_argument("--test-subjects", nargs="+", type=int, default=list(range(81, 110)))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--out", default="checkpoints")
    args = parser.parse_args()
    train(args.subjects, args.test_subjects, args.epochs, args.out)
