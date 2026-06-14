"""Evaluate MC dropout accuracy per test subject. Run after training.

Usage:
    python scripts/screen_subjects.py --checkpoint checkpoints/eegnet.pt
"""
import argparse
import numpy as np
from src.preprocessing.pipeline import load_and_epoch_subject
from src.models.mc_dropout import MCDropoutDecoder, STATES


def evaluate_subject(decoder: MCDropoutDecoder, subject: int) -> float | None:
    try:
        X, y = load_and_epoch_subject(subject)
    except Exception as e:
        print(f"  Subject {subject}: SKIP ({e})")
        return None

    if len(X) == 0:
        print(f"  Subject {subject}: SKIP (no epochs survived rejection)")
        return None

    correct = 0
    for i in range(len(X)):
        state, confidence = decoder.predict(X[i].astype(np.float32))
        if confidence >= 0.5:
            pred = STATES.index(state) if state in STATES else -1
            if pred == y[i]:
                correct += 1

    acc = correct / len(X)
    print(f"  Subject {subject}: {acc:.2%} ({len(X)} epochs)")
    return acc, len(X)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--min-epochs", type=int, default=40,
        help="Minimum epochs for a subject to be eligible as a demo recommendation. "
             "Guards against tiny-sample subjects with noisy (lucky) accuracy.",
    )
    args = parser.parse_args()

    decoder = MCDropoutDecoder(args.checkpoint, n_passes=50, device="cpu")

    print("Evaluating test subjects 81-109...")
    results = []
    for subject in range(81, 110):
        out = evaluate_subject(decoder, subject)
        if out is not None:
            acc, n = out
            results.append((subject, acc, n))

    # Rank only subjects with enough epochs for the accuracy to be meaningful.
    eligible = [r for r in results if r[2] >= args.min_epochs]
    eligible.sort(key=lambda x: x[1], reverse=True)

    print(f"\n=== Top 5 subjects for demo (>= {args.min_epochs} epochs) ===")
    for subject, acc, n in eligible[:5]:
        print(f"  Subject {subject}: {acc:.2%} ({n} epochs)")

    excluded = sorted([r for r in results if r[2] < args.min_epochs],
                      key=lambda x: x[1], reverse=True)
    if excluded:
        print(f"\n(Excluded {len(excluded)} subjects with < {args.min_epochs} epochs — "
              "accuracy too noisy to trust, e.g. "
              + ", ".join(f"S{s} {a:.0%}/{n}ep" for s, a, n in excluded[:3]) + ")")

    if not eligible:
        print("\nNo subject meets the minimum-epoch bar; lower --min-epochs.")
        return

    best = eligible[0][0]
    print("\nRecommended demo recording: subject", best)
    print("Use run 4 (motor imagery left/right fist, first repeat)")
    print(f"  python -m src.server.server --checkpoint {args.checkpoint} "
          f"--subject {best} --run 4")


if __name__ == "__main__":
    main()
