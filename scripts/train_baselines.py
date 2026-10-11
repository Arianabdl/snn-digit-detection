"""Train the gradient-based comparison models (CNN, snnTorch) and report MNIST test accuracy.

Checkpoints are written to ``checkpoints/<model>.pt`` and reused by notebook 03.

Example:
    python scripts/train_baselines.py --model cnn
    python scripts/train_baselines.py --model snntorch --epochs 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from snn_detection import CHECKPOINT_DIR
from snn_detection.baselines import MODELS, predict_proba, train
from snn_detection.data import load_mnist


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x_train, y_train, x_test, y_test = load_mnist(normalize=False)
    spec = MODELS[args.model]
    print(f"Training {args.model} on {args.device} for {args.epochs} epochs...")
    model = train(spec["cls"](), x_train, y_train, epochs=args.epochs, lr=spec["lr"],
                  batch_size=spec["batch_size"], device=args.device)

    predictions = predict_proba(model, x_test, args.device).argmax(axis=1)
    print(f"Test accuracy, first 1000 images: {100 * np.mean(predictions[:1000] == y_test[:1000]):.2f}%")
    print(f"Test accuracy, all 10000 images:  {100 * np.mean(predictions == y_test):.2f}%")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    path = CHECKPOINT_DIR / f"{args.model}.pt"
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint to {path}")


if __name__ == "__main__":
    main()
