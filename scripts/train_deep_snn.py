"""Train the two-layer SNN (STDP encoder + R-STDP classifier) on MNIST and save its weights.

Example:
    python scripts/train_deep_snn.py                      # 1k STDP + 10k R-STDP images
    python scripts/train_deep_snn.py --n-rstdp 5000 --seed 1
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import numpy as np

from snn_detection import OUTPUT_DIR, WEIGHTS_DIR
from snn_detection.data import load_mnist
from snn_detection.deep_snn import evaluate, save_weights, train_deep_snn
from snn_detection.visualization import plot_filters


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-stdp", type=int, default=1000, help="images for unsupervised STDP (phase 1)")
    parser.add_argument("--n-rstdp", type=int, default=10000, help="images for R-STDP (phase 2)")
    parser.add_argument("--n-test", type=int, default=1000, help="test images to evaluate on (max 10000)")
    parser.add_argument("--seed", type=int, default=0, help="NumPy random seed")
    parser.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR)
    parser.add_argument("--no-save", action="store_true", help="do not overwrite the saved weights")
    args = parser.parse_args()

    np.random.seed(args.seed)
    x_train, y_train, x_test, y_test = load_mnist()

    start = time.time()
    encoder, classifier, _ = train_deep_snn(x_train, y_train, n_stdp=args.n_stdp, n_rstdp=args.n_rstdp)
    print(f"Training finished in {time.time() - start:.1f}s")

    accuracy = evaluate(encoder, classifier, x_test[:args.n_test], y_test[:args.n_test])
    print(f"Test accuracy on {args.n_test} unseen images: {accuracy * 100:.2f}%")

    if not args.no_save:
        save_weights(encoder, classifier, args.weights_dir)
        print(f"Saved weights to {args.weights_dir}")
    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_filters(encoder.weights).savefig(OUTPUT_DIR / "filters.png", dpi=150)
    print(f"Saved filter plot to {OUTPUT_DIR / 'filters.png'}")


if __name__ == "__main__":
    main()
