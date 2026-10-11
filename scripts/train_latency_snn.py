"""Hierarchical latency-coded SNN (C1 conv -> S1 pooling -> output layer), an exploratory experiment.

By default the pretrained weights in ``weights/latency_snn.npz`` are loaded and the
network is only visualised and tested. ``--retrain`` trains from scratch, which is
slow because the simulation is written as plain Python loops (C1 on 1000 images
can take hours).

Example:
    python scripts/train_latency_snn.py              # load weights, test on 10 images
    python scripts/train_latency_snn.py --n-test 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from snn_detection import OUTPUT_DIR, WEIGHTS_DIR
from snn_detection.data import load_mnist
from snn_detection.encoding import latency_encode
from snn_detection.latency_snn import ConvolutionalLayer, FullyConnectedLayer, gabor_filters, max_pool_spikes

N_FILTERS, FILTER_SIZE, POOL_SIZE, POOL_STRIDE, N_OUTPUT = 4, 5, 2, 2, 100
SIM_DURATION = 100  # latency-coding window (time steps)


def plot_filters_row(weights, title, path, cmap):
    fig, axes = plt.subplots(1, len(weights), figsize=(12, 3))
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axes):
        ax.imshow(weights[i].reshape(FILTER_SIZE, FILTER_SIZE), cmap=cmap)
        ax.set_title(f"Filter {i}")
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--n-train-c1", type=int, default=1000)
    parser.add_argument("--n-train-out", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", type=Path, default=WEIGHTS_DIR / "latency_snn.npz")
    args = parser.parse_args()

    np.random.seed(args.seed)
    OUTPUT_DIR.mkdir(exist_ok=True)
    x_train, y_train, x_test, y_test = load_mnist(normalize=False)
    c1 = ConvolutionalLayer((28, 28), N_FILTERS, FILTER_SIZE)
    c1_shape = (N_FILTERS, c1.output_dim, c1.output_dim)
    s1_dim = (c1.output_dim - POOL_SIZE) // POOL_STRIDE + 1
    out = FullyConnectedLayer(N_FILTERS * s1_dim * s1_dim, N_OUTPUT)

    def s1_spikes(image):
        c1_spikes = c1.run(latency_encode(image, SIM_DURATION), SIM_DURATION, training=False)
        return max_pool_spikes(c1_spikes, c1_shape, POOL_SIZE, POOL_STRIDE)[0]

    if args.weights.exists() and not args.retrain:
        state = np.load(args.weights)
        c1.weights, out.weights, neuron_labels = state["c1_weights"], state["output_weights"], state["neuron_labels"]
        print(f"Loaded weights from {args.weights}")
    else:
        c1.weights = gabor_filters(N_FILTERS, FILTER_SIZE)
        plot_filters_row(c1.weights, "Initial C1 receptive fields (Gabor)",
                         OUTPUT_DIR / "latency_initial_gabor.png", "gray")
        for image in tqdm(x_train[:args.n_train_c1], desc="Training C1"):
            c1.run(latency_encode(image, SIM_DURATION), SIM_DURATION, training=True)
        for image in tqdm(x_train[:args.n_train_out], desc="Training output layer"):
            spikes = s1_spikes(image)
            if len(spikes):
                out.run(spikes, SIM_DURATION, training=True)
        votes = np.zeros((N_OUTPUT, 10), dtype=int)
        for image, label in tqdm(list(zip(x_train[:args.n_train_out], y_train[:args.n_train_out])), desc="Labelling"):
            counts = out.run(s1_spikes(image), SIM_DURATION, training=False)
            if counts.sum() > 0:
                votes[np.argmax(counts), label] += 1
        neuron_labels = np.where(votes.sum(axis=1) > 0, votes.argmax(axis=1), -1)
        args.weights.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.weights, c1_weights=c1.weights, output_weights=out.weights, neuron_labels=neuron_labels)
        print(f"Saved weights to {args.weights}")

    plot_filters_row(c1.weights, "Final C1 receptive fields", OUTPUT_DIR / "latency_final_c1.png", "hot")

    # S1 activity for one test image
    pooled, s1_shape = max_pool_spikes(c1.run(latency_encode(x_test[0], SIM_DURATION), SIM_DURATION, training=False),
                                       c1_shape, POOL_SIZE, POOL_STRIDE)
    activity = np.bincount(pooled[:, 0], minlength=int(np.prod(s1_shape))).reshape(s1_shape)
    fig, axes = plt.subplots(1, N_FILTERS + 1, figsize=(15, 4))
    fig.suptitle(f"Test image (label {y_test[0]}) and S1 spike counts per filter", fontsize=16)
    axes[0].imshow(x_test[0], cmap="gray")
    axes[0].axis("off")
    for f in range(N_FILTERS):
        im = axes[f + 1].imshow(activity[f], cmap="hot", interpolation="nearest")
        axes[f + 1].set_title(f"S1, filter {f}")
        axes[f + 1].axis("off")
        fig.colorbar(im, ax=axes[f + 1])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "latency_s1_activity.png", dpi=150)
    plt.close(fig)

    correct = answered = 0
    for image, label in tqdm(list(zip(x_test[:args.n_test], y_test[:args.n_test])), desc="Testing"):
        counts = out.run(s1_spikes(image), SIM_DURATION, training=False)
        if counts.sum() > 0 and neuron_labels[np.argmax(counts)] != -1:
            answered += 1
            correct += neuron_labels[np.argmax(counts)] == label
    print(f"Answered {answered}/{args.n_test} test images; "
          f"accuracy on answered: {100 * correct / max(answered, 1):.2f}% ({correct}/{answered})")
    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
