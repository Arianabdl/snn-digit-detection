"""Train the single-layer LIF network (thesis sections 4.3-4.4) and report test accuracy.

Rules:
    stdp   unsupervised STDP, neurons initialised from random training images and
           labelled after training by majority vote  (thesis: 40.6%)
    rstdp  reward-modulated STDP, 10 neurons per digit initialised from images of
           that digit                                (thesis: 77.9%)

Example:
    python scripts/train_single_layer.py --rule rstdp                  # < 1 min
    python scripts/train_single_layer.py --rule stdp --n-train 10000   # ~5 min
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snn_detection import CHECKPOINT_DIR, OUTPUT_DIR
from snn_detection.data import load_mnist
from snn_detection.encoding import rate_encode
from snn_detection.single_layer import SingleLayerSNN, assign_labels, smart_init_random, smart_init_targeted
from snn_detection.visualization import plot_receptive_fields

DEFAULT_N_TRAIN = {"stdp": 10000, "rstdp": 1000}
DURATION = 350
RULE_NAMES = {"stdp": "STDP", "rstdp": "R-STDP"}


def plot_spike_distribution(counts, neuron_labels):
    """Total test spikes per neuron, grouped by assigned digit."""
    order = np.argsort(neuron_labels, kind="stable")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(range(len(counts)), counts[order], color="royalblue")
    for boundary in np.flatnonzero(np.diff(neuron_labels[order])) + 0.5:
        ax.axvline(boundary, color="red", linestyle="--", lw=1)
    ax.set_xlabel("Neurons grouped by assigned digit (-1 = never won)")
    ax.set_ylabel("Spikes during testing")
    ax.set_title("Output spikes per neuron")
    ax.set_xticks([])
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rule", choices=["stdp", "rstdp"], required=True)
    parser.add_argument("--n-train", type=int, help="training images (default: 10000 for stdp, 1000 for rstdp)")
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--n-neurons", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="checkpoint path (default: checkpoints/single_layer_<rule>.npz)")
    args = parser.parse_args()
    n_train = args.n_train or DEFAULT_N_TRAIN[args.rule]
    output = args.output or CHECKPOINT_DIR / f"single_layer_{args.rule}.npz"

    np.random.seed(args.seed)
    x_train, y_train, x_test, y_test = load_mnist(normalize=False)

    if args.rule == "stdp":
        net = SingleLayerSNN(n_neurons=args.n_neurons, initial_weights=smart_init_random(x_train, args.n_neurons))
    else:
        weights, targets = smart_init_targeted(x_train, y_train, args.n_neurons)
        net = SingleLayerSNN(n_neurons=args.n_neurons, initial_weights=weights, neuron_labels=targets)
    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_receptive_fields(net.weights, title="Initial receptive fields (smart initialisation)").savefig(
        OUTPUT_DIR / f"single_layer_{args.rule}_initial_rf.png", dpi=100)

    print(f"Training with {RULE_NAMES[args.rule]} on {n_train} images...")
    start = time.time()
    for i in range(n_train):
        net.run(rate_encode(x_train[i]), DURATION, rule=args.rule, label=y_train[i])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n_train} images, {time.time() - start:.0f}s")

    if args.rule == "stdp":
        print("Labelling neurons by their responses to the training images...")
        net.neuron_labels = assign_labels(net, x_train[:n_train], y_train[:n_train], rate_encode, DURATION)
    output.parent.mkdir(parents=True, exist_ok=True)
    net.save(output)
    print(f"Saved model to {output}")

    print(f"Testing on {args.n_test} images...")
    correct = np.zeros(10, dtype=int)
    total = np.zeros(10, dtype=int)
    responded = 0
    test_counts = np.zeros(net.n_neurons)
    for image, label in zip(x_test[:args.n_test], y_test[:args.n_test]):
        counts = net.run(rate_encode(image), DURATION)
        test_counts += counts
        total[label] += 1
        if counts.sum() > 0 and net.neuron_labels[np.argmax(counts)] != -1:
            responded += 1
            correct[label] += net.neuron_labels[np.argmax(counts)] == label

    print(f"\nAccuracy: {100 * correct.sum() / total.sum():.2f}% ({correct.sum()}/{total.sum()} test images)")
    if responded:
        print(f"Accuracy on the {responded} images that produced a labelled response: "
              f"{100 * correct.sum() / responded:.2f}%")
    for digit in range(10):
        print(f"  digit {digit}: {100 * correct[digit] / max(total[digit], 1):6.2f}% ({correct[digit]}/{total[digit]})")
    print(f"Dead neurons (no spike during testing): {int((test_counts == 0).sum())}/{net.n_neurons}")

    plot_receptive_fields(net.weights, titles=net.neuron_labels,
                          title=f"Final receptive fields ({RULE_NAMES[args.rule]}), titles = digit label").savefig(
        OUTPUT_DIR / f"single_layer_{args.rule}_final_rf.png", dpi=100)
    plot_spike_distribution(test_counts, net.neuron_labels).savefig(
        OUTPUT_DIR / f"single_layer_{args.rule}_spikes.png", dpi=100)
    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
