"""Visualise rate coding and latency (time-to-first-spike) coding of one MNIST digit.

Example:
    python scripts/demo_spike_encoding.py --index 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snn_detection import OUTPUT_DIR
from snn_detection.data import load_mnist
from snn_detection.encoding import latency_encode, rate_encode


def plot_encoding(image, label, spikes, duration, title):
    """Original image on top; raster plot and pixel intensities side by side below."""
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    ax_img = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax_raster = plt.subplot2grid((2, 2), (1, 0))
    ax_pixels = plt.subplot2grid((2, 2), (1, 1))

    ax_img.imshow(image, cmap="gray_r")
    ax_img.set_title(f"Original image (label: {label})")
    ax_img.axis("off")

    ax_raster.scatter(spikes[:, 1], spikes[:, 0], s=2, color="black", marker="|")
    ax_raster.set(title="Spike train (raster plot)", xlabel="Time (ms)", ylabel="Input neuron (pixel)",
                  xlim=(0, duration), ylim=(784, 0))

    ax_pixels.barh(np.arange(784), image.flatten(), height=1.0, color="darkgray")
    ax_pixels.set(title="Pixel intensity", xlabel="Intensity (0-255)", ylabel="Input neuron (pixel)",
                  xlim=(0, 255), ylim=(784, 0))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=int, default=0, help="index into the MNIST training set")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    x_train, y_train, _, _ = load_mnist(normalize=False)
    image, label = x_train[args.index], y_train[args.index]
    OUTPUT_DIR.mkdir(exist_ok=True)

    rate_spikes = rate_encode(image, duration=350, max_rate=64, min_rate=0)
    print(f"Rate coding: {len(rate_spikes)} spikes in 350 ms")
    plot_encoding(image, label, rate_spikes, 350, "Rate coding (350 ms window)").savefig(
        OUTPUT_DIR / "rate_coding.png", dpi=150)

    latency_spikes = latency_encode(image, max_time=100)
    print(f"Latency coding: {len(latency_spikes)} spikes in 100 ms (one per pixel)")
    plot_encoding(image, label, latency_spikes, 100, "Latency coding (100 ms window, brighter = earlier)").savefig(
        OUTPUT_DIR / "latency_coding.png", dpi=150)
    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
