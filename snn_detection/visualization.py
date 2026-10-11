"""Matplotlib helpers for filters, receptive fields and detection results."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_filters(weights, kernel_size=5, ncols=8, title="Learned filters (STDP encoder)", cmap="gray"):
    """Plot each row of ``weights`` as a ``kernel_size x kernel_size`` image."""
    nrows = int(np.ceil(len(weights) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2.4 * nrows + 0.4))
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(np.ravel(axes)):
        if i < len(weights):
            ax.imshow(weights[i].reshape(kernel_size, kernel_size), cmap=cmap)
            ax.set_title(f"Filter {i}", fontsize=12)
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_receptive_fields(weights, titles=None, side=28, ncols=10, title="Receptive fields", cmap="hot"):
    """Plot each column of an ``(n_inputs, n_neurons)`` weight matrix as an image."""
    n_neurons = weights.shape[1]
    nrows = int(np.ceil(n_neurons / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.2 * ncols, 1.2 * nrows + 0.6))
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(np.ravel(axes)):
        if i < n_neurons:
            ax.imshow(weights[:, i].reshape(side, side), cmap=cmap)
            if titles is not None:
                ax.set_title(str(titles[i]), fontsize=8)
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def draw_detections(ax, canvas, detections, ground_truths=(), title=None, fontsize=10):
    """Show ``canvas`` with ground truths in green and detections in red."""
    ax.imshow(canvas, cmap="gray")
    for gt in ground_truths:
        x1, y1, x2, y2 = gt["bbox"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor="lime", facecolor="none"))
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.5, edgecolor="red", facecolor="none"))
        ax.text(min(x2 + 2, canvas.shape[1] - 2), (y1 + y2) / 2, f"{det['label']} ({det['score']:.3g})",
                color="red", fontsize=fontsize, weight="bold", va="center")
    if title:
        ax.set_title(title)
    ax.axis("off")
