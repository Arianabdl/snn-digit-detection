"""Simulate one adaptive LIF neuron under a step current and plot potential, threshold and spikes.

Shows integration, firing, threshold adaptation after each spike (homeostasis),
threshold decay back to baseline and the refractory period.

Example:
    python scripts/demo_lif_neuron.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snn_detection import OUTPUT_DIR
from snn_detection.neurons import AdaptiveLIF


def main():
    duration = 500  # ms, dt = 1 ms
    current = np.zeros(duration)
    current[20:120] = 40.0

    neuron = AdaptiveLIF()
    potential, threshold, spike_times = [], [], []
    for t in range(duration):
        spike, v = neuron.step(current[t])
        potential.append(v)
        threshold.append(neuron.v_th)
        if spike:
            spike_times.append(t)
            print(f"spike at t = {t} ms, new threshold = {neuron.v_th:.2f}")

    time = np.arange(duration)
    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax1.plot(time, potential, color="b", label="Membrane potential", zorder=10)
    ax1.plot(time, threshold, color="r", linestyle="--", label="Adaptive threshold ($V_{th}$)")
    for i, t in enumerate(spike_times):
        ax1.axvline(t, color="orange", linestyle="--", label="Spike" if i == 0 else None)
    ax1.set_xlabel("Time (ms)", fontsize=12)
    ax1.set_ylabel("Membrane potential (mV)", color="b", fontsize=12)
    ax1.set_ylim(-5, max(threshold) * 1.2)
    ax1.grid(True, linestyle=":")
    ax2 = ax1.twinx()
    ax2.plot(time, current, color="g", linestyle=":", label="Input current (I)")
    ax2.set_ylabel("Input current", color="g", fontsize=12)
    ax2.set_ylim(-5, current.max() * 1.5)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.suptitle("LIF neuron response with adaptive threshold", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_DIR / "lif_neuron.png", dpi=150)
    print(f"Saved plot to {OUTPUT_DIR / 'lif_neuron.png'}")


if __name__ == "__main__":
    main()
