"""Exploratory hierarchical (HMAX-like) SNN driven by latency-coded input.

Architecture:

1. C1 - convolutional layer of adaptive LIF neurons (Gabor-initialised 5x5
   filters) with WTA lateral inhibition between filters at each location,
   trained with STDP.
2. S1 - untrained max-pooling over spike events (spatial invariance).
3. Output - fully connected adaptive LIF layer trained with STDP; neurons are
   labelled afterwards by the digit they respond to most.

This is a slow, loop-based reference simulation kept for documentation of the
latency-coding experiments; it was only evaluated on a handful of images.
"""

import numpy as np

from snn_detection.neurons import AdaptiveLIF, stdp_dw


def gabor_filters(n_filters, size, sigma=2.0, wavelength=5.0, gamma=0.5, psi=0.0):
    """``n_filters`` Gabor kernels at evenly spaced orientations, each scaled to [0, 1]."""
    filters = np.zeros((n_filters, size * size))
    c = size // 2
    x, y = np.meshgrid(np.arange(-c, c + 1), np.arange(-c, c + 1))
    for i, theta in enumerate(np.linspace(0, np.pi, n_filters, endpoint=False)):
        x_t = x * np.cos(theta) + y * np.sin(theta)
        y_t = -x * np.sin(theta) + y * np.cos(theta)
        g = np.exp(-.5 * (x_t ** 2 + gamma ** 2 * y_t ** 2) / sigma ** 2) * np.cos(2 * np.pi * x_t / wavelength + psi)
        filters[i] = ((g - g.min()) / (g.max() - g.min())).flatten()
    return filters


def _bucket_by_time(spike_train, time_steps):
    buckets = [[] for _ in range(time_steps)]
    for i, t in np.asarray(spike_train).reshape(-1, 2):
        if t < time_steps:
            buckets[t].append(i)
    return buckets


class ConvolutionalLayer:
    """C1 layer: per-location LIF columns with WTA, learning by STDP."""

    def __init__(self, input_shape, n_filters, filter_size, weight_scale=400.0):
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.output_dim = input_shape[0] - filter_size + 1
        self.weights = np.random.rand(n_filters, filter_size * filter_size) * 0.5 + 0.2
        self.weight_scale = weight_scale  # large because latency coding gives each pixel one spike
        self.neurons = [[AdaptiveLIF() for _ in range(self.output_dim ** 2)] for _ in range(n_filters)]
        self.last_pre_spike_time = np.zeros(input_shape[0] * input_shape[1])
        # For each output location: {global input index: index inside the filter}.
        self.rf_maps = {}
        for r in range(self.output_dim):
            for c in range(self.output_dim):
                self.rf_maps[r * self.output_dim + c] = {
                    i * input_shape[1] + j: (i - r) * filter_size + (j - c)
                    for i in range(r, r + filter_size) for j in range(c, c + filter_size)}

    def run(self, input_spikes, time_steps, training=True):
        """Returns output spikes as rows ``[filter, row, col, t]``."""
        for column in self.neurons:
            for neuron in column:
                neuron.reset()
        spikes_at_t = _bucket_by_time(input_spikes, time_steps)
        output = []
        for t in range(time_steps):
            for i in spikes_at_t[t]:
                self.last_pre_spike_time[i] = t
            for r in range(self.output_dim):
                for c in range(self.output_dim):
                    loc = r * self.output_dim + c
                    column = [self.neurons[f][loc] for f in range(self.n_filters)]
                    if any(n.refractory_time > 0 for n in column):
                        for n in column:
                            n.step(0)
                        continue

                    rf_map = self.rf_maps[loc]
                    local = [rf_map[i] for i in spikes_at_t[t] if i in rf_map]
                    potentials = []
                    for f, n in enumerate(column):
                        current = self.weights[f, local].sum() * self.weight_scale if local else 0
                        potentials.append(n.step(current)[1])

                    if max(potentials) >= column[0].v_th_base:
                        winner = int(np.argmax(potentials))
                        w = column[winner]
                        w.v = w.v_rest
                        w.refractory_time = w.refractory_period
                        w.v_th += w.theta_plus
                        output.append([winner, r, c, t])
                        if training:
                            for pre, idx in rf_map.items():
                                if self.last_pre_spike_time[pre] > 0:
                                    dw = stdp_dw(t - self.last_pre_spike_time[pre])
                                    self.weights[winner, idx] = np.clip(self.weights[winner, idx] + dw, 0.0, 1.0)
                        for f, n in enumerate(column):
                            if f != winner:
                                n.v = n.v_rest
        return np.array(output, dtype=int).reshape(-1, 4)


class FullyConnectedLayer:
    """Output layer of adaptive LIF neurons trained with STDP."""

    def __init__(self, n_input, n_neurons, weight_scale=60.0):
        self.n_input, self.n_neurons = n_input, n_neurons
        self.neurons = [AdaptiveLIF() for _ in range(n_neurons)]
        self.weights = np.random.rand(n_input, n_neurons)
        self.weight_scale = weight_scale
        self.last_pre_spike_time = np.zeros(n_input)

    def run(self, input_spikes, time_steps, training=True):
        """Returns the spike count of every output neuron."""
        for neuron in self.neurons:
            neuron.reset()
        counts = np.zeros(self.n_neurons)
        spikes_at_t = _bucket_by_time(input_spikes, time_steps)
        for t in range(time_steps):
            for i in spikes_at_t[t]:
                self.last_pre_spike_time[i] = t
            for j, neuron in enumerate(self.neurons):
                spike, _ = neuron.step(self.weights[spikes_at_t[t], j].sum() * self.weight_scale)
                if spike:
                    counts[j] += 1
                    if training:
                        seen = self.last_pre_spike_time > 0
                        dw = stdp_dw(t - self.last_pre_spike_time[seen])
                        self.weights[seen, j] = np.clip(self.weights[seen, j] + dw, 0.0, 1.0)
        return counts


def max_pool_spikes(input_spikes, input_shape, pool_size=2, stride=2):
    """S1 layer: max-pooling over C1 spike events ``[filter, row, col, t]``.

    Returns ``(spikes as [neuron_index, t] rows, output_shape)``; at most one spike
    per pooled neuron and time step is emitted.
    """
    n_filters = input_shape[0]
    out_dim = (input_shape[1] - pool_size) // stride + 1
    seen, output = set(), []
    for f, r, c, t in np.asarray(input_spikes).reshape(-1, 4):
        key = (f, r // stride, c // stride, t)
        if key not in seen:
            seen.add(key)
            output.append([f * out_dim ** 2 + (r // stride) * out_dim + c // stride, t])
    return np.array(output, dtype=int).reshape(-1, 2), (n_filters, out_dim, out_dim)
