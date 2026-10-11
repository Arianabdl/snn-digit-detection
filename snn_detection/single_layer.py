"""Single-layer, event-driven LIF network (784 inputs -> N neurons), as in thesis sections 4.3-4.4.

Inputs are rate-coded spike trains simulated for 350 time steps (1 ms each). The
output neurons compete through winner-take-all inhibition and have adaptive
thresholds. Two training rules are supported:

* ``"stdp"`` - unsupervised STDP; neurons are labelled afterwards by the digit
  they respond to most (:func:`assign_labels`).
* ``"rstdp"`` - reward-modulated STDP; each neuron has a fixed target digit and
  the update is multiplied by +1 / -1 depending on whether the winner's target
  matches the label.

Both rules renormalise the winner's incoming weights after every update.
This is a vectorized port of the original per-neuron/per-synapse loops; it
follows the same update order and produces the same results up to
floating-point summation order.
"""

import numpy as np

from snn_detection.neurons import stdp_dw


class SingleLayerSNN:
    def __init__(self, n_input=784, n_neurons=100, initial_weights=None, neuron_labels=None,
                 weight_scale=15.0, base_threshold=35.0, tau=10.0, resistance=1.0,
                 refractory_period=3, theta_plus=0.05, tau_theta=1e7, w_min=0.0, w_max=1.0,
                 a_plus=0.1, a_minus=0.1, tau_plus=5.0, tau_minus=5.0):
        self.n_input, self.n_neurons = n_input, n_neurons
        if initial_weights is None:
            initial_weights = np.random.rand(n_input, n_neurons) * 0.5 + 0.2
        self.weights = np.array(initial_weights, dtype=float)
        # Digit each neuron votes for (-1 = unassigned).
        self.neuron_labels = None if neuron_labels is None else np.asarray(neuron_labels)
        self.weight_scale = weight_scale
        self.base_threshold = base_threshold
        self.thresholds = np.full(n_neurons, base_threshold)
        self.tau, self.resistance = tau, resistance
        self.refractory_period = refractory_period
        self.theta_plus, self.tau_theta = theta_plus, tau_theta
        self.w_min, self.w_max = w_min, w_max
        self.stdp_params = dict(a_plus=a_plus, a_minus=a_minus, tau_plus=tau_plus, tau_minus=tau_minus)
        self.norm_factor = n_input / 4
        # Not reset between images (as in the original implementation), so the
        # first spikes of an image can pair with inputs from the previous one.
        self.last_pre_spike_time = np.zeros(n_input)

    def run(self, spike_train, duration=350, rule=None, label=None):
        """Simulate one image and return the spike count of every output neuron.

        ``rule`` is ``None`` (inference), ``"stdp"`` or ``"rstdp"`` (needs ``label``).
        """
        if rule == "rstdp" and (label is None or self.neuron_labels is None):
            raise ValueError("R-STDP needs a label and neuron target labels")
        learn = rule is not None
        v = np.zeros(self.n_neurons)
        refractory = np.zeros(self.n_neurons, dtype=int)
        counts = np.zeros(self.n_neurons)

        spikes_at_t = [[] for _ in range(duration)]
        for pixel, t in np.asarray(spike_train).reshape(-1, 2):
            if t < duration:
                spikes_at_t[t].append(pixel)

        for t in range(duration):
            inputs = spikes_at_t[t]
            self.last_pre_spike_time[inputs] = t

            resting = refractory > 0
            refractory[resting] -= 1
            v[resting] = 0.0
            current = self.weights[inputs].sum(axis=0) * self.weight_scale
            active = ~resting
            v[active] += (-v[active] + self.resistance * current[active]) / self.tau
            potentials = np.where(active, v, 0.0)

            candidates = potentials >= self.thresholds
            if candidates.any():
                winner = int(np.argmax(np.where(candidates, potentials, -np.inf)))
                refractory[winner] = self.refractory_period
                counts[winner] += 1
                others = np.arange(self.n_neurons) != winner
                v[others] = 0.0  # lateral inhibition
                if learn:
                    reward = 1.0
                    if rule == "rstdp":
                        reward = 1.0 if self.neuron_labels[winner] == label else -1.0
                    self._update_winner(winner, t, reward)

            if learn:
                self.thresholds += (self.base_threshold - self.thresholds) / self.tau_theta
        return counts

    def _update_winner(self, winner, t, reward):
        self.thresholds[winner] += self.theta_plus
        seen = self.last_pre_spike_time > 0
        dw = stdp_dw(t - self.last_pre_spike_time[seen], **self.stdp_params)
        column = self.weights[seen, winner] + dw * reward
        self.weights[seen, winner] = np.clip(column, self.w_min, self.w_max)
        total = self.weights[:, winner].sum()
        if total > 0:
            self.weights[:, winner] *= self.norm_factor / total

    def predict(self, spike_train, duration=350):
        """Label of the most active neuron, or -1 if nothing fired / it is unassigned."""
        counts = self.run(spike_train, duration)
        if counts.sum() == 0:
            return -1
        return int(self.neuron_labels[int(np.argmax(counts))])

    def save(self, path):
        np.savez(path, weights=self.weights, thresholds=self.thresholds, neuron_labels=self.neuron_labels)

    @classmethod
    def load(cls, path):
        state = np.load(path)
        net = cls(n_input=state["weights"].shape[0], n_neurons=state["weights"].shape[1],
                  initial_weights=state["weights"], neuron_labels=state["neuron_labels"])
        net.thresholds = state["thresholds"]
        return net


def smart_init_random(images, n_neurons=100, noise=0.01):
    """Initialise each neuron with a random training image plus a little noise."""
    indices = np.random.choice(len(images), n_neurons, replace=False)
    weights = np.zeros((images[0].size, n_neurons))
    for i in range(n_neurons):
        weights[:, i] = images[indices[i]].flatten() / 255.0 + np.random.rand(images[0].size) * noise
    return weights


def smart_init_targeted(images, labels, n_neurons=100, noise=0.01):
    """Assign ``n_neurons / 10`` neurons to each digit and initialise them from images of that digit.

    Returns ``(weights, neuron_labels)``.
    """
    per_digit = n_neurons // 10
    neuron_labels = np.floor(np.arange(n_neurons) / per_digit).astype(int)
    by_digit = [np.where(labels == d)[0] for d in range(10)]
    weights = np.zeros((images[0].size, n_neurons))
    for i in range(n_neurons):
        sample = np.random.choice(by_digit[neuron_labels[i]])
        weights[:, i] = images[sample].flatten() / 255.0 + np.random.rand(images[0].size) * noise
    return weights, neuron_labels


def assign_labels(network, images, labels, encode, duration=350):
    """Label each neuron with the digit for which it most often wins (-1 if it never wins)."""
    votes = np.zeros((network.n_neurons, 10), dtype=int)
    for image, label in zip(images, labels):
        counts = network.run(encode(image), duration)
        if counts.sum() > 0:
            votes[int(np.argmax(counts)), label] += 1
    return np.where(votes.sum(axis=1) > 0, votes.argmax(axis=1), -1)


def group_purity(counts, n_classes=10):
    """For R-STDP networks with contiguous digit groups: ``(winning_digit, purity)``.

    Purity is the share of all output spikes produced by the most active group.
    """
    group_counts = counts.reshape(n_classes, -1).sum(axis=1)
    digit = int(np.argmax(group_counts))
    return digit, group_counts[digit] / counts.sum()
