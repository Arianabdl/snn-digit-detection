"""Two-layer convolutional SNN: unsupervised STDP encoder + R-STDP readout (pure NumPy).

This is the final model of the project:

1. :class:`STDPEncoder` - 16 convolutional 5x5 filters learned without labels.
   Each image location is a cortical column whose neurons compete through
   winner-take-all (WTA) lateral inhibition, so at most one filter spikes per
   location. Weights are kept at a constant total "energy" (homeostasis).
2. :class:`RSTDPClassifier` - 10 output neurons fully connected to the
   16 x 24 x 24 encoder spike map, trained with reward-modulated STDP: a global
   +1/-1 reward (dopamine-like signal) gates the Hebbian update.

Both layers are evaluated as a single integrate-and-fire step per image, which
keeps training fast (seconds for 10k images).
"""

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def extract_patches(image, k=5):
    """Return all ``k x k`` patches (stride 1) as rows of a ``(n_patches, k*k)`` array."""
    return sliding_window_view(image, (k, k)).reshape(-1, k * k)


class STDPEncoder:
    """Convolutional layer trained with unsupervised STDP and WTA inhibition."""

    def __init__(self, n_filters=16, kernel_size=5, threshold=2.0, filter_energy=5.0,
                 a_plus=0.05, tau_plus=5.0, decay=0.001):
        self.k = kernel_size
        self.threshold = threshold
        self.filter_energy = filter_energy
        self.a_plus = a_plus
        self.tau_plus = tau_plus
        self.decay = decay
        self.weights = np.random.rand(n_filters, kernel_size ** 2) + 0.1
        self.normalize_weights()

    def normalize_weights(self):
        """Homeostasis: rescale each filter to the same total weight, then clip to [0, 1]."""
        w_sum = self.weights.sum(axis=1, keepdims=True)
        w_sum[w_sum == 0] = 1.0
        self.weights = np.clip(self.weights / w_sum * self.filter_energy, 0.0, 1.0)

    def stdp_update(self, spikes, patches):
        """Vectorized STDP.

        Every (input pixel, winning filter) coincidence is treated as a causal
        pre-before-post pair with a fixed ``delta_t = 2`` ms, so LTP is
        ``a_plus * exp(-2 / tau_plus)`` scaled by the normalized coincidence count.
        A small constant decay stands in for LTD on uncorrelated inputs.
        """
        dw_ltp = self.a_plus * np.exp(-2.0 / self.tau_plus)
        coincidences = spikes @ patches
        self.weights += dw_ltp * (coincidences / (coincidences.max() + 1e-6))
        self.weights -= self.decay
        self.weights = np.clip(self.weights, 0.0, 1.0)

    def forward(self, image, learn=False):
        """Return the binary spike map of shape ``(n_filters, n_locations)``."""
        patches = extract_patches(image, self.k)
        voltages = self.weights @ patches.T
        is_winner = (voltages == voltages.max(axis=0)) & (voltages >= self.threshold)
        spikes = is_winner.astype(float)
        if learn:
            self.stdp_update(spikes, patches)
            self.normalize_weights()
        return spikes


class RSTDPClassifier:
    """Fully connected readout trained with reward-modulated STDP."""

    def __init__(self, input_size, n_classes=10, a_plus=0.1, tau_plus=5.0,
                 w_min=-1.0, w_max=1.0):
        self.a_plus = a_plus
        self.tau_plus = tau_plus
        self.w_min = w_min  # negative weights act as inhibition
        self.w_max = w_max
        self.weights = np.random.rand(n_classes, input_size) * 0.1

    def rstdp_update(self, x, neuron, reward):
        """``w += reward * STDP`` on the synapses of ``neuron`` whose input spiked."""
        dw = self.a_plus * np.exp(-2.0 / self.tau_plus)
        row = self.weights[neuron]
        row[x > 0] += dw * reward
        self.weights[neuron] = np.clip(row, self.w_min, self.w_max)

    def train_step(self, x, label):
        """One R-STDP step. Returns 1 if the prediction was correct, else 0.

        The winning neuron gets reward +1 (correct) or -1 (punishment). On a
        mistake the correct neuron is additionally rewarded, which speeds up
        convergence.
        """
        prediction = int(np.argmax(self.weights @ x))
        correct = prediction == label
        self.rstdp_update(x, prediction, 1.0 if correct else -1.0)
        if not correct:
            self.rstdp_update(x, label, 1.0)
        return int(correct)

    def voltages(self, x):
        return self.weights @ x

    def predict(self, x):
        return int(np.argmax(self.weights @ x))


def train_deep_snn(x_train, y_train, n_stdp=1000, n_rstdp=10000, n_filters=16,
                   log_every=1000, verbose=True):
    """Phase 1: unsupervised STDP on ``n_stdp`` images. Phase 2: R-STDP on ``n_rstdp`` images.

    Returns ``(encoder, classifier, history)`` where ``history`` holds a 0/1
    correctness flag per R-STDP step.
    """
    encoder = STDPEncoder(n_filters=n_filters)
    n_locations = (x_train.shape[1] - encoder.k + 1) * (x_train.shape[2] - encoder.k + 1)
    classifier = RSTDPClassifier(input_size=n_filters * n_locations)

    if verbose:
        print(f"Phase 1: unsupervised STDP on {n_stdp} images...")
    for i in range(n_stdp):
        encoder.forward(x_train[i], learn=True)

    if verbose:
        print(f"Phase 2: supervised R-STDP on {n_rstdp} images (encoder frozen)...")
    history = []
    for i in range(n_rstdp):
        spikes = encoder.forward(x_train[i])
        history.append(classifier.train_step(spikes.ravel(), y_train[i]))
        if verbose and log_every and i > 0 and i % log_every == 0:
            print(f"  step {i}/{n_rstdp} | accuracy on last 500: {np.mean(history[-500:]) * 100:.2f}%")
    return encoder, classifier, history


def predict(encoder, classifier, image):
    return classifier.predict(encoder.forward(image).ravel())


def evaluate(encoder, classifier, images, labels):
    """Classification accuracy in ``[0, 1]``."""
    predictions = [predict(encoder, classifier, img) for img in images]
    return float(np.mean(np.array(predictions) == np.asarray(labels)))


def save_weights(encoder, classifier, weights_dir):
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    np.save(weights_dir / "encoder_weights.npy", encoder.weights)
    np.save(weights_dir / "classifier_weights.npy", classifier.weights)


def load_weights(weights_dir):
    """Rebuild an encoder/classifier pair from ``encoder_weights.npy`` and ``classifier_weights.npy``."""
    weights_dir = Path(weights_dir)
    encoder_w = np.load(weights_dir / "encoder_weights.npy")
    classifier_w = np.load(weights_dir / "classifier_weights.npy")
    encoder = STDPEncoder(n_filters=encoder_w.shape[0], kernel_size=int(np.sqrt(encoder_w.shape[1])))
    encoder.weights = encoder_w
    classifier = RSTDPClassifier(input_size=classifier_w.shape[1], n_classes=classifier_w.shape[0])
    classifier.weights = classifier_w
    return encoder, classifier
