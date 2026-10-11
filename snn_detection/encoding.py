"""Image preprocessing and image-to-spike encoders.

Spike trains are returned as integer arrays of shape ``(n_spikes, 2)`` holding
``[pixel_index, time_step]`` pairs, which is the format the event-driven
single-layer and latency models consume.
"""

import numpy as np


def rate_encode(image, duration=350, max_rate=60.0, min_rate=4.0, max_intensity=255.0):
    """Rate-code an image: brighter pixels emit more spikes.

    Each non-zero pixel fires ``int(rate / 1000 * duration)`` spikes at uniformly
    random time steps, where ``rate = pixel / max_intensity * max_rate + min_rate``
    (in Hz, one time step = 1 ms). The defaults match the single-layer models;
    the encoding demo uses ``max_rate=64, min_rate=0``.
    """
    rates = image.flatten() / max_intensity * max_rate
    rates[rates > 0] += min_rate
    spikes = []
    for pixel, rate in enumerate(rates):
        n_spikes = int(rate / 1000 * duration) if rate > 0 else 0
        if n_spikes > 0:
            for t in np.random.randint(0, duration, size=n_spikes):
                spikes.append([pixel, t])
    return np.array(spikes, dtype=int).reshape(-1, 2)


def latency_encode(image, max_time=100):
    """Time-to-first-spike coding: each pixel fires once, brighter pixels earlier.

    Expects uint8-range intensities. Every pixel below 255 emits a spike, so the
    black background fires at the last time step ``max_time - 1``. Spikes are
    sorted by time.
    """
    inverted = 255 - image.flatten()
    times = np.floor(inverted / 255.0 * (max_time - 1)).astype(int)
    pixels = np.flatnonzero(inverted > 0)
    spikes = np.stack([pixels, times[pixels]], axis=1)
    return spikes[np.argsort(spikes[:, 1], kind="stable")]


def dog_filter(image, sigma_center=1.0, sigma_surround=2.0):
    """Difference-of-Gaussians retina model (ON-center cells), scaled to ``[0, 1]``."""
    import cv2

    img = image.astype(np.float32)
    center = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma_center)
    surround = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma_surround)
    dog = np.maximum(center - surround, 0)
    if dog.max() > 0:
        dog = dog / dog.max()
    return dog
