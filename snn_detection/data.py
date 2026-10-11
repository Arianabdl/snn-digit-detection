"""MNIST loading. torchvision is only used to download and parse the raw files."""

import numpy as np

from snn_detection import DATA_DIR


def load_mnist(data_dir=DATA_DIR, normalize=True):
    """Return ``(x_train, y_train, x_test, y_test)`` as NumPy arrays.

    Images have shape ``(N, 28, 28)``. With ``normalize=True`` they are float64 in
    ``[0, 1]``; otherwise the original uint8 values in ``[0, 255]`` are returned.
    The dataset is downloaded to ``data_dir`` on first use (~55 MB).
    """
    from torchvision import datasets

    train = datasets.MNIST(str(data_dir), train=True, download=True)
    test = datasets.MNIST(str(data_dir), train=False, download=True)
    x_train, y_train = train.data.numpy(), train.targets.numpy()
    x_test, y_test = test.data.numpy(), test.targets.numpy()
    if normalize:
        x_train = x_train / 255.0
        x_test = x_test / 255.0
    return x_train, y_train, x_test, y_test
