"""Bio-inspired spiking neural networks (SNNs) for MNIST digit recognition and detection.

The core models are written in plain NumPy. PyTorch/torchvision are only used to
download MNIST and for the comparison baselines in :mod:`snn_detection.baselines`.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
WEIGHTS_DIR = REPO_ROOT / "weights"
OUTPUT_DIR = REPO_ROOT / "outputs"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
