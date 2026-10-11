"""Fast unit tests (no dataset download needed). Run with: python -m pytest"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snn_detection.deep_snn import RSTDPClassifier, STDPEncoder, extract_patches, load_weights, save_weights
from snn_detection.detection import (detect_with, evaluate_detections, iou, make_fixed_canvas,
                                     make_random_canvas, non_max_suppression, sliding_windows)
from snn_detection.encoding import dog_filter, latency_encode, rate_encode
from snn_detection.neurons import AdaptiveLIF, stdp_dw
from snn_detection.single_layer import SingleLayerSNN, group_purity


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(0)


def test_extract_patches_matches_naive_loop():
    image = np.random.rand(28, 28)
    naive = np.array([image[i:i + 5, j:j + 5].flatten() for i in range(24) for j in range(24)])
    assert np.array_equal(extract_patches(image), naive)


def test_encoder_fires_at_most_one_filter_per_location():
    encoder = STDPEncoder()
    np.testing.assert_allclose(encoder.weights.sum(axis=1), 5.0)
    spikes = encoder.forward(np.random.rand(28, 28), learn=True)
    assert spikes.shape == (16, 576)
    assert spikes.sum(axis=0).max() <= 1
    assert encoder.weights.min() >= 0 and encoder.weights.max() <= 1


def test_rstdp_rewards_correct_and_punishes_wrong_neuron():
    clf = RSTDPClassifier(input_size=4, n_classes=2)
    clf.weights = np.array([[0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1]])
    x = np.array([1.0, 0.0, 1.0, 0.0])
    assert clf.train_step(x, label=1) == 0  # neuron 0 wins wrongly
    assert clf.weights[0, 0] < 0.5 and clf.weights[0, 1] == 0.5  # punished on active inputs only
    assert clf.weights[1, 0] > 0.1 and clf.weights[1, 1] == 0.1  # correct neuron rewarded


def test_save_and_load_weights_roundtrip(tmp_path):
    encoder, classifier = STDPEncoder(), RSTDPClassifier(16 * 576)
    save_weights(encoder, classifier, tmp_path)
    enc2, clf2 = load_weights(tmp_path)
    assert np.array_equal(enc2.weights, encoder.weights) and np.array_equal(clf2.weights, classifier.weights)


def test_rate_encode_spike_counts():
    image = np.zeros((28, 28), dtype=np.uint8)
    image[0, 0] = 255
    spikes = rate_encode(image, duration=350)
    assert len(spikes) == int(64 / 1000 * 350)  # (60 + 4) Hz for 350 ms
    assert set(spikes[:, 0]) == {0} and spikes[:, 1].max() < 350


def test_latency_encode_brighter_fires_earlier():
    image = np.zeros((28, 28), dtype=np.uint8)
    image[0, 0], image[0, 1] = 250, 100
    spikes = latency_encode(image, max_time=100)
    times = dict(map(tuple, spikes))
    assert times[0] < times[1] < times[2]  # pixel 2 is black background
    assert np.all(np.diff(spikes[:, 1]) >= 0)


def test_dog_filter_range():
    image = np.zeros((28, 28))
    image[10:18, 10:18] = 1.0
    out = dog_filter(image)
    assert out.min() >= 0 and np.isclose(out.max(), 1.0)


def test_adaptive_lif_threshold_rises_after_spikes():
    neuron = AdaptiveLIF()
    spikes = [neuron.step(40.0)[0] for _ in range(100)]
    assert sum(spikes) > 1
    assert neuron.v_th > neuron.v_th_base


def test_stdp_sign():
    dw = stdp_dw(np.array([2.0, -2.0, 0.0]))
    assert dw[0] > 0 and dw[1] < 0 and dw[2] == 0


def test_iou_and_nms():
    a, b, c = (0, 0, 10, 10), (5, 0, 15, 10), (50, 50, 60, 60)
    assert np.isclose(iou(a, a), 1.0) and np.isclose(iou(a, b), 1 / 3, atol=1e-6) and iou(a, c) == 0
    dets = [{"bbox": a, "label": 1, "score": 2.0}, {"bbox": b, "label": 1, "score": 3.0},
            {"bbox": c, "label": 2, "score": 1.0}]
    for metric in ("iou", "overlap"):
        kept = non_max_suppression(dets, threshold=0.3, metric=metric)
        assert [d["score"] for d in kept] == [3.0, 1.0]


def test_sliding_windows_cover_last_position():
    positions = [(x, y) for x, y, _ in sliding_windows(np.zeros((100, 100)), 28, 2)]
    assert (72, 72) in positions and len(positions) == 37 * 37


def test_canvases_and_evaluation():
    images = np.random.rand(20, 28, 28)
    labels = np.arange(20) % 10
    canvas, truths = make_fixed_canvas(images, labels, num_digits=3)
    assert canvas.shape == (100, 100) and len(truths) == 3
    canvas, truths = make_random_canvas((images * 255).astype(np.uint8), labels, num_digits=2)
    assert canvas.dtype == np.uint8 and len(truths) == 2

    perfect = [{"bbox": gt["bbox"], "label": gt["label"], "score": 1.0} for gt in truths]
    wrong = [{"bbox": (0, 0, 28, 28), "label": 99, "score": 1.0}]
    result = evaluate_detections(perfect + wrong, truths)
    assert result["found"] == [0, 1] and result["missed"] == [] and result["false_positives"] == [2]


def test_detect_with_skips_background():
    canvas = np.zeros((60, 60))
    canvas[10:38, 10:38] = 1.0
    dets = detect_with(canvas, lambda crop: (7, crop.mean()), stride=4, min_mean=0.5)
    assert dets and all(d["label"] == 7 for d in dets)
    assert all(d["score"] >= 0.5 for d in dets)  # background windows were skipped


def test_single_layer_network_runs_and_learns(tmp_path):
    labels = np.repeat(np.arange(10), 2)
    net = SingleLayerSNN(n_neurons=20, neuron_labels=labels)
    image = np.zeros((28, 28), dtype=np.uint8)
    image[5:20, 10:14] = 255
    before = net.weights.copy()
    counts = net.run(rate_encode(image), rule="rstdp", label=3)
    assert counts.sum() > 0 and not np.array_equal(before, net.weights)
    net.save(tmp_path / "net.npz")
    loaded = SingleLayerSNN.load(tmp_path / "net.npz")
    assert np.array_equal(loaded.weights, net.weights) and np.array_equal(loaded.neuron_labels, labels)
    with pytest.raises(ValueError):
        net.run(rate_encode(image), rule="rstdp")  # missing label


def test_group_purity():
    counts = np.zeros(100)
    counts[30:40] = 3  # digit 3 group
    counts[0] = 10
    digit, purity = group_purity(counts)
    assert digit == 3 and np.isclose(purity, 30 / 40)
