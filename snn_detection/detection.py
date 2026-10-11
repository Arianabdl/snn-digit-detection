"""Sliding-window digit detection on a larger canvas, with NMS and evaluation.

Boxes are ``(x1, y1, x2, y2)`` tuples with exclusive end coordinates. A detection
is a dict ``{"bbox": box, "label": int, "score": float}`` and a ground truth is
``{"bbox": box, "label": int}``.
"""

import numpy as np

DIGIT_SIZE = 28
# Top-left (y, x) anchors used by the deep-SNN experiment; shuffled per canvas.
FIXED_POSITIONS = [(10, 10), (50, 15), (25, 55), (50, 50)]


def make_fixed_canvas(images, labels, num_digits=3, canvas_size=100):
    """Place ``num_digits`` random digits at shuffled fixed anchor points (max-blended)."""
    canvas = np.zeros((canvas_size, canvas_size))
    indices = np.random.choice(len(images), num_digits, replace=False)
    positions = list(FIXED_POSITIONS)
    np.random.shuffle(positions)
    ground_truths = []
    for idx, (y, x) in zip(indices, positions[:num_digits]):
        region = canvas[y:y + DIGIT_SIZE, x:x + DIGIT_SIZE]
        canvas[y:y + DIGIT_SIZE, x:x + DIGIT_SIZE] = np.maximum(region, images[idx])
        ground_truths.append({"bbox": (x, y, x + DIGIT_SIZE, y + DIGIT_SIZE), "label": int(labels[idx])})
    return canvas, ground_truths


def _boxes_touch(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def make_random_canvas(images, labels, num_digits=1, canvas_size=100, max_attempts=50):
    """Place digits at random, non-overlapping positions (keeps the image dtype)."""
    canvas = np.zeros((canvas_size, canvas_size), dtype=images.dtype)
    ground_truths = []
    for _ in range(num_digits):
        for _attempt in range(max_attempts):
            idx = np.random.randint(0, len(images))
            x = np.random.randint(0, canvas_size - DIGIT_SIZE)
            y = np.random.randint(0, canvas_size - DIGIT_SIZE)
            box = (x, y, x + DIGIT_SIZE, y + DIGIT_SIZE)
            if not any(_boxes_touch(box, gt["bbox"]) for gt in ground_truths):
                canvas[y:y + DIGIT_SIZE, x:x + DIGIT_SIZE] = images[idx]
                ground_truths.append({"bbox": box, "label": int(labels[idx])})
                break
    return canvas, ground_truths


def iou(box_a, box_b):
    """Intersection over union of two boxes."""
    inter_w = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    inter_h = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    inter = inter_w * inter_h
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / float(area_a + area_b - inter + 1e-6)


def non_max_suppression(detections, threshold=0.3, metric="iou"):
    """Class-agnostic greedy NMS; keeps the highest-scoring box of each overlapping group.

    ``metric="iou"`` suppresses boxes whose IoU with a kept box is ``>= threshold``
    (used by the deep SNN detector). ``metric="overlap"`` uses intersection over
    the *suppressed* box's area with inclusive pixel coordinates and suppresses
    when it is ``> threshold`` (the variant used by the comparison detectors).
    """
    if not detections:
        return []
    if metric == "iou":
        remaining = sorted(detections, key=lambda d: d["score"], reverse=True)
        keep = []
        while remaining:
            best = remaining.pop(0)
            keep.append(best)
            remaining = [d for d in remaining if iou(best["bbox"], d["bbox"]) < threshold]
        return keep
    if metric != "overlap":
        raise ValueError(f"unknown NMS metric: {metric}")

    boxes = np.array([d["bbox"] for d in detections], dtype=float)
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort([d["score"] for d in detections])
    keep = []
    while len(order) > 0:
        i, rest = order[-1], order[:-1]
        keep.append(detections[i])
        w = np.maximum(0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]) + 1)
        h = np.maximum(0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]) + 1)
        order = rest[(w * h) / area[rest] <= threshold]
    return keep


def sliding_windows(image, window=DIGIT_SIZE, stride=2):
    """Yield ``(x, y, crop)`` for every window position, including the last row/column."""
    height, width = image.shape
    for y in range(0, height - window + 1, stride):
        for x in range(0, width - window + 1, stride):
            yield x, y, image[y:y + window, x:x + window]


def detect_with(canvas, classify, stride=2, min_mean=None, window=DIGIT_SIZE):
    """Run ``classify(crop) -> (label, score) | None`` on every window.

    Windows whose mean intensity is below ``min_mean`` are skipped as background.
    """
    detections = []
    for x, y, crop in sliding_windows(canvas, window, stride):
        if min_mean is not None and crop.mean() < min_mean:
            continue
        result = classify(crop)
        if result is not None:
            label, score = result
            detections.append({"bbox": (x, y, x + window, y + window), "label": int(label), "score": float(score)})
    return detections


def detect_digits(canvas, encoder, classifier, stride=2, threshold=93.0, border=2, border_max=0.25):
    """Deep-SNN sliding-window detector (before NMS).

    A window is scored by the voltage of the winning output neuron and kept if it
    reaches ``threshold``. Boundary constraint: windows with any pixel brighter
    than ``border_max`` in their outer ``border`` pixels are rejected, because
    they cut through a digit and produce off-centre false positives.
    """
    border_mask = np.ones((DIGIT_SIZE, DIGIT_SIZE), dtype=bool)
    border_mask[border:-border, border:-border] = False

    def classify(crop):
        if crop[border_mask].max() > border_max:
            return None
        voltages = classifier.voltages(encoder.forward(crop).ravel())
        label = int(np.argmax(voltages))
        return (label, voltages[label]) if voltages[label] >= threshold else None

    return detect_with(canvas, classify, stride=stride)


def evaluate_detections(detections, ground_truths, iou_threshold=0.3):
    """Match detections to ground truths (same label and IoU >= ``iou_threshold``).

    Returns a dict with the indices of ``found`` and ``missed`` ground truths and
    of ``false_positives`` (detections matching no ground truth).
    """
    def matches(det, gt):
        return det["label"] == gt["label"] and iou(det["bbox"], gt["bbox"]) >= iou_threshold

    found = [g for g, gt in enumerate(ground_truths) if any(matches(d, gt) for d in detections)]
    missed = [g for g in range(len(ground_truths)) if g not in found]
    false_positives = [i for i, d in enumerate(detections) if not any(matches(d, gt) for gt in ground_truths)]
    return {"found": found, "missed": missed, "false_positives": false_positives}
