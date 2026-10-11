"""Detect several MNIST digits on a 100x100 canvas with a sliding window + NMS.

By default the trained weights in ``weights/`` are used; ``--retrain`` trains a
fresh model first (1k STDP + 5k R-STDP images, as in the original experiment).

Example:
    python scripts/detect_digits.py
    python scripts/detect_digits.py --runs 20 --threshold 90 --seed 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snn_detection import OUTPUT_DIR, WEIGHTS_DIR
from snn_detection.data import load_mnist
from snn_detection.deep_snn import load_weights, train_deep_snn
from snn_detection.detection import detect_digits, evaluate_detections, make_fixed_canvas, non_max_suppression
from snn_detection.visualization import draw_detections


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=10, help="number of random canvases")
    parser.add_argument("--digits", type=int, default=3, choices=[1, 2, 3, 4], help="digits per canvas")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=93.0, help="minimum output voltage for a detection")
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--match-iou", type=float, default=0.3, help="IoU needed to count a ground truth as found")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR)
    parser.add_argument("--retrain", action="store_true", help="train a new model instead of loading weights")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "detection_results.png")
    args = parser.parse_args()

    np.random.seed(args.seed)
    x_train, y_train, x_test, y_test = load_mnist()
    if args.retrain:
        encoder, classifier, _ = train_deep_snn(x_train, y_train, n_stdp=1000, n_rstdp=5000, log_every=0)
    else:
        encoder, classifier = load_weights(args.weights_dir)

    ncols = min(args.runs, 5)
    nrows = int(np.ceil(args.runs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.5 * nrows), squeeze=False)
    fig.suptitle(f"Deep SNN: detecting {args.digits} digits per image "
                 "(red = detection, green = missed ground truth)", fontsize=16)

    totals = {"found": 0, "missed": 0, "false_positives": 0}
    for run, ax in enumerate(axes.flat):
        if run >= args.runs:
            ax.axis("off")
            continue
        canvas, ground_truths = make_fixed_canvas(x_test, y_test, num_digits=args.digits)
        raw = detect_digits(canvas, encoder, classifier, stride=args.stride, threshold=args.threshold)
        detections = non_max_suppression(raw, threshold=args.nms_iou)
        result = evaluate_detections(detections, ground_truths, iou_threshold=args.match_iou)
        for key in totals:
            totals[key] += len(result[key])

        truth = [gt["label"] for gt in ground_truths]
        predicted = [d["label"] for d in detections]
        print(f"run {run + 1:2d}: digits {truth} | detections {predicted} | found {len(result['found'])}/"
              f"{len(ground_truths)} | false positives {len(result['false_positives'])}")
        missed = [ground_truths[g] for g in result["missed"]]
        draw_detections(ax, canvas, detections, missed, fontsize=13,
                        title=f"Run {run + 1}: found {len(result['found'])}/{len(ground_truths)}")

    n_gt = totals["found"] + totals["missed"]
    print(f"\nTotal: found {totals['found']}/{n_gt} digits ({100 * totals['found'] / n_gt:.1f}%), "
          f"false positives: {totals['false_positives']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.output, dpi=120)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
