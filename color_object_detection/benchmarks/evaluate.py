"""
evaluate.py
===========
Comprehensive benchmark and evaluation harness for the Color-Based Object
Detection System.

Metrics computed
----------------
1. IoU (Intersection over Union) between detected bounding boxes and
   ground-truth boxes.
2. Precision, Recall, F1-Score.
3. Per-stage latency (ms) for each pipeline step.
4. Throughput (FPS) measured across QVGA, VGA, HD 720p, FHD 1080p.

Output artefacts
----------------
- benchmark_curves.png   : FPS vs. resolution + F1/IoU vs. Lux
- confusion_matrix.png   : Per-class precision/recall/F1 heatmap
- pipeline_stages.png    : Grouped bar chart of per-stage latency

Usage
-----
    python -m benchmarks.evaluate
    python -m benchmarks.evaluate --out ./results/
    python -m benchmarks.evaluate --frames 500
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Add parent directory to path so we can import the src package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.tracker import ColorTracker
from benchmarks.synthetic_data import (
    GroundTruthObject,
    LuxLevel,
    SyntheticSceneGenerator,
)


# ---------------------------------------------------------------------------
# IoU and matching helpers
# ---------------------------------------------------------------------------

def compute_iou(box_a, box_b):
    """Compute the Intersection-over-Union of two axis-aligned bounding boxes.

    Parameters
    ----------
    box_a, box_b : tuple
        Bounding boxes as (x, y, w, h).

    Returns
    -------
    float
        IoU in [0.0, 1.0].
    """
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b

    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    intersection = inter_w * inter_h

    if intersection == 0:
        return 0.0

    union = aw * ah + bw * bh - intersection
    return intersection / max(union, 1e-6)


def match_detections_to_gt(detected_bboxes, gt_objects, iou_threshold=None):
    """Greedy maximum-IoU matching of detections to ground-truth objects.

    Parameters
    ----------
    detected_bboxes : list of (color_name, bbox)
    gt_objects : list of GroundTruthObject
    iou_threshold : float, optional

    Returns
    -------
    (true_positives, false_positives, false_negatives)
    """
    if iou_threshold is None:
        iou_threshold = config.IOU_TP_THRESHOLD

    matched_gt = set()
    tp = 0
    fp = 0

    for det_name, det_box in detected_bboxes:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_objects):
            if idx in matched_gt:
                continue
            if gt.color_name != det_name:
                continue
            iou = compute_iou(det_box, gt.bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(gt_objects) - len(matched_gt)
    return tp, fp, fn


def precision_recall_f1(tp, fp, fn):
    """Compute precision, recall, and F1-score from confusion counts."""
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = (2 * precision * recall) / max(precision + recall, 1e-6)
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Pipeline stage latency benchmark
# ---------------------------------------------------------------------------

def benchmark_stage_latency(resolutions, frames_per_res, color_name="Green",
                             lux=None):
    """Measure mean per-stage latency (ms) at each resolution.

    Returns
    -------
    dict : resolution -> stage_name -> mean_ms
    """
    if lux is None:
        lux = LuxLevel.NORMAL
    results = {}

    for w, h in resolutions:
        print("  Latency benchmark: {}x{} ...".format(w, h))
        gen     = SyntheticSceneGenerator(width=w, height=h, seed=0)
        tracker = ColorTracker(color_names=[color_name])

        accum = {
            "gaussian_ms":        [],
            "hsv_convert_ms":     [],
            "inrange_morph_ms":   [],
            "contour_moments_ms": [],
        }

        for frame, _ in gen.stream(num_frames=frames_per_res, lux=lux):
            result = tracker.process_frame(frame)
            for key in accum:
                if key in result.stage_times_ms:
                    accum[key].append(result.stage_times_ms[key])

        results[(w, h)] = {
            k: float(np.mean(v)) if v else 0.0 for k, v in accum.items()
        }

    return results


# ---------------------------------------------------------------------------
# Throughput (FPS) benchmark
# ---------------------------------------------------------------------------

def benchmark_throughput(resolutions, frames_per_res, lux=None):
    """Measure end-to-end throughput (FPS) across resolutions.

    Returns
    -------
    dict : resolution -> mean_fps
    """
    if lux is None:
        lux = LuxLevel.NORMAL
    fps_map = {}

    for w, h in resolutions:
        print("  Throughput benchmark: {}x{} ...".format(w, h))
        gen     = SyntheticSceneGenerator(width=w, height=h, seed=1)
        tracker = ColorTracker()

        t_start = time.time()
        for frame, _ in gen.stream(num_frames=frames_per_res, lux=lux):
            tracker.process_frame(frame)
        elapsed = time.time() - t_start

        fps_map[(w, h)] = frames_per_res / max(elapsed, 1e-9)

    return fps_map


# ---------------------------------------------------------------------------
# Accuracy benchmark (IoU, P/R/F1) across illuminance levels
# ---------------------------------------------------------------------------

def benchmark_accuracy(resolution, frames_per_lux, lux_levels):
    """Compute IoU, Precision, Recall, F1 at each illuminance level.

    Returns
    -------
    dict : lux_value -> {mean_iou, precision, recall, f1}
    """
    w, h = resolution
    acc_results = {}

    for lux in lux_levels:
        print("  Accuracy benchmark: lux={} ...".format(lux))
        gen     = SyntheticSceneGenerator(width=w, height=h, seed=2)
        tracker = ColorTracker()

        iou_scores = []
        total_tp = total_fp = total_fn = 0

        for frame, gt_objects in gen.stream(num_frames=frames_per_lux, lux=lux):
            result   = tracker.process_frame(frame)
            detected = [(b.color_name, b.bbox) for b in result.blobs]

            for det_name, det_box in detected:
                best_iou = 0.0
                for gt in gt_objects:
                    if gt.color_name == det_name:
                        best_iou = max(best_iou, compute_iou(det_box, gt.bbox))
                if best_iou > 0:
                    iou_scores.append(best_iou)

            tp, fp, fn = match_detections_to_gt(detected, gt_objects)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
        p, r, f1 = precision_recall_f1(total_tp, total_fp, total_fn)

        acc_results[lux] = {
            "mean_iou":  mean_iou,
            "precision": p,
            "recall":    r,
            "f1":        f1,
        }

    return acc_results


# ---------------------------------------------------------------------------
# Per-class accuracy
# ---------------------------------------------------------------------------

def benchmark_per_class_accuracy(resolution=None, frames=150, lux=None):
    """Compute P/R/F1 per colour class for the confusion-matrix plot.

    Returns
    -------
    dict : color_name -> {precision, recall, f1}
    """
    if resolution is None:
        resolution = (640, 480)
    if lux is None:
        lux = LuxLevel.NORMAL

    w, h    = resolution
    gen     = SyntheticSceneGenerator(width=w, height=h, seed=3)
    tracker = ColorTracker()

    class_counts = {
        name: {"tp": 0, "fp": 0, "fn": 0}
        for name in config.COLOR_PRESETS
    }

    for frame, gt_objects in gen.stream(num_frames=frames, lux=lux):
        result   = tracker.process_frame(frame)
        detected = [(b.color_name, b.bbox) for b in result.blobs]

        for color_name in config.COLOR_PRESETS:
            class_det = [(n, b) for n, b in detected if n == color_name]
            class_gt  = [g for g in gt_objects if g.color_name == color_name]

            tp, fp, fn = match_detections_to_gt(class_det, class_gt)
            class_counts[color_name]["tp"] += tp
            class_counts[color_name]["fp"] += fp
            class_counts[color_name]["fn"] += fn

    out = {}
    for name, counts in class_counts.items():
        p, r, f1 = precision_recall_f1(
            counts["tp"], counts["fp"], counts["fn"]
        )
        out[name] = {"precision": p, "recall": r, "f1": f1}

    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_DARK_BG = "#0f0f1a"
_PALETTE = ["#e94560", "#00d4aa", "#4a90d9", "#f5a623"]


def _apply_dark_style(fig, axs):
    fig.patch.set_facecolor(_DARK_BG)
    try:
        ax_list = list(axs)
    except TypeError:
        ax_list = [axs]
    for ax in ax_list:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#aaaacc")
        ax.xaxis.label.set_color("#aaaacc")
        ax.yaxis.label.set_color("#aaaacc")
        ax.title.set_color("#ccccff")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")


def plot_benchmark_curves(throughput, accuracy, out_dir):
    """Generate benchmark_curves.png."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Color Detection - Benchmark Curves",
        color="#ffffff", fontsize=14, fontweight="bold",
    )

    # Subplot 1: Throughput
    labels   = ["{}x{}".format(w, h) for w, h in throughput]
    fps_vals = list(throughput.values())
    bars     = ax1.bar(labels, fps_vals, color=_PALETTE, edgecolor="#333355", width=0.5)
    ax1.set_xlabel("Resolution")
    ax1.set_ylabel("FPS")
    ax1.set_title("Throughput vs. Resolution")
    for bar, val in zip(bars, fps_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            "{:.1f}".format(val),
            ha="center", va="bottom", color="#aaaacc", fontsize=9,
        )

    # Subplot 2: F1 and IoU vs. Lux
    lux_vals = sorted(accuracy.keys())
    f1_vals  = [accuracy[l]["f1"]       for l in lux_vals]
    iou_vals = [accuracy[l]["mean_iou"] for l in lux_vals]

    ax2.plot(lux_vals, f1_vals,  "o-",  color="#e94560", label="F1-Score",  lw=2)
    ax2.plot(lux_vals, iou_vals, "s--", color="#00d4aa", label="Mean IoU",  lw=2)
    ax2.set_xlabel("Illuminance (Lux)")
    ax2.set_ylabel("Score")
    ax2.set_title("Detection Quality vs. Illuminance")
    ax2.set_ylim(0, 1.05)
    ax2.legend(facecolor="#1e1e2e", labelcolor="#ccccff", edgecolor="#333355")
    ax2.grid(True, linestyle="--", alpha=0.3, color="#555577")

    _apply_dark_style(fig, [ax1, ax2])
    path = out_dir / "benchmark_curves.png"
    fig.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: {}".format(path))


def plot_confusion_matrix(per_class, out_dir):
    """Generate confusion_matrix.png (precision/recall/F1 heatmap)."""
    color_names = list(per_class.keys())
    metrics     = ["Precision", "Recall", "F1"]
    data        = np.array(
        [[per_class[c]["precision"], per_class[c]["recall"], per_class[c]["f1"]]
         for c in color_names]
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        "Per-Class Detection Metrics (Heatmap)",
        color="#ffffff", fontsize=13, fontweight="bold",
    )

    cmap = LinearSegmentedColormap.from_list(
        "custom", ["#1a1a2e", "#e94560", "#00d4aa"]
    )
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(range(len(color_names)))
    ax.set_yticklabels(color_names)

    for i in range(len(color_names)):
        for j in range(len(metrics)):
            val        = data[i, j]
            text_color = "white" if val < 0.6 else "#0f0f1a"
            ax.text(j, i, "{:.2f}".format(val), ha="center", va="center",
                    color=text_color, fontsize=11, fontweight="bold")

    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cb.ax.tick_params(colors="#aaaacc")

    _apply_dark_style(fig, ax)
    path = out_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: {}".format(path))


def plot_pipeline_stages(stage_latency, out_dir):
    """Generate pipeline_stages.png (grouped bar chart of stage latencies)."""
    resolutions   = list(stage_latency.keys())
    stage_names   = ["gaussian_ms", "hsv_convert_ms", "inrange_morph_ms", "contour_moments_ms"]
    display_names = ["Gaussian", "HSV Convert", "InRange+Morph", "Contour+Moments"]

    x         = np.arange(len(resolutions))
    bar_width = 0.18
    fig, ax   = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        "Per-Stage Pipeline Latency vs. Resolution",
        color="#ffffff", fontsize=13, fontweight="bold",
    )

    for idx, (stage, disp_name) in enumerate(zip(stage_names, display_names)):
        vals   = [stage_latency[res].get(stage, 0.0) for res in resolutions]
        offset = (idx - len(stage_names) / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width,
               label=disp_name, color=_PALETTE[idx % len(_PALETTE)],
               edgecolor="#333355")

    ax.set_xticks(x)
    ax.set_xticklabels(["{}x{}".format(w, h) for w, h in resolutions])
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_title("Pipeline Stage Latency Breakdown")
    ax.legend(facecolor="#1e1e2e", labelcolor="#ccccff", edgecolor="#333355")
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#555577")

    _apply_dark_style(fig, ax)
    path = out_dir / "pipeline_stages.png"
    fig.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: {}".format(path))


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_full_benchmark(out_dir, frames_per_res=None):
    """Execute all benchmarks and save result plots.

    Parameters
    ----------
    out_dir : Path
        Directory where PNG files will be written.
    frames_per_res : int, optional
        How many synthetic frames to evaluate per resolution.
    """
    if frames_per_res is None:
        frames_per_res = config.BENCHMARK_FRAMES_PER_RESOLUTION

    out_dir.mkdir(parents=True, exist_ok=True)
    resolutions = config.BENCHMARK_RESOLUTIONS
    lux_levels  = config.BENCHMARK_LUX_LEVELS

    print("=" * 60)
    print("  Color Detection System -- Full Benchmark Suite")
    print("=" * 60)

    print("\n[1/4] Stage latency benchmark ...")
    stage_latency = benchmark_stage_latency(resolutions, frames_per_res)

    print("\n[2/4] Throughput (FPS) benchmark ...")
    throughput = benchmark_throughput(resolutions, frames_per_res)

    print("\n[3/4] Accuracy vs. illuminance benchmark (640x480) ...")
    accuracy = benchmark_accuracy(
        resolution=(640, 480),
        frames_per_lux=frames_per_res // 2,
        lux_levels=lux_levels,
    )

    print("\n[4/4] Per-class accuracy (confusion matrix) ...")
    per_class = benchmark_per_class_accuracy(
        resolution=(640, 480),
        frames=frames_per_res,
    )

    print("\nGenerating plots ...")
    plot_benchmark_curves(throughput, accuracy, out_dir)
    plot_confusion_matrix(per_class, out_dir)
    plot_pipeline_stages(stage_latency, out_dir)

    # Console summary
    print("\n" + "=" * 60)
    print("  THROUGHPUT SUMMARY")
    print("=" * 60)
    for (w, h), fps in throughput.items():
        print("  {:4d}x{:<4d}  ->  {:6.1f} FPS".format(w, h, fps))

    print("\n  ACCURACY SUMMARY (640x480)")
    print("  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}".format(
        "Lux", "IoU", "P", "R", "F1"))
    print("  " + "-" * 38)
    for lux in sorted(accuracy):
        m = accuracy[lux]
        print("  {:>6}  {:>6.3f}  {:>6.3f}  {:>6.3f}  {:>6.3f}".format(
            lux, m["mean_iou"], m["precision"], m["recall"], m["f1"]
        ))

    print("\n  PER-CLASS ACCURACY (640x480, normal light)")
    print("  {:>10}  {:>6}  {:>6}  {:>6}".format("Class", "P", "R", "F1"))
    print("  " + "-" * 32)
    for name, m in per_class.items():
        print("  {:>10}  {:>6.3f}  {:>6.3f}  {:>6.3f}".format(
            name, m["precision"], m["recall"], m["f1"]
        ))

    print("\n  All plots saved to:", str(out_dir.resolve()))
    print("=" * 60)


def _parse_args():
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.evaluate",
        description="Color Detection - Benchmark Suite",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("."),
        help="Directory to save output plots.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=config.BENCHMARK_FRAMES_PER_RESOLUTION,
        help="Frames per resolution/lux.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_full_benchmark(out_dir=args.out, frames_per_res=args.frames)
