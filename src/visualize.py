"""
The three figures in the results folder: error distribution, best-pair
overlay, and the worst-pair failure breakdown. evaluate.py decides which
pairs to plot and saves the images -- these functions just build them.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.metrics import DEFAULT_SUBPIXEL_THRESHOLD_PX, DEFAULT_THRESHOLDS_PX, pass_rate

PRED_COLOR = "red"
TRUE_COLOR = "lime"


def load_image_gray(path):
    """cv2.imread grayscale wrapper that raises instead of silently handing
    back None -- a bad path here should fail loudly, not surface later as a
    confusing matplotlib error.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def _mark_points(ax, pred_xy, true_xy):
    ax.plot(pred_xy[0], pred_xy[1], marker="x", color=PRED_COLOR, markersize=12,
             markeredgewidth=2.5, linestyle="None",
             label=f"predicted ({pred_xy[0]:.2f}, {pred_xy[1]:.2f})")
    ax.plot(true_xy[0], true_xy[1], marker="+", color=TRUE_COLOR, markersize=14,
             markeredgewidth=2.5, linestyle="None",
             label=f"ground truth ({true_xy[0]:.2f}, {true_xy[1]:.2f})")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)


def plot_overlay(reference_img, search_img, pred_xy, true_xy, error_px=None,
                  title=None):
    """The core "is it right" visual for a single pair: search image with
    predicted and ground-truth centres marked, reference image alongside.
    """
    fig, (ax_ref, ax_search) = plt.subplots(1, 2, figsize=(11, 5.5))

    ax_ref.imshow(reference_img, cmap="gray")
    ax_ref.set_title("Reference (template)")
    ax_ref.axis("off")

    ax_search.imshow(search_img, cmap="gray")
    _mark_points(ax_search, pred_xy, true_xy)
    subtitle = f" -- error {error_px:.2f} px" if error_px is not None else ""
    ax_search.set_title(f"Search image{subtitle}")
    ax_search.axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_error_distribution(errors, thresholds=DEFAULT_THRESHOLDS_PX,
                             subpixel_threshold_px=DEFAULT_SUBPIXEL_THRESHOLD_PX,
                             title=None):
    """Three panels showing how the errors are spread out: a histogram, a
    full-range cumulative curve, and the same curve zoomed in near zero --
    the zoomed view is needed because the pass thresholds (0.5-5px) would
    otherwise be squeezed unreadably close together next to the handful of
    much larger outlier errors.
    """
    errors = np.asarray(errors, dtype=float)
    all_thresholds = sorted(set(list(thresholds) + [subpixel_threshold_px]))
    rates_by_threshold = {t: pass_rate(errors, t) for t in all_thresholds}

    fig, (ax_hist, ax_cdf_full, ax_cdf_zoom) = plt.subplots(3, 1, figsize=(7.5, 12))

    def draw_threshold_lines(ax, with_dots=False):
        for t in all_thresholds:
            is_subpixel = t == subpixel_threshold_px
            color = "darkorange" if is_subpixel else "crimson"
            ax.axvline(t, linestyle="--", linewidth=1.1, color=color, alpha=0.85)
            if with_dots:
                p = rates_by_threshold[t]
                if not np.isnan(p):
                    ax.plot(t, min(p, 1.0), marker="o", color=color, markersize=5, zorder=5)

    n_bins = min(30, max(5, errors.size))
    ax_hist.hist(errors, bins=n_bins, color="steelblue", edgecolor="black", linewidth=0.5)
    draw_threshold_lines(ax_hist)
    ax_hist.set_xlabel("Euclidean error (px)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("Error histogram (full range)")

    sorted_err = np.sort(errors) if errors.size else errors
    cdf = np.arange(1, sorted_err.size + 1) / sorted_err.size if sorted_err.size else sorted_err

    ax_cdf_full.step(sorted_err, cdf, where="post", color="steelblue")
    draw_threshold_lines(ax_cdf_full, with_dots=True)
    ax_cdf_full.set_xlabel("Euclidean error (px)")
    ax_cdf_full.set_ylabel("cumulative fraction of pairs")
    ax_cdf_full.set_ylim(0, 1.05)
    ax_cdf_full.set_title("Error CDF (full range)")

    zoom_max = max(all_thresholds) * 1.4 if all_thresholds else 1.0
    if errors.size:
        zoom_max = max(zoom_max, float(np.min(errors)) * 1.1, 1e-6)
    ax_cdf_zoom.step(sorted_err, cdf, where="post", color="steelblue")
    draw_threshold_lines(ax_cdf_zoom, with_dots=True)
    ax_cdf_zoom.set_xlim(0, zoom_max)
    ax_cdf_zoom.set_ylim(0, 1.05)
    ax_cdf_zoom.set_xlabel("Euclidean error (px)")
    ax_cdf_zoom.set_ylabel("cumulative fraction of pairs")
    ax_cdf_zoom.set_title(f"Error CDF zoomed to [0, {zoom_max:.3g}] px -- pass rate at each threshold")

    legend_lines = []
    for t in sorted(all_thresholds, reverse=True):
        p = rates_by_threshold[t]
        tag = " (sub-px)" if t == subpixel_threshold_px else ""
        legend_lines.append(f"{t:g}px{tag}: {p * 100:.1f}% pass" if not np.isnan(p)
                             else f"{t:g}px{tag}: n/a")
    ax_cdf_zoom.text(0.98, 0.04, "\n".join(legend_lines), transform=ax_cdf_zoom.transAxes,
                      ha="right", va="bottom", fontsize=8,
                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"))

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def _crop_around(img, cx, cy, half_size):
    h, w = img.shape[:2]
    half_size = max(1, int(round(half_size)))
    x0, x1 = int(max(0, cx - half_size)), int(min(w, cx + half_size))
    y0, y1 = int(max(0, cy - half_size)), int(min(h, cy + half_size))
    return img[y0:y1, x0:x1]


def plot_failure_case(reference_img, search_img, pred_xy, true_xy, error_px,
                       confidence_ratio, n_valid=None, crop_half_size=50,
                       ambiguous_confidence_ratio=1.15, title=None):
    """The failure figure for the worst-scoring pair. Crops the search
    image at both the predicted and true locations and shows them side by
    side -- in a repeating pattern they usually look nearly identical,
    which is the actual visual reason the match got confused, not just a
    label saying "wrong". confidence_ratio backs this up with a number:
    close to 1.0 means the top two candidates were essentially tied.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10.5))
    ax_ref, ax_search, ax_crop_true, ax_crop_pred = (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    )

    ax_ref.imshow(reference_img, cmap="gray")
    ax_ref.set_title("Reference (template)")
    ax_ref.axis("off")

    ax_search.imshow(search_img, cmap="gray")
    _mark_points(ax_search, pred_xy, true_xy)
    ax_search.set_title(f"Search image -- error {error_px:.2f} px")
    ax_search.axis("off")

    crop_true = _crop_around(search_img, true_xy[0], true_xy[1], crop_half_size)
    crop_pred = _crop_around(search_img, pred_xy[0], pred_xy[1], crop_half_size)
    ax_crop_true.imshow(crop_true, cmap="gray")
    ax_crop_true.set_title("Crop at ground truth")
    ax_crop_true.axis("off")
    ax_crop_pred.imshow(crop_pred, cmap="gray")
    ax_crop_pred.set_title("Crop at prediction")
    ax_crop_pred.axis("off")

    is_ambiguous = confidence_ratio <= ambiguous_confidence_ratio
    note = (" -- best and runner-up match scores were nearly tied: consistent "
            "with a periodic pattern producing multiple near-identical "
            "candidates" if is_ambiguous else "")
    caption = f"confidence_ratio = {confidence_ratio:.3f}{note}"
    if n_valid is not None:
        caption += f"\nn_valid = {n_valid} candidate(s) within tolerance of the best score"
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=9, wrap=True)

    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    return fig
