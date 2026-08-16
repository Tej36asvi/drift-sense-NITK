"""
Accuracy and runtime number-crunching, kept separate from file reading and
plotting so it's easy to test with small made-up examples.

"Within 5px" means error 5px or less, boundary included -- not stricter
than that.

The problem statement asks for "sub-pixel" accuracy without saying exactly
what counts, so 0.5px here is just a reasonable guess (half a pixel), not
an official number -- kept as a parameter everywhere instead of hardcoded.
"""

import re

import numpy as np

DEFAULT_THRESHOLDS_PX = (5.0, 4.0, 2.0, 1.0)
DEFAULT_SUBPIXEL_THRESHOLD_PX = 0.5


def euclidean_errors(pred_x, pred_y, true_x, true_y):
    """Per-pair Euclidean error in pixels. Accepts scalars or array-likes
    of equal length; always returns a numpy array of floats.
    """
    pred_x = np.asarray(pred_x, dtype=float)
    pred_y = np.asarray(pred_y, dtype=float)
    true_x = np.asarray(true_x, dtype=float)
    true_y = np.asarray(true_y, dtype=float)
    return np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)


def pass_rate(errors, threshold, inclusive=True):
    """Fraction of `errors` at or within `threshold` px (inclusive, see
    module docstring). Returns NaN for an empty input rather than raising,
    so a breakdown bucket with zero rows can propagate NaN instead of
    crashing.
    """
    errors = np.asarray(errors, dtype=float)
    if errors.size == 0:
        return float("nan")
    passed = errors <= threshold if inclusive else errors < threshold
    return float(np.count_nonzero(passed)) / errors.size


def pass_rates_at_thresholds(errors, thresholds=DEFAULT_THRESHOLDS_PX, inclusive=True):
    """dict of {"pass_rate_<T>px": rate} for each threshold in `thresholds`."""
    return {
        f"pass_rate_{_fmt(t)}px": pass_rate(errors, t, inclusive=inclusive)
        for t in thresholds
    }


def subpixel_pass_rate(errors, subpixel_threshold_px=DEFAULT_SUBPIXEL_THRESHOLD_PX,
                        inclusive=True):
    """Pass rate at the (named, non-magic) sub-pixel threshold."""
    return pass_rate(errors, subpixel_threshold_px, inclusive=inclusive)


def error_summary(errors):
    """dict with n, mean_error_px, median_error_px, worst_error_px.
    worst_error_px is the max (worst case is the largest error, not the
    smallest). NaNs for an empty input rather than raising.
    """
    errors = np.asarray(errors, dtype=float)
    if errors.size == 0:
        return {
            "n": 0,
            "mean_error_px": float("nan"),
            "median_error_px": float("nan"),
            "worst_error_px": float("nan"),
        }
    return {
        "n": int(errors.size),
        "mean_error_px": float(np.mean(errors)),
        "median_error_px": float(np.median(errors)),
        "worst_error_px": float(np.max(errors)),
    }


def runtime_summary(runtimes):
    """Average and slowest runtime, in seconds."""
    runtimes = np.asarray(runtimes, dtype=float)
    if runtimes.size == 0:
        return {"mean_runtime_sec": float("nan"), "worst_runtime_sec": float("nan")}
    return {
        "mean_runtime_sec": float(np.mean(runtimes)),
        "worst_runtime_sec": float(np.max(runtimes)),
    }


def full_metrics(errors, runtimes=None, thresholds=DEFAULT_THRESHOLDS_PX,
                  subpixel_threshold_px=DEFAULT_SUBPIXEL_THRESHOLD_PX,
                  inclusive=True):
    """Everything above, bundled into one dict. evaluate.py calls this once
    for the overall numbers and once per breakdown group, so both are
    computed exactly the same way.
    """
    out = {}
    out.update(error_summary(errors))
    out.update(pass_rates_at_thresholds(errors, thresholds, inclusive=inclusive))
    out[f"subpixel_pass_rate_{_fmt(subpixel_threshold_px)}px"] = subpixel_pass_rate(
        errors, subpixel_threshold_px, inclusive=inclusive
    )
    if runtimes is not None:
        out.update(runtime_summary(runtimes))
    return out


def _fmt(threshold):
    # Turn a numeric threshold into a JSON/dict-key-safe fragment:
    # 5 -> "5", 0.5 -> "0p5". Keeps keys stable and readable in metrics.json.
    s = f"{threshold:g}"
    return s.replace(".", "p").replace("-", "neg")


def bin_column(values, n_bins=5, method="equal_width", max_categorical=12):
    """Turn a column of values into group labels for group_by_column.

    Columns like "architecture" that aren't really numbers (or only have a
    handful of distinct values) are left as-is. Columns that are genuinely
    numeric and have many different values (like scale_ratio) get sliced
    into `n_bins` ranges instead, e.g. "[9.2, 9.6)", so there's something
    sensible to group by.

    Bins are equal-width by default (equal-sized ranges, not equal-sized
    groups) since that matches what "noise level" or "scale" naturally
    means when reading a report. Pass method="quantile" for equal-sized
    groups instead, if a breakdown needs balanced bucket counts.
    """
    n = len(values)
    if n == 0:
        return []

    numeric = []
    all_numeric = True
    for v in values:
        try:
            numeric.append(float(v))
        except (TypeError, ValueError):
            all_numeric = False
            break

    if not all_numeric:
        return [str(v) for v in values]

    unique_vals = set(numeric)
    if len(unique_vals) <= max_categorical:
        # numeric but low-cardinality (e.g. a small fixture, or an integer
        # count column): treat each distinct value as its own category, but
        # format with %g instead of str() so a derived float column (e.g.
        # distance-from-centre) prints as "141.4", not a full float repr
        # like "141.4213562373095".
        return [f"{v:.4g}" for v in numeric]

    arr = np.asarray(numeric, dtype=float)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if lo == hi:
        return [f"[{lo:.3g}]"] * n

    if method == "quantile":
        edges = np.unique(np.quantile(arr, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size < 2:
            edges = np.array([lo, hi])
    else:
        edges = np.linspace(lo, hi, n_bins + 1)

    idx = np.clip(np.searchsorted(edges, arr, side="right") - 1, 0, len(edges) - 2)
    labels = []
    for i, v in zip(idx, arr):
        e0, e1 = edges[i], edges[i + 1]
        closer = "]" if i == len(edges) - 2 else ")"
        labels.append(f"[{e0:.3g}, {e1:.3g}{closer}")
    return labels


def _bin_sort_key(label):
    # Sort numeric-looking bin labels ("[9.2, 9.6)") by their lower edge;
    # fall back to alphabetical for categorical labels ("dram"/"finfet") so
    # the printed breakdown has a stable, readable order either way.
    m = re.match(r"\[(-?[0-9.eE+-]+)", label)
    if m:
        try:
            return (0, float(m.group(1)), label)
        except ValueError:
            pass
    return (1, 0.0, label)


def group_by_column(rows, column, error_key="error", runtime_key=None,
                     thresholds=DEFAULT_THRESHOLDS_PX,
                     subpixel_threshold_px=DEFAULT_SUBPIXEL_THRESHOLD_PX,
                     inclusive=True, n_bins=5, method="equal_width",
                     max_categorical=12):
    """Split `rows` into groups by `column` and compute accuracy/runtime for
    each group separately. One function handles every "broken down by X"
    report (noise level, target position, scale, rotation, ...) -- just
    pass in whichever column to group by.

    Returns a dict from group label to its own set of metrics. Empty
    groups just don't appear, rather than showing up blank.
    """
    if not rows:
        return {}
    raw_values = [r[column] for r in rows]
    labels = bin_column(raw_values, n_bins=n_bins, method=method,
                         max_categorical=max_categorical)

    buckets = {}
    for label, row in zip(labels, rows):
        buckets.setdefault(label, []).append(row)

    result = {}
    for label in sorted(buckets, key=_bin_sort_key):
        sub_rows = buckets[label]
        errors = [r[error_key] for r in sub_rows]
        runtimes = [r[runtime_key] for r in sub_rows] if runtime_key else None
        result[label] = full_metrics(errors, runtimes, thresholds,
                                      subpixel_threshold_px, inclusive)
    return result
