#!/usr/bin/env python3
"""
Drift-Sense evaluation CLI.

    python evaluate.py --manifest results/dataset/manifest.csv \
                        --predictions results/predictions.csv \
                        --out-dir results/

Matches up the manifest (ground truth) and predictions CSVs, works out
accuracy and runtime, prints a summary, and saves metrics.json plus a
few figures.

Only reads the two CSV files -- it doesn't call the localizer directly --
so it works on any predictions CSV with the right columns, including the
small hand-made ones used in the tests.
"""

import argparse
import csv
import json
import os
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless -- this just writes PNG files, no GUI needed

from src import metrics
from src import visualize

# Both images are 1000x1000px. Exposed as a CLI flag rather than a bare
# constant so the target-position breakdown still works if that changes.
DEFAULT_SEARCH_IMAGE_SIZE_PX = 1000

MANIFEST_NUMERIC_COLUMNS = [
    "gt_x", "gt_y", "seed", "dose_search", "dose_reference",
    "shear_amplitude_px", "drift_jitter_px", "beam_spot_size_nm",
    "speckle_sigma", "salt_pepper_prob", "charging_streak_prob",
    "astigmatism_ratio", "gamma", "vignette_strength", "barrel_distortion_k",
]
PREDICTIONS_NUMERIC_COLUMNS = [
    "pred_x", "pred_y", "score", "phase_x", "phase_y", "shear_slope",
    "runtime_sec",
]

GROUP_BY_SPECS = [
    # (column, human label) -- breakdown axes: noise level, target
    # position, scale, rotation
    ("architecture", "architecture"),
    ("condition", "acquisition condition"),
    ("dose_search", "search-image dose (lower = noisier)"),
    ("shear_amplitude_px", "raster drift amplitude (px)"),
    ("dist_from_center_px", "target position: distance from search-image centre (px)"),
]


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_numeric(rows, columns):
    for row in rows:
        for col in columns:
            if col in row:
                row[col] = to_float(row[col])


def resolve_path(base_dir, rel_path):
    # manifest paths are relative to the manifest's own directory, always --
    # never the predictions CSV's directory, even though predictions rows
    # also carry reference_path/search_path
    p = Path(rel_path)
    return p if p.is_absolute() else Path(base_dir) / p


def join_manifest_predictions(manifest_rows, predictions_rows):
    manifest_by_id = {r["pair_id"]: r for r in manifest_rows}
    predictions_by_id = {r["pair_id"]: r for r in predictions_rows}

    common_ids = [pid for pid in manifest_by_id if pid in predictions_by_id]
    missing_predictions = sorted(set(manifest_by_id) - set(predictions_by_id))
    missing_manifest = sorted(set(predictions_by_id) - set(manifest_by_id))

    joined = []
    for pid in common_ids:
        row = dict(manifest_by_id[pid])
        pred = predictions_by_id[pid]
        for key in ("pred_x", "pred_y", "confidence_ratio", "n_valid",
                    "coarse_scale", "refined_scale", "refined_rotation_deg",
                    "runtime_sec"):
            row[key] = pred.get(key)
        row["pair_id"] = pid
        joined.append(row)
    return joined, missing_predictions, missing_manifest


def collect_environment_facts():
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine() or "unknown",
        "machine": platform.machine(),
        "cpu_count_logical": os.cpu_count(),
        "timing_method": (
            "runtime_sec column from the predictions CSV: per-pair wall-clock "
            "time as measured by the localization pipeline (localize.py / "
            "src/localizer.py / src/refine.py). evaluate.py does not re-time "
            "inference itself, it only aggregates the reported values."
        ),
    }


def fmt_pct(x):
    return "n/a" if x != x else f"{x * 100:.1f}%"  # x != x is the NaN check


def fmt_px(x):
    return "n/a" if x != x else f"{x:.2f}px"


def print_metrics_block(label, m, indent="  "):
    print(f"{indent}n={m['n']}  mean={fmt_px(m['mean_error_px'])}  "
          f"median={fmt_px(m['median_error_px'])}  worst={fmt_px(m['worst_error_px'])}")
    pass_bits = "  ".join(
        f"@{k.replace('pass_rate_', '').replace('px', '')}px={fmt_pct(v)}"
        for k, v in m.items() if k.startswith("pass_rate_")
    )
    subpx_bits = "  ".join(
        f"sub-px({k.replace('subpixel_pass_rate_', '').replace('px', '')}px)={fmt_pct(v)}"
        for k, v in m.items() if k.startswith("subpixel_pass_rate_")
    )
    print(f"{indent}{pass_bits}  {subpx_bits}")
    if "mean_runtime_sec" in m:
        print(f"{indent}runtime: mean={m['mean_runtime_sec']:.4f}s  "
              f"worst={m['worst_runtime_sec']:.4f}s")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Drift-Sense evaluation CLI")
    p.add_argument("--manifest", required=True, help="path to manifest.csv")
    p.add_argument("--predictions", required=True, help="path to predictions.csv")
    p.add_argument("--out-dir", required=True, help="directory to write metrics.json and figures into")
    p.add_argument("--thresholds", type=float, nargs="+", default=list(metrics.DEFAULT_THRESHOLDS_PX),
                    help="pixel pass-rate thresholds, default 5 4 2 1")
    p.add_argument("--subpixel-threshold", type=float, default=metrics.DEFAULT_SUBPIXEL_THRESHOLD_PX,
                    help="sub-pixel pass-rate cutoff in px (no fixed spec value, see metrics.py)")
    p.add_argument("--n-bins", type=int, default=5, help="bins for auto-binned continuous breakdown columns")
    p.add_argument("--search-image-size", type=float, default=DEFAULT_SEARCH_IMAGE_SIZE_PX,
                    help="search image side length in px, used for the target-position breakdown")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    manifest_path = Path(args.manifest).resolve()
    predictions_path = Path(args.predictions).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = manifest_path.parent

    manifest_rows = read_csv_rows(manifest_path)
    predictions_rows = read_csv_rows(predictions_path)
    coerce_numeric(manifest_rows, MANIFEST_NUMERIC_COLUMNS)
    coerce_numeric(predictions_rows, PREDICTIONS_NUMERIC_COLUMNS)

    joined, missing_predictions, missing_manifest = join_manifest_predictions(
        manifest_rows, predictions_rows
    )
    if not joined:
        print("ERROR: no pair_id is present in both CSVs, nothing to evaluate.", file=sys.stderr)
        return 1

    center = args.search_image_size / 2.0
    for row in joined:
        row["error"] = float(metrics.euclidean_errors(
            row["pred_x"], row["pred_y"], row["gt_x"], row["gt_y"]
        ))
        row["dist_from_center_px"] = float(metrics.euclidean_errors(
            row["gt_x"], row["gt_y"], center, center
        ))

    all_errors = [row["error"] for row in joined]
    all_runtimes = [row["runtime_sec"] for row in joined]
    overall = metrics.full_metrics(
        all_errors, all_runtimes, thresholds=args.thresholds,
        subpixel_threshold_px=args.subpixel_threshold,
    )

    breakdowns = {}
    for column, _label in GROUP_BY_SPECS:
        if all(column in row for row in joined):
            breakdowns[column] = metrics.group_by_column(
                joined, column, error_key="error", runtime_key="runtime_sec",
                thresholds=args.thresholds, subpixel_threshold_px=args.subpixel_threshold,
                n_bins=args.n_bins,
            )

    env_facts = collect_environment_facts()

    # ---- stdout summary ----
    print("=" * 70)
    print("DRIFT-SENSE EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Manifest:     {manifest_path}  ({len(manifest_rows)} pairs)")
    print(f"Predictions:  {predictions_path}  ({len(predictions_rows)} pairs)")
    print(f"Joined pairs: {len(joined)}  "
          f"({len(missing_predictions)} in manifest w/o prediction, "
          f"{len(missing_manifest)} predicted w/o manifest row)")
    if missing_predictions:
        print(f"  missing predictions for pair_id: {missing_predictions}")
    if missing_manifest:
        print(f"  predictions with no manifest row: {missing_manifest}")

    print("\nEnvironment")
    print(f"  Python:    {env_facts['python_version']} ({env_facts['python_implementation']})")
    print(f"  Platform:  {env_facts['platform']}")
    print(f"  Processor: {env_facts['processor']}  |  CPU count (logical): {env_facts['cpu_count_logical']}")
    print(f"  Timing:    {env_facts['timing_method']}")

    print(f"\nOverall accuracy (n={overall['n']})")
    print_metrics_block("overall", overall)

    for column, label in GROUP_BY_SPECS:
        if column not in breakdowns:
            continue
        print(f"\nBreakdown by {label}")
        for bin_label, m in breakdowns[column].items():
            print(f"  {bin_label}")
            print_metrics_block(bin_label, m, indent="    ")

    # ---- figures ----
    best_row = min(joined, key=lambda r: (r["error"], r["pair_id"]))
    worst_row = max(joined, key=lambda r: (r["error"], r["pair_id"]))

    figure_paths = []

    def load_pair_images(row):
        ref_img = visualize.load_image_gray(resolve_path(manifest_dir, row["reference_path"]))
        search_img = visualize.load_image_gray(resolve_path(manifest_dir, row["search_path"]))
        return ref_img, search_img

    dist_fig = visualize.plot_error_distribution(
        all_errors, thresholds=args.thresholds, subpixel_threshold_px=args.subpixel_threshold,
        title=f"Error distribution across {len(joined)} pairs",
    )
    dist_path = out_dir / "error_distribution.png"
    dist_fig.savefig(dist_path, dpi=150)
    figure_paths.append(dist_path)

    best_ref, best_search = load_pair_images(best_row)
    overlay_fig = visualize.plot_overlay(
        best_ref, best_search, (best_row["pred_x"], best_row["pred_y"]),
        (best_row["gt_x"], best_row["gt_y"]), error_px=best_row["error"],
        title=f"Best pair: {best_row['pair_id']}",
    )
    overlay_path = out_dir / "overlay_best_pair.png"
    overlay_fig.savefig(overlay_path, dpi=150)
    figure_paths.append(overlay_path)

    worst_ref, worst_search = load_pair_images(worst_row)
    # template footprint in the search image is roughly reference_size /
    # scale_ratio -- use that as the crop window instead of a fixed guess
    scale_ratio = worst_row.get("scale_ratio") or 10.0
    crop_half = max(10.0, worst_ref.shape[1] / (2.0 * scale_ratio))
    failure_fig = visualize.plot_failure_case(
        worst_ref, worst_search, (worst_row["pred_x"], worst_row["pred_y"]),
        (worst_row["gt_x"], worst_row["gt_y"]), error_px=worst_row["error"],
        confidence_ratio=worst_row.get("confidence_ratio", float("nan")),
        n_valid=worst_row.get("n_valid"), crop_half_size=crop_half,
        title=f"Worst pair (failure case): {worst_row['pair_id']}",
    )
    failure_path = out_dir / "failure_worst_pair.png"
    failure_fig.savefig(failure_path, dpi=150)
    figure_paths.append(failure_path)

    print("\nFigures written:")
    for p in figure_paths:
        print(f"  {p}")

    # ---- metrics.json ----
    report = {
        "manifest_path": str(manifest_path),
        "predictions_path": str(predictions_path),
        "n_manifest_pairs": len(manifest_rows),
        "n_predictions_pairs": len(predictions_rows),
        "n_joined_pairs": len(joined),
        "missing_predictions_pair_ids": missing_predictions,
        "missing_manifest_pair_ids": missing_manifest,
        "thresholds_px": args.thresholds,
        "subpixel_threshold_px": args.subpixel_threshold,
        "environment": env_facts,
        "overall": overall,
        "breakdowns": breakdowns,
        "best_pair_id": best_row["pair_id"],
        "worst_pair_id": worst_row["pair_id"],
        "figures": [str(p) for p in figure_paths],
    }
    metrics_json_path = out_dir / "metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=False)
    print(f"\nMetrics JSON: {metrics_json_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
