"""
CLI entry point for Drift-Sense localization.

Two modes:

    python localize.py --reference REF.png --search SEARCH.png
    python localize.py --manifest M.csv --out predictions.csv

Single-pair mode prints "x.xx,y.yy" to stdout and nothing else -- this is
the graded interface, so stdout has to stay clean. Batch mode writes a
predictions CSV, one row per manifest row, with per-pair runtime_sec
measured by time.perf_counter().

Orchestrates:
    src.localizer.find_candidates    stage 1: coarse candidates
    src.localizer.select_winner      stage 2: filter + closest-to-centre
    src.refine.refine_subpixel       stage 3: sub-pixel refinement

localize_pair() is a pure function (two image paths in, a result dict out)
kept separate from argument parsing, so evaluate.py and the tests can call
it directly without going through argparse or stdout.
"""

import argparse
import csv
import os
import sys
import time

import cv2

from src.localizer import locate

# predictions CSV column order -- keep in sync with evaluate.py
PREDICTIONS_FIELDNAMES = [
    "pair_id", "reference_path", "search_path", "pred_x", "pred_y",
    "score", "phase_x", "phase_y", "shear_slope", "runtime_sec",
]


def localize_pair(reference_path: str, search_path: str) -> dict:
    """Two image paths in, a full result dict out.

    Runs the phase-aligned match plus raster-drift correction, timed end to
    end with time.perf_counter().

    -> {"x", "y", "score", "phase_x", "phase_y", "shear_slope", "runtime_sec"}
    """
    start = time.perf_counter()

    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    if reference is None:
        raise FileNotFoundError(f"could not read reference image: {reference_path}")
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if search is None:
        raise FileNotFoundError(f"could not read search image: {search_path}")

    result = locate(reference, search)
    result["runtime_sec"] = time.perf_counter() - start
    return result


def run_single(reference_path: str, search_path: str) -> None:
    """Single-pair mode: print "x.xx,y.yy" to stdout, nothing else."""
    result = localize_pair(reference_path, search_path)
    print(f"{result['x']:.2f},{result['y']:.2f}")


def run_batch(manifest_path: str, out_path: str) -> None:
    """Batch mode: read the generator's manifest CSV, write predictions.

    Paths in the manifest are relative to the manifest's own directory,
    not the caller's cwd, so a dataset stays portable.
    """
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))

    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTIONS_FIELDNAMES)
        writer.writeheader()

        for row in rows:
            reference_path = os.path.join(manifest_dir, row["reference_path"])
            search_path = os.path.join(manifest_dir, row["search_path"])

            try:
                result = localize_pair(reference_path, search_path)
            except Exception as exc:
                # one bad pair shouldn't kill the whole batch -- log it
                # and leave that row's numeric fields blank instead
                print(f"warning: pair {row.get('pair_id')} failed: {exc}",
                      file=sys.stderr)
                writer.writerow({
                    "pair_id": row.get("pair_id", ""),
                    "reference_path": row.get("reference_path", ""),
                    "search_path": row.get("search_path", ""),
                    "pred_x": "", "pred_y": "", "score": "",
                    "phase_x": "", "phase_y": "", "shear_slope": "",
                    "runtime_sec": "",
                })
                continue

            writer.writerow({
                "pair_id": row.get("pair_id", ""),
                "reference_path": row["reference_path"],
                "search_path": row["search_path"],
                "pred_x": f"{result['x']:.4f}",
                "pred_y": f"{result['y']:.4f}",
                "score": round(result["score"], 6),
                "phase_x": result["phase_x"],
                "phase_y": result["phase_y"],
                "shear_slope": round(result["shear_slope"], 8),
                "runtime_sec": f"{result['runtime_sec']:.4f}",
            })


def main():
    parser = argparse.ArgumentParser(
        description="Locate a reference pattern inside a larger search image.")
    parser.add_argument("--reference", dest="reference_path",
                         help="path to the reference image (single-pair mode)")
    parser.add_argument("--search", dest="search_path",
                         help="path to the search image (single-pair mode)")
    parser.add_argument("--manifest", dest="manifest_path",
                         help="path to the generator's manifest CSV (batch mode)")
    parser.add_argument("--out", dest="out_path",
                         help="path to write the predictions CSV (batch mode)")
    args = parser.parse_args()

    single_requested = bool(args.reference_path or args.search_path)
    batch_requested = bool(args.manifest_path or args.out_path)

    if single_requested and batch_requested:
        parser.error("use either --reference/--search or --manifest/--out, not both")
    elif single_requested:
        if not (args.reference_path and args.search_path):
            parser.error("--reference and --search are both required in single-pair mode")
        run_single(args.reference_path, args.search_path)
    elif batch_requested:
        if not (args.manifest_path and args.out_path):
            parser.error("--manifest and --out are both required in batch mode")
        run_batch(args.manifest_path, args.out_path)
    else:
        parser.error("specify either --reference/--search or --manifest/--out")


if __name__ == "__main__":
    main()
