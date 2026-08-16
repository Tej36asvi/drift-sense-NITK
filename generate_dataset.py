#!/usr/bin/env python3
"""CLI to generate a Drift-Sense synthetic dataset.

Uses the Applied Materials starter resource for the actual image
generation (see README.md for why) -- this script just calls it and saves
the output in the flat layout localize.py and evaluate.py expect.

Example:
    python generate_dataset.py --num-samples 30 --output-dir results/dataset --seed 42
"""

import argparse
import csv
import os

import cv2
import numpy as np

from dataclasses import replace as dataclass_replace

from src.pipeline import GenerationParams, SEARCH_VARIANTS, generate_sample
from src.presets import PRESETS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-samples", type=int, default=30)
    p.add_argument("--architectures", nargs="+", default=list(PRESETS.keys()), choices=list(PRESETS.keys()))
    p.add_argument("--output-dir", default="results/dataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beam-spot-size-nm", type=float, default=GenerationParams.beam_spot_size_nm)
    p.add_argument("--collapse-threshold-nm", type=float, default=GenerationParams.collapse_threshold_nm)
    p.add_argument("--dose-reference", type=float, default=GenerationParams.dose_reference)
    p.add_argument("--dose-search", type=float, default=GenerationParams.dose_search)
    p.add_argument("--shear-amplitude-px", type=float, default=GenerationParams.shear_amplitude_px)
    p.add_argument("--drift-jitter-px", type=float, default=GenerationParams.drift_jitter_px)
    p.add_argument("--astigmatism-ratio", type=float, default=GenerationParams.astigmatism_ratio)
    p.add_argument("--vignette-strength", type=float, default=GenerationParams.vignette_strength)
    p.add_argument("--gamma", type=float, default=GenerationParams.gamma)
    p.add_argument("--barrel-distortion-k", type=float, default=GenerationParams.barrel_distortion_k)
    p.add_argument("--charging-streak-prob", type=float, default=GenerationParams.charging_streak_prob)
    p.add_argument("--charging-streak-intensity", type=float, default=GenerationParams.charging_streak_intensity)
    p.add_argument("--speckle-sigma", type=float, default=GenerationParams.speckle_sigma)
    p.add_argument("--salt-pepper-prob", type=float, default=GenerationParams.salt_pepper_prob)
    p.add_argument("--linewidth-bias-nm", type=float, default=GenerationParams.linewidth_bias_nm)
    p.add_argument("--corner-rounding-px", type=float, default=GenerationParams.corner_rounding_px)
    p.add_argument("--mat-size-nm", type=float, default=GenerationParams.mat_size_nm)
    p.add_argument("--strip-width-nm", type=float, default=GenerationParams.strip_width_nm)
    p.add_argument("--boundary-bias", type=float, default=GenerationParams.boundary_bias)
    p.add_argument("--vary-conditions", action="store_true",
                    help="cycle pairs through different scan conditions "
                         "(clean, low dose, heavy drift, speckle/salt-pepper, charging) "
                         "instead of using the same one for every pair")
    return p.parse_args()


def _conditions(base_params, vary):
    """Which scan conditions to use for each pair.

    Without --vary-conditions, every pair uses the same settings, so there
    would be no real noise-level variety to compare results across.
    """
    if not vary:
        return [("default", base_params)]
    out = [("default", base_params)]
    for variant in SEARCH_VARIANTS:
        overrides = {k: v for k, v in variant.items() if k != "label"}
        out.append((variant["label"], dataclass_replace(base_params, **overrides)))
    return out


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    params = GenerationParams(
        beam_spot_size_nm=args.beam_spot_size_nm,
        collapse_threshold_nm=args.collapse_threshold_nm,
        dose_reference=args.dose_reference,
        dose_search=args.dose_search,
        shear_amplitude_px=args.shear_amplitude_px,
        drift_jitter_px=args.drift_jitter_px,
        astigmatism_ratio=args.astigmatism_ratio,
        vignette_strength=args.vignette_strength,
        gamma=args.gamma,
        barrel_distortion_k=args.barrel_distortion_k,
        charging_streak_prob=args.charging_streak_prob,
        charging_streak_intensity=args.charging_streak_intensity,
        speckle_sigma=args.speckle_sigma,
        salt_pepper_prob=args.salt_pepper_prob,
        linewidth_bias_nm=args.linewidth_bias_nm,
        corner_rounding_px=args.corner_rounding_px,
        mat_size_nm=args.mat_size_nm,
        strip_width_nm=args.strip_width_nm,
        boundary_bias=args.boundary_bias,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    fieldnames = [
        "pair_id", "reference_path", "search_path", "gt_x", "gt_y",
        "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h", "architecture",
        "beam_spot_size_nm", "collapse_threshold_nm", "dose_reference",
        "dose_search", "shear_amplitude_px", "drift_jitter_px",
        "astigmatism_ratio", "vignette_strength", "gamma", "barrel_distortion_k",
        "charging_streak_prob", "charging_streak_intensity",
        "speckle_sigma", "salt_pepper_prob",
        "linewidth_bias_nm", "corner_rounding_px",
        "mat_size_nm", "strip_width_nm", "boundary_bias", "seed", "condition",
    ]

    conditions = _conditions(params, args.vary_conditions)

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for i in range(args.num_samples):
            architecture = args.architectures[int(rng.integers(0, len(args.architectures)))]
            cond_label, cond_params = conditions[i % len(conditions)]
            sample = generate_sample(architecture, rng, cond_params)

            # plain filenames next to the manifest, nothing nested
            ref_name = f"reference_{i:03d}.png"
            search_name = f"search_{i:03d}.png"
            cv2.imwrite(os.path.join(args.output_dir, ref_name), sample["reference_img"])
            cv2.imwrite(os.path.join(args.output_dir, search_name), sample["search_img"])

            gx0, gy0, gw, gh = sample["gt_box"]
            writer.writerow({
                "pair_id": f"pair_{i:03d}",
                "reference_path": ref_name,
                "search_path": search_name,
                "gt_x": sample["gt_x"],
                "gt_y": sample["gt_y"],
                "gt_box_x": gx0, "gt_box_y": gy0, "gt_box_w": gw, "gt_box_h": gh,
                "architecture": architecture,
                **sample["params"],
                "seed": args.seed,
                "condition": cond_label,
            })
            print(f"[{i + 1}/{args.num_samples}] {architecture} [{cond_label}] "
                   f"-> gt=({sample['gt_x']:.1f}, {sample['gt_y']:.1f})")

    print(f"Wrote {args.num_samples} pairs to {args.output_dir}")


if __name__ == "__main__":
    main()
