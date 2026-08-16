"""
Tests for src/metrics.py.

Run with: python -m pytest tests/test_metrics.py -v
(run from the repo root so "src" resolves on sys.path).

All expected values here are computed by hand in the comments next to
each assertion -- the point is to catch a metrics bug that would silently
inflate reported accuracy, not just trust the function under test to
grade itself.
"""

import math

import pytest

from src.metrics import (
    bin_column,
    error_summary,
    euclidean_errors,
    full_metrics,
    group_by_column,
    pass_rate,
    pass_rates_at_thresholds,
    runtime_summary,
    subpixel_pass_rate,
)


# ---------------------------------------------------------------------------
# Hand-verified fixture, reused across several tests.
#
# errors:   3-4-5 triangles and exact threshold boundaries, chosen so every
#           value is exactly checkable by hand:
#             pair 0: pred == true                          -> error = 0.0
#             pair 1: dx=0.3, dy=0.4                         -> error = 0.5
#             pair 2: dx=0.6, dy=0.8                         -> error = 1.0
#             pair 3: dx=3,   dy=4                           -> error = 5.0
#             pair 4: dx=4,   dy=0                           -> error = 4.0
#             pair 5: dx=140, dy=0 (periodic-pattern jump)   -> error = 140.0
#           sorted:  [0.0, 0.5, 1.0, 4.0, 5.0, 140.0]
#           mean   = 150.5 / 6            = 25.083333...
#           median = (1.0 + 4.0) / 2      = 2.5
#           worst  = 140.0
# runtimes: [0.050, 0.060, 0.055, 0.070, 0.045, 0.090]
#           mean = 0.370 / 6 = 0.0616666...
#           worst = 0.090
# ---------------------------------------------------------------------------
FIXTURE_ERRORS = [0.0, 0.5, 1.0, 5.0, 4.0, 140.0]
FIXTURE_RUNTIMES = [0.050, 0.060, 0.055, 0.070, 0.045, 0.090]
FIXTURE_ARCHITECTURE = ["dram", "dram", "finfet", "finfet", "dram", "finfet"]


# ---------------------------------------------------------------------------
# euclidean_errors
# ---------------------------------------------------------------------------

def test_euclidean_errors_known_3_4_5_triangle():
    # classic 3-4-5 right triangle -> hypotenuse 5.0 exactly
    err = euclidean_errors(pred_x=3.0, pred_y=4.0, true_x=0.0, true_y=0.0)
    assert err == pytest.approx(5.0)


def test_euclidean_errors_zero_when_equal():
    err = euclidean_errors(pred_x=10.0, pred_y=-3.5, true_x=10.0, true_y=-3.5)
    assert err == pytest.approx(0.0)


def test_euclidean_errors_vectorized_matches_fixture():
    pred_x = [500.0, 300.3, 700.6, 153.0, 604.0, 340.0]
    pred_y = [500.0, 300.4, 200.8, 854.0, 600.0, 800.0]
    true_x = [500.0, 300.0, 700.0, 150.0, 600.0, 200.0]
    true_y = [500.0, 300.0, 200.0, 850.0, 600.0, 800.0]
    errs = euclidean_errors(pred_x, pred_y, true_x, true_y)
    for computed, expected in zip(errs, FIXTURE_ERRORS):
        assert computed == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# pass_rate: edge cases (single pair, all pass, all fail) and the threshold
# boundary convention (documented as inclusive in metrics.py).
# ---------------------------------------------------------------------------

def test_pass_rate_single_pair_passes():
    assert pass_rate([2.0], threshold=5.0) == pytest.approx(1.0)


def test_pass_rate_single_pair_fails():
    assert pass_rate([9.0], threshold=5.0) == pytest.approx(0.0)


def test_pass_rate_all_pairs_pass():
    assert pass_rate([0.1, 0.2, 1.0, 4.9], threshold=5.0) == pytest.approx(1.0)


def test_pass_rate_all_pairs_fail():
    assert pass_rate([5.1, 10.0, 100.0], threshold=5.0) == pytest.approx(0.0)


def test_pass_rate_empty_is_nan():
    assert math.isnan(pass_rate([], threshold=5.0))


def test_threshold_boundary_is_inclusive():
    # documented convention: error == threshold counts as a PASS.
    assert pass_rate([5.0], threshold=5.0, inclusive=True) == pytest.approx(1.0)
    # one ULP-scale over the threshold must fail either way.
    assert pass_rate([5.0000001], threshold=5.0, inclusive=True) == pytest.approx(0.0)
    # exclusive mode (opt-in) must reject the exact boundary.
    assert pass_rate([5.0], threshold=5.0, inclusive=False) == pytest.approx(0.0)


def test_pass_rates_at_thresholds_matches_hand_computation():
    # FIXTURE_ERRORS sorted: [0.0, 0.5, 1.0, 4.0, 5.0, 140.0], n=6
    # <=5: {0.0,0.5,1.0,4.0,5.0}          -> 5/6
    # <=4: {0.0,0.5,1.0,4.0}              -> 4/6
    # <=2: {0.0,0.5,1.0}                  -> 3/6
    # <=1: {0.0,0.5,1.0}                  -> 3/6
    rates = pass_rates_at_thresholds(FIXTURE_ERRORS, thresholds=(5.0, 4.0, 2.0, 1.0))
    assert rates["pass_rate_5px"] == pytest.approx(5 / 6)
    assert rates["pass_rate_4px"] == pytest.approx(4 / 6)
    assert rates["pass_rate_2px"] == pytest.approx(3 / 6)
    assert rates["pass_rate_1px"] == pytest.approx(3 / 6)


def test_subpixel_pass_rate_default_threshold():
    # <=0.5: {0.0, 0.5} -> 2/6
    assert subpixel_pass_rate(FIXTURE_ERRORS) == pytest.approx(2 / 6)


def test_subpixel_pass_rate_custom_threshold_is_a_named_parameter():
    # threshold is a parameter, not a hardcoded constant: a tighter cutoff
    # (0.2px) should only pass the exact-zero pair -> 1/6.
    assert subpixel_pass_rate(FIXTURE_ERRORS, subpixel_threshold_px=0.2) == pytest.approx(1 / 6)


# ---------------------------------------------------------------------------
# error_summary / runtime_summary
# ---------------------------------------------------------------------------

def test_error_summary_matches_hand_computation():
    s = error_summary(FIXTURE_ERRORS)
    assert s["n"] == 6
    assert s["mean_error_px"] == pytest.approx(150.5 / 6, rel=1e-9)
    assert s["median_error_px"] == pytest.approx(2.5, rel=1e-9)
    assert s["worst_error_px"] == pytest.approx(140.0, rel=1e-9)


def test_error_summary_single_pair():
    s = error_summary([3.7])
    assert s["n"] == 1
    assert s["mean_error_px"] == pytest.approx(3.7)
    assert s["median_error_px"] == pytest.approx(3.7)
    assert s["worst_error_px"] == pytest.approx(3.7)


def test_error_summary_empty_is_nan_not_a_crash():
    s = error_summary([])
    assert s["n"] == 0
    assert math.isnan(s["mean_error_px"])
    assert math.isnan(s["median_error_px"])
    assert math.isnan(s["worst_error_px"])


def test_runtime_summary_matches_hand_computation():
    r = runtime_summary(FIXTURE_RUNTIMES)
    assert r["mean_runtime_sec"] == pytest.approx(0.370 / 6, rel=1e-9)
    assert r["worst_runtime_sec"] == pytest.approx(0.090, rel=1e-9)


# ---------------------------------------------------------------------------
# full_metrics: the exact bundle evaluate.py relies on for both the overall
# summary and every breakdown bucket.
# ---------------------------------------------------------------------------

def test_full_metrics_matches_hand_computation():
    m = full_metrics(FIXTURE_ERRORS, FIXTURE_RUNTIMES)
    assert m["n"] == 6
    assert m["mean_error_px"] == pytest.approx(150.5 / 6, rel=1e-9)
    assert m["median_error_px"] == pytest.approx(2.5, rel=1e-9)
    assert m["worst_error_px"] == pytest.approx(140.0, rel=1e-9)
    assert m["pass_rate_5px"] == pytest.approx(5 / 6)
    assert m["pass_rate_4px"] == pytest.approx(4 / 6)
    assert m["pass_rate_2px"] == pytest.approx(3 / 6)
    assert m["pass_rate_1px"] == pytest.approx(3 / 6)
    assert m["subpixel_pass_rate_0p5px"] == pytest.approx(2 / 6)
    assert m["mean_runtime_sec"] == pytest.approx(0.370 / 6, rel=1e-9)
    assert m["worst_runtime_sec"] == pytest.approx(0.090, rel=1e-9)


def test_full_metrics_without_runtimes_omits_runtime_keys():
    m = full_metrics(FIXTURE_ERRORS)
    assert "mean_runtime_sec" not in m
    assert "worst_runtime_sec" not in m


# ---------------------------------------------------------------------------
# bin_column / group_by_column
# ---------------------------------------------------------------------------

def test_bin_column_categorical_passthrough():
    labels = bin_column(["dram", "dram", "finfet", "finfet", "dram", "finfet"])
    assert labels == ["dram", "dram", "finfet", "finfet", "dram", "finfet"]


def test_bin_column_continuous_auto_bins_into_requested_count():
    values = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0, 30.0]
    labels = bin_column(values, n_bins=3, max_categorical=5)
    assert len(set(labels)) == 3  # exactly 3 distinct equal-width buckets used
    assert len(labels) == len(values)


def test_bin_column_constant_column_single_bin():
    labels = bin_column([7.0, 7.0, 7.0, 7.0], max_categorical=1)
    assert len(set(labels)) == 1


def test_group_by_column_categorical_matches_hand_computation():
    rows = [
        {"pair_id": str(i), "error": e, "architecture": a}
        for i, (e, a) in enumerate(zip(FIXTURE_ERRORS, FIXTURE_ARCHITECTURE))
    ]
    # dram rows (indices 0,1,4): errors [0.0, 0.5, 4.0] -> mean 1.5, median 0.5, worst 4.0
    # finfet rows (indices 2,3,5): errors [1.0, 5.0, 140.0] -> mean 48.6667, median 5.0, worst 140.0
    grouped = group_by_column(rows, "architecture")
    assert set(grouped.keys()) == {"dram", "finfet"}
    assert grouped["dram"]["n"] == 3
    assert grouped["dram"]["mean_error_px"] == pytest.approx(1.5)
    assert grouped["dram"]["median_error_px"] == pytest.approx(0.5)
    assert grouped["dram"]["worst_error_px"] == pytest.approx(4.0)
    assert grouped["finfet"]["n"] == 3
    assert grouped["finfet"]["mean_error_px"] == pytest.approx((1.0 + 5.0 + 140.0) / 3)
    assert grouped["finfet"]["median_error_px"] == pytest.approx(5.0)
    assert grouped["finfet"]["worst_error_px"] == pytest.approx(140.0)


def test_group_by_column_continuous_auto_binning_covers_all_rows():
    rows = [
        {"pair_id": str(i), "error": e, "scale_ratio": s}
        for i, (e, s) in enumerate(zip(
            FIXTURE_ERRORS, [10.0, 9.5, 10.5, 11.0, 9.0, 10.0]
        ))
    ]
    grouped = group_by_column(rows, "scale_ratio", n_bins=5, max_categorical=2)
    total_n = sum(bucket["n"] for bucket in grouped.values())
    assert total_n == len(rows)  # every row lands in exactly one bucket
    assert len(grouped) <= 5


def test_group_by_column_empty_rows_returns_empty_dict():
    assert group_by_column([], "architecture") == {}


def test_group_by_column_with_runtime_key():
    rows = [
        {"pair_id": str(i), "error": e, "runtime_sec": r, "architecture": a}
        for i, (e, r, a) in enumerate(zip(FIXTURE_ERRORS, FIXTURE_RUNTIMES, FIXTURE_ARCHITECTURE))
    ]
    grouped = group_by_column(rows, "architecture", runtime_key="runtime_sec")
    # dram runtimes (indices 0,1,4): [0.050, 0.060, 0.045] -> mean 0.05166..., worst 0.060
    assert grouped["dram"]["mean_runtime_sec"] == pytest.approx((0.050 + 0.060 + 0.045) / 3)
    assert grouped["dram"]["worst_runtime_sec"] == pytest.approx(0.060)
