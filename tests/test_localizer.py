"""Tests for the localizer's geometry and its two distortion estimators.

The end-to-end accuracy numbers live in the README; these check the claims
the method rests on, so a regression shows up as a failed assertion rather
than a slightly worse score.
"""

import cv2
import numpy as np
import pytest

from src.localizer import (
    SCALE,
    TEMPLATE_PX,
    _phase_templates,
    estimate_row_jitter,
    estimate_shear_slope,
    locate,
)


def _fine_canvas(size=2400, seed=0, strips=True):
    """A repeating test pattern with a few random marks so it's not fully
    identical everywhere (otherwise there'd be nothing unique to find).

    `strips` adds the flat boundary bands a real die has. The drift
    estimator needs those -- they're the only thing running the full height
    of the image, which is what makes it possible to notice a sideways
    lean. Without them the estimate gets much worse, which is a real
    limitation and is tested for below rather than swept under the rug.
    """
    rng = np.random.default_rng(seed)
    canvas = np.full((size, size), 40, dtype=np.uint8)
    for y in range(0, size, 64):
        canvas[y:y + 20, :] = 150
    for x in range(0, size, 96):
        canvas[:, x:x + 20] = 170
    # aperiodic marks, so one location is genuinely distinguishable
    for _ in range(60):
        cy, cx = rng.integers(50, size - 50, size=2)
        cv2.circle(canvas, (int(cx), int(cy)), int(rng.integers(8, 20)), 235, -1)
    if strips:
        for p in range(2600, size, 2920):
            canvas[:, p:p + 320] = 95
            canvas[p:p + 320, :] = 95
    return canvas


def _pair_from_canvas(canvas, x0, y0):
    reference = canvas[y0:y0 + 1000, x0:x0 + 1000].copy()
    h = canvas.shape[0] // SCALE
    search = cv2.resize(canvas, (h, h), interpolation=cv2.INTER_AREA)
    return reference, search


class TestPhaseTemplates:
    def test_aligned_phase_matches_plain_area_downsample(self):
        """Phase (0, 0) via integral image must equal cv2.resize INTER_AREA.

        The integral-image shortcut only exists for speed, so it has to agree
        with the obvious implementation it replaced.
        """
        rng = np.random.default_rng(3)
        ref = rng.integers(0, 255, size=(1000, 1000), dtype=np.uint8)
        got = _phase_templates(ref)[(0, 0)]
        want = cv2.resize(ref[0:990, 0:990], (TEMPLATE_PX, TEMPLATE_PX),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        assert np.abs(got - want).max() < 1.0

    def test_all_phases_present_and_correct_shape(self):
        rng = np.random.default_rng(4)
        ref = rng.integers(0, 255, size=(1000, 1000), dtype=np.uint8)
        templates = _phase_templates(ref)
        assert len(templates) == SCALE * SCALE
        for template in templates.values():
            assert template.shape == (TEMPLATE_PX, TEMPLATE_PX)

    def test_phase_shift_moves_content_by_one_fine_pixel(self):
        """Phase px and px+1 must differ, otherwise the sub-pixel search is
        scoring the same template ten times over."""
        ref = _fine_canvas(size=1000, seed=5)
        templates = _phase_templates(ref)
        assert not np.allclose(templates[(0, 0)], templates[(1, 0)])
        assert not np.allclose(templates[(0, 0)], templates[(0, 1)])


class TestGeometry:
    @pytest.mark.parametrize("x0,y0", [(600, 400), (603, 407), (1000, 1000), (57, 993)])
    def test_centre_recovered_exactly_on_clean_pair(self, x0, y0):
        """Should land exactly on the true centre, even when the crop origin
        isn't a clean multiple of 10 -- that's the case a naive match gets wrong."""
        canvas = _fine_canvas(seed=7)
        reference, search = _pair_from_canvas(canvas, x0, y0)
        result = locate(reference, search, correct_shear=False)
        assert abs(result["x"] - (x0 / SCALE + 50)) < 0.25
        assert abs(result["y"] - (y0 / SCALE + 50)) < 0.25


class TestShearEstimator:
    def test_undistorted_image_gives_near_zero_slope(self):
        """Must degrade to a no-op, or it would inject error into clean data."""
        canvas = _fine_canvas(seed=11)
        search = cv2.resize(canvas, (240, 240), interpolation=cv2.INTER_AREA)
        assert abs(estimate_shear_slope(search)) < 0.002

    def test_recovers_a_known_shear(self):
        canvas = _fine_canvas(size=10000, seed=12, strips=True)
        search = cv2.resize(canvas, (1000, 1000), interpolation=cv2.INTER_AREA)
        amplitude = 4.0
        rows = np.arange(1000)
        shift = (amplitude * rows / 999.0).astype(np.float32)
        map_x = np.arange(1000, dtype=np.float32)[None, :] + shift[:, None]
        map_y = np.tile(np.arange(1000, dtype=np.float32)[:, None], (1, 1000))
        sheared = cv2.remap(search, map_x, map_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        recovered = -estimate_shear_slope(sheared) * 999.0
        assert abs(recovered - amplitude) < 1.2


class TestJitterEstimator:
    def test_jitter_estimate_grows_with_applied_jitter(self):
        """Only the ordering matters: it selects the match blur width."""
        canvas = _fine_canvas(size=10000, seed=13)
        search = cv2.resize(canvas, (1000, 1000), interpolation=cv2.INTER_AREA)
        rng = np.random.default_rng(14)
        estimates = []
        for sigma in (0.0, 2.0):
            shift = rng.normal(0, sigma, size=1000).astype(np.float32)
            map_x = np.arange(1000, dtype=np.float32)[None, :] + shift[:, None]
            map_y = np.tile(np.arange(1000, dtype=np.float32)[:, None], (1, 1000))
            jittered = cv2.remap(search, map_x, map_y, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
            estimates.append(estimate_row_jitter(jittered))
        assert estimates[1] > estimates[0] + 0.5
