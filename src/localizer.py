"""
Finds where the small reference image sits inside the larger search image.

The search image is just the reference shrunk down by exactly 10x, so most
of the work here is being careful about that shrink: lining the pixel grids
up correctly, undoing some scan wobble, and picking the right match when a
few spots look almost identical.
"""

import cv2
import numpy as np
from skimage.registration import phase_cross_correlation

# reference is 10x bigger (finer) than search
SCALE = 10
TEMPLATE_PX = 99  # size of the shrunk-down template, with a little room to spare

# gentle blur before comparing images, so small misalignments don't matter too much
MATCH_BLUR_SIGMA = 1.0

# extra horizontal blur, scaled to how shaky the scan looks (see estimate_row_jitter)
BLUR_SIGMA_X_MIN = 1.0
BLUR_SIGMA_X_MAX = 3.2
BLUR_SIGMA_X_GAIN = 1.25

# a few starting offsets used to get a rough position quickly, before the precise pass
COARSE_PHASES = ((0, 0), (5, 0), (0, 5), (5, 5))

# how many rough matches to keep, and how far apart they need to be
COARSE_TOP_K = 25
COARSE_NMS_RADIUS_PX = 6

# how far to search around each rough match to find the precise spot
FINE_RADIUS_PX = 4

# scores this close to the best one count as tied; ties are broken by centre distance
VALID_TOLERANCE = 0.002

# Small-rotation pre-check. The problem statement says rotation may be
# 1-2 degrees. A rotation error doesn't reliably show up as a lower match
# score (checked directly: 1 degree of uncorrected rotation gave scores of
# 0.87-0.97, indistinguishable from a clean match, while the actual
# position error was 6-35px) -- so unlike scale, we can't just detect this
# from a weak score and fall back. Instead we always check a small set of
# rotation candidates upfront and use whichever one matches best.
ROTATION_CANDIDATES_DEG = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)

# Scale fallback. First version of this only ran when the fast method's
# score looked weak (checked once: a 7% scale error dropped score from
# ~0.93-0.99 down to 0.37-0.74). Checked more broadly afterwards and that
# didn't hold up -- a scale mismatch can also score misleadingly high, the
# same problem rotation has -- so this now always runs and we just keep
# whichever of the two answers scores better. Costs about 0.7s extra per
# pair; worth it since a gated version was silently trusting wrong answers.
SCALE_FALLBACK_RANGE = (9.0, 11.0)
SCALE_FALLBACK_STEPS = 9


def _phase_templates(reference):
    """Build the reference image shrunk down at all 100 possible pixel offsets.

    We don't know exactly where the "shrink by 10" grid lines up, so we try
    every starting offset. Uses a running-sum trick instead of resizing the
    image 100 separate times.
    """
    ref = reference.astype(np.float64)
    ii = cv2.integral(ref)  # running sum of everything above-left of each pixel
    n = TEMPLATE_PX
    out = {}
    for py in range(SCALE):
        for px in range(SCALE):
            r0 = py + SCALE * np.arange(n)
            c0 = px + SCALE * np.arange(n)
            r1, c1 = r0 + SCALE, c0 + SCALE
            block = (ii[np.ix_(r1, c1)] - ii[np.ix_(r0, c1)]
                     - ii[np.ix_(r1, c0)] + ii[np.ix_(r0, c0)])
            out[(px, py)] = (block / (SCALE * SCALE)).astype(np.float32)
    return out


def _blur(img, sigma):
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma)


def _local_peaks(surface, top_k, radius):
    """Pick out the strongest, spread-out matches instead of a pile of near-duplicates."""
    flat = surface.ravel()
    order = np.argsort(flat)[::-1][:max(top_k * 40, 200)]
    kept = []
    for idx in order:
        y, x = divmod(int(idx), surface.shape[1])
        if all((x - kx) ** 2 + (y - ky) ** 2 > radius ** 2 for kx, ky in kept):
            kept.append((x, y))
        if len(kept) >= top_k:
            break
    return kept


def _best_in_window(search_blur, template, cx, cy, radius):
    """Best match score in a small area around one point.

    Lets OpenCV do the searching in one call instead of a slow Python loop.
    -> (score, x0, y0), the winning top-left corner.
    """
    n = template.shape[0]
    h, w = search_blur.shape
    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    x1 = min(w, cx + radius + n)
    y1 = min(h, cy + radius + n)
    if x1 - x0 < n or y1 - y0 < n:
        return -np.inf, 0, 0
    window = search_blur[y0:y1, x0:x1]
    surface = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(surface)
    return float(score), x0 + loc[0], y0 + loc[1]


def _subpixel_profile_shift(a, b, max_shift):
    """How far does b need to shift to line up with a? Down to a fraction of a pixel.

    Kept to a small search range on purpose -- scan drift is only ever a
    couple of pixels, and a wider search risks locking onto the wrong
    repeat of the pattern instead.
    """
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0, 0.0
    a, b = a / na, b / nb
    shifts = np.arange(-max_shift, max_shift + 1)
    cc = np.array([float(np.dot(a, np.roll(b, -int(s)))) for s in shifts])
    k = int(np.argmax(cc))
    peak = float(cc[k])
    if k == 0 or k == len(cc) - 1:
        return float(shifts[k]), peak
    y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
    den = y0 - 2 * y1 + y2
    delta = 0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0
    return float(shifts[k] + delta), peak


def estimate_shear_slope(search, n_bands=8, max_shift=4):
    """Estimate how much the scan drifts sideways as it goes down the image.

    That drift is too small to notice inside one small patch, so instead we
    look at the strip boundaries between memory blocks, which run the full
    height of the image, and measure how much they visually "lean". Returns
    0 for an image with no drift, so this safely does nothing on clean data.
    """
    h, w = search.shape[:2]
    f = search.astype(np.float32)
    texture = cv2.GaussianBlur(np.abs(f - cv2.GaussianBlur(f, (0, 0), 2.0)), (0, 0), 4.0)

    band_h = h // n_bands
    profiles, centres = [], []
    for i in range(n_bands):
        prof = texture[i * band_h:(i + 1) * band_h, :].mean(axis=0)
        prof = prof - cv2.GaussianBlur(prof.reshape(-1, 1), (0, 0), 30).ravel()
        profiles.append(prof)
        centres.append((i + 0.5) * band_h)

    mid = n_bands // 2
    shifts, weights = [], []
    for prof in profiles:
        s, peak = _subpixel_profile_shift(profiles[mid], prof, max_shift)
        shifts.append(s)
        weights.append(max(peak, 0.0))

    weights = np.array(weights)
    if weights.sum() < 1e-6:
        return 0.0
    return float(np.polyfit(np.array(centres), np.array(shifts), 1, w=weights)[0])


def _rotate_reference(reference, angle_deg):
    """Rotate the reference at full resolution, about its own centre."""
    if angle_deg == 0.0:
        return reference
    h, w = reference.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    return cv2.warpAffine(reference, M, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)


def _detect_rotation(reference, search):
    """Which small rotation (if any) best lines the reference up with the
    search image? One cheap, single-phase check per candidate angle across
    the whole image -- rough on purpose, just enough to pick the best
    candidate and hand it to the precise phase search afterwards.
    """
    best_angle, best_score = 0.0, -np.inf
    n = TEMPLATE_PX
    for angle in ROTATION_CANDIDATES_DEG:
        rotated = _rotate_reference(reference, angle)
        crop = rotated[:SCALE * n, :SCALE * n]
        template = cv2.resize(crop, (n, n), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)
        if score > best_score:
            best_score, best_angle = float(score), angle
    return best_angle


def _scale_fallback(reference, search):
    """Fallback for when the main method's score looks weak: sweep a range
    of scale ratios.

    The first version of this just kept the single best match at each
    scale -- and lost to periodic decoys almost immediately, the exact
    same mistake already fixed once earlier in this project. Fixed the
    same way: pool several candidates per scale, not just the top one,
    then use the same tight-tolerance closest-to-centre tie-break as the
    main method to choose between them, instead of blind argmax.

    Finishes with a sub-pixel refinement pass on the winner via
    phase_cross_correlation, since the exact-arithmetic trick the main
    method relies on only applies at exactly scale 10 -- off that, this
    needs a real (if approximate) sub-pixel step of its own.

    -> (score, x, y), or None if nothing could be scored.
    """
    h, w = reference.shape[:2]
    search_h, search_w = search.shape[:2]

    pool = []
    for scale in np.linspace(*SCALE_FALLBACK_RANGE, SCALE_FALLBACK_STEPS):
        tw, th = int(round(w / scale)), int(round(h / scale))
        if tw >= search_w or th >= search_h:
            continue
        template = cv2.resize(reference, (tw, th), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        for x, y in _local_peaks(result, top_k=5, radius=COARSE_NMS_RADIUS_PX):
            pool.append((float(result[y, x]), x, y, tw, th))

    if not pool:
        return None

    pool.sort(key=lambda c: -c[0])
    kept = []
    for cand in pool:
        _, x, y, tw, th = cand
        cx, cy = x + tw / 2.0, y + th / 2.0
        too_close = any(
            (cx - (k[1] + k[3] / 2.0)) ** 2 + (cy - (k[2] + k[4] / 2.0)) ** 2
            <= COARSE_NMS_RADIUS_PX ** 2
            for k in kept
        )
        if not too_close:
            kept.append(cand)
        if len(kept) >= COARSE_TOP_K:
            break

    best_score = kept[0][0]
    valid = [c for c in kept if c[0] >= best_score - VALID_TOLERANCE]
    cx0, cy0 = search_w / 2.0, search_h / 2.0

    def centre_distance(c):
        _, x, y, tw, th = c
        return (x + tw / 2.0 - cx0) ** 2 + (y + th / 2.0 - cy0) ** 2

    score, x, y, tw, th = min(valid, key=centre_distance)

    # sub-pixel polish on the winner only
    template = cv2.resize(reference, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float64)
    margin = 4
    y0, y1 = max(0, y - margin), min(search_h, y + th + margin)
    x0, x1 = max(0, x - margin), min(search_w, x + tw + margin)
    dx = dy = 0.0
    if (y1 - y0) >= th + margin and (x1 - x0) >= tw + margin:
        window = search[y0:y1, x0:x1].astype(np.float64)
        crop = window[y - y0:y - y0 + th, x - x0:x - x0 + tw]
        try:
            shift, _err, _phase = phase_cross_correlation(template, crop, upsample_factor=20)
            dy, dx = float(shift[0]), float(shift[1])
        except Exception:
            dy = dx = 0.0

    return (score, x + tw / 2.0 + dx, y + th / 2.0 + dy)


def _locate_at_blur(reference, search, templates, template_sigma_x, base_sigma):
    """Run one full search pass at a given blur setting.

    -> (score, corner_x, corner_y, phase_x, phase_y, n_candidates)
    """
    search_blur = cv2.GaussianBlur(search.astype(np.float32), (0, 0),
                                    sigmaX=template_sigma_x, sigmaY=base_sigma)
    blurred = {ph: cv2.GaussianBlur(t, (0, 0), sigmaX=template_sigma_x, sigmaY=base_sigma)
               for ph, t in templates.items()}

    # quick pass: check a few offsets across the whole image, keep the best of each spot
    combined = None
    for ph in COARSE_PHASES:
        surface = cv2.matchTemplate(search_blur, blurred[ph], cv2.TM_CCOEFF_NORMED)
        combined = surface if combined is None else np.maximum(combined, surface)

    candidates = _local_peaks(combined, COARSE_TOP_K, COARSE_NMS_RADIUS_PX)
    if not candidates:
        raise ValueError("no correlation peak found")

    # precise pass: try every offset, but only near the rough matches from above
    scored = []
    for cx, cy in candidates:
        local = (-np.inf, 0, 0, 0, 0)
        for (px, py), template in blurred.items():
            score, ax, ay = _best_in_window(search_blur, template, cx, cy, FINE_RADIUS_PX)
            if score > local[0]:
                local = (score, ax, ay, px, py)
        if local[0] > -np.inf:
            scored.append(local)

    if not scored:
        raise ValueError("no candidate could be scored")

    # if a few spots score almost the same, go with whichever is closest to
    # the centre -- this is the tie-break rule the problem statement asks for
    best_score = max(s[0] for s in scored)
    valid = [s for s in scored if s[0] >= best_score - VALID_TOLERANCE]
    h, w = search.shape[:2]
    cx0, cy0 = w / 2.0, h / 2.0

    def centre_distance(entry):
        _, ax, ay, _, _ = entry
        return (ax + TEMPLATE_PX / 2.0 - cx0) ** 2 + (ay + TEMPLATE_PX / 2.0 - cy0) ** 2

    score, ax, ay, px, py = min(valid, key=centre_distance)
    return score, ax, ay, px, py, len(candidates)


def locate(reference, search, correct_shear=True, correct_rotation=True,
           allow_scale_fallback=True):
    """Find where the reference sits in the search image.

    Checks a small set of rotation candidates upfront (see module
    docstring for why this can't just be detected from a low score
    afterwards), then runs the precise phase search using whichever
    rotation matched best. Also always tries a range of scale ratios as a
    second opinion and keeps whichever of the two answers scores higher --
    a low score on the fast method turned out not to be a reliable enough
    signal to only check sometimes, so this always runs both instead.

    Blurs a little more when the scan looks shaky, and skips the drift
    correction entirely if no drift is detected.

    -> {"x", "y", "score", "phase_x", "phase_y", "shear_slope",
        "row_jitter_px", "n_candidates", "detected_rotation_deg",
        "used_scale_fallback"}
    """
    detected_rotation = _detect_rotation(reference, search) if correct_rotation else 0.0
    working_reference = _rotate_reference(reference, detected_rotation)

    templates = _phase_templates(working_reference)

    jitter = estimate_row_jitter(search)
    sigma_x = float(np.clip(BLUR_SIGMA_X_GAIN * jitter,
                             BLUR_SIGMA_X_MIN, BLUR_SIGMA_X_MAX))
    score, ax, ay, px, py, n_cand = _locate_at_blur(
        working_reference, search, templates, sigma_x, MATCH_BLUR_SIGMA)

    centre_x = ax + TEMPLATE_PX / 2.0 + 0.5 - px / SCALE
    centre_y = ay + TEMPLATE_PX / 2.0 + 0.5 - py / SCALE

    used_fallback = False
    if allow_scale_fallback:
        fallback = _scale_fallback(working_reference, search)
        if fallback is not None and fallback[0] > score:
            score, centre_x, centre_y = fallback
            used_fallback = True

    shear_slope = estimate_shear_slope(search) if correct_shear else 0.0
    centre_x = centre_x - shear_slope * centre_y

    return {
        "x": float(centre_x),
        "y": float(centre_y),
        "score": float(score),
        "phase_x": int(px),
        "phase_y": int(py),
        "shear_slope": float(shear_slope),
        "row_jitter_px": float(jitter),
        "n_candidates": int(n_cand),
        "detected_rotation_deg": float(detected_rotation),
        "used_scale_fallback": bool(used_fallback),
    }


def estimate_row_jitter(search, n_rows=300, max_shift=6):
    """How shaky is the scan, row by row?

    Neighbouring rows show almost the same content, so whatever shift lines
    up row r with row r+1 is mostly just the scan's own wobble. About 0.2px
    on a clean scan, about 2.5px on a shaky one -- a clear enough gap to
    tell them apart.
    """
    f = search.astype(np.float32)
    f = f - cv2.GaussianBlur(f, (0, 0), 15)  # drop slow shading, keep structure
    h = f.shape[0]
    indices = np.linspace(5, h - 6, n_rows).astype(int)

    shifts = []
    for r in indices:
        a, b = f[r], f[r + 1]
        a = a - a.mean()
        b = b - b.mean()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            continue
        a, b = a / na, b / nb
        cc = np.array([float(np.dot(a, np.roll(b, -s)))
                       for s in range(-max_shift, max_shift + 1)])
        k = int(np.argmax(cc))
        if k == 0 or k == len(cc) - 1:
            continue
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        den = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0
        shifts.append(k - max_shift + delta)

    return float(np.std(shifts)) if shifts else 0.0
