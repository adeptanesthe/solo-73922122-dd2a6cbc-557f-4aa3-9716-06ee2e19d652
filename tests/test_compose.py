"""Exactness tests for the solver against a straightforward reference.

The reference enumerates every alpha and every foreground value per channel
(a few hundred thousand small checks per pixel) and is only used on tiny
images.
"""

import numpy as np
import pytest

from app.compose import blend, compose


def reference_pixel(s1, s2, bg1, bg2, bnew):
    """Return (feasible, unique, output) for one pixel."""
    bg1 = np.asarray(bg1, dtype=np.int64)
    bg2 = np.asarray(bg2, dtype=np.int64)
    bnew = np.asarray(bnew, dtype=np.int64)
    s1 = np.asarray(s1, dtype=np.int64)
    s2 = np.asarray(s2, dtype=np.int64)
    fg = np.arange(256, dtype=np.int64)
    out_sets = [set() for _ in range(3)]
    feasible = False
    for alpha in range(256):
        allowed = []
        ok = True
        for c in range(3):
            o1 = (alpha * fg + (255 - alpha) * bg1[c] + 127) // 255
            o2 = (alpha * fg + (255 - alpha) * bg2[c] + 127) // 255
            vals = fg[(o1 == s1[c]) & (o2 == s2[c])]
            if vals.size == 0:
                ok = False
                break
            allowed.append(vals)
        if not ok:
            continue
        feasible = True
        for c in range(3):
            outs = (alpha * allowed[c] + (255 - alpha) * bnew[c] + 127) // 255
            out_sets[c].update(int(v) for v in np.unique(outs))
    if not feasible:
        return False, False, None
    unique = all(len(s) == 1 for s in out_sets)
    output = tuple(next(iter(s)) for s in out_sets) if unique else None
    return True, unique, output


def solve_pixels(s1, s2, bg1, bg2, bnew):
    s1 = np.asarray(s1, dtype=np.uint8).reshape(-1, 3)
    s2 = np.asarray(s2, dtype=np.uint8).reshape(-1, 3)
    bnew = np.asarray(bnew, dtype=np.uint8).reshape(-1, 3)
    return compose(s1.reshape(1, -1, 3), s2.reshape(1, -1, 3), bg1, bg2,
                   bnew.reshape(1, -1, 3), chunk_rows=7)


def assert_matches_reference(s1, s2, bg1, bg2, bnew, result):
    s1 = np.asarray(s1, dtype=np.uint8).reshape(-1, 3)
    s2 = np.asarray(s2, dtype=np.uint8).reshape(-1, 3)
    bnew = np.asarray(bnew, dtype=np.uint8).reshape(-1, 3)
    for i in range(s1.shape[0]):
        feasible, unique, output = reference_pixel(
            s1[i], s2[i], bg1, bg2, bnew[i])
        assert bool(result.unique.reshape(-1)[i]) == unique, (
            bg1, bg2, s1[i], s2[i], bnew[i], feasible, unique, output)
        assert bool(result.contradiction.reshape(-1)[i]) == (not feasible)
        assert bool(result.ambiguous.reshape(-1)[i]) == (feasible and not unique)
        if unique:
            assert tuple(result.image.reshape(-1, 3)[i]) == output


def test_blend_rounding_is_half_up():
    assert blend((0, 0, 0), 0, (0, 0, 0)) == (0, 0, 0)
    assert blend((255, 255, 255), 255, (0, 0, 0)) == (255, 255, 255)
    # 128 * 1 / 255 = 0.502 -> 1 ; 127 * 1 / 255 = 0.498 -> 0
    assert blend((128, 127, 128), 1, (0, 0, 0)) == (1, 0, 1)
    # 128 * 255 / 255 = 128 exactly ; (255 - 128) * 255 / 255 = 127 exactly
    assert blend((0, 255, 0), 128, (255, 0, 0)) == (127, 128, 0)
    assert blend((0, 0, 0), 0, (17, 34, 51)) == (17, 34, 51)
    with pytest.raises(ValueError):
        blend((0, 0, 0), 256, (0, 0, 0))


def test_alpha_zero_pixel_is_ambiguous_with_low_alpha():
    # At alpha 0 the samples equal the backgrounds; at alpha 1 the same
    # samples are still reproducible with several foreground values, so the
    # new background does not determine one RGB triple.
    bg1 = (10, 20, 30)
    bg2 = (200, 100, 50)
    result = solve_pixels([[10, 20, 30]], [[200, 100, 50]], bg1, bg2,
                          [[7, 8, 9]])
    assert_matches_reference([[10, 20, 30]], [[200, 100, 50]], bg1, bg2,
                             [[7, 8, 9]], result)
    assert result.ambiguous[0, 0]


def test_replacement_with_same_background_returns_sample_where_unique():
    rng = np.random.default_rng(0)
    bg = (40, 120, 200)
    fg = rng.integers(0, 256, (64, 3), dtype=np.uint8)
    alpha = rng.integers(0, 256, (64, 1), dtype=np.uint8)
    bg_arr = np.array(bg, dtype=np.int64)
    sample = ((alpha.astype(np.int64) * fg.astype(np.int64)
               + (255 - alpha.astype(np.int64)) * bg_arr + 127) // 255)
    sample = sample.astype(np.uint8)
    result = solve_pixels(sample, sample, bg, bg, sample)
    assert_matches_reference(sample, sample, bg, bg, sample, result)
    unique = result.unique.reshape(-1)
    np.testing.assert_array_equal(result.image.reshape(-1, 3)[unique],
                                  sample[unique])


def test_random_pixels_match_reference():
    rng = np.random.default_rng(1234)
    for _ in range(5):
        bg1 = tuple(int(v) for v in rng.integers(0, 256, 3))
        bg2 = tuple(int(v) for v in rng.integers(0, 256, 3))
        count = 40
        s1 = rng.integers(0, 256, (count, 3), dtype=np.uint8)
        s2 = rng.integers(0, 256, (count, 3), dtype=np.uint8)
        bnew = rng.integers(0, 256, (count, 3), dtype=np.uint8)
        # Make about half the pixels genuine composites so both outcomes occur.
        alpha = rng.integers(0, 256, count)[:, None]
        fg = rng.integers(0, 256, (count, 3))
        genuine = rng.random(count) < 0.5
        comp1 = ((alpha * fg + (255 - alpha) * np.array(bg1) + 127) // 255)
        comp2 = ((alpha * fg + (255 - alpha) * np.array(bg2) + 127) // 255)
        s1[genuine] = comp1[genuine].astype(np.uint8)
        s2[genuine] = comp2[genuine].astype(np.uint8)
        result = solve_pixels(s1, s2, bg1, bg2, bnew)
        assert_matches_reference(s1, s2, bg1, bg2, bnew, result)


def test_constructed_alpha_sweep_matches_reference():
    rng = np.random.default_rng(99)
    bg1 = (3, 250, 77)
    bg2 = (240, 5, 160)
    count = 120
    alpha = np.arange(count, dtype=np.int64)
    fg = rng.integers(0, 256, (count, 3))
    s1 = ((alpha[:, None] * fg + (255 - alpha[:, None]) * np.array(bg1)
           + 127) // 255).astype(np.uint8)
    s2 = ((alpha[:, None] * fg + (255 - alpha[:, None]) * np.array(bg2)
           + 127) // 255).astype(np.uint8)
    bnew = np.tile(np.array(bg1, dtype=np.uint8), (count, 1))
    result = solve_pixels(s1, s2, bg1, bg2, bnew)
    assert_matches_reference(s1, s2, bg1, bg2, bnew, result)


def test_alpha_zero_foreground_is_fully_free():
    # Both samples are exactly their backgrounds, so alpha 0 is feasible.
    bg1 = (30, 60, 90)
    bg2 = (90, 60, 30)
    s = (30, 60, 90)
    t = (90, 60, 30)
    result = solve_pixels([s], [t], bg1, bg2, [[10, 20, 30]])
    assert result.ambiguous[0, 0]
    assert not result.contradiction[0, 0]


def test_mixed_chunk_sizes_agree():
    rng = np.random.default_rng(77)
    h, w = 23, 17
    bg1 = (11, 22, 33)
    bg2 = (222, 111, 44)
    s1 = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    s2 = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    bnew = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    base = compose(s1, s2, bg1, bg2, bnew, chunk_rows=h)
    for chunk in (1, 2, 3, 5, 64):
        other = compose(s1, s2, bg1, bg2, bnew, chunk_rows=chunk)
        np.testing.assert_array_equal(base.image, other.image)
        np.testing.assert_array_equal(base.unique, other.unique)
        np.testing.assert_array_equal(base.contradiction, other.contradiction)
        np.testing.assert_array_equal(base.ambiguous, other.ambiguous)


def test_candidate_alpha_interval_never_excludes_feasible_alpha():
    """The analytic candidate interval must be a superset of feasible alphas."""
    from app.compose import _candidate_alphas

    rng = np.random.default_rng(5)
    s1 = rng.integers(0, 256, (400, 3), dtype=np.int64)
    s2 = rng.integers(0, 256, (400, 3), dtype=np.int64)
    bg1 = rng.integers(0, 256, 3)
    bg2 = rng.integers(0, 256, 3)
    lo, hi, ok = _candidate_alphas(s1, s2, bg1, bg2)
    for i in range(s1.shape[0]):
        for alpha in range(256):
            feasible = True
            for c in range(3):
                fg = np.arange(256, dtype=np.int64)
                o1 = ((alpha * fg + (255 - alpha) * bg1[c] + 127) // 255)
                o2 = ((alpha * fg + (255 - alpha) * bg2[c] + 127) // 255)
                if not np.any((o1 == s1[i, c]) & (o2 == s2[i, c])):
                    feasible = False
                    break
            if feasible and alpha > 0:
                assert ok[i] and lo[i] <= alpha <= hi[i], (
                    "feasible alpha excluded", i, alpha, lo[i], hi[i], ok[i])


def test_shape_validation():
    a = np.zeros((2, 3, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        compose(a, a, (0, 0, 0), (1, 1, 1), np.zeros((2, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        compose(a, a, (0, 0, 0), (1, 1, 1), np.zeros((2, 3), dtype=np.uint8))
