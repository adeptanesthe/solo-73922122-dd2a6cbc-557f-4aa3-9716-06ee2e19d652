"""Exact alpha/foreground analysis and background replacement.

Model
-----
Both registered samples are composites of the *same* frame over two solid
backgrounds::

    sample_c = round((A * F_c + (255 - A) * background_c) / 255)

where ``A`` is one 8-bit alpha shared by the three channels of a pixel,
``F_c`` is the 8-bit foreground channel and the division is rounded half up
(``(x + 127) // 255``).

For one channel and one sample the rounded value ``s`` over background ``b``
means::

    255*s - 127 <= A*F + (255 - A)*b <= 255*s + 127
    <=>  g(A) - 127 <= A*F <= g(A) + 127   with g(A) = 255*(s - b) + A*b

For two samples the allowed ``A*F`` window is the intersection of two such
windows.  For a fixed ``A`` the allowed foreground values are therefore one
integer interval, and the resulting output over the new background is also
one integer interval.  A pixel is feasible when, for at least one ``A``, all
three channels have a non-empty foreground interval.  Its result is unique
exactly when, over every feasible ``A``, each output channel has a single
value (the channel minima and maxima coincide).

The implementation never materialises ``height * width * 256`` candidates:

* the candidate alpha interval of every pixel is derived analytically from
  linear inequalities, which keeps memory at a handful of 8/16-bit planes;
* exact alpha checks are done one alpha at a time on row chunks;
* pixels are dropped as soon as non-uniqueness is proven, so wide candidate
  ranges do not have to be scanned to the end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Rounding table for round(x / 255) with round-half-up, x in [0, 65025].
MAX_BLEND = 255 * 255
_ROUND_LUT = ((np.arange(MAX_BLEND + 1, dtype=np.int64) + 127) // 255).astype(np.uint8)

# Candidate alphas are the 8-bit values 1..255; alpha 0 is handled separately
# because the foreground colour is unobservable there.
A_MIN = 1
A_MAX = 255


def blend(foreground, alpha, background):
    """Reference scalar blend used by tests and examples.

    ``foreground``/``background`` are RGB triples, ``alpha`` is an int.
    Returns an RGB tuple rounded half up, matching the vectorised solver.
    """
    fg = np.asarray(foreground, dtype=np.int64)
    bg = np.asarray(background, dtype=np.int64)
    a = int(alpha)
    if a < 0 or a > 255:
        raise ValueError("alpha must be in [0, 255]")
    if np.any((fg < 0) | (fg > 255)) or np.any((bg < 0) | (bg > 255)):
        raise ValueError("colour channels must be in [0, 255]")
    return tuple(int(v) for v in _ROUND_LUT[a * fg + (255 - a) * bg])


@dataclass
class ComposeResult:
    """Result planes, all of shape (height, width)."""

    image: np.ndarray  # (H, W, 3) uint8, meaningful where unique
    unique: np.ndarray  # (H, W) bool
    contradiction: np.ndarray  # (H, W) bool: no (F, A) reproduces both samples
    ambiguous: np.ndarray  # (H, W) bool: feasible but new RGB is not unique

    @property
    def all_unique(self) -> bool:
        return bool(self.unique.all())


def _ceil_div(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Element-wise ceil(p / q) for q != 0, matching Python floor semantics."""
    return -((-p) // q)


def _apply_le(a, b, lo, hi, ok):
    """Intersect x with ``a*x <= b``.  ``lo``/``hi``/``ok`` are updated in place."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    safe = np.where(a == 0, 1, a)
    pos = a > 0
    neg = a < 0
    zero = ~(pos | neg)
    hi = np.where(pos, np.minimum(hi, b // safe), hi)
    lo = np.where(neg, np.maximum(lo, _ceil_div(b, safe)), lo)
    ok &= ~zero | (b >= 0)
    return lo, hi, ok


def _apply_ge(a, b, lo, hi, ok):
    """Intersect x with ``a*x >= b``.  ``lo``/``hi``/``ok`` are updated in place."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    safe = np.where(a == 0, 1, a)
    pos = a > 0
    neg = a < 0
    zero = ~(pos | neg)
    lo = np.where(pos, np.maximum(lo, _ceil_div(b, safe)), lo)
    hi = np.where(neg, np.minimum(hi, b // safe), hi)
    ok &= ~zero | (b <= 0)
    return lo, hi, ok


def _candidate_alphas(s1, s2, bg1, bg2):
    """Analytic alpha interval [lo, hi] per pixel for A in [1, 255].

    ``s1``/``s2`` are (N, 3) integer arrays, backgrounds are length-3.
    Returns int64 arrays ``lo``, ``hi`` and a boolean ``ok``.  These are only
    necessary conditions (the existence of an integer foreground is checked
    per alpha later), so they never exclude a feasible alpha.
    """
    n = s1.shape[0]
    lo = np.full(n, A_MIN, dtype=np.int64)
    hi = np.full(n, A_MAX, dtype=np.int64)
    ok = np.ones(n, dtype=bool)

    for c in range(3):
        s1c = s1[:, c]
        s2c = s2[:, c]
        b1 = int(bg1[c])
        b2 = int(bg2[c])
        # |g1(A) - g2(A)| <= 254  <=>  both differences <= 254.
        d = b1 - b2
        lo, hi, ok = _apply_le(d, 254 - 255 * s1c + 255 * s2c + 255 * d, lo, hi, ok)
        lo, hi, ok = _apply_le(-d, 254 - 255 * s2c + 255 * s1c - 255 * d, lo, hi, ok)
        # g1, g2 >= -127  (so the upper end of the window is non-negative).
        lo, hi, ok = _apply_ge(b1, -127 - 255 * s1c + 255 * b1, lo, hi, ok)
        lo, hi, ok = _apply_ge(b2, -127 - 255 * s2c + 255 * b2, lo, hi, ok)
        # g1, g2 <= 255*A + 127  (so the lower end fits below alpha * 255).
        lo, hi, ok = _apply_le(b1 - 255, 127 - 255 * s1c + 255 * b1, lo, hi, ok)
        lo, hi, ok = _apply_le(b2 - 255, 127 - 255 * s2c + 255 * b2, lo, hi, ok)

    ok &= lo <= hi
    np.clip(lo, A_MIN, A_MAX, out=lo)
    np.clip(hi, A_MIN, A_MAX, out=hi)
    ok &= lo <= hi
    return lo, hi, ok


def _outputs_for_alpha(s1, s2, bnew, bg1, bg2, alpha):
    """Exact feasibility and output bounds for one alpha on (m, 3) pixels.

    Returns ``feasible`` (m,), ``out_min`` (m, 3) uint8 and ``out_max`` (m, 3)
    uint8.  Values are arbitrary where ``feasible`` is false.
    """
    b1 = np.asarray(bg1, dtype=np.int64)
    b2 = np.asarray(bg2, dtype=np.int64)
    bnew = np.asarray(bnew, dtype=np.int64)
    s1 = np.asarray(s1, dtype=np.int64)
    s2 = np.asarray(s2, dtype=np.int64)
    g1 = alpha * b1 + 255 * (s1 - b1)
    g2 = alpha * b2 + 255 * (s2 - b2)
    lower = np.maximum(g1, g2) - 127
    upper = np.minimum(g1, g2) + 127
    np.maximum(lower, 0, out=lower)
    np.minimum(upper, 255 * alpha, out=upper)
    f_min = _ceil_div(lower, np.int64(alpha))
    f_max = upper // alpha
    feasible = np.all(f_min <= f_max, axis=1)
    # Only feasible rows have bounded intervals (0 <= f_min <= f_max <= 255);
    # clamp the rest so the lookup-table index stays in range.
    safe_min = np.where(feasible[:, None], f_min, 0)
    safe_max = np.where(feasible[:, None], f_max, 0)
    out_min = _ROUND_LUT[alpha * safe_min + (255 - alpha) * bnew]
    out_max = _ROUND_LUT[alpha * safe_max + (255 - alpha) * bnew]
    return feasible, out_min, out_max


def compose(
    s1: np.ndarray,
    s2: np.ndarray,
    bg1,
    bg2,
    bnew: np.ndarray,
    chunk_rows: int | None = None,
    chunk_pixels: int = 128 * 1024,
) -> ComposeResult:
    """Analyse every pixel and compose over the new background.

    Parameters
    ----------
    s1, s2 : (H, W, 3) uint8 arrays
        The two samples, same size.
    bg1, bg2 : sequence of 3 ints
        Solid backgrounds of the two samples.
    bnew : (H, W, 3) uint8 array
        New background, same size as the samples.
    chunk_rows : int or None
        Rows processed together.  When omitted, rows are grouped so that a
        chunk holds about ``chunk_pixels`` pixels.  Peak memory is a handful
        of planes of that size; alpha candidates are never expanded for the
        whole image.
    chunk_pixels : int
        Target pixels per chunk when ``chunk_rows`` is not given.
    """
    if s1.shape != s2.shape or s1.shape != bnew.shape:
        raise ValueError("samples and new background must have the same shape")
    if s1.ndim != 3 or s1.shape[2] != 3:
        raise ValueError("expected HxWx3 RGB arrays")
    h, w, _ = s1.shape
    bg1 = np.asarray(bg1, dtype=np.int64).reshape(3)
    bg2 = np.asarray(bg2, dtype=np.int64).reshape(3)
    if chunk_rows is None:
        chunk_rows = max(1, chunk_pixels // max(1, w))

    image = np.zeros((h, w, 3), dtype=np.uint8)
    unique = np.zeros((h, w), dtype=bool)
    contradiction = np.zeros((h, w), dtype=bool)
    ambiguous = np.zeros((h, w), dtype=bool)

    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        rows = y1 - y0
        count = rows * w
        s1c = s1[y0:y1].reshape(count, 3).astype(np.int64)
        s2c = s2[y0:y1].reshape(count, 3).astype(np.int64)
        bnc = bnew[y0:y1].reshape(count, 3).astype(np.int64)

        cand_lo, cand_hi, cand_ok = _candidate_alphas(s1c, s2c, bg1, bg2)

        # Alpha 0: foreground is invisible, so the samples must be exactly the
        # two backgrounds and the result is exactly the new background.
        a0 = np.all(s1c == bg1, axis=1) & np.all(s2c == bg2, axis=1)

        out_lo = np.full((count, 3), 255, dtype=np.uint8)
        out_hi = np.zeros((count, 3), dtype=np.uint8)
        feasible = a0.copy()
        out_lo[a0] = bnc[a0]
        out_hi[a0] = bnc[a0]

        pending = cand_ok.copy()
        for alpha in range(A_MIN, A_MAX + 1):
            if not pending.any():
                break
            selected = pending & (cand_lo <= alpha) & (alpha <= cand_hi)
            if not selected.any():
                continue
            idx = np.flatnonzero(selected)
            feas, lo_vals, hi_vals = _outputs_for_alpha(
                s1c[idx], s2c[idx], bnc[idx], bg1, bg2, alpha
            )
            if not feas.any():
                continue
            good = idx[feas]
            new_lo = lo_vals[feas]
            new_hi = hi_vals[feas]
            out_lo[good] = np.minimum(out_lo[good], new_lo)
            out_hi[good] = np.maximum(out_hi[good], new_hi)
            feasible[good] = True
            # Once any channel differs between its minimum and maximum, at
            # least two distinct output RGB triples are producible.  Such a
            # pixel can never become unique again, so stop scanning it.
            non_unique = np.any(out_lo[good] != out_hi[good], axis=1)
            if non_unique.any():
                pending[good[non_unique]] = False

        amb = feasible & np.any(out_lo != out_hi, axis=1)
        uniq = feasible & ~amb
        contra = ~feasible

        image[y0:y1] = np.where(uniq[:, None], out_lo, 0).reshape(rows, w, 3)
        unique[y0:y1] = uniq.reshape(rows, w)
        contradiction[y0:y1] = contra.reshape(rows, w)
        ambiguous[y0:y1] = amb.reshape(rows, w)

    return ComposeResult(
        image=image,
        unique=unique,
        contradiction=contradiction,
        ambiguous=ambiguous,
    )
