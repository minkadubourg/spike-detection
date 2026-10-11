"""Signal-processing stages: gap repair, filtering, noise estimation, threshold
crossings and trough alignment. Each stage is a pure function of arrays."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt

MAD_TO_SIGMA = 1 / 0.6745   # sigma of Gaussian noise = MAD / 0.6745


def infer_fs(time_s) -> float:
    """Sampling rate from timestamps (median step, robust to jitter and gaps)."""
    time_s = np.asarray(time_s, float)
    if len(time_s) < 2:
        raise ValueError("need at least two timestamps to infer the sampling rate")
    return float(1.0 / np.median(np.diff(time_s)))


def repair_gaps(x, causal=False, last_valid=0.0):
    """Fill NaN samples so the filter can run. Returns (repaired, NaN mask).

    Offline: linear interpolation between the neighbouring valid samples.
    Causal: hold the last valid value (the next one is not known yet);
    `last_valid` carries that value across blocks. All-NaN input gives zeros."""
    x = np.asarray(x, float)
    nan = np.isnan(x)
    if not nan.any():
        return x, nan
    if causal:
        idx = np.where(~nan, np.arange(len(x)), -1)
        idx = np.maximum.accumulate(idx)
        out = np.where(idx >= 0, x[np.maximum(idx, 0)], last_valid)
        return out, nan
    if nan.all():
        return np.zeros_like(x), nan
    idx = np.arange(len(x))
    out = x.copy()
    out[nan] = np.interp(idx[nan], idx[~nan], x[~nan])
    return out, nan


def design_bandpass(fs, band_hz, order):
    lo, hi = band_hz
    hi = min(hi, 0.45 * fs)
    if not 0 < lo < hi:
        raise ValueError(f"invalid band {band_hz} for fs = {fs} Hz")
    return butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")


def bandpass(x, fs, band_hz=(300.0, 3500.0), order=4, causal=False):
    """Zero-phase (sosfiltfilt) or causal (sosfilt, initial state at rest) band-pass."""
    sos = design_bandpass(fs, band_hz, order)
    x = np.asarray(x, float)
    if causal:
        return sosfilt(sos, x)
    padlen = min(3 * (2 * len(sos) + 1), len(x) - 1)
    return sosfiltfilt(sos, x, padlen=max(padlen, 0))


def mad_sigma(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(np.median(np.abs(x - np.median(x))) * MAD_TO_SIGMA)


def threshold_trace(x, fs, k, window_s=None, calibration_s=None):
    """Per-sample threshold  median - k * sigma  (sigma from the MAD).

    * default: one estimate from the whole recording;
    * ``calibration_s``: one estimate from the first seconds only (what a
      streaming detector can know), applied everywhere;
    * ``window_s``: one estimate per consecutive window, so the threshold
      follows slow changes in noise such as a degrading electrode. A last
      window shorter than half a window is merged into the previous one.
    """
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return np.zeros(0)

    def level(seg):
        s = mad_sigma(seg)
        return np.median(seg) - k * s if s > 0 else -np.inf   # flat signal: nothing can cross

    if calibration_s is not None:
        return np.full(n, level(x[: max(2, int(round(calibration_s * fs)))]))
    if window_s is None or n < 2 * int(round(window_s * fs)):
        return np.full(n, level(x))
    w = int(round(window_s * fs))
    edges = list(range(0, n, w)) + [n]
    if edges[-1] - edges[-2] < w // 2:
        edges.pop(-2)
    out = np.empty(n)
    for s, e in zip(edges[:-1], edges[1:]):
        out[s:e] = level(x[s:e])
    return out


def crossings(x, threshold, starts_above=True):
    """Indices where x goes from above to at-or-below the threshold.

    The sample before the first one is assumed above the threshold, so a
    recording that starts inside a spike still yields an onset at index 0
    (pass ``starts_above=False`` when x continues an earlier block)."""
    below = np.asarray(x) <= threshold
    if len(below) == 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.flatnonzero(below[1:] & ~below[:-1]) + 1
    if starts_above and below[0]:
        idx = np.concatenate([[0], idx])
    return idx


def align_to_trough(x, onsets, window):
    """For each crossing, the index of the minimum in [onset, onset + window),
    and the sub-sample offset of the parabola through the three samples around it."""
    x = np.asarray(x, float)
    n = len(x)
    troughs = np.empty(len(onsets), dtype=np.int64)
    offsets = np.zeros(len(onsets))
    for i, s in enumerate(onsets):
        e = min(n, s + window)
        m = s + int(np.argmin(x[s:e]))
        troughs[i] = m
        if 0 < m < n - 1:
            y0, y1, y2 = x[m - 1], x[m], x[m + 1]
            d = y0 - 2 * y1 + y2
            offsets[i] = 0.5 * (y0 - y2) / d if d > 0 else 0.0
    return troughs, offsets


def width_below(x, troughs, threshold, cap):
    """Number of consecutive samples at or below the threshold around each
    trough, searched at most `cap` samples on each side (so the result only
    depends on a bounded neighbourhood, which streaming relies on)."""
    x = np.asarray(x, float)
    thr = np.broadcast_to(threshold, x.shape)
    widths = np.zeros(len(troughs), dtype=np.int64)
    n = len(x)
    for i, m in enumerate(troughs):
        lo = m
        while lo > 0 and m - lo < cap and x[lo - 1] <= thr[lo - 1]:
            lo -= 1
        hi = m
        while hi < n - 1 and hi - m < cap and x[hi + 1] <= thr[hi + 1]:
            hi += 1
        widths[i] = hi - lo + 1
    return widths
