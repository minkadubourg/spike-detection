"""Offline and multichannel spike detection.

    raw --repair NaN--> --band-pass--> filtered --MAD threshold--> crossings
        --align to trough--> events --flags--> event table

Every event that crosses the threshold is returned, with flags explaining
why it would be rejected; `accepted` is the subset a downstream user would
keep by default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from . import signal as sig
from .config import DetectorConfig
from .events import REJECTION_FLAGS, empty_events, make_events


def ms_to_samples(ms: float, fs: float) -> int:
    return max(1, int(round(ms * 1e-3 * fs)))


def detect_rails(raw, min_run: int = 3):
    """Amplifier rails: the extreme values if the raw signal sits on them for
    at least `min_run` consecutive samples. Returns (low, high), None if absent."""
    raw = np.asarray(raw, float)
    raw = raw[np.isfinite(raw)]
    out = []
    for v in (raw.min(initial=np.inf), raw.max(initial=-np.inf)):
        at = np.concatenate([[0], (raw == v).astype(int), [0]])
        runs = np.flatnonzero(np.diff(at) == -1) - np.flatnonzero(np.diff(at) == 1)
        out.append(v if len(runs) and runs.max() >= min_run else None)
    return tuple(out)


def events_from_block(f, thr, fs, cfg: DetectorConfig, *, raw=None, nan=None, base=0,
                      n_total=None, starts_above=True, rails=(None, None), channel=0):
    """Detect and flag events in a block of filtered signal.

    `base` is the absolute index of f[0]; `n_total` the length of the whole
    recording (None if unknown yet, as in streaming). Every flag depends only on
    a bounded neighbourhood of the event (see `context_samples`), which is what
    lets the streaming detector reproduce this function exactly.
    """
    f = np.asarray(f, float)
    n = len(f)
    A = ms_to_samples(cfg.align_window_ms, fs)
    W = ms_to_samples(cfg.artifact_window_ms, fs)
    R = ms_to_samples(cfg.refractory_ms, fs)
    cap = 5 * ms_to_samples(cfg.max_width_ms, fs)

    onsets = sig.crossings(f, thr, starts_above=starts_above)
    troughs, offsets = sig.align_to_trough(f, onsets, A)
    troughs, first = np.unique(troughs, return_index=True)   # two crossings, one trough
    offsets = offsets[first]
    if len(troughs) == 0:
        return empty_events()

    amp = f[troughs]
    thr_b = np.broadcast_to(thr, f.shape)
    widths = sig.width_below(f, troughs, thr, cap)

    a = base + troughs
    edge = a < W
    if n_total is not None:
        edge |= a + W >= n_total

    # artifact tests
    # 1) polarity: a bigger positive peak nearby (skipped when the window is truncated)
    polarity = (not cfg.causal) if cfg.polarity_test is None else cfg.polarity_test
    if polarity:
        pos_peak = np.array([f[max(0, m - W):m + W + 1].max() for m in troughs])
        artifact = (pos_peak > np.abs(amp)) & ~edge
    else:
        artifact = np.zeros(len(troughs), bool)
    # 2) width: below threshold for longer than an action potential lasts
    artifact |= widths / fs * 1e3 > cfg.max_width_ms
    # 3) slow excursion: the event sits on a broad, much larger deflection of the raw signal
    #    (motion, stimulation or saturation artifacts are mostly removed by the band-pass,
    #    but their edges leave spike-sized lobes in the spike band)
    if raw is not None:
        artifact |= slow_excursion(raw, troughs, amp, fs, cfg)
    # 4) optional absolute ceiling
    if cfg.max_amplitude_uv is not None:
        artifact |= np.abs(amp) > cfg.max_amplitude_uv

    dropped = np.zeros(len(troughs), bool) if nan is None else np.asarray(nan)[troughs]

    clipped = np.zeros(len(troughs), bool)
    lo_rail, hi_rail = rails
    if cfg.clip_level_uv is not None:
        lo_rail, hi_rail = -cfg.clip_level_uv, cfg.clip_level_uv
    if raw is not None and (lo_rail is not None or hi_rail is not None):
        raw = np.asarray(raw, float)
        for i, m in enumerate(troughs):
            w = raw[max(0, m - W):m + W + 1]
            clipped[i] = (lo_rail is not None and np.any(w <= lo_rail)) or \
                         (hi_rail is not None and np.any(w >= hi_rail))

    # refractory: an event is a duplicate if a deeper candidate lies within R samples,
    # or a much deeper one within the ringing window
    # (ringing is tested against every event, artifacts included: they ring too)
    G = ms_to_samples(cfg.ringing_ms, fs)
    dup = np.zeros(len(troughs), bool)
    every = np.arange(len(troughs))
    for i in np.flatnonzero(~artifact):
        d = np.abs(troughs - troughs[i])
        for j in every[(d <= max(R, G)) & (every != i)]:
            close = d[j] <= R and not artifact[j]
            deeper = amp[j] < amp[i] or (amp[j] == amp[i] and troughs[j] < troughs[i])
            if (close and deeper) or amp[j] < cfg.ringing_ratio * amp[i]:
                dup[i] = True
                break

    return make_events(channel=np.full(len(troughs), channel), sample=a,
                       time_s=a / fs, subsample_offset=offsets, amplitude_uv=amp,
                       threshold_uv=thr_b[troughs], width_ms=widths / fs * 1e3,
                       artifact=artifact, edge=edge, dropped_sample=dropped, clipped=clipped,
                       refractory_duplicate=dup, common_artifact=np.zeros(len(troughs), bool))


def slow_excursion(raw, troughs, amp, fs, cfg: DetectorConfig):
    """True where the event lies on a broad, much larger deflection of the raw signal.

    The raw signal minus its rolling median is searched within
    ±excursion_context_ms of the event. The largest deflection counts as an
    artifact if (a) it is more than excursion_ratio times the event amplitude,
    (b) it is wider than max_width_ms at half height, and (c) the event lies
    within its extent (above a tenth of its height, ± artifact_window_ms).
    Condition (c) keeps genuine spikes that merely follow an artifact."""
    raw = np.asarray(raw, float)
    B = ms_to_samples(cfg.excursion_baseline_ms, fs) | 1
    C = ms_to_samples(cfg.excursion_context_ms, fs)
    W = ms_to_samples(cfg.artifact_window_ms, fs)
    max_w = cfg.max_width_ms * 1e-3 * fs
    out = np.zeros(len(troughs), bool)
    if len(raw) < 3:
        return out

    def run(mask, k):
        left = right = k
        while left > 0 and mask[left - 1]:
            left -= 1
        while right < len(mask) - 1 and mask[right + 1]:
            right += 1
        return left, right

    for i, m in enumerate(troughs):
        lo, hi = max(0, m - C - B), min(len(raw), m + C + B + 1)
        seg = raw[lo:hi]
        r = seg - median_filter(seg, size=B, mode="reflect")
        c0, c1 = max(0, m - C) - lo, min(len(raw), m + C + 1) - lo
        k = c0 + int(np.argmax(np.abs(r[c0:c1])))
        peak = abs(r[k])
        if peak <= cfg.excursion_ratio * abs(amp[i]):
            continue
        signed = np.sign(r[k]) * r
        l2, r2 = run(signed >= peak / 2, k)
        if r2 - l2 + 1 <= max_w:
            continue
        l4, r4 = run(signed >= peak / 10, k)
        out[i] = l4 - W <= m - lo <= r4 + W
    return out


def context_samples(cfg: DetectorConfig, fs: float) -> int:
    """Neighbourhood an event's flags depend on, on each side of its trough."""
    A = ms_to_samples(cfg.align_window_ms, fs)
    W = ms_to_samples(cfg.artifact_window_ms, fs)
    R = max(ms_to_samples(cfg.refractory_ms, fs), ms_to_samples(cfg.ringing_ms, fs))
    cap = 5 * ms_to_samples(cfg.max_width_ms, fs)
    C = ms_to_samples(cfg.excursion_context_ms, fs) + ms_to_samples(cfg.excursion_baseline_ms, fs)
    return A + W + R + max(cap, C) + 2


def preprocess(voltage, fs, cfg: DetectorConfig):
    """Gap repair, band-pass and threshold. Returns (repaired, filtered, threshold, nan mask)."""
    repaired, nan = sig.repair_gaps(voltage, causal=cfg.causal)
    f = sig.bandpass(repaired, fs, cfg.band_hz, cfg.filter_order, causal=cfg.causal)
    thr = sig.threshold_trace(f, fs, cfg.k, cfg.noise_window_s, cfg.noise_calibration_s)
    return repaired, f, thr, nan


def detect(voltage, fs: float | None = None, time_s=None, config: DetectorConfig | None = None,
           channel: int = 0) -> pd.DataFrame:
    """Detect spikes in one channel. Give `fs`, or `time_s` to infer it."""
    cfg = config or DetectorConfig()
    voltage = np.asarray(voltage, float)
    if len(voltage) == 0:
        return empty_events()
    if fs is None:
        if time_s is None:
            raise ValueError("give fs or time_s")
        fs = sig.infer_fs(time_s)
    if np.isnan(voltage).all() or len(voltage) < 4 * (2 * cfg.filter_order + 1):
        return empty_events()
    repaired, f, thr, nan = preprocess(voltage, fs, cfg)
    rails = detect_rails(voltage) if cfg.clip_level_uv is None else (None, None)
    ev = events_from_block(f, thr, fs, cfg, raw=repaired, nan=nan, n_total=len(voltage),
                           rails=rails, channel=channel)
    if time_s is not None and len(ev):
        ev["time_s"] = np.asarray(time_s, float)[ev["sample"].to_numpy()]
    return ev


def flag_common_artifacts(events: pd.DataFrame, n_channels: int, fs: float,
                          coincidence_ms: float = 0.5, min_fraction: float = 0.75) -> pd.DataFrame:
    """Mark events that occur within ±coincidence_ms on at least `min_fraction`
    of the channels: a spike is local to a few electrodes, a shared artifact
    (reference glitch, motion) hits all of them at once."""
    ev = events.copy()
    if ev.empty or n_channels < 2:
        return ev
    tol = ms_to_samples(coincidence_ms, fs)
    need = int(np.ceil(min_fraction * n_channels))
    s, ch = ev["sample"].to_numpy(), ev["channel"].to_numpy()
    order = np.argsort(s, kind="stable")
    s_sorted, ch_sorted = s[order], ch[order]
    common = np.zeros(len(ev), bool)
    for i in range(len(ev)):
        lo = np.searchsorted(s_sorted, s[i] - tol, side="left")
        hi = np.searchsorted(s_sorted, s[i] + tol, side="right")
        common[i] = len(set(ch_sorted[lo:hi])) >= need
    ev["common_artifact"] = common
    ev["accepted"] = ~ev[REJECTION_FLAGS].any(axis=1)
    return ev


def detect_multichannel(voltages, fs: float, config: DetectorConfig | None = None,
                        coincidence_ms: float = 0.5, min_fraction: float = 0.75) -> pd.DataFrame:
    """Detect on each row of a (channels, samples) array, then flag events that
    are coincident across most channels."""
    voltages = np.atleast_2d(np.asarray(voltages, float))
    parts = [detect(v, fs, config=config, channel=c) for c, v in enumerate(voltages)]
    ev = pd.concat([p for p in parts if len(p)] or [empty_events()], ignore_index=True)
    ev = ev.astype(empty_events().dtypes.to_dict())
    return flag_common_artifacts(ev, len(voltages), fs, coincidence_ms, min_fraction)
